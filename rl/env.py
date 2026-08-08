"""FLArmsRaceEnv — the federated-learning environment for the RL arms race.

Holds the global model, the per-client benign updates for the current round,
and the FedAvg aggregator + evaluation oracle. It deliberately knows nothing
about LLMs: it consumes already-parsed poisoned state_dicts and detection
verdicts, and exposes ground truth (the per-round poisoned set) for reward
computation.

Round protocol (driven by the schedule / inference loop):

    env.reset(global, client_weights, baseline_acc)
    ctx = env.begin_round()                 # fixes this round's defense algorithm + poison
                                            # quota (from the TrainingCurriculum, or drawn),
                                            # builds honest updates, and exposes the
                                            # attacker's controllable pool (ctx.pool_benign)
    # the attacker SELECTS exactly ctx.budget clients from the pool and poisons them
    updates = env.build_updates(poisoned_by_client)
    acc = env.evaluate_updates(updates, verdicts)   # no commit (used to score rollouts)
    ...
    env.set_committed_poison(chosen_ids)            # record the committed poison set
    new_acc = env.commit(updates, verdicts)         # measure (and, if not frozen, install)

SIMULATED (frozen-anchor) ROUNDS — ``fl.freeze_global_in_phase2: true``, the
default. Phase 2 does not run a continuing federation; it runs independent
attacker-learning episodes that all branch off the SAME Phase-1 final model:

    for each round:
        the frozen Phase-1 global is sent to every client
        every client trains on NEW local data  (data.round_sampler)
        the attacker picks `budget` of its pool and poisons them
        server aggregates poisoned + honest -> candidate global
        that candidate is evaluated on the test set  -> reward -> GRPO
        the candidate is DISCARDED; the next round starts from the anchor again

so ``commit`` measures the round without advancing the model, and the global the
clients see never drifts. Set ``freeze_global_in_phase2: false`` to restore the
original continuing federation, where each committed aggregate becomes the next
round's starting point.
"""

import copy
import logging

from clients.benign_client import BenignClient
from core.types import ModelUpdate, DetectionVerdict
from detector.features import compute_client_features
from server.aggregation import FedAvgAggregator
from server.fed_server import FedServer

logger = logging.getLogger(__name__)


def _accepts_honest_majority(verdicts) -> bool:
    """Did the defense keep more than half of an entirely HONEST cohort?

    Applied to the clean counterfactual's verdicts, where every update is honest by
    construction, so anything it flags is a false positive. A robust aggregator that
    rejects the honest majority of an unpoisoned cohort is misconfigured, and every
    number the round produces afterwards — the counterfactual accuracy, the post-attack
    accuracy, the drop, the reward — describes that malfunction rather than the
    attacker (see ``server.algo_defender.resolve_root_epochs`` for the mechanism, and
    ``rl.schedule._warn_defense_malfunction`` for the post-hoc detector this
    complements).

    Testing it on the CLEAN updates rather than the poisoned ones matters: it makes
    the check independent of what the attacker did, so a genuinely strong attack that
    provokes lots of flags can never be mistaken for a broken defense, and the verdict
    is available before the rollouts are scored.
    """
    if not verdicts:
        return True
    accepted = sum(1 for v in verdicts if not v.is_suspicious)
    return accepted * 2 > len(verdicts)


class RoundContext:
    """Per-round observation handed to the agents.

    The attacker controls a fixed ``pool`` of clients and must poison exactly
    ``budget`` of them; it CHOOSES which (see ``AttackerAgent.select_and_apply``),
    so ``poisoned_ids`` is empty here and filled in once the choice is committed.
    """

    def __init__(self, round_num, global_accuracy, pool_ids, pool_benign, budget,
                 goal=None, clean_accuracy=None, clean_measured=True,
                 defense_sane=True):
        self.round_num = round_num
        self.global_accuracy = global_accuracy            # accuracy of the CURRENT global model
        self.pool_ids = pool_ids                          # list[int] controllable pool
        self.pool_benign = pool_benign                    # {cid: state_dict} for the pool
        self.budget = budget                              # exact poison-client quota
        self.goal = goal                                  # this round's attack goal (maybe sampled)
        self.poisoned_ids = []                            # set at commit (attacker's choice)
        # The clean counterfactual: what this round's aggregate scores with NO
        # poison. This — not ``global_accuracy`` — is what the attacker's damage
        # is measured against (see ``FLArmsRaceEnv.clean_reference_accuracy``).
        self.clean_accuracy = clean_accuracy
        # False when the defense produced NO clean aggregate, so the counterfactual
        # could not actually be measured and ``clean_accuracy`` is only a fallback.
        # The attacker's damage term is meaningless on such a round — see
        # ``FLArmsRaceEnv.clean_reference_measured``.
        self.clean_measured = clean_measured
        # False when the round's defense rejected the honest MAJORITY of an
        # entirely unpoisoned cohort. It aggregated something, so the numbers look
        # ordinary, but they measure the defense malfunctioning — see
        # ``FLArmsRaceEnv.clean_defense_sane``.
        self.defense_sane = defense_sane


class FLArmsRaceEnv:
    def __init__(self, config: dict, client_loaders, test_loader, rng, defense=None,
                 curriculum=None, round_data=None):
        """``defense`` is an optional :class:`server.algo_defender.AlgorithmicDefender`.

        When present the defender LLM is disabled and the server side of every
        round is run by one published defense algorithm, selected per round in
        :meth:`begin_round` and used for the clean counterfactual, every scored
        rollout and the commit. That algorithm also produces the round's
        AGGREGATE (FLTrust re-weights and rescales, DeFL Beta-weights, ...), so
        callers use :meth:`defend` + :meth:`evaluate_state` / :meth:`commit_state`
        instead of the verdict-driven :meth:`evaluate_updates` / :meth:`commit`.
        ``defense=None`` keeps the original FedAvg-over-unflagged path.

        ``curriculum`` is an optional :class:`rl.curriculum.TrainingCurriculum`.
        When present it REPLACES the two per-round random draws — the defense
        algorithm and the exact poison quota — with a deterministic sweep that
        holds one (algorithm, #poisoners) pair for a whole block of consecutive
        rounds, so every pair gets an equal, contiguous share of training. The
        goal/target draw is unaffected (see :meth:`_round_goal`).

        ``round_data`` is an optional :class:`data.round_sampler.RoundDataSampler`.
        When present, every client is handed a fresh slice of its own shard at the
        start of each round — the "clients train with new data" step of a simulated
        round. Without it the clients replay their whole fixed shard every round.
        """
        fl = config["fl"]
        attack = config.get("attack", {})
        self.n_clients = int(fl["n_clients"])
        self.device = fl.get("device", "cpu")
        self.benign_retrain = bool(fl.get("benign_retrain_each_round", True))
        self.training_rounds = int(fl.get("training_rounds", 0))
        # SIMULATED ROUNDS. Phase 2 does not continue the federation: every round is
        # an independent episode branching off the frozen Phase-1 model. The round's
        # aggregate is measured (that measurement IS the attacker's reward) and then
        # discarded, so round n+1 sends the clients the same anchor as round n.
        #
        # Why: in a continuing federation the attacker's own damage becomes the next
        # round's starting point, so what a plan is worth depends on the wreckage
        # left by every plan before it — the same attack scores differently
        # depending on run history, and the reward drifts with the environment
        # rather than measuring the action. Freezing the anchor makes each round a
        # controlled experiment against a fixed reference, which is what GRPO's
        # damage term (clean_reference_accuracy - post_accuracy) assumes.
        self.freeze_global = bool(fl.get("freeze_global_in_phase2", True))
        # A frozen anchor plus frozen client weights would make every round
        # byte-identical, so the retrain step is not optional here — it is the
        # "clients train with new data" half of the round.
        if self.freeze_global and not self.benign_retrain:
            logger.warning(
                "fl.benign_retrain_each_round is false but Phase 2 runs frozen "
                "simulated rounds — replaying frozen Phase-1 client weights against a "
                "frozen global makes every round identical. Forcing benign retraining on."
            )
            self.benign_retrain = True
        # Per-round local data for the clients (None = replay the whole fixed shard).
        self.round_data = round_data
        self.goal = attack.get(
            "goal", {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}
        )
        # Attacker is a partial insider: it may only touch clients [0 .. n_compromisable).
        self.n_compromisable = max(1, min(
            int(fl.get("n_compromisable", self.n_clients)), self.n_clients))
        # Per-round poison budget (exact number of clients to poison). Training
        # randomizes it in [1, budget_cap]; eval fixes it (set sample_budget=False,
        # budget_cap=<desired>). These attributes are overridable by the benchmark.
        self.budget_cap = max(1, min(
            int(attack.get("max_poison_clients", self.n_compromisable)), self.n_compromisable))
        self.sample_budget = bool(attack.get("sample_budget_in_training", True))
        # Per-round attack-goal target sampling (untargeted_degrade): when on, draw
        # target_accuracy_drop from target_choices each round so the policy generalizes
        # across targets instead of overfitting one. Eval fixes it (sample_target=False ->
        # the configured attack.goal). Overridable by the benchmark (env.sample_target).
        self.sample_target = bool(attack.get("sample_target_in_training", False))
        self.target_choices = [float(x) for x in attack.get(
            "target_choices", [0.05, 0.10, 0.20, 0.30])]

        self.test_loader = test_loader
        self.rng = rng
        self.aggregator = FedAvgAggregator()
        self.server = FedServer(device=self.device)
        # Algorithmic (non-LLM) defense, or None when the defender LLM defends.
        self.defense = defense
        # Deterministic (algorithm, #poisoners) sweep, or None for random draws.
        self.curriculum = curriculum

        # Benign clients (used only when benign_retrain is True).
        self._clients = [
            BenignClient(
                client_id=i,
                data_loader=client_loaders[i],
                lr=float(fl["lr"]),
                local_epochs=int(fl["local_epochs"]),
                device=self.device,
            )
            for i in range(self.n_clients)
        ] if client_loaders is not None else None

        # Set by reset().
        self.client_weights: list[dict] = []
        self.baseline_accuracy: float = 0.0
        self.current_accuracy: float = 0.0
        self.round_index: int = 0
        # The Phase-1 final model. In frozen mode this is what every simulated round
        # sends to the clients, and what :meth:`commit_state` rewinds to.
        self._anchor_weights: dict | None = None
        # The most recent COMMITTED round's post-aggregation accuracy. In frozen mode
        # this is the only place it survives, since it never becomes the global's.
        self.last_round_accuracy: float = 0.0

        # Set by begin_round().
        self.honest_updates: list[ModelUpdate] = []
        self.pool_ids: list[int] = []
        self.pool_benign: dict[int, dict] = {}
        self.round_budget: int = 1
        self.round_goal: dict = dict(self.goal)           # this round's (maybe sampled) goal
        self.poisoned_ids: list[int] = []                 # attacker's committed choice
        self._clean_ref_acc: float | None = None          # cached per-round clean counterfactual
        self._clean_ref_measured: bool = False            # was that counterfactual real?
        self._clean_defense_sane: bool = True             # did it keep the honest majority?
        self.round_defense: str | None = None             # this round's defense algorithm (if any)
        self.round_curriculum = None                      # this round's CurriculumSlot (if any)

    # ------------------------------------------------------------------
    def reset(self, global_weights, client_weights, baseline_accuracy):
        self.server.set_global_weights(copy.deepcopy(global_weights))
        # Keep an untouchable copy: in frozen mode this is the model every simulated
        # round hands the clients, so it must survive whatever a round aggregates.
        self._anchor_weights = copy.deepcopy(global_weights)
        self.client_weights = [copy.deepcopy(w) for w in client_weights]
        self.baseline_accuracy = float(baseline_accuracy)
        self.current_accuracy = float(baseline_accuracy)
        self.last_round_accuracy = float(baseline_accuracy)
        self.round_index = 0
        budget_src = ("curriculum" if self.curriculum is not None
                      else ("sampled" if self.sample_budget else "fixed"))
        logger.info(
            f"Env reset — n_clients={self.n_clients}, n_compromisable={self.n_compromisable}, "
            f"budget_cap={self.budget_cap}, budget={budget_src}, "
            f"benign_retrain={self.benign_retrain}, baseline_acc={baseline_accuracy:.4f}"
        )
        if self.freeze_global:
            logger.info(
                "Phase 2 = SIMULATED rounds on the frozen Phase-1 global: every round "
                "restarts the clients from this anchor (acc=%.4f), scores the round's "
                "aggregate, then discards it. The between-phase benign FL round is "
                "disabled — there is no shared FL state to advance.",
                self.baseline_accuracy,
            )
        else:
            logger.info("Phase 2 = CONTINUING federation: each committed aggregate "
                        "becomes the next round's global (freeze_global_in_phase2: false)")

    # ------------------------------------------------------------------
    def snapshot_fl_state(self) -> dict:
        """Serializable snapshot of the LIVE shared FL state — the evolving global
        model, the current per-client benign weights, the running accuracy, and the
        FL round counter — for a faithful Phase-2 resume.

        Without this, a resume rewinds the shared model to the Phase-1 baseline
        (``reset``) while the adapters + round counters continue, so all attacker
        damage / defender recovery accumulated so far is silently erased.
        ``baseline_accuracy`` is intentionally NOT saved here — it is the fixed
        Phase-1 reference and is re-supplied by ``reset``.
        """
        return {
            "global_weights": copy.deepcopy(self.server.get_global_weights()),
            "client_weights": [copy.deepcopy(w) for w in self.client_weights],
            "current_accuracy": float(self.current_accuracy),
            "round_index": int(self.round_index),
        }

    def restore_fl_state(self, state: dict) -> None:
        """Restore a snapshot from :meth:`snapshot_fl_state` (called on resume,
        AFTER ``reset``, so the shared model continues from the checkpoint instead
        of the Phase-1 baseline). ``round_index`` is restored here too, though the
        driver also re-supplies it from the progress file."""
        self.server.set_global_weights(copy.deepcopy(state["global_weights"]))
        self.client_weights = [copy.deepcopy(w) for w in state["client_weights"]]
        self.current_accuracy = float(state["current_accuracy"])
        self.round_index = int(state.get("round_index", self.round_index))
        if self.freeze_global and self._anchor_weights is not None:
            # Frozen mode has no drifting global to restore, and a checkpoint written
            # BEFORE the flag was turned on holds one that did drift — re-assert the
            # Phase-1 anchor so the resumed rounds start where fresh ones do.
            self.server.set_global_weights(copy.deepcopy(self._anchor_weights))
            self.current_accuracy = self.baseline_accuracy
        self._clean_ref_acc = None        # stale: the shared model just changed
        self._clean_ref_measured = False
        self._clean_defense_sane = True
        logger.info(
            f"Restored Phase-2 FL state — round_index={self.round_index}, "
            f"current_accuracy={self.current_accuracy:.4f} "
            f"(baseline stays {self.baseline_accuracy:.4f})"
        )

    # ------------------------------------------------------------------
    def _round_budget(self, slot=None) -> int:
        """This round's exact poison quota.

        A curriculum ``slot`` pins it to the block's count (clamped to the
        controllable pool, which the attacker's selection clamps to anyway).
        Otherwise it is drawn in [1, cap] when ``sample_budget`` is on, or fixed
        at the cap (evaluation).
        """
        if slot is not None:
            return max(1, min(int(slot.n_poisoners), self.n_compromisable))
        if self.sample_budget:
            return self.rng.randint(1, self.budget_cap)
        return self.budget_cap

    def _round_defense(self, slot=None) -> str | None:
        """This round's defense algorithm, or ``None`` when the defender LLM defends.

        A curriculum ``slot`` pins it to the block's algorithm (via
        ``AlgorithmicDefender.select``, so the defender's ``current`` stays in
        sync); otherwise the defender draws one per ``defense.selection``.
        """
        if self.defense is None:
            return None
        if slot is not None and slot.algorithm is not None:
            return self.defense.select(slot.algorithm)
        return self.defense.choose()

    def _round_goal(self) -> dict:
        """This round's attack goal. When target sampling is on (untargeted_degrade),
        draw ``target_accuracy_drop`` from ``target_choices`` so the policy becomes
        TARGET-AWARE and generalizes across targets; otherwise the fixed config goal.
        The returned dict is the CLEAN goal shown to the LLM and used by the reward."""
        if self.sample_target and self.goal.get("type") == "untargeted_degrade":
            return {"type": "untargeted_degrade",
                    "target_accuracy_drop": self.rng.choice(self.target_choices)}
        return self.goal

    def _honest_update(self, cid: int) -> ModelUpdate:
        if self.benign_retrain and self._clients is not None:
            return self._clients[cid].train(self.server.model)
        # Replay frozen Phase-1 weights.
        return ModelUpdate(client_id=cid, weights=copy.deepcopy(self.client_weights[cid]))

    def _refresh_client_data(self, round_index: int) -> None:
        """Hand every client its data for this round — a fresh slice of its OWN shard.

        This is the step that makes consecutive simulated rounds differ: the global
        they start from is frozen, so without new local data every round would
        reproduce the same honest updates. No-op when no sampler is configured
        (``fl.client_data_refresh: none``), where the clients keep their full shard.
        """
        if self.round_data is None or self._clients is None:
            return
        for client, loader in zip(self._clients,
                                  self.round_data.loaders_for_round(round_index)):
            client.set_data_loader(loader)

    def begin_round(self) -> RoundContext:
        """Produce this round's honest updates and expose the attacker's controllable
        pool + exact poison quota. The poisoned SET is chosen by the attacker, not here."""
        self.round_index += 1
        round_num = self.training_rounds + self.round_index

        # New local data first, THEN train: the honest updates (and so the clean
        # counterfactual, every scored rollout and the commit) all come from this
        # round's data, off the frozen global.
        self._refresh_client_data(self.round_index)
        self.honest_updates = [self._honest_update(cid) for cid in range(self.n_clients)]
        self.pool_ids = list(range(self.n_compromisable))
        self.pool_benign = {cid: self.honest_updates[cid].weights for cid in self.pool_ids}
        # Consume exactly ONE curriculum slot per round, here — the FL interlude
        # between phases does not go through begin_round(), so it correctly does
        # not eat a training round out of the block.
        slot = self.curriculum.advance() if self.curriculum is not None else None
        self.round_curriculum = slot
        self.round_budget = self._round_budget(slot)
        self.round_goal = self._round_goal()
        self.poisoned_ids = []                            # attacker decides at commit
        self._clean_ref_acc = None                        # recomputed lazily for this round
        self._clean_ref_measured = False
        self._clean_defense_sane = True
        # Fix this round's defense BEFORE anything is scored (the clean
        # counterfactual below already goes through it), so the counterfactual,
        # every rollout and the commit all face the SAME algorithm.
        self.round_defense = self._round_defense(slot)

        clean_acc = self.clean_reference_accuracy()
        logger.info(
            f"Round {round_num}: controllable_pool={self.pool_ids} "
            f"budget={self.round_budget} goal={self.round_goal} "
            f"defense={self.round_defense or 'llm'} "
            + (f"curriculum=block{slot.block}[{slot.block_round + 1}/"
               f"{self.curriculum.rounds_per_block}] cycle={slot.cycle} " if slot else "")
            + f"(global_acc={self.current_accuracy:.4f} clean_ref={clean_acc:.4f}"
            + ("" if self._clean_ref_measured else " UNMEASURED")
            + ("" if self._clean_defense_sane else " DEFENSE-MALFUNCTION") + ")"
        )
        return RoundContext(
            round_num=round_num,
            global_accuracy=self.current_accuracy,
            pool_ids=list(self.pool_ids),
            pool_benign=self.pool_benign,
            budget=self.round_budget,
            goal=self.round_goal,
            clean_accuracy=clean_acc,
            clean_measured=self._clean_ref_measured,
            defense_sane=self._clean_defense_sane,
        )

    # ------------------------------------------------------------------
    def clean_reference_accuracy(self) -> float:
        """Accuracy of THIS round's aggregate with **no poison and no flags**.

        This is the counterfactual the attacker's damage is scored against:
        ``drop = clean_reference_accuracy() - post_accuracy`` isolates what the
        attack actually cost, independent of what happened in previous rounds.

        Why it is not ``current_accuracy``: with
        ``benign_retrain_each_round: false`` the honest updates are frozen
        Phase-1 replays, so every round's aggregate is rebuilt from the same
        benign weights and the environment carries no memory of past damage.
        Scoring against the previous round's post-attack accuracy therefore
        measured the round-over-round *change* — an identical attack repeated
        twice scored high once and ~0 after — which made the schedule's
        ``success_streak`` gate unreachable. With ``benign_retrain_each_round:
        true`` this value tracks the recovering model instead, so the same
        definition works in both modes.

        Computed lazily and cached for the round (one extra test-set evaluation
        per round, against the G+1 the attacker's rollouts already cost).

        With an algorithmic defense the unpoisoned updates are run through THIS
        round's algorithm too (without committing its state), so the reference is
        "what the defended aggregate scores with no poison" — the drop then
        isolates the attack rather than the defense's own cost in honest rounds.

        **When the defense produces no clean aggregate at all** (FLTrust zeroing
        every trust score, DeFL removing everyone during a CLP) there IS no
        counterfactual to measure. This returns ``current_accuracy`` as a placeholder
        and sets :attr:`clean_reference_measured` to False; callers must check it.
        Silently returning ``current_accuracy`` was a real training bug: it is the
        PREVIOUS round's post-attack accuracy — exactly the quantity this method
        exists not to be — and when the poisoned round also produced no aggregate the
        post accuracy was that same number, so ``drop`` was identically 0.0000 by
        construction. About a quarter of rounds in a recorded run looked like a
        perfectly measured "the attack achieved nothing" when in truth nothing had
        been measured, and GRPO trained on the resulting zero damage term.
        """
        if self._clean_ref_acc is None:
            updates = self.build_updates({})
            if self.defense is not None:
                _verdicts, state = self.defend(updates, commit=False)
            else:
                _verdicts = [DetectionVerdict(u.client_id, False, 0.0, "clean_ref")
                             for u in updates]
                state = self.aggregator.aggregate(updates, _verdicts)
            self._clean_ref_measured = state is not None
            # Every update here is honest, so anything flagged is a false positive.
            # Losing the majority of them means the DEFENSE is broken this round, and
            # the accuracies below describe that, not the attack.
            self._clean_defense_sane = _accepts_honest_majority(_verdicts)
            if not self._clean_defense_sane:
                n_flagged = sum(1 for v in _verdicts if v.is_suspicious)
                logger.warning(
                    "Defense MALFUNCTION: %s flagged %d of %d entirely HONEST clients "
                    "on the clean counterfactual (FPR=%.2f). Every accuracy this round "
                    "is a measurement of the defense, not of the attack — the round "
                    "still advances the environment but applies no policy gradient.",
                    self.round_defense or "the aggregator", n_flagged, len(_verdicts),
                    n_flagged / max(1, len(_verdicts)),
                )
            if state is None:
                logger.warning(
                    "Clean counterfactual UNMEASURABLE: %s produced no aggregate from "
                    "the unpoisoned updates — falling back to the current global's "
                    "accuracy (%.4f). The attacker's damage term is meaningless this "
                    "round and no policy update will be applied.",
                    self.round_defense or "the aggregator", self.current_accuracy,
                )
            self._clean_ref_acc = self._eval_state(state)
        return self._clean_ref_acc

    @property
    def clean_reference_measured(self) -> bool:
        """Did :meth:`clean_reference_accuracy` actually MEASURE the counterfactual?

        False means the round's defense declined to aggregate the unpoisoned updates,
        so the cached reference is the current global's accuracy standing in for a
        value that does not exist. A round in that state cannot score an attack: see
        ``rl.schedule._run_learner_round``, which skips the GRPO update rather than
        training on a structurally-zero damage term.
        """
        return self._clean_ref_measured

    @property
    def clean_defense_sane(self) -> bool:
        """Did this round's defense keep the honest majority of an UNPOISONED cohort?

        False means it rejected most of a cohort in which every update was honest, so
        the aggregate it built — and therefore the clean counterfactual, the
        post-attack accuracy and the drop between them — is a reading of the defense's
        own false-positive rate. The round is not a valid observation of the attacker:
        ``rl.schedule._run_learner_round`` skips the GRPO update on it, exactly as it
        does for an unmeasurable counterfactual.

        This was silent for a whole recorded run: 45 of 262 rounds (17%) rejected the
        honest majority under FLTrust (mean FPR 0.43, peak 0.81), a warning was logged
        after the fact, and the policy was trained on every one of them anyway.
        """
        return self._clean_defense_sane

    def set_committed_poison(self, chosen_ids) -> None:
        """Record which clients the attacker actually poisoned (for logs/metrics)."""
        self.poisoned_ids = sorted(int(c) for c in chosen_ids)

    # ------------------------------------------------------------------
    def build_updates(self, poisoned_by_client: dict[int, dict]) -> list[ModelUpdate]:
        """Assemble the full client update list for a candidate attacker action."""
        updates = []
        for cid in range(self.n_clients):
            if cid in poisoned_by_client:
                w = poisoned_by_client[cid]
                meta = {"poisoned": True}
            else:
                w = self.honest_updates[cid].weights
                meta = {"poisoned": False}
            updates.append(ModelUpdate(client_id=cid, weights=w, metadata=meta))
        return updates

    @property
    def global_weights(self) -> dict:
        return self.server.get_global_weights()

    def features(self, updates: list[ModelUpdate]) -> dict[int, dict]:
        return compute_client_features(updates, self.server.get_global_weights())

    def _eval_state(self, state: dict | None) -> float:
        """Evaluate a candidate aggregated state without committing it."""
        if state is None:
            return self.current_accuracy
        backup = self.server.get_global_weights()
        self.server.set_global_weights(state)
        acc = self.server.evaluate(self.test_loader)
        self.server.set_global_weights(backup)
        return acc

    def evaluate_updates(self, updates, verdicts) -> float:
        """Post-aggregation accuracy for these updates+verdicts (no commit)."""
        candidate = self.aggregator.aggregate(updates, verdicts)
        return self._eval_state(candidate)

    def commit(self, updates, verdicts) -> float:
        """Aggregate, update the global model, and advance the round."""
        return self.commit_state(self.aggregator.aggregate(updates, verdicts))

    # ------------------------------------------------------------------
    # Algorithmic-defense path (defender LLM disabled)
    # ------------------------------------------------------------------
    def defend(self, updates, *, commit: bool = False):
        """Run THIS round's defense algorithm over ``updates``.

        Returns ``(verdicts, aggregated_state)``. Unlike the FedAvg path the
        algorithm emits BOTH — its verdicts and its own aggregate — so the two
        must come from the same call; the caller passes the state to
        :meth:`evaluate_state` (scoring) or :meth:`commit_state` (committing).
        ``aggregated_state`` is ``None`` when the defense declined to update the
        global (e.g. every client removed).

        ``commit=False`` rolls back the algorithm's cross-round memory afterwards,
        so all G rollouts in a group are graded against an identical defense.
        """
        if self.defense is None:
            raise RuntimeError(
                "env.defend() requires an algorithmic defense; this env is running "
                "the defender-LLM path (defense.mode: llm)"
            )
        outcome = self.defense.run(
            updates, self.server.get_global_weights(),
            commit=commit, algorithm=self.round_defense,
        )
        return outcome.verdicts, outcome.new_global

    def evaluate_state(self, state: dict | None) -> float:
        """Accuracy of an already-aggregated state, without committing it.
        ``None`` (defense skipped the round) scores the unchanged global."""
        return self._eval_state(state)

    def commit_state(self, state: dict | None) -> float:
        """Close the round: measure the aggregate and return its test accuracy.

        In FROZEN mode (``fl.freeze_global_in_phase2``, the default) that is ALL this
        does — the aggregate is scored and thrown away, and the global stays the
        Phase-1 anchor so the next round sends the clients exactly the same model.
        The returned accuracy is still this round's post-attack accuracy, which is
        what the reward is computed from (``clean_reference_accuracy() - post``); it
        simply no longer becomes round n+1's starting point.

        Otherwise the aggregate is installed as the new global (a continuing
        federation) and the returned accuracy is the new global's.
        """
        if self.freeze_global:
            # ``_eval_state`` evaluates without installing and always restores the
            # global, so the anchor is intact when this returns. ``state is None``
            # (the defense declined to aggregate) scores the unchanged anchor.
            self.last_round_accuracy = self._eval_state(state)
            if state is None:
                logger.warning("Round commit: no aggregate produced — "
                               "scoring the unchanged frozen global")
            return self.last_round_accuracy

        if state is not None:
            self.server.set_global_weights(state)
            self.current_accuracy = self.server.evaluate(self.test_loader)
        else:
            logger.warning("Round commit: no aggregate produced — global model unchanged")
        self.last_round_accuracy = self.current_accuracy
        return self.current_accuracy

    # ------------------------------------------------------------------
    def run_benign_fl_round(self) -> dict | None:
        """Run ONE honest FedAvg round (exactly like a Phase-1 round) to advance
        the shared FL state between arms-race phases.

        Every client trains locally from the CURRENT global model; the updates are
        FedAvg-aggregated with NO attacker and NO detector (all clients benign)
        into a new global, and the resulting per-client local weights REPLACE the
        frozen benign references in ``self.client_weights``. This is the
        ``benign_retrain`` path run once, on demand: afterwards the next learner
        (attacker or defender), the frozen opponent, and the aggregator all operate
        on freshly trained client weights + a new global — with
        ``benign_retrain_each_round=False`` these refreshed weights become the
        honest updates the attacker poisons and the defender inspects every
        following round.

        Advances ``round_index`` so the interlude gets its own sequential round
        number, and returns a summary dict for logging. Returns ``None`` — and
        changes nothing — when the env has no client loaders (e.g. some unit tests),
        or when Phase 2 runs frozen simulated rounds, where advancing the shared
        model is precisely what must not happen.
        """
        if self.freeze_global:
            logger.info(
                "run_benign_fl_round: SKIPPED — Phase 2 runs simulated rounds on the "
                "frozen Phase-1 global, so there is no shared FL state to advance "
                "(fl.freeze_global_in_phase2)."
            )
            return None
        if self._clients is None:
            logger.warning("run_benign_fl_round: no client loaders — FL round skipped")
            return None

        self.round_index += 1
        round_num = self.training_rounds + self.round_index
        prev_accuracy = self.current_accuracy

        self._refresh_client_data(self.round_index)
        updates = [c.train(self.server.model) for c in self._clients]
        # All clients are honest here — every verdict is benign, so FedAvg averages
        # the full set (mirrors Phase 1's clean aggregation).
        clean = [DetectionVerdict(u.client_id, False, 0.0, "benign_fl") for u in updates]
        new_global = self.aggregator.aggregate(updates, clean)
        if new_global is not None:
            self.server.set_global_weights(new_global)
        self.current_accuracy = self.server.evaluate(self.test_loader)

        # Refresh the per-client benign references the rest of Phase 2 consumes.
        self.client_weights = [copy.deepcopy(u.weights) for u in updates]
        self._clean_ref_acc = None        # stale: global + benign references just changed
        self._clean_ref_measured = False
        self._clean_defense_sane = True

        logger.info(
            f"[FL round {round_num}] benign FedAvg over {len(updates)} clients "
            f"(honest, no attacker/detector): accuracy {prev_accuracy:.4f} -> "
            f"{self.current_accuracy:.4f}"
        )
        return {
            "round_num": round_num,
            "prev_accuracy": prev_accuracy,
            "post_accuracy": self.current_accuracy,
            "updates": updates,
            "n_clients": len(updates),
        }
