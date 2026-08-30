"""FLArmsRaceEnv — the federated-learning environment for the RL arms race.

Holds the global model, this round's client updates (honest and poisoned), and
the FedAvg aggregator + evaluation oracle. It deliberately knows nothing about
LLMs: it consumes detection verdicts and exposes ground truth (the per-round
poisoned set) for reward computation.

THE ATTACK IS LABEL FLIPPING, and it lives here rather than in an agent. A fixed,
configurable set of insider clients (``attack.poison_client_ids``, default
``[0]``) trains each round on its own data with a fraction of the labels flipped
symmetrically (``y -> 9-y``). How large that fraction is comes from a
detection-adaptive ladder — full poison, backing off a step every time the
defense catches it, resetting to full once it bottoms out — see
:mod:`agents.label_flip_attacker`. There is no attacker policy and nothing on the
attack side is learned; the only learner is the defender.

Round protocol (driven by the schedule / inference loop):

    env.reset(global, client_weights, baseline_acc)
    ctx = env.begin_round()            # fixes this round's defense algorithm, trains
                                       # every client honestly, re-trains the poisoned
                                       # clients on flipped labels, and exposes the
                                       # round's ground truth (ctx.poisoned_ids)
    updates = env.build_updates()      # honest cohort with the poisoned clients swapped in
    acc = env.evaluate_updates(updates, verdicts)   # no commit (used to score rollouts)
    ...
    new_acc = env.commit(updates, verdicts)         # measure (and, if not frozen, install)
    env.record_detection(verdicts)                  # feed the ladder — ONCE per round

SIMULATED (frozen-anchor) ROUNDS — ``fl.freeze_global_in_phase2: true``, the
default. Phase 2 does not run a continuing federation; it runs independent
defender-learning episodes that all branch off the SAME Phase-1 final model:

    for each round:
        the frozen Phase-1 global is sent to every client
        every client trains on NEW local data  (data.round_sampler)
        the poisoned clients re-train the same data with flipped labels
        the defender classifies every client, the server aggregates the un-flagged
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
from clients.malicious_client import LabelFlipClient
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
    accuracy, the drop — describes that malfunction rather than the attack (see
    ``server.algo_defender.resolve_root_epochs`` for the mechanism).

    Testing it on the CLEAN updates rather than the poisoned ones matters: it makes
    the check independent of the attack, so a genuinely strong attack that provokes
    lots of flags can never be mistaken for a broken defense, and the verdict is
    available before the rollouts are scored.
    """
    if not verdicts:
        return True
    accepted = sum(1 for v in verdicts if not v.is_suspicious)
    return accepted * 2 > len(verdicts)


class RoundContext:
    """Per-round observation handed to the defender + the logging path.

    The attack is fully determined before the round starts (the ladder's fraction
    applied to the configured insider set), so unlike the old attacker-policy
    design ``poisoned_ids`` is ground truth from the outset rather than something
    filled in at commit.
    """

    def __init__(self, round_num, global_accuracy, poisoned_ids, flip_plan,
                 flip_fraction, goal=None, clean_accuracy=None, clean_measured=True,
                 defense_sane=True):
        self.round_num = round_num
        self.global_accuracy = global_accuracy            # accuracy of the CURRENT global model
        self.poisoned_ids = list(poisoned_ids)            # GROUND TRUTH for this round
        self.flip_plan = dict(flip_plan)                  # {cid: #labels flipped}
        self.flip_fraction = float(flip_fraction)         # the ladder level this round was sent at
        self.goal = goal                                  # this round's attack goal (for the damage bar)
        # The clean counterfactual: what this round's aggregate scores with NO
        # poison. This — not ``global_accuracy`` — is what the attack's damage is
        # measured against (see ``FLArmsRaceEnv.clean_reference_accuracy``).
        self.clean_accuracy = clean_accuracy
        # False when the defense produced NO clean aggregate, so the counterfactual
        # could not actually be measured and ``clean_accuracy`` is only a fallback.
        self.clean_measured = clean_measured
        # False when the round's defense rejected the honest MAJORITY of an
        # entirely unpoisoned cohort. It aggregated something, so the numbers look
        # ordinary, but they measure the defense malfunctioning.
        self.defense_sane = defense_sane

    @property
    def n_poisoned(self) -> int:
        return len(self.poisoned_ids)


class FLArmsRaceEnv:
    def __init__(self, config: dict, client_loaders, test_loader, rng, defense=None,
                 curriculum=None, round_data=None, attacker=None):
        """``attacker`` is the :class:`agents.label_flip_attacker.LabelFlipAttacker`
        that decides how many labels each insider client flips this round. When it
        is ``None`` one is built from ``config['attack']``; pass an explicit one to
        share its ladder state with the resume path.

        ``defense`` is an optional :class:`server.algo_defender.AlgorithmicDefender`.
        When present the defender LLM is disabled and the server side of every
        round is run by one published defense algorithm, selected per round in
        :meth:`begin_round` and used for the clean counterfactual, every scored
        rollout and the commit. That algorithm also produces the round's
        AGGREGATE (FLTrust re-weights and rescales, DeFL Beta-weights, ...), so
        callers use :meth:`defend` + :meth:`evaluate_state` / :meth:`commit_state`
        instead of the verdict-driven :meth:`evaluate_updates` / :meth:`commit`.
        ``defense=None`` keeps the FedAvg-over-unflagged path the defender LLM uses.

        ``curriculum`` is an optional :class:`rl.curriculum.TrainingCurriculum`,
        which under an algorithmic defense sweeps the defense algorithm in blocks
        instead of drawing it per round.

        ``round_data`` is an optional :class:`data.round_sampler.RoundDataSampler`.
        When present, every client is handed a fresh slice of its own shard at the
        start of each round — the "clients train with new data" step of a simulated
        round. Without it the clients replay their whole fixed shard every round.
        """
        fl = config["fl"]
        attack = config.get("attack", {})
        self.n_clients = int(fl["n_clients"])
        self.device = fl.get("device", "cpu")
        self.training_rounds = int(fl.get("training_rounds", 0))
        # SIMULATED ROUNDS. Phase 2 does not continue the federation: every round is
        # an independent episode branching off the frozen Phase-1 model. The round's
        # aggregate is measured (that measurement IS the reward) and then discarded,
        # so round n+1 sends the clients the same anchor as round n. That makes each
        # round a controlled experiment against a fixed reference, which is what the
        # damage term (clean_reference_accuracy - post_accuracy) assumes.
        self.freeze_global = bool(fl.get("freeze_global_in_phase2", True))
        # LABEL FLIPPING REQUIRES LOCAL TRAINING. The poison is in the data, so a
        # poisoned update only exists if the client actually runs SGD on the flipped
        # labels — there is no frozen state_dict to replay and no weight edit to
        # apply. Replaying frozen Phase-1 weights for the honest clients while the
        # poisoned ones train fresh would also be a giveaway for entirely the wrong
        # reason (staleness, not the flipped labels), so the retrain is mandatory.
        self.benign_retrain = bool(fl.get("benign_retrain_each_round", True))
        if not self.benign_retrain:
            logger.warning(
                "fl.benign_retrain_each_round is false, but the label-flipping attack "
                "needs every client to train locally each round (the poison is in the "
                "DATA, not in the weights). Forcing benign retraining on."
            )
            self.benign_retrain = True
        # Per-round local data for the clients (None = replay the whole fixed shard).
        self.round_data = round_data
        self.goal = attack.get(
            "goal", {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}
        )
        # Per-round attack-goal target sampling: when on, draw target_accuracy_drop
        # from target_choices each round. It only scales the reported damage bar (the
        # attack itself is the ladder), so it is off by default.
        self.sample_target = bool(attack.get("sample_target_in_training", False))
        self.target_choices = [float(x) for x in attack.get(
            "target_choices", [0.05, 0.10, 0.20, 0.30])]

        self.test_loader = test_loader
        self.rng = rng
        self.aggregator = FedAvgAggregator()
        self.server = FedServer(device=self.device)
        # Algorithmic (non-LLM) defense, or None when the defender LLM defends.
        self.defense = defense
        # Deterministic defense-algorithm sweep, or None for random draws.
        self.curriculum = curriculum

        # The attack: which clients flip labels, and the detection-adaptive ladder
        # that sets how many of their labels are flipped each round.
        if attacker is None:
            from agents.label_flip_attacker import build_attacker
            attacker = build_attacker(config, n_clients=self.n_clients,
                                      seed=int(fl.get("poison_seed", 0)))
        self.attacker = attacker
        self.poison_client_ids = list(attacker.poison_client_ids)
        if len(self.poison_client_ids) * 2 >= self.n_clients:
            logger.warning(
                f"{len(self.poison_client_ids)} of {self.n_clients} clients flip labels "
                f"every round ({len(self.poison_client_ids) / self.n_clients:.0%}) — at or "
                f"past half the federation, so the 'honest majority' the robust "
                f"aggregators are proved under no longer holds. Raise fl.n_clients or "
                f"shorten attack.poison_client_ids to keep the majority honest."
            )

        # Honest clients (one per id) and the label-flipping twins of the insiders.
        # Both draw from the SAME per-round loader, so the poisoned update differs
        # from the honest one only by the flipped labels.
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
        self._flip_clients = {
            cid: LabelFlipClient(
                client_id=cid,
                data_loader=client_loaders[cid],
                lr=float(fl["lr"]),
                local_epochs=int(fl["local_epochs"]),
                device=self.device,
                n_classes=int(attack.get("n_classes", 10)),
            )
            for cid in self.poison_client_ids
        } if client_loaders is not None else {}

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
        self.poisoned_updates: dict[int, ModelUpdate] = {}
        self.poisoned_ids: list[int] = []                 # ground truth for THIS round
        self.flip_plan: dict[int, int] = {}               # {cid: #labels flipped}
        self.flip_fraction: float = 0.0                   # the ladder level this round
        self.round_goal: dict = dict(self.goal)           # this round's (maybe sampled) goal
        self._clean_ref_acc: float | None = None          # cached per-round clean counterfactual
        self._clean_ref_measured: bool = False            # was that counterfactual real?
        self._clean_defense_sane: bool = True             # did it keep the honest majority?
        self.round_defense: str | None = None             # this round's defense algorithm (if any)
        self.round_curriculum = None                      # this round's CurriculumSlot (if any)
        self._detection_recorded: bool = False            # ladder fed for the current round?

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
        logger.info(
            f"Env reset — n_clients={self.n_clients}, "
            f"poisoned={self.poison_client_ids}, {self.attacker.describe()}, "
            f"baseline_acc={baseline_accuracy:.4f}"
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
        """This round's attack goal — the target drop the reported damage is scaled
        against. It does not change what the attack DOES (the ladder decides that);
        it only sets the bar the round's damage is judged by."""
        if self.sample_target and self.goal.get("type") == "untargeted_degrade":
            return {"type": "untargeted_degrade",
                    "target_accuracy_drop": self.rng.choice(self.target_choices)}
        return self.goal

    def _honest_update(self, cid: int) -> ModelUpdate:
        if self._clients is not None:
            return self._clients[cid].train(self.server.model)
        # No client loaders (unit tests): replay the stored benign weights.
        return ModelUpdate(client_id=cid, weights=copy.deepcopy(self.client_weights[cid]),
                           metadata={"poisoned": False})

    def _refresh_client_data(self, round_index: int) -> None:
        """Hand every client its data for this round — a fresh slice of its OWN shard.

        This is the step that makes consecutive simulated rounds differ: the global
        they start from is frozen, so without new local data every round would
        reproduce the same honest updates. The label-flipping twins get the SAME
        loader as their honest counterparts, so the only difference between the two
        updates is the flipped labels. No-op when no sampler is configured
        (``fl.client_data_refresh: none``), where the clients keep their full shard.
        """
        if self.round_data is None or self._clients is None:
            return
        for client, loader in zip(self._clients,
                                  self.round_data.loaders_for_round(round_index)):
            client.set_data_loader(loader)
            twin = self._flip_clients.get(client.client_id)
            if twin is not None:
                twin.set_data_loader(loader)

    def _client_sample_count(self, cid: int) -> int:
        """How many examples client ``cid`` trains on THIS round.

        The ladder's fraction is applied to this, not to the whole shard, because
        this is the data whose labels can actually be flipped — with the per-round
        refresh on, a client sees only ``fl.client_round_fraction`` of its shard.
        """
        if self._clients is None:
            return 0
        return len(self._clients[cid].data_loader.dataset)

    def _poisoned_update(self, cid: int, n_flip: int) -> ModelUpdate:
        """This round's poisoned update: the same local training, flipped labels."""
        client = self._flip_clients[cid]
        client.set_flip(n_flip, self.attacker.flip_seed(self.round_index, cid))
        return client.train(self.server.model)

    def begin_round(self) -> RoundContext:
        """Produce this round's honest updates, apply the label-flip ladder to the
        insider clients, and expose the round's ground truth."""
        self.round_index += 1
        round_num = self.training_rounds + self.round_index

        # New local data first, THEN train: the honest updates (and so the clean
        # counterfactual, every scored rollout and the commit) all come from this
        # round's data, off the frozen global.
        self._refresh_client_data(self.round_index)
        self.honest_updates = [self._honest_update(cid) for cid in range(self.n_clients)]

        # The attack. The ladder's fraction is applied to each insider's own
        # per-round sample count, and only clients that end up with at least one
        # flipped label count as poisoned — a level that rounds to zero flips sends
        # a genuinely honest update, and calling it poison would corrupt both the
        # defender's reward and the ladder's own feedback.
        self.flip_fraction = self.attacker.fraction
        self.flip_plan = self.attacker.plan(
            {cid: self._client_sample_count(cid) for cid in self.poison_client_ids})
        self.poisoned_updates = {
            cid: self._poisoned_update(cid, n_flip)
            for cid, n_flip in sorted(self.flip_plan.items()) if n_flip > 0
        }
        self.poisoned_ids = sorted(self.poisoned_updates)
        self._detection_recorded = False

        # Consume exactly ONE curriculum slot per round, here — the FL interlude
        # between phases does not go through begin_round(), so it correctly does
        # not eat a training round out of the block.
        slot = self.curriculum.advance() if self.curriculum is not None else None
        self.round_curriculum = slot
        self.round_goal = self._round_goal()
        self._clean_ref_acc = None                        # recomputed lazily for this round
        self._clean_ref_measured = False
        self._clean_defense_sane = True
        # Fix this round's defense BEFORE anything is scored (the clean
        # counterfactual below already goes through it), so the counterfactual,
        # every rollout and the commit all face the SAME algorithm.
        self.round_defense = self._round_defense(slot)

        clean_acc = self.clean_reference_accuracy()
        flips = ", ".join(f"{cid}:{n}/{self._client_sample_count(cid)}"
                          for cid, n in sorted(self.flip_plan.items()))
        logger.info(
            f"Round {round_num}: label_flip {self.flip_fraction:.0%} "
            f"(level {self.attacker.ladder.level}/{self.attacker.ladder.n_levels - 1}, "
            f"cycle {self.attacker.ladder.cycle}) poisoned={self.poisoned_ids} "
            f"flips[{flips}] goal={self.round_goal} "
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
            poisoned_ids=list(self.poisoned_ids),
            flip_plan=dict(self.flip_plan),
            flip_fraction=self.flip_fraction,
            goal=self.round_goal,
            clean_accuracy=clean_acc,
            clean_measured=self._clean_ref_measured,
            defense_sane=self._clean_defense_sane,
        )

    # ------------------------------------------------------------------
    def clean_reference_accuracy(self) -> float:
        """Accuracy of THIS round's aggregate with **no poison and no flags**.

        This is the counterfactual the attack's damage is scored against:
        ``drop = clean_reference_accuracy() - post_accuracy`` isolates what the
        flipped labels actually cost, independent of previous rounds. The
        counterfactual is exact here in a way it never was for the weight-space
        attacker: the poisoned clients' HONEST updates for this very round are
        computed alongside their poisoned ones, from the same data and the same
        starting model, so the two aggregates differ by nothing but the labels.

        Computed lazily and cached for the round (one extra test-set evaluation
        per round).

        With an algorithmic defense the unpoisoned updates are run through THIS
        round's algorithm too (without committing its state), so the reference is
        "what the defended aggregate scores with no poison" — the drop then
        isolates the attack rather than the defense's own cost in honest rounds.

        **When the defense produces no clean aggregate at all** (FLTrust zeroing
        every trust score, DeFL removing everyone during a CLP) there IS no
        counterfactual to measure. This returns ``current_accuracy`` as a placeholder
        and sets :attr:`clean_reference_measured` to False; callers must check it.
        """
        if self._clean_ref_acc is None:
            updates = self.build_updates(include_poison=False)
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
                    "is a measurement of the defense, not of the attack.",
                    self.round_defense or "the aggregator", n_flagged, len(_verdicts),
                    n_flagged / max(1, len(_verdicts)),
                )
            if state is None:
                logger.warning(
                    "Clean counterfactual UNMEASURABLE: %s produced no aggregate from "
                    "the unpoisoned updates — falling back to the current global's "
                    "accuracy (%.4f). The damage measured this round is meaningless.",
                    self.round_defense or "the aggregator", self.current_accuracy,
                )
            self._clean_ref_acc = self._eval_state(state)
        return self._clean_ref_acc

    @property
    def clean_reference_measured(self) -> bool:
        """Did :meth:`clean_reference_accuracy` actually MEASURE the counterfactual?

        False means the round's defense declined to aggregate the unpoisoned updates,
        so the cached reference is the current global's accuracy standing in for a
        value that does not exist, and the round's ``induced_drop`` is not a
        measurement. Reported in the round log so such rounds can be sliced out.
        """
        return self._clean_ref_measured

    @property
    def clean_defense_sane(self) -> bool:
        """Did this round's defense keep the honest majority of an UNPOISONED cohort?

        False means it rejected most of a cohort in which every update was honest, so
        the aggregate it built — and therefore the clean counterfactual, the
        post-attack accuracy and the drop between them — is a reading of the defense's
        own false-positive rate rather than of the attack.
        """
        return self._clean_defense_sane

    # ------------------------------------------------------------------
    def record_detection(self, verdicts) -> dict:
        """Feed the COMMITTED round's verdicts back into the label-flip ladder.

        This is the feedback edge of the whole design: caught -> back off a step,
        missed -> hold, caught at the floor -> reset to full poison. Returns the
        ladder's transition record for the round log.

        Guarded so it fires at most once per round. The GRPO loop scores G defender
        rollouts against the same poisoned cohort, and letting each of them advance
        the ladder would make the attack schedule depend on ``rl.G`` — a sampling
        hyperparameter — rather than on whether the defense actually caught anything.
        """
        if self._detection_recorded:
            logger.debug("record_detection called twice in round %d — ignoring the "
                         "second call (the ladder advances once per committed round)",
                         self.round_index)
            return {}
        self._detection_recorded = True
        return self.attacker.record_round(verdicts, self.poisoned_ids)

    # ------------------------------------------------------------------
    def build_updates(self, include_poison: bool = True) -> list[ModelUpdate]:
        """Assemble the full client update list for this round.

        ``include_poison=False`` returns the all-honest cohort — the clean
        counterfactual, i.e. the same clients trained on the same data with their
        real labels.
        """
        updates = []
        for cid in range(self.n_clients):
            if include_poison and cid in self.poisoned_updates:
                u = self.poisoned_updates[cid]
                meta = dict(u.metadata or {})
                meta["poisoned"] = True
            else:
                u = self.honest_updates[cid]
                meta = dict(u.metadata or {})
                meta["poisoned"] = False
            updates.append(ModelUpdate(client_id=cid, weights=u.weights, metadata=meta))
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
        FedAvg-aggregated with NO poison and NO detector into a new global, and the
        resulting per-client local weights REPLACE the stored benign references in
        ``self.client_weights``.

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
            f"(honest, no poison/detector): accuracy {prev_accuracy:.4f} -> "
            f"{self.current_accuracy:.4f}"
        )
        return {
            "round_num": round_num,
            "prev_accuracy": prev_accuracy,
            "post_accuracy": self.current_accuracy,
            "updates": updates,
            "n_clients": len(updates),
        }
