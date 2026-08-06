"""FLArmsRaceEnv — the federated-learning environment for the RL arms race.

Holds the global model, the per-client benign updates for the current round,
and the FedAvg aggregator + evaluation oracle. It deliberately knows nothing
about LLMs: it consumes already-parsed poisoned state_dicts and detection
verdicts, and exposes ground truth (the per-round poisoned set) for reward
computation.

Round protocol (driven by the schedule / inference loop):

    env.reset(global, client_weights, baseline_acc)
    ctx = env.begin_round()                 # builds honest updates; exposes the attacker's
                                            # controllable pool (ctx.pool_benign) + budget
    # the attacker SELECTS exactly ctx.budget clients from the pool and poisons them
    updates = env.build_updates(poisoned_by_client)
    acc = env.evaluate_updates(updates, verdicts)   # no commit (used to score rollouts)
    ...
    env.set_committed_poison(chosen_ids)            # record the committed poison set
    new_acc = env.commit(updates, verdicts)         # advance the global model one round
"""

import copy
import logging

from clients.benign_client import BenignClient
from core.types import ModelUpdate, DetectionVerdict
from detector.features import compute_client_features
from server.aggregation import FedAvgAggregator
from server.fed_server import FedServer

logger = logging.getLogger(__name__)


class RoundContext:
    """Per-round observation handed to the agents.

    The attacker controls a fixed ``pool`` of clients and must poison exactly
    ``budget`` of them; it CHOOSES which (see ``AttackerAgent.select_and_apply``),
    so ``poisoned_ids`` is empty here and filled in once the choice is committed.
    """

    def __init__(self, round_num, global_accuracy, pool_ids, pool_benign, budget,
                 goal=None, clean_accuracy=None):
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


class FLArmsRaceEnv:
    def __init__(self, config: dict, client_loaders, test_loader, rng, defense=None):
        """``defense`` is an optional :class:`server.algo_defender.AlgorithmicDefender`.

        When present the defender LLM is disabled and the server side of every
        round is run by one published defense algorithm, drawn per round in
        :meth:`begin_round` and used for the clean counterfactual, every scored
        rollout and the commit. That algorithm also produces the round's
        AGGREGATE (FLTrust re-weights and rescales, DeFL Beta-weights, ...), so
        callers use :meth:`defend` + :meth:`evaluate_state` / :meth:`commit_state`
        instead of the verdict-driven :meth:`evaluate_updates` / :meth:`commit`.
        ``defense=None`` keeps the original FedAvg-over-unflagged path.
        """
        fl = config["fl"]
        attack = config.get("attack", {})
        self.n_clients = int(fl["n_clients"])
        self.device = fl.get("device", "cpu")
        self.benign_retrain = bool(fl.get("benign_retrain_each_round", True))
        self.training_rounds = int(fl.get("training_rounds", 0))
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

        # Set by begin_round().
        self.honest_updates: list[ModelUpdate] = []
        self.pool_ids: list[int] = []
        self.pool_benign: dict[int, dict] = {}
        self.round_budget: int = 1
        self.round_goal: dict = dict(self.goal)           # this round's (maybe sampled) goal
        self.poisoned_ids: list[int] = []                 # attacker's committed choice
        self._clean_ref_acc: float | None = None          # cached per-round clean counterfactual
        self.round_defense: str | None = None             # this round's defense algorithm (if any)

    # ------------------------------------------------------------------
    def reset(self, global_weights, client_weights, baseline_accuracy):
        self.server.set_global_weights(copy.deepcopy(global_weights))
        self.client_weights = [copy.deepcopy(w) for w in client_weights]
        self.baseline_accuracy = float(baseline_accuracy)
        self.current_accuracy = float(baseline_accuracy)
        self.round_index = 0
        logger.info(
            f"Env reset — n_clients={self.n_clients}, n_compromisable={self.n_compromisable}, "
            f"budget_cap={self.budget_cap}, sample_budget={self.sample_budget}, "
            f"benign_retrain={self.benign_retrain}, baseline_acc={baseline_accuracy:.4f}"
        )

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
        self._clean_ref_acc = None        # stale: the shared model just changed
        logger.info(
            f"Restored Phase-2 FL state — round_index={self.round_index}, "
            f"current_accuracy={self.current_accuracy:.4f} "
            f"(baseline stays {self.baseline_accuracy:.4f})"
        )

    # ------------------------------------------------------------------
    def _round_budget(self) -> int:
        """Draw this round's exact poison quota in [1, cap], or use the cap."""
        if self.sample_budget:
            return self.rng.randint(1, self.budget_cap)
        return self.budget_cap

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

    def begin_round(self) -> RoundContext:
        """Produce this round's honest updates and expose the attacker's controllable
        pool + exact poison quota. The poisoned SET is chosen by the attacker, not here."""
        self.round_index += 1
        round_num = self.training_rounds + self.round_index

        self.honest_updates = [self._honest_update(cid) for cid in range(self.n_clients)]
        self.pool_ids = list(range(self.n_compromisable))
        self.pool_benign = {cid: self.honest_updates[cid].weights for cid in self.pool_ids}
        self.round_budget = self._round_budget()
        self.round_goal = self._round_goal()
        self.poisoned_ids = []                            # attacker decides at commit
        self._clean_ref_acc = None                        # recomputed lazily for this round
        # Draw this round's defense BEFORE anything is scored (the clean
        # counterfactual below already goes through it), so the counterfactual,
        # every rollout and the commit all face the SAME algorithm.
        self.round_defense = self.defense.choose() if self.defense is not None else None

        clean_acc = self.clean_reference_accuracy()
        logger.info(
            f"Round {round_num}: controllable_pool={self.pool_ids} "
            f"budget={self.round_budget} goal={self.round_goal} "
            f"defense={self.round_defense or 'llm'} "
            f"(global_acc={self.current_accuracy:.4f} clean_ref={clean_acc:.4f})"
        )
        return RoundContext(
            round_num=round_num,
            global_accuracy=self.current_accuracy,
            pool_ids=list(self.pool_ids),
            pool_benign=self.pool_benign,
            budget=self.round_budget,
            goal=self.round_goal,
            clean_accuracy=clean_acc,
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
        """
        if self._clean_ref_acc is None:
            updates = self.build_updates({})
            if self.defense is not None:
                _verdicts, state = self.defend(updates, commit=False)
            else:
                clean = [DetectionVerdict(u.client_id, False, 0.0, "clean_ref")
                         for u in updates]
                state = self.aggregator.aggregate(updates, clean)
            self._clean_ref_acc = self._eval_state(state)
        return self._clean_ref_acc

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
        """Install an already-aggregated state as the new global and re-evaluate."""
        if state is not None:
            self.server.set_global_weights(state)
            self.current_accuracy = self.server.evaluate(self.test_loader)
        else:
            logger.warning("Round commit: no aggregate produced — global model unchanged")
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
        number, and returns a summary dict for logging. Returns ``None`` when the
        env has no client loaders (e.g. some unit tests) so callers can no-op.
        """
        if self._clients is None:
            logger.warning("run_benign_fl_round: no client loaders — FL round skipped")
            return None

        self.round_index += 1
        round_num = self.training_rounds + self.round_index
        prev_accuracy = self.current_accuracy

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
