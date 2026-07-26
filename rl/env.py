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
    # the attacker SELECTS <= ctx.budget clients from the pool and poisons them
    updates = env.build_updates(poisoned_by_client)
    acc = env.evaluate_updates(updates, verdicts)   # no commit (used to score rollouts)
    ...
    env.set_committed_poison(chosen_ids)            # record the committed poison set
    new_acc = env.commit(updates, verdicts)         # advance the global model one round
"""

import copy
import logging

from clients.benign_client import BenignClient
from core.types import ClassEval, ModelUpdate, DetectionVerdict
from detector.features import compute_client_features
from server.aggregation import FedAvgAggregator
from server.fed_server import FedServer

logger = logging.getLogger(__name__)


class RoundContext:
    """Per-round observation handed to the agents.

    The attacker controls a fixed ``pool`` of clients and may poison up to
    ``budget`` of them; it CHOOSES which (see ``AttackerAgent.select_and_apply``),
    so ``poisoned_ids`` is empty here and filled in once the choice is committed.
    """

    def __init__(self, round_num, global_accuracy, pool_ids, pool_benign, budget,
                 goal=None, clean_accuracy=None, clean_eval=None):
        self.round_num = round_num
        self.global_accuracy = global_accuracy            # accuracy of the CURRENT global model
        self.pool_ids = pool_ids                          # list[int] controllable pool
        self.pool_benign = pool_benign                    # {cid: state_dict} for the pool
        self.budget = budget                              # max clients that may be poisoned
        self.goal = goal                                  # this round's attack goal (maybe sampled)
        self.poisoned_ids = []                            # set at commit (attacker's choice)
        # The clean counterfactual: what this round's aggregate scores with NO
        # poison. This — not ``global_accuracy`` — is what the attacker's damage
        # is measured against (see ``FLArmsRaceEnv.clean_reference_accuracy``).
        self.clean_accuracy = clean_accuracy
        # The SAME counterfactual broken down per class (``core.types.ClassEval``).
        # The targeted goal scores the drop in ``clean_eval.per_class[label]``
        # against the poisoned round's recall for that label, and charges the drop
        # in every OTHER class as collateral damage.
        self.clean_eval = clean_eval


class FLArmsRaceEnv:
    def __init__(self, config: dict, client_loaders, test_loader, rng):
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
        # Per-round poison budget (max clients the attacker may poison). Training
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
        # --- Targeted (per-label) poisoning -------------------------------
        # How many classes the task has, so every evaluation can be broken down
        # per class (``FedServer.evaluate_per_class``).
        self.n_classes = int(config.get("data", {}).get("n_classes", 10))
        # The pool of labels a ``targeted_label`` goal may be aimed at. When
        # ``sample_target_in_training`` is on we draw ONE label from this list each
        # round, so the policy learns "attack the class named in the goal" instead
        # of memorizing a single row of the output layer. Eval fixes the label to
        # ``attack.goal.label`` (``sample_target`` off).
        self.target_labels = [int(x) for x in attack.get(
            "target_labels", list(range(self.n_classes)))]

        self.test_loader = test_loader
        self.rng = rng
        self.aggregator = FedAvgAggregator()
        self.server = FedServer(device=self.device)

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
        self._clean_ref_eval: ClassEval | None = None     # cached per-round clean counterfactual
        self.current_eval: ClassEval | None = None        # per-class view of the live global

    # ------------------------------------------------------------------
    def reset(self, global_weights, client_weights, baseline_accuracy):
        self.server.set_global_weights(copy.deepcopy(global_weights))
        self.client_weights = [copy.deepcopy(w) for w in client_weights]
        self.baseline_accuracy = float(baseline_accuracy)
        self.current_accuracy = float(baseline_accuracy)
        self.current_eval = None          # measured lazily by current_class_eval()
        self._clean_ref_eval = None
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
        self._clean_ref_eval = None       # stale: the shared model just changed
        self.current_eval = None          # ditto — re-measured on demand
        logger.info(
            f"Restored Phase-2 FL state — round_index={self.round_index}, "
            f"current_accuracy={self.current_accuracy:.4f} "
            f"(baseline stays {self.baseline_accuracy:.4f})"
        )

    # ------------------------------------------------------------------
    def _round_budget(self) -> int:
        """This round's poison budget: randomized in [1, cap] when sampling, else the cap."""
        if self.sample_budget:
            return self.rng.randint(1, self.budget_cap)
        return self.budget_cap

    def _round_goal(self) -> dict:
        """This round's attack goal. When target sampling is on, randomize the part of
        the goal the policy should GENERALIZE over, so it learns to read the goal
        instead of memorizing one setting. The returned dict is the CLEAN goal shown
        to the LLM and used by the reward.

        * ``untargeted_degrade`` — draw ``target_accuracy_drop`` from
          ``target_choices``; the policy becomes TARGET-AWARE.
        * ``targeted_label`` — draw ``label`` from ``target_labels``; the policy
          becomes LABEL-AWARE. This is what makes a run trained over labels
          0..5 work on whichever single label evaluation asks for: the label is
          never constant, so the only way to score is to read it off the goal and
          attack the corresponding output unit.

        Eval always fixes the goal (``sample_target`` off)."""
        if not self.sample_target:
            return self.goal
        gtype = self.goal.get("type")
        if gtype == "untargeted_degrade":
            return {"type": "untargeted_degrade",
                    "target_accuracy_drop": self.rng.choice(self.target_choices)}
        if gtype == "targeted_label" and self.target_labels:
            # Keep every other field of the configured goal (target_class_drop,
            # max_collateral, ...) and swap only the label being attacked.
            goal = dict(self.goal)
            goal["label"] = int(self.rng.choice(self.target_labels))
            return goal
        return self.goal

    def _honest_update(self, cid: int) -> ModelUpdate:
        if self.benign_retrain and self._clients is not None:
            return self._clients[cid].train(self.server.model)
        # Replay frozen Phase-1 weights.
        return ModelUpdate(client_id=cid, weights=copy.deepcopy(self.client_weights[cid]))

    def begin_round(self) -> RoundContext:
        """Produce this round's honest updates and expose the attacker's controllable
        pool + poison budget. The poisoned SET is chosen by the attacker, not here."""
        self.round_index += 1
        round_num = self.training_rounds + self.round_index

        self.honest_updates = [self._honest_update(cid) for cid in range(self.n_clients)]
        self.pool_ids = list(range(self.n_compromisable))
        self.pool_benign = {cid: self.honest_updates[cid].weights for cid in self.pool_ids}
        self.round_budget = self._round_budget()
        self.round_goal = self._round_goal()
        self.poisoned_ids = []                            # attacker decides at commit
        self._clean_ref_eval = None                       # recomputed lazily for this round

        clean_eval = self.clean_reference_eval()
        extra = ""
        label = self.round_goal.get("label")
        if self.round_goal.get("type") == "targeted_label" and label is not None:
            extra = (f" clean_recall[{label}]={clean_eval.recall(int(label)):.4f}"
                     f" others={clean_eval.others_mean(int(label)):.4f}")
        logger.info(
            f"Round {round_num}: controllable_pool={self.pool_ids} "
            f"budget={self.round_budget} goal={self.round_goal} "
            f"(global_acc={self.current_accuracy:.4f} clean_ref={clean_eval.overall:.4f}{extra})"
        )
        return RoundContext(
            round_num=round_num,
            global_accuracy=self.current_accuracy,
            pool_ids=list(self.pool_ids),
            pool_benign=self.pool_benign,
            budget=self.round_budget,
            goal=self.round_goal,
            clean_accuracy=clean_eval.overall,
            clean_eval=clean_eval,
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
        """
        return self.clean_reference_eval().overall

    def clean_reference_eval(self) -> ClassEval:
        """:meth:`clean_reference_accuracy` broken down PER CLASS.

        Same single counterfactual aggregate, same single test-set pass — the
        per-class recalls just fall out of it (see ``FedServer.evaluate_per_class``).
        This is the reference a ``targeted_label`` round is scored against: the
        attack must drive ``per_class[label]`` down from *this* value while leaving
        the other classes at *these* values.
        """
        if self._clean_ref_eval is None:
            updates = self.build_updates({})
            clean = [DetectionVerdict(u.client_id, False, 0.0, "clean_ref") for u in updates]
            self._clean_ref_eval = self._eval_state_full(
                self.aggregator.aggregate(updates, clean))
        return self._clean_ref_eval

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

    def _eval_state_full(self, state: dict | None) -> ClassEval:
        """Evaluate a candidate aggregated state (per class) without committing it.

        ``state is None`` means the aggregator produced nothing (every client
        flagged), so the model is unchanged — we return the live model's own
        evaluation rather than re-measuring it.
        """
        if state is None:
            return self.current_class_eval()
        backup = self.server.get_global_weights()
        self.server.set_global_weights(state)
        ev = self.server.evaluate_per_class(self.test_loader, self.n_classes)
        self.server.set_global_weights(backup)
        return ev

    def _eval_state(self, state: dict | None) -> float:
        """Evaluate a candidate aggregated state without committing it."""
        return self._eval_state_full(state).overall

    def current_class_eval(self) -> ClassEval:
        """Per-class evaluation of the LIVE global model (measured once, cached).

        ``commit`` refreshes this, so the only time it has to measure is right
        after a ``reset``/``restore`` — i.e. the first round of a run.
        """
        if self.current_eval is None:
            self.current_eval = self.server.evaluate_per_class(
                self.test_loader, self.n_classes)
            self.current_accuracy = self.current_eval.overall
        return self.current_eval

    def evaluate_updates(self, updates, verdicts) -> float:
        """Post-aggregation accuracy for these updates+verdicts (no commit)."""
        return self.evaluate_updates_full(updates, verdicts).overall

    def evaluate_updates_full(self, updates, verdicts) -> ClassEval:
        """:meth:`evaluate_updates`, broken down per class (same single test pass).

        This is what scores a targeted rollout: compare its ``per_class[label]``
        against ``clean_reference_eval().per_class[label]``.
        """
        candidate = self.aggregator.aggregate(updates, verdicts)
        return self._eval_state_full(candidate)

    def commit(self, updates, verdicts) -> float:
        """Aggregate, update the global model, and advance the round."""
        candidate = self.aggregator.aggregate(updates, verdicts)
        if candidate is not None:
            self.server.set_global_weights(candidate)
            self.current_eval = self.server.evaluate_per_class(
                self.test_loader, self.n_classes)
            self.current_accuracy = self.current_eval.overall
            logger.info(f"Global model test accuracy: {self.current_accuracy:.4f}")
        else:
            logger.warning("Round commit: all clients flagged — global model unchanged")
        return self.current_accuracy

    def commit_full(self, updates, verdicts) -> ClassEval:
        """:meth:`commit`, returning the committed model's per-class evaluation."""
        self.commit(updates, verdicts)
        return self.current_class_eval()

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
        self.current_eval = self.server.evaluate_per_class(self.test_loader, self.n_classes)
        self.current_accuracy = self.current_eval.overall

        # Refresh the per-client benign references the rest of Phase 2 consumes.
        self.client_weights = [copy.deepcopy(u.weights) for u in updates]
        self._clean_ref_eval = None       # stale: global + benign references just changed

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
