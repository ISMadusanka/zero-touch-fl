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
from core.types import ModelUpdate, DetectionVerdict
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

    def __init__(self, round_num, global_accuracy, pool_ids, pool_benign, budget, target_neuron_indices=None):
        self.round_num = round_num
        self.global_accuracy = global_accuracy
        self.pool_ids = pool_ids                          # list[int] controllable pool
        self.pool_benign = pool_benign                    # {cid: state_dict} for the pool
        self.budget = budget                              # max clients that may be poisoned
        self.target_neuron_indices = target_neuron_indices # {cid: {layer: [indices]}}
        self.poisoned_ids = []                            # set at commit (attacker's choice)


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
        self.poisoned_ids: list[int] = []                 # attacker's committed choice

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
    def _round_budget(self) -> int:
        """This round's poison budget: randomized in [1, cap] when sampling, else the cap."""
        if self.sample_budget:
            return self.rng.randint(1, self.budget_cap)
        return self.budget_cap

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
        self.poisoned_ids = []                            # attacker decides at commit

        logger.info(
            f"Round {round_num}: controllable_pool={self.pool_ids} "
            f"budget={self.round_budget} (global_acc={self.current_accuracy:.4f})"
        )
        self.target_neuron_indices = None
        if self.goal.get("type") == "targeted_label" and self._clients is not None:
            target_class = int(self.goal.get("label", 0))
            from attacks.neuron_importance import compute_neuron_importance
            self.target_neuron_indices = {}
            for cid in self.pool_ids:
                self.target_neuron_indices[str(cid)] = compute_neuron_importance(
                    self.server.model, self._clients[cid].data_loader, target_class, device=self.device
                )

        return RoundContext(
            round_num=round_num,
            global_accuracy=self.current_accuracy,
            pool_ids=list(self.pool_ids),
            pool_benign=self.pool_benign,
            budget=self.round_budget,
            target_neuron_indices=self.target_neuron_indices,
        )

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
        candidate = self.aggregator.aggregate(updates, verdicts)
        if candidate is not None:
            self.server.set_global_weights(candidate)
            self.current_accuracy = self.server.evaluate(self.test_loader)
        else:
            logger.warning("Round commit: all clients flagged — global model unchanged")
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
