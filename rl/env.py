"""FLArmsRaceEnv — the federated-learning environment for the RL arms race.

Holds the global model, the per-client benign updates for the current round,
and the FedAvg aggregator + evaluation oracle. It deliberately knows nothing
about LLMs: it consumes already-parsed poisoned state_dicts and detection
verdicts, and exposes ground truth (the per-round poisoned set) for reward
computation.

Round protocol (driven by the schedule / inference loop):

    env.reset(global, client_weights, baseline_acc)
    ctx = env.begin_round()                 # samples poisoned set, builds honest updates
    updates = env.build_updates(poisoned_by_client)
    acc = env.evaluate_updates(updates, verdicts)   # no commit (used to score rollouts)
    ...
    new_acc = env.commit(updates, verdicts)         # advance the global model one round
"""

import copy
import logging

from clients.benign_client import BenignClient
from core.types import ModelUpdate
from detector.features import compute_client_features
from server.aggregation import FedAvgAggregator
from server.fed_server import FedServer

logger = logging.getLogger(__name__)


class RoundContext:
    """Per-round observation handed to the agents."""

    def __init__(self, round_num, global_accuracy, poisoned_ids, benign_by_poisoned):
        self.round_num = round_num
        self.global_accuracy = global_accuracy
        self.poisoned_ids = poisoned_ids                  # list[int]
        self.benign_by_poisoned = benign_by_poisoned      # {cid: state_dict}


class FLArmsRaceEnv:
    def __init__(self, config: dict, client_loaders, test_loader, rng):
        fl = config["fl"]
        self.n_clients = int(fl["n_clients"])
        self.device = fl.get("device", "cpu")
        self.poison_fraction = float(fl.get("poison_fraction", 0.2))
        self.benign_retrain = bool(fl.get("benign_retrain_each_round", True))
        self.training_rounds = int(fl.get("training_rounds", 0))
        self.goal = config.get("attack", {}).get(
            "goal", {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}
        )

        self.test_loader = test_loader
        self.rng = rng
        self.aggregator = FedAvgAggregator()
        self.server = FedServer(device=self.device)
        # Materialize the fixed test set on-GPU once — the reward oracle evaluates
        # G+1 times per round, so this removes the per-batch DataLoader overhead.
        self.server.build_eval_cache(test_loader)

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
        self.poisoned_ids: list[int] = []
        self.benign_by_poisoned: dict[int, dict] = {}

    # ------------------------------------------------------------------
    def reset(self, global_weights, client_weights, baseline_accuracy):
        self.server.set_global_weights(copy.deepcopy(global_weights))
        self.client_weights = [copy.deepcopy(w) for w in client_weights]
        self.baseline_accuracy = float(baseline_accuracy)
        self.current_accuracy = float(baseline_accuracy)
        self.round_index = 0
        logger.info(
            f"Env reset — n_clients={self.n_clients}, poison_fraction={self.poison_fraction}, "
            f"benign_retrain={self.benign_retrain}, baseline_acc={baseline_accuracy:.4f}"
        )

    # ------------------------------------------------------------------
    def _num_poisoned(self) -> int:
        """Poison count, clamped to keep a strict benign majority."""
        k = round(self.poison_fraction * self.n_clients)
        majority_cap = (self.n_clients - 1) // 2
        return max(1, min(k, majority_cap))

    def _honest_update(self, cid: int) -> ModelUpdate:
        if self.benign_retrain and self._clients is not None:
            return self._clients[cid].train(self.server.model)
        # Replay frozen Phase-1 weights.
        return ModelUpdate(client_id=cid, weights=copy.deepcopy(self.client_weights[cid]))

    def begin_round(self) -> RoundContext:
        """Sample the poisoned subset and produce this round's honest updates."""
        self.round_index += 1
        round_num = self.training_rounds + self.round_index

        k = self._num_poisoned()
        self.poisoned_ids = sorted(self.rng.sample(range(self.n_clients), k))

        self.honest_updates = [self._honest_update(cid) for cid in range(self.n_clients)]
        self.benign_by_poisoned = {
            cid: self.honest_updates[cid].weights for cid in self.poisoned_ids
        }
        logger.info(
            f"Round {round_num}: poisoned_ids={self.poisoned_ids} "
            f"(global_acc={self.current_accuracy:.4f})"
        )
        return RoundContext(
            round_num=round_num,
            global_accuracy=self.current_accuracy,
            poisoned_ids=list(self.poisoned_ids),
            benign_by_poisoned=self.benign_by_poisoned,
        )

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
