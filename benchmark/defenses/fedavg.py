"""No-defense baseline: plain FedAvg over ALL clients (flags nobody).

This is both (a) the 'what the attack does with no protection' reference and
(b) the state the attacker observes each round in the harness. It reuses the
existing FedAvgAggregator unchanged.
"""
from core.types import DetectionVerdict
from server.aggregation import FedAvgAggregator

from benchmark.defenses.base import Defense, StepResult


class NoDefense(Defense):
    name = "fedavg"

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._agg = FedAvgAggregator()

    def step(self, updates, poisoned_ids) -> StepResult:
        verdicts = [DetectionVerdict(u.client_id, False, 0.0, "no-defense") for u in updates]
        new_global = self._agg.aggregate(updates, verdicts)
        if new_global is not None:
            self._global = new_global
        return StepResult(new_global, verdicts)
