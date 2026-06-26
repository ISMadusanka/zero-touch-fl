"""Oracle defense — flags EXACTLY the ground-truth poisoned clients.

Not a real defense (it cheats by reading the ground truth); included as an UPPER
BOUND on detection (TPR=1, FPR=0) and on robustness, to contextualise the others.
"""
from core.types import DetectionVerdict
from server.aggregation import FedAvgAggregator

from benchmark.defenses.base import Defense, StepResult


class Oracle(Defense):
    name = "oracle"

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._agg = FedAvgAggregator()

    def step(self, updates, poisoned_ids) -> StepResult:
        verdicts = [
            DetectionVerdict(u.client_id, u.client_id in poisoned_ids, 1.0, "oracle")
            for u in updates
        ]
        new_global = self._agg.aggregate(updates, verdicts)
        if new_global is not None:
            self._global = new_global
        return StepResult(new_global, verdicts)
