"""FedAvg aggregation — filters out detected malicious clients."""

import copy
import logging
import torch

from core.types import ModelUpdate, DetectionVerdict
from core.interfaces import BaseAggregator

logger = logging.getLogger(__name__)


class FedAvgAggregator(BaseAggregator):
    """Federated averaging that excludes suspicious clients."""

    def aggregate(
        self, updates: list[ModelUpdate], verdicts: list[DetectionVerdict]
    ) -> dict:
        """Average the weights of non-suspicious clients."""
        verdict_map = {v.client_id: v for v in verdicts}
        clean_updates = [
            u for u in updates if not verdict_map.get(u.client_id, DetectionVerdict(0, False, 0, "")).is_suspicious
        ]

        if not clean_updates:
            logger.warning("Aggregator: ALL clients flagged — using all updates as fallback")
            clean_updates = updates

        n = len(clean_updates)
        logger.info(f"Aggregator: averaging {n}/{len(updates)} client updates")

        avg_state = copy.deepcopy(clean_updates[0].weights)
        for key in avg_state:
            stacked = torch.stack([u.weights[key].float() for u in clean_updates])
            avg_state[key] = stacked.mean(dim=0)

        return avg_state
