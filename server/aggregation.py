"""Aggregation — FedAvg over the clients the defender labelled benign.

The old FLTrust trust-weighting branch and the LLM-chosen "method" dispatch are
gone (the defender now emits a direct benign/malicious classification, not a
detector-method choice). Aggregation simply averages the weights of clients the
defender did NOT flag.
"""

import copy
import logging

import torch

from core.types import ModelUpdate, DetectionVerdict
from core.interfaces import BaseAggregator

logger = logging.getLogger(__name__)


class FedAvgAggregator(BaseAggregator):
    """Average the weights of all non-flagged clients."""

    def aggregate(
        self,
        updates: list[ModelUpdate],
        verdicts: list[DetectionVerdict],
        strategy: dict | None = None,   # kept for interface compatibility; ignored
    ) -> dict | None:
        """Return the averaged state_dict, or None if every client was flagged."""
        flagged = {v.client_id for v in verdicts if v.is_suspicious}
        clean = [u for u in updates if u.client_id not in flagged]

        if not clean:
            logger.warning(
                "Aggregator: ALL clients flagged — round skipped (global unchanged)"
            )
            return None

        logger.info(f"Aggregator (FedAvg): averaging {len(clean)}/{len(updates)} updates")
        avg = copy.deepcopy(clean[0].weights)
        for key in avg:
            stacked = torch.stack([u.weights[key].float() for u in clean])
            avg[key] = stacked.mean(dim=0)
        return avg
