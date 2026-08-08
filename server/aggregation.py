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
    """Average the weights of all non-flagged clients.

    ``clip_multiplier`` (optional, per call, default ``None`` = off): when set,
    every ACCEPTED client's weights are first anchored to the coordinate-wise
    median of ALL updates this round (accepted + rejected -- the controllable
    pool is a strict minority of the total client set, so this median stays a
    robust "typical" reference even with poisoned clients present), then
    clamped so their deviation from that median has L2 norm at most
    ``clip_multiplier`` times the MEDIAN deviation norm among accepted clients.

    This bounds how much damage a single accepted-but-actually-poisoned update
    can do, independent of whichever verdict admitted it -- a lightweight,
    defense-agnostic backstop underneath verdict-based filtering, in the same
    spirit as FLTrust's rescale-to-trusted-norm step but without needing a
    clean root dataset. Left OFF by default: this is opt-in per call site so it
    does not silently change the behaviour of every consumer of this class
    (notably the "no defense" FedAvg baseline, which must stay unclipped to
    remain a meaningful reference point).
    """

    def aggregate(
        self,
        updates: list[ModelUpdate],
        verdicts: list[DetectionVerdict],
        strategy: dict | None = None,   # kept for interface compatibility; ignored
        clip_multiplier: float | None = None,
    ) -> dict | None:
        """Return the averaged state_dict, or None if every client was flagged."""
        flagged = {v.client_id for v in verdicts if v.is_suspicious}
        clean = [u for u in updates if u.client_id not in flagged]

        if not clean:
            logger.warning(
                "Aggregator: ALL clients flagged — round skipped (global unchanged)"
            )
            return None

        weights = [u.weights for u in clean]
        if clip_multiplier is not None and len(updates) > 1:
            weights = _clip_to_median(updates, weights, clip_multiplier)

        logger.info(
            f"Aggregator (FedAvg): averaging {len(clean)}/{len(updates)} updates"
            + (f" (median-clip x{clip_multiplier})" if clip_multiplier is not None else "")
        )
        avg = copy.deepcopy(weights[0])
        for key in avg:
            stacked = torch.stack([w[key].float() for w in weights])
            avg[key] = stacked.mean(dim=0)
        return avg


def _clip_to_median(all_updates: list[ModelUpdate], accepted_weights: list[dict],
                    clip_multiplier: float) -> list[dict]:
    """Clamp each accepted client's deviation from the coordinate-wise median
    (over ALL updates this round, accepted or not) to at most
    ``clip_multiplier`` times the MEDIAN deviation norm among the accepted
    clients. Returns new state_dicts; never mutates the inputs.
    """
    keys = list(all_updates[0].weights.keys())
    all_stacked = {k: torch.stack([u.weights[k].float() for u in all_updates]) for k in keys}
    median = {k: all_stacked[k].median(dim=0).values for k in keys}

    def _dev_norm(w: dict) -> float:
        return float(torch.cat([(w[k].float() - median[k]).flatten() for k in keys]).norm())

    dev_norms = [_dev_norm(w) for w in accepted_weights]
    ref_norm = float(torch.tensor(dev_norms).median()) if dev_norms else 0.0
    threshold = clip_multiplier * ref_norm

    clipped = []
    for w, dnorm in zip(accepted_weights, dev_norms):
        if threshold <= 0.0 or dnorm <= threshold:
            clipped.append(w)
            continue
        scale = threshold / (dnorm + 1e-8)
        new_w = {k: (median[k] + (w[k].float() - median[k]) * scale).to(w[k].dtype) for k in keys}
        clipped.append(new_w)
    return clipped
