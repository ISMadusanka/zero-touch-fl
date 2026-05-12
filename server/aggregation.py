"""Aggregation strategies — FedAvg + FLTrust weighted aggregation.

FedAvg: Standard federated averaging that excludes flagged clients.
FLTrust: Cao et al. (NDSS 2021) — trust-weighted aggregation using
         ReLU(cosine_similarity) as trust scores and magnitude normalization.
"""

import copy
import logging
import torch
import numpy as np

from core.types import ModelUpdate, DetectionVerdict
from core.interfaces import BaseAggregator

logger = logging.getLogger(__name__)


class FedAvgAggregator(BaseAggregator):
    """Federated averaging that excludes suspicious clients.

    When ``strategy`` is ``"fltrust"``, performs FLTrust weighted aggregation
    (Cao et al. 2021) instead of simple averaging.
    """

    def aggregate(
        self, updates: list[ModelUpdate], verdicts: list[DetectionVerdict],
        strategy: dict | None = None,
    ) -> dict | None:
        """Aggregate client updates.

        Args:
            updates: All client model updates.
            verdicts: Detection verdicts (one per client).
            strategy: The defender's strategy dict. If method=="fltrust",
                      uses trust-weighted aggregation.

        Returns None if ALL clients are flagged (round should be skipped).
        """
        method = (strategy or {}).get("method", "fedavg")

        if method == "fltrust":
            return self._fltrust_aggregate(updates, verdicts)

        return self._fedavg_aggregate(updates, verdicts)

    def _fedavg_aggregate(
        self, updates: list[ModelUpdate], verdicts: list[DetectionVerdict]
    ) -> dict | None:
        """Standard FedAvg: average the weights of non-suspicious clients.

        Returns None if ALL clients are flagged (round should be skipped).
        """
        verdict_map = {v.client_id: v for v in verdicts}
        clean_updates = [
            u for u in updates if not verdict_map.get(u.client_id, DetectionVerdict(0, False, 0, "")).is_suspicious
        ]

        if not clean_updates:
            logger.warning(
                "Aggregator: ALL clients flagged — skipping round "
                "(global model unchanged)"
            )
            return None

        n = len(clean_updates)
        logger.info(f"Aggregator (FedAvg): averaging {n}/{len(updates)} client updates")

        avg_state = copy.deepcopy(clean_updates[0].weights)
        for key in avg_state:
            stacked = torch.stack([u.weights[key].float() for u in clean_updates])
            avg_state[key] = stacked.mean(dim=0)

        return avg_state

    def _fltrust_aggregate(
        self, updates: list[ModelUpdate], verdicts: list[DetectionVerdict]
    ) -> dict | None:
        """FLTrust weighted aggregation (Cao et al. NDSS 2021).

        Algorithm:
          1. Compute server reference update Δ_server = mean of all updates.
          2. For each client i:
             a. Trust score TS_i = ReLU(cos_sim(Δi, Δ_server))
             b. Normalize magnitude: Δ̂i = (‖Δ_server‖ / ‖Δi‖) × Δi
          3. Aggregated update = Σ(TS_i × Δ̂i) / Σ(TS_i)

        FLTrust does NOT filter by verdicts — it uses trust scores as continuous
        weights. Clients with negative cosine similarity get TS=0 (ignored).
        """
        if not updates:
            return None

        # Flatten all updates to compute reference and trust scores
        keys = list(updates[0].weights.keys())
        deltas_per_key = {key: [] for key in keys}
        for u in updates:
            for key in keys:
                deltas_per_key[key].append(u.weights[key].float())

        # Server reference: mean of all updates (proxy for root dataset)
        server_ref = {}
        for key in keys:
            server_ref[key] = torch.stack(deltas_per_key[key]).mean(dim=0)

        # Compute trust scores and normalize magnitudes
        # Flatten for cosine similarity computation
        flat_updates = []
        for u in updates:
            flat = torch.cat([u.weights[k].flatten().float() for k in keys])
            flat_updates.append(flat)

        flat_server = torch.cat([server_ref[k].flatten() for k in keys])
        server_norm = flat_server.norm().item()

        trust_scores = []
        for flat_u in flat_updates:
            cos_sim = torch.nn.functional.cosine_similarity(
                flat_u.unsqueeze(0), flat_server.unsqueeze(0)
            ).item()
            # ReLU clipping: trust = max(0, cos_sim)
            ts = max(0.0, cos_sim)
            trust_scores.append(ts)

        total_trust = sum(trust_scores)
        if total_trust < 1e-8:
            logger.warning("FLTrust: all trust scores are ~0 — skipping round")
            return None

        n_trusted = sum(1 for ts in trust_scores if ts > 0)
        logger.info(
            f"Aggregator (FLTrust): {n_trusted}/{len(updates)} clients with positive trust, "
            f"trust_scores={[round(ts, 4) for ts in trust_scores]}"
        )

        # Weighted aggregation with magnitude normalization
        result = {}
        for key in keys:
            weighted_sum = torch.zeros_like(server_ref[key])
            for i, u in enumerate(updates):
                if trust_scores[i] < 1e-8:
                    continue
                u_param = u.weights[key].float()
                # Normalize magnitude to match server reference
                u_norm = u_param.flatten().norm().item()
                s_norm = server_ref[key].flatten().norm().item()
                if u_norm > 1e-8 and s_norm > 1e-8:
                    normalized = u_param * (s_norm / u_norm)
                else:
                    normalized = u_param
                weighted_sum += trust_scores[i] * normalized
            result[key] = weighted_sum / total_trust

        return result
