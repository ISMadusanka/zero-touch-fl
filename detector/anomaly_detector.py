"""Anomaly detector — statistical analysis + LLM-driven thresholds.

Has NO prior knowledge of which client is malicious. Uses only statistical
features of the weight updates to make detection decisions.
"""

import logging
import torch
import numpy as np

from core.types import ModelUpdate, DetectionVerdict
from core.interfaces import BaseDetector

logger = logging.getLogger(__name__)


class AnomalyDetector(BaseDetector):
    """Computes update statistics and applies the defender agent's strategy."""

    def analyze(
        self, updates: list[ModelUpdate], global_weights: dict, strategy: dict
    ) -> list[DetectionVerdict]:
        """Analyze all updates using the given strategy. Returns one verdict per client."""
        features = self._compute_features(updates, global_weights)
        method = strategy.get("method", "norm_threshold")
        params = strategy.get("params", {})

        logger.info(f"Detector: method={method}, params={params}")
        logger.info(f"Detector: features={self._summarize_features(features)}")

        verdicts = []
        for i, update in enumerate(updates):
            suspicious, confidence, reason = self._apply_method(
                method, params, features, i
            )
            verdict = DetectionVerdict(
                client_id=update.client_id,
                is_suspicious=suspicious,
                confidence=confidence,
                reason=reason,
            )
            verdicts.append(verdict)
            if suspicious:
                logger.warning(
                    f"Detector: client {update.client_id} flagged — {reason} (conf={confidence:.3f})"
                )

        return verdicts

    def get_features(self, updates: list[ModelUpdate], global_weights: dict) -> dict:
        """Public access to computed features for the defender agent."""
        return self._summarize_features(self._compute_features(updates, global_weights))

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_features(self, updates: list[ModelUpdate], global_weights: dict) -> dict:
        """Compute per-client statistical features."""
        deltas = []
        for u in updates:
            flat = torch.cat([
                (u.weights[k] - global_weights[k]).flatten().float()
                for k in global_weights
            ])
            deltas.append(flat)

        norms = [d.norm().item() for d in deltas]

        # Cosine similarities with mean update
        mean_delta = torch.stack(deltas).mean(dim=0)
        cosines = []
        for d in deltas:
            cos = torch.nn.functional.cosine_similarity(
                d.unsqueeze(0), mean_delta.unsqueeze(0)
            ).item()
            cosines.append(cos)

        # Pairwise distances
        pairwise = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                pairwise.append((deltas[i] - deltas[j]).norm().item())

        return {
            "l2_norms": norms,
            "cosine_similarities": cosines,
            "mean_pairwise_distance": float(np.mean(pairwise)) if pairwise else 0.0,
            "client_ids": [u.client_id for u in updates],
        }

    def _summarize_features(self, features: dict) -> dict:
        """Round features for logging / LLM consumption."""
        return {
            "l2_norms": {
                cid: round(n, 4)
                for cid, n in zip(features["client_ids"], features["l2_norms"])
            },
            "cosine_similarities": {
                cid: round(c, 4)
                for cid, c in zip(features["client_ids"], features["cosine_similarities"])
            },
            "mean_pairwise_distance": round(features["mean_pairwise_distance"], 4),
        }

    def _apply_method(self, method: str, params: dict, features: dict, idx: int):
        """Apply detection method to a single client. Returns (suspicious, confidence, reason)."""
        norm = features["l2_norms"][idx]
        cosine = features["cosine_similarities"][idx]
        mean_norm = float(np.mean(features["l2_norms"]))

        if method == "norm_threshold":
            threshold = params.get("threshold", 2.0)
            ratio = norm / (mean_norm + 1e-8)
            suspicious = ratio > threshold
            return suspicious, min(ratio / threshold, 1.0), f"norm_ratio={ratio:.3f} vs threshold={threshold}"

        elif method == "cosine_threshold":
            threshold = params.get("threshold", 0.5)
            suspicious = cosine < threshold
            confidence = max(0, 1 - cosine) if suspicious else 0.0
            return suspicious, confidence, f"cosine={cosine:.3f} vs threshold={threshold}"

        elif method == "combined":
            norm_t = params.get("norm_threshold", 2.0)
            cos_t = params.get("cosine_threshold", 0.5)
            ratio = norm / (mean_norm + 1e-8)
            norm_flag = ratio > norm_t
            cos_flag = cosine < cos_t
            suspicious = norm_flag or cos_flag
            confidence = max(ratio / norm_t, (1 - cosine) if cos_flag else 0)
            reasons = []
            if norm_flag:
                reasons.append(f"norm_ratio={ratio:.3f}>{norm_t}")
            if cos_flag:
                reasons.append(f"cosine={cosine:.3f}<{cos_t}")
            return suspicious, min(confidence, 1.0), "; ".join(reasons) or "clean"

        else:
            logger.warning(f"Unknown method '{method}' — defaulting to norm_threshold")
            return self._apply_method("norm_threshold", params, features, idx)
