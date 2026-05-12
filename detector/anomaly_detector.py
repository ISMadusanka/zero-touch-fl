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

        # Pairwise distances and Multi-Krum scores
        n = len(deltas)
        pairwise_matrix = torch.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = (deltas[i] - deltas[j]).norm().item()
                pairwise_matrix[i, j] = dist
                pairwise_matrix[j, i] = dist

        # Krum selects n - f - 2 closest neighbors (assume f=20% max attackers)
        f = max(1, int(n * 0.2))
        k = max(1, n - f - 2)
        krum_scores = []
        for i in range(n):
            sorted_dists = torch.sort(pairwise_matrix[i]).values
            # Ignore self (distance 0 at index 0), sum the next k distances
            score = sorted_dists[1:k+1].sum().item()
            krum_scores.append(score)

        # DnC Scores (Spectral Analysis)
        centered = torch.stack(deltas) - mean_delta
        try:
            # Compute SVD to find the principal component (direction of max variance)
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            top_v = Vh[0] 
            dnc_scores = [(c_i @ top_v).item() ** 2 for c_i in centered]
        except Exception as e:
            logger.error(f"SVD failed for DnC: {e}")
            dnc_scores = [0.0] * n

        return {
            "l2_norms": norms,
            "cosine_similarities": cosines,
            "mean_pairwise_distance": float(pairwise_matrix.sum() / (n * (n-1))) if n > 1 else 0.0,
            "krum_scores": krum_scores,
            "dnc_scores": dnc_scores,
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
            "krum_scores": {
                cid: round(s, 4)
                for cid, s in zip(features["client_ids"], features.get("krum_scores", []))
            },
            "dnc_scores": {
                cid: round(s, 4)
                for cid, s in zip(features["client_ids"], features.get("dnc_scores", []))
            },
            "mean_pairwise_distance": round(features["mean_pairwise_distance"], 4),
        }

    def _apply_method(self, method: str, params: dict, features: dict, idx: int):
        """Apply detection method to a single client. Returns (suspicious, confidence, reason)."""
        norm = features["l2_norms"][idx]
        cosine = features["cosine_similarities"][idx]
        mean_norm = float(np.mean(features["l2_norms"]))

        if method == "norm_threshold":
            # Aligned with the equation: uses an absolute threshold (tau) on the L2 norm
            threshold = params.get("threshold", 10.0)
            suspicious = norm > threshold
            confidence = min(norm / threshold, 1.0) if suspicious else 0.0
            return suspicious, confidence, f"norm={norm:.3f} vs threshold={threshold}"

        elif method == "cosine_threshold":
            threshold = params.get("threshold", 0.5)
            suspicious = cosine < threshold
            confidence = max(0, 1 - cosine) if suspicious else 0.0
            return suspicious, confidence, f"cosine={cosine:.3f} vs threshold={threshold}"

        elif method == "multi_krum":
            krum_score = features["krum_scores"][idx]
            median_krum = float(np.median(features["krum_scores"]))
            threshold_multiplier = params.get("threshold", 1.5)
            threshold = median_krum * threshold_multiplier
            suspicious = krum_score > threshold
            confidence = min(krum_score / (threshold + 1e-8), 1.0) if suspicious else 0.0
            return suspicious, confidence, f"krum_score={krum_score:.3f} vs threshold={threshold:.3f}"

        elif method == "dnc":
            dnc_score = features["dnc_scores"][idx]
            mean_dnc = float(np.mean(features["dnc_scores"]))
            std_dnc = float(np.std(features["dnc_scores"]))
            threshold_z = params.get("threshold", 2.0)
            threshold = mean_dnc + threshold_z * std_dnc
            suspicious = dnc_score > threshold
            confidence = min(dnc_score / (threshold + 1e-8), 1.0) if suspicious else 0.0
            return suspicious, confidence, f"dnc_score={dnc_score:.3f} vs threshold={threshold:.3f}"

        else:
            logger.warning(f"Unknown method '{method}' — defaulting to norm_threshold")
            return self._apply_method("norm_threshold", params, features, idx)
