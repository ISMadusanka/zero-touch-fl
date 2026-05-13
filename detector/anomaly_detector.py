"""Anomaly detector — adaptive, research-grounded defenses.

Implements 5 defense algorithms with fully adaptive (auto-configured)
parameters. No hardcoded thresholds — all bounds are derived from the
statistical distribution of the current round's updates.

Defense methods and their papers:
  - norm_threshold : Sun et al., "Can You Really Backdoor FL?" (2019)
  - dnc            : Shejwalkar & Houmansadr, "Manipulating the Byzantine" (NDSS 2021)
  - fltrust        : Cao et al., "FLTrust: Byzantine-robust FL via Trust Bootstrapping" (NDSS 2021)
  - foolsgold      : Fung et al., "Limitations of FL in Sybil Settings" (RAID 2020)
  - flame          : Nguyen et al., "FLAME: Taming Backdoors in FL" (USENIX Security 2022)

Has NO prior knowledge of which client is malicious. Uses only statistical
features of the weight updates to make detection decisions.
"""

import logging
import torch
import numpy as np

from core.types import ModelUpdate, DetectionVerdict
from core.interfaces import BaseDetector

logger = logging.getLogger(__name__)


def _median_absolute_deviation(values: np.ndarray) -> float:
    """Compute MAD — a robust measure of spread (resistant to outliers).

    MAD = median(|x_i - median(x)|)

    Used instead of std to prevent outliers from inflating the threshold.
    """
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


class AnomalyDetector(BaseDetector):
    """Computes update statistics and applies the defender agent's strategy.

    All thresholds are adaptive — computed from the data distribution of the
    current round. The LLM defender agent controls a single ``sensitivity``
    parameter (z-score multiplier) that governs how aggressive detection is.
    """

    def analyze(
        self, updates: list[ModelUpdate], global_weights: dict, strategy: dict
    ) -> list[DetectionVerdict]:
        """Analyze all updates using the given strategy. Returns one verdict per client."""
        features = self._compute_features(updates, global_weights)
        method = strategy.get("method", "norm_threshold")
        params = strategy.get("params", {})

        logger.info(f"Detector: method={method}, params={params}")
        logger.info(f"Detector: features={self._summarize_features(features)}")

        # Pre-compute FLAME cluster labels once (not per-client)
        if method == "flame":
            features["flame_labels"] = self._compute_flame_labels(params, features)

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
    # Internal: Feature computation
    # ------------------------------------------------------------------

    def _compute_features(self, updates: list[ModelUpdate], global_weights: dict) -> dict:
        """Compute per-client statistical features used by all defense methods."""
        deltas = []
        for u in updates:
            flat = torch.cat([
                (u.weights[k] - global_weights[k]).flatten().float()
                for k in global_weights
            ])
            deltas.append(flat)

        n = len(deltas)
        norms = [d.norm().item() for d in deltas]

        # ---- Cosine similarities with mean update ----
        mean_delta = torch.stack(deltas).mean(dim=0)
        cosines = []
        for d in deltas:
            cos = torch.nn.functional.cosine_similarity(
                d.unsqueeze(0), mean_delta.unsqueeze(0)
            ).item()
            cosines.append(cos)

        # ---- Pairwise cosine similarity matrix (for FoolsGold / FLAME) ----
        delta_stack = torch.stack(deltas)  # (n, d)
        delta_norms_t = delta_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = delta_stack / delta_norms_t
        pairwise_cosine = (normalized @ normalized.T).cpu().numpy().astype(np.float64)  # (n, n)

        # ---- Pairwise L2 distances ----
        pairwise_matrix = torch.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = (deltas[i] - deltas[j]).norm().item()
                pairwise_matrix[i, j] = dist
                pairwise_matrix[j, i] = dist

        # ---- DnC scores: SVD-based spectral outlier detection ----
        # Shejwalkar & Houmansadr (NDSS 2021) — project centered updates
        # onto top right singular vector, squared projection = outlier score.
        centered = torch.stack(deltas) - mean_delta
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            top_v = Vh[0]
            dnc_scores = [(c_i @ top_v).item() ** 2 for c_i in centered]
        except Exception as e:
            logger.error(f"SVD failed for DnC: {e}")
            dnc_scores = [0.0] * n

        # ---- FLTrust scores ----
        # Cao et al. (NDSS 2021) — ReLU(cosine_sim(Δi, Δserver))
        # We use mean_delta as the server reference update (proxy for root dataset).
        fltrust_scores = [max(0.0, c) for c in cosines]

        # ---- FoolsGold: max pairwise cosine similarity ----
        # Fung et al. (RAID 2020) — cs_i = max_{j≠i} cos_sim(Δi, Δj)
        foolsgold_max_cs = []
        for i in range(n):
            row = pairwise_cosine[i].copy()
            row[i] = -2.0  # exclude self
            foolsgold_max_cs.append(float(np.max(row)))

        return {
            "deltas": deltas,
            "l2_norms": norms,
            "cosine_similarities": cosines,
            "pairwise_cosine": pairwise_cosine,
            "mean_pairwise_distance": float(pairwise_matrix.sum() / (n * (n - 1))) if n > 1 else 0.0,
            "dnc_scores": dnc_scores,
            "fltrust_scores": fltrust_scores,
            "foolsgold_max_cs": foolsgold_max_cs,
            "client_ids": [u.client_id for u in updates],
            "mean_delta": mean_delta,
            "mean_norm": float(mean_delta.norm().item()),
        }

    def _summarize_features(self, features: dict) -> dict:
        """Round features for logging / LLM consumption."""
        cids = features["client_ids"]
        summary = {
            "l2_norms": {
                cid: round(n, 4)
                for cid, n in zip(cids, features["l2_norms"])
            },
            "cosine_similarities": {
                cid: round(c, 4)
                for cid, c in zip(cids, features["cosine_similarities"])
            },
            "dnc_scores": {
                cid: round(s, 4)
                for cid, s in zip(cids, features.get("dnc_scores", []))
            },
            "fltrust_scores": {
                cid: round(s, 4)
                for cid, s in zip(cids, features.get("fltrust_scores", []))
            },
            "foolsgold_max_cs": {
                cid: round(s, 4)
                for cid, s in zip(cids, features.get("foolsgold_max_cs", []))
            },
            "mean_pairwise_distance": round(features["mean_pairwise_distance"], 4),
        }
        return summary

    # ------------------------------------------------------------------
    # Internal: Detection methods (all adaptive)
    # ------------------------------------------------------------------

    def _apply_method(self, method: str, params: dict, features: dict, idx: int):
        """Route to the appropriate defense. Returns (suspicious, confidence, reason)."""
        dispatch = {
            "norm_threshold": self._norm_threshold,
            "dnc": self._dnc,
            "fltrust": self._fltrust,
            "foolsgold": self._foolsgold,
            "flame": self._flame,
        }
        handler = dispatch.get(method)
        if handler is None:
            logger.warning(f"Unknown method '{method}' — falling back to norm_threshold")
            handler = self._norm_threshold
        return handler(params, features, idx)

    # ---------- 1. Adaptive Norm Threshold ----------
    # Sun et al. (2019) — "Can You Really Backdoor Federated Learning?"
    #
    # Original: clip updates whose ‖Δi‖ > M (fixed M).
    # Adaptive: M = median(norms) + sensitivity × MAD(norms)
    #   where MAD = median absolute deviation (robust to outliers).
    #   The LLM tunes ``sensitivity`` (default 2.0).
    # --------------------------------------------------
    def _norm_threshold(self, params: dict, features: dict, idx: int):
        sensitivity = params.get("sensitivity", 2.0)
        norms = np.array(features["l2_norms"])
        norm_i = norms[idx]

        median_norm = float(np.median(norms))
        mad = _median_absolute_deviation(norms)
        # Prevent degenerate case when all norms are identical (MAD=0)
        mad = max(mad, 1e-8)

        threshold = median_norm + sensitivity * mad
        suspicious = norm_i > threshold
        # Confidence: how far above the threshold (capped at 1.0)
        if suspicious:
            confidence = min((norm_i - threshold) / (mad + 1e-8), 1.0)
        else:
            confidence = 0.0

        return (
            suspicious,
            confidence,
            f"norm={norm_i:.4f}, threshold={threshold:.4f} "
            f"(median={median_norm:.4f} + {sensitivity}×MAD={mad:.4f})",
        )

    # ---------- 2. DnC — Divide-and-Conquer Spectral Analysis ----------
    # Shejwalkar & Houmansadr (NDSS 2021)
    #
    # Algorithm:
    #   1. Center updates: Δ̃i = Δi − mean(Δ)
    #   2. SVD → top right singular vector v₁
    #   3. Outlier score s_i = (Δ̃i · v₁)²
    #   4. Adaptive threshold: median(s) + sensitivity × MAD(s)
    #
    # The paper uses sub-sampling (divide step) for scalability; with small
    # n_clients we operate on the full set (equivalent to a single partition).
    # --------------------------------------------------
    def _dnc(self, params: dict, features: dict, idx: int):
        sensitivity = params.get("sensitivity", 2.0)
        scores = np.array(features["dnc_scores"])
        score_i = scores[idx]

        median_score = float(np.median(scores))
        mad = _median_absolute_deviation(scores)
        mad = max(mad, 1e-8)

        threshold = median_score + sensitivity * mad
        suspicious = score_i > threshold

        if suspicious:
            confidence = min((score_i - threshold) / (mad + 1e-8), 1.0)
        else:
            confidence = 0.0

        return (
            suspicious,
            confidence,
            f"dnc_score={score_i:.4f}, threshold={threshold:.4f} "
            f"(median={median_score:.4f} + {sensitivity}×MAD={mad:.4f})",
        )

    # ---------- 3. FLTrust — Trust Bootstrapping ----------
    # Cao et al. (NDSS 2021)
    #
    # Algorithm:
    #   1. Server computes reference update Δ_server (here: mean of all Δ).
    #   2. Trust score: TS_i = ReLU(cos_sim(Δi, Δ_server))
    #   3. Normalize magnitudes: Δ̂i = (‖Δ_server‖ / ‖Δi‖) × Δi
    #   4. Weighted aggregation: Σ(TS_i × Δ̂i) / Σ(TS_i)
    #
    # For detection: flag clients with TS < adaptive_threshold.
    # Threshold = max(0, median(TS) - sensitivity × MAD(TS))
    # Clients with TS=0 (negative cosine) are always flagged.
    # --------------------------------------------------
    def _fltrust(self, params: dict, features: dict, idx: int):
        sensitivity = params.get("sensitivity", 2.0)
        scores = np.array(features["fltrust_scores"])
        ts_i = scores[idx]

        median_ts = float(np.median(scores))
        mad = _median_absolute_deviation(scores)
        mad = max(mad, 1e-8)

        # Lower tail detection: flag clients whose trust is anomalously low
        threshold = max(0.0, median_ts - sensitivity * mad)
        suspicious = ts_i < threshold

        if suspicious:
            # Confidence: how far below the threshold
            confidence = min((threshold - ts_i) / (mad + 1e-8), 1.0)
        else:
            confidence = 0.0

        return (
            suspicious,
            confidence,
            f"trust_score={ts_i:.4f}, threshold={threshold:.4f} "
            f"(median={median_ts:.4f} - {sensitivity}×MAD={mad:.4f})",
        )

    # ---------- 4. FoolsGold — Sybil-Resistant Scoring ----------
    # Fung et al. (RAID 2020)
    #
    # Algorithm:
    #   1. Pairwise cosine similarity matrix C, where C_ij = cos(Δi, Δj).
    #   2. Contribution score for client i: cs_i = max_{j≠i} C_ij
    #      → high cs_i means the client looks very similar to another
    #        (indicative of Sybil/colluding behavior).
    #   3. Weight w_i = 1 - cs_i  (penalize high similarity).
    #   4. In the paper, a logit transform κ and normalization are applied.
    #
    # For detection: flag clients whose cs_i is anomalously HIGH.
    # Threshold = median(cs) + sensitivity × MAD(cs)
    #
    # Note: In a single-attacker setting this detects the client whose
    # update direction differs most from the majority (low weight → flagged).
    # --------------------------------------------------
    def _foolsgold(self, params: dict, features: dict, idx: int):
        sensitivity = params.get("sensitivity", 2.0)
        max_cs = np.array(features["foolsgold_max_cs"])
        cs_i = max_cs[idx]

        # Compute FoolsGold weight: w_i = 1 - cs_i
        weights = 1.0 - max_cs

        # Apply logit-like transformation (from the paper):
        #   κ(w) = log(w / (1 - w)) rescaled to [0, 1]
        # This amplifies differences near the extremes.
        eps = 1e-6
        weights_clipped = np.clip(weights, eps, 1.0 - eps)
        logit_weights = np.log(weights_clipped / (1.0 - weights_clipped))
        # Rescale logit weights to [0, 1]
        lw_min, lw_max = logit_weights.min(), logit_weights.max()
        if lw_max - lw_min > eps:
            logit_weights = (logit_weights - lw_min) / (lw_max - lw_min)
        else:
            logit_weights = np.ones_like(logit_weights)

        weight_i = logit_weights[idx]

        # Flag clients with anomalously LOW FoolsGold weight
        median_w = float(np.median(logit_weights))
        mad = _median_absolute_deviation(logit_weights)
        mad = max(mad, 1e-8)

        threshold = max(0.0, median_w - sensitivity * mad)
        suspicious = weight_i < threshold

        if suspicious:
            confidence = min((threshold - weight_i) / (mad + 1e-8), 1.0)
        else:
            confidence = 0.0

        return (
            suspicious,
            confidence,
            f"fg_weight={weight_i:.4f}, max_cs={cs_i:.4f}, threshold={threshold:.4f} "
            f"(median_w={median_w:.4f} - {sensitivity}×MAD={mad:.4f})",
        )

    # ---------- 5. FLAME — HDBSCAN + Adaptive Clipping ----------
    # Nguyen et al. (USENIX Security 2022)
    #
    # Algorithm:
    #   1. Compute pairwise cosine distance matrix: d_ij = 1 - cos(Δi, Δj).
    #   2. Cluster updates using HDBSCAN with min_cluster_size = ⌊n/2⌋ + 1
    #      (ensures the largest cluster is a majority).
    #   3. Identify the largest cluster as "benign".
    #   4. Flag all clients NOT in the largest cluster (outliers/noise/minority clusters).
    #   5. Adaptive clipping bound: median of L2 norms in the benign cluster.
    #
    # Clustering is pre-computed once in analyze() and cached in
    # features["flame_labels"] to avoid redundant computation per client.
    #
    # Fallback: If HDBSCAN is unavailable, use agglomerative clustering.
    #           If scipy is also unavailable, use cosine-distance z-score.
    # --------------------------------------------------

    def _compute_flame_labels(self, params: dict, features: dict) -> np.ndarray:
        """Run FLAME clustering ONCE and return the label array.

        Called from analyze() before the per-client loop so that clustering
        is not redundantly re-computed for every client.
        """
        sensitivity = params.get("sensitivity", 2.0)
        n = len(features["l2_norms"])
        pairwise_cosine = features["pairwise_cosine"]

        # Cosine distance matrix
        cosine_dist = 1.0 - pairwise_cosine
        np.fill_diagonal(cosine_dist, 0.0)

        try:
            import hdbscan
            # min_cluster_size ensures majority cluster ≥ ⌊n/2⌋ + 1
            min_cs = max(2, n // 2 + 1)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cs,
                metric="precomputed",
                allow_single_cluster=True,
            )
            labels = clusterer.fit_predict(cosine_dist)
            logger.info(f"FLAME (HDBSCAN): cluster labels = {labels.tolist()}")
            return labels

        except ImportError:
            logger.info("HDBSCAN not installed — using agglomerative clustering fallback")
            try:
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import squareform
                condensed = squareform(cosine_dist)
                Z = linkage(condensed, method="average")
                # Cut the dendrogram adaptively: median linkage distance + sensitivity × MAD
                merge_dists = Z[:, 2]
                median_d = float(np.median(merge_dists))
                mad_d = _median_absolute_deviation(merge_dists)
                cut_threshold = median_d + sensitivity * max(mad_d, 1e-8)
                labels = fcluster(Z, t=cut_threshold, criterion="distance") - 1  # 0-indexed
                logger.info(f"FLAME (agglomerative): cluster labels = {labels.tolist()}")
                return labels

            except ImportError:
                logger.info("scipy not available — FLAME will use cosine distance z-score fallback")
                # Return None sentinel — _flame() will use the z-score fallback
                return None

    def _flame(self, params: dict, features: dict, idx: int):
        sensitivity = params.get("sensitivity", 2.0)
        labels = features.get("flame_labels")

        if labels is not None:
            # Use pre-computed cluster labels
            return self._flame_cluster_verdict(labels, features, idx)

        # Fallback: no clustering library available — use cosine distance z-score
        n = len(features["l2_norms"])
        pairwise_cosine = features["pairwise_cosine"]
        cosine_dist = 1.0 - pairwise_cosine
        np.fill_diagonal(cosine_dist, 0.0)

        mean_cos_dist = cosine_dist.sum(axis=1) / max(n - 1, 1)
        score_i = mean_cos_dist[idx]
        median_score = float(np.median(mean_cos_dist))
        mad = _median_absolute_deviation(mean_cos_dist)
        mad = max(mad, 1e-8)
        threshold = median_score + sensitivity * mad
        suspicious = score_i > threshold
        confidence = min((score_i - threshold) / (mad + 1e-8), 1.0) if suspicious else 0.0
        return (
            suspicious,
            confidence,
            f"flame_fallback: cos_dist={score_i:.4f}, threshold={threshold:.4f} "
            f"(median={median_score:.4f} + {sensitivity}×MAD={mad:.4f})",
        )

    def _flame_cluster_verdict(
        self, labels: np.ndarray, features: dict, idx: int
    ):
        """Given cluster labels, flag clients not in the largest cluster."""
        n = len(labels)
        # Find the largest cluster (excluding noise label -1)
        unique_labels, counts = np.unique(labels, return_counts=True)
        # Filter out noise (-1) when determining the majority cluster
        valid_mask = unique_labels >= 0
        if valid_mask.any():
            valid_labels = unique_labels[valid_mask]
            valid_counts = counts[valid_mask]
            majority_label = valid_labels[np.argmax(valid_counts)]
        else:
            # All noise — flag everyone as not suspicious (degenerate case)
            return (False, 0.0, "flame: all points labeled as noise — no flagging")

        is_in_majority = labels[idx] == majority_label
        suspicious = not is_in_majority

        if suspicious:
            # Confidence based on how isolated this client is
            # Use the client's mean cosine similarity to the majority cluster
            majority_indices = np.where(labels == majority_label)[0]
            pairwise_cosine = features["pairwise_cosine"]
            mean_sim_to_majority = float(np.mean(pairwise_cosine[idx, majority_indices]))
            confidence = min(max(1.0 - mean_sim_to_majority, 0.0), 1.0)
        else:
            confidence = 0.0

        cluster_label = int(labels[idx])
        majority_size = int(np.sum(labels == majority_label))
        return (
            suspicious,
            confidence,
            f"flame: cluster={cluster_label}, majority_cluster={int(majority_label)} "
            f"(size={majority_size}/{n}), in_majority={is_in_majority}",
        )
