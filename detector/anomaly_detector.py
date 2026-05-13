"""Anomaly detector — non-IID-aware statistical analysis.

Has NO prior knowledge of which client is malicious. Uses only statistical
features of the weight updates (and, for FLTrust, a server-side reference
update) to make detection decisions.

Defense methods implemented here are all drawn from published research and
were chosen to remain meaningful under non-IID client data, where honest
clients naturally diverge from each other:

- norm_threshold:   absolute L2-norm cap (Sun et al., 2019, "Can You Really
                    Backdoor Federated Learning?"). Scale defense is
                    orthogonal to data heterogeneity.
- dnc:              Divide-and-Conquer spectral filter (Shejwalkar &
                    Houmansadr, NDSS 2021, "Manipulating the Byzantine").
                    Evaluated by the authors under non-IID.
- fltrust:          Server-anchored trust scoring (Cao et al., NDSS 2021,
                    "FLTrust: Byzantine-robust Federated Learning via Trust
                    Bootstrapping"). Cosine is taken against a server-side
                    root update, so divergent honest clients are NOT
                    penalised for differing from each other.
- foolsgold:        Historical pairwise cosine (Fung et al., RAID 2020,
                    "The Limitations of Federated Learning in Sybil
                    Settings"). Under non-IID, honest clients SHOULD be
                    dissimilar — high pairwise similarity exposes sybils
                    / colluders.
- flame:            HDBSCAN clustering on cosine matrix + median-norm
                    clipping (Nguyen et al., USENIX Security 2022,
                    "FLAME: Taming Backdoors in Federated Learning").
                    Density clustering keeps honest-but-divergent clients
                    together instead of thresholding against a global mean.

The previous `cosine_threshold` (cosine-to-mean) and `multi_krum` methods
were removed: both anchor on peer consensus and are documented to break
under non-IID (Cao et al. 2021; Fung et al. 2020; Nguyen et al. 2022).
"""

import logging
import math
import torch
import numpy as np

from core.types import ModelUpdate, DetectionVerdict
from core.interfaces import BaseDetector

logger = logging.getLogger(__name__)


class AnomalyDetector(BaseDetector):
    """Computes update statistics and applies the defender agent's strategy."""

    def __init__(self):
        # FoolsGold requires per-client *historical* update accumulators.
        # client_id -> running sum of flattened deltas across rounds.
        self._history: dict[int, torch.Tensor] = {}

    def analyze(
        self,
        updates: list[ModelUpdate],
        global_weights: dict,
        strategy: dict,
        server_delta: torch.Tensor | None = None,
    ) -> list[DetectionVerdict]:
        """Analyze all updates using the given strategy. Returns one verdict per client.

        Args:
            updates: client model updates for this round.
            global_weights: current global model state dict.
            strategy: {"method": str, "params": dict} from defender agent.
            server_delta: flat tensor of the server's own root-update delta
                (required for FLTrust; ignored by other methods).
        """
        features = self._compute_features(updates, global_weights, server_delta)
        # FoolsGold history is updated AFTER scoring so the current round's
        # contribution does not artificially inflate self-similarity.
        self._update_history(features["client_ids"], features["deltas"])

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

    def get_features(
        self,
        updates: list[ModelUpdate],
        global_weights: dict,
        server_delta: torch.Tensor | None = None,
    ) -> dict:
        """Public access to computed features for the defender agent."""
        return self._summarize_features(
            self._compute_features(updates, global_weights, server_delta)
        )

    # ------------------------------------------------------------------
    # Internal: feature computation
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        updates: list[ModelUpdate],
        global_weights: dict,
        server_delta: torch.Tensor | None,
    ) -> dict:
        """Compute per-client statistical features."""
        deltas = []
        for u in updates:
            flat = torch.cat([
                (u.weights[k] - global_weights[k]).flatten().float()
                for k in global_weights
            ])
            deltas.append(flat)

        n = len(deltas)
        norms = [d.norm().item() for d in deltas]

        # Mean update (used only for diagnostics now — no peer-cosine method
        # is exposed to the defender any more).
        mean_delta = torch.stack(deltas).mean(dim=0)

        # ---- DnC (Shejwalkar & Houmansadr, NDSS 2021) --------------------
        centered = torch.stack(deltas) - mean_delta
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            top_v = Vh[0]
            dnc_scores = [(c_i @ top_v).item() ** 2 for c_i in centered]
        except Exception as e:
            logger.error(f"SVD failed for DnC: {e}")
            dnc_scores = [0.0] * n

        # ---- FLTrust (Cao et al., NDSS 2021) -----------------------------
        # Trust = ReLU(cos(client_delta, server_delta)).  Without a server
        # delta this defense is undefined and all scores fall back to 0,
        # which means the method will refuse to make decisions (logged).
        fltrust_scores = [0.0] * n
        if server_delta is not None:
            sd = server_delta.float()
            sd_norm = sd.norm().item()
            if sd_norm > 1e-12:
                for i, d in enumerate(deltas):
                    d_norm = d.norm().item()
                    if d_norm < 1e-12:
                        fltrust_scores[i] = 0.0
                    else:
                        cos = float((d @ sd).item() / (d_norm * sd_norm))
                        fltrust_scores[i] = max(0.0, cos)  # ReLU

        # ---- FoolsGold (Fung et al., RAID 2020) --------------------------
        # Pairwise cosine on HISTORICAL accumulated updates.  We build the
        # matrix from prior-round history (this round is added later).  On
        # the first round there is no history yet → all scores 0 (no
        # information).  Each client's score = max cosine to any peer.
        foolsgold_scores = self._foolsgold_scores(
            [u.client_id for u in updates], deltas
        )

        # ---- FLAME (Nguyen et al., USENIX Security 2022) -----------------
        # Pairwise cosine between current-round deltas, then HDBSCAN
        # clustering.  Clients outside the largest cluster are anomalies.
        cosine_matrix = self._cosine_matrix(deltas)
        flame_labels = self._flame_cluster(cosine_matrix)
        # Median-norm clipping reference is a FLAME side-output we expose
        # for logging and for potential aggregator use.
        median_norm = float(np.median(norms)) if norms else 0.0

        return {
            "deltas": deltas,
            "l2_norms": norms,
            "mean_pairwise_distance": self._mean_pairwise_distance(deltas),
            "dnc_scores": dnc_scores,
            "fltrust_scores": fltrust_scores,
            "foolsgold_scores": foolsgold_scores,
            "flame_labels": flame_labels,
            "flame_median_norm": median_norm,
            "client_ids": [u.client_id for u in updates],
        }

    def _summarize_features(self, features: dict) -> dict:
        """Round features for logging / LLM consumption."""
        cids = features["client_ids"]
        return {
            "l2_norms": {
                cid: round(v, 4) for cid, v in zip(cids, features["l2_norms"])
            },
            "dnc_scores": {
                cid: round(v, 4) for cid, v in zip(cids, features["dnc_scores"])
            },
            "fltrust_scores": {
                cid: round(v, 4) for cid, v in zip(cids, features["fltrust_scores"])
            },
            "foolsgold_scores": {
                cid: round(v, 4) for cid, v in zip(cids, features["foolsgold_scores"])
            },
            "flame_labels": {
                cid: int(v) for cid, v in zip(cids, features["flame_labels"])
            },
            "flame_median_norm": round(features["flame_median_norm"], 4),
            "mean_pairwise_distance": round(features["mean_pairwise_distance"], 4),
        }

    # ------------------------------------------------------------------
    # Internal: method dispatch
    # ------------------------------------------------------------------

    def _apply_method(self, method: str, params: dict, features: dict, idx: int):
        """Apply detection method to a single client. Returns (suspicious, confidence, reason)."""
        norm = features["l2_norms"][idx]

        if method == "norm_threshold":
            # Sun et al., 2019. Absolute L2 cap on the update.
            threshold = float(params.get("threshold", 10.0))
            suspicious = norm > threshold
            confidence = min(norm / threshold, 1.0) if suspicious else 0.0
            return suspicious, confidence, f"norm={norm:.3f} vs threshold={threshold}"

        elif method == "dnc":
            # Shejwalkar & Houmansadr, NDSS 2021.  Flag if projection on the
            # top singular direction is a positive z-score outlier.
            dnc_score = features["dnc_scores"][idx]
            mean_dnc = float(np.mean(features["dnc_scores"]))
            std_dnc = float(np.std(features["dnc_scores"]))
            threshold_z = float(params.get("threshold", 2.0))
            threshold = mean_dnc + threshold_z * std_dnc
            suspicious = dnc_score > threshold
            confidence = min(dnc_score / (threshold + 1e-8), 1.0) if suspicious else 0.0
            return suspicious, confidence, f"dnc_score={dnc_score:.3f} vs threshold={threshold:.3f}"

        elif method == "fltrust":
            # Cao et al., NDSS 2021.  Trust = ReLU(cos(client, server_root)).
            # Default threshold of 0.0 flags any client whose update is
            # anti-aligned with the server's root update (negative cosine
            # → clipped to 0 by ReLU).  Defender can raise this to enforce
            # stronger alignment.
            trust = features["fltrust_scores"][idx]
            threshold = float(params.get("threshold", 0.0))
            if all(s == 0.0 for s in features["fltrust_scores"]):
                # No server_delta was provided this round.
                return False, 0.0, "fltrust: no server_delta available"
            suspicious = trust <= threshold
            confidence = (1.0 - trust) if suspicious else 0.0
            return suspicious, confidence, f"fltrust_trust={trust:.3f} vs threshold={threshold:.3f}"

        elif method == "foolsgold":
            # Fung et al., RAID 2020.  Flag clients with high max-pairwise
            # cosine on historical updates — under non-IID this signals
            # collusion / sybils, since honest clients should look distinct.
            score = features["foolsgold_scores"][idx]
            threshold = float(params.get("threshold", 0.95))
            # When history is empty the scores are all 0.0 (no info).
            if all(s == 0.0 for s in features["foolsgold_scores"]):
                return False, 0.0, "foolsgold: no history yet"
            suspicious = score > threshold
            confidence = score if suspicious else 0.0
            return suspicious, confidence, f"foolsgold_score={score:.3f} vs threshold={threshold:.3f}"

        elif method == "flame":
            # Nguyen et al., USENIX Security 2022.  Flag clients outside the
            # largest HDBSCAN cluster; -1 (noise) is always flagged.
            label = features["flame_labels"][idx]
            labels = features["flame_labels"]
            # Find the largest non-noise cluster.
            counts: dict[int, int] = {}
            for lbl in labels:
                if lbl == -1:
                    continue
                counts[lbl] = counts.get(lbl, 0) + 1
            if not counts:
                # No structure found — be conservative and flag nothing.
                return False, 0.0, "flame: no clusters formed"
            majority = max(counts, key=counts.get)
            suspicious = label != majority
            confidence = 1.0 if suspicious else 0.0
            return (
                suspicious,
                confidence,
                f"flame: cluster={label}, majority={majority}",
            )

        else:
            logger.warning(f"Unknown method '{method}' — defaulting to norm_threshold")
            return self._apply_method("norm_threshold", params, features, idx)

    # ------------------------------------------------------------------
    # Internal: helpers for the new defenses
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pairwise_distance(deltas: list[torch.Tensor]) -> float:
        n = len(deltas)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += (deltas[i] - deltas[j]).norm().item()
                count += 1
        return total / count if count else 0.0

    @staticmethod
    def _cosine_matrix(deltas: list[torch.Tensor]) -> np.ndarray:
        """n×n cosine similarity matrix on the current-round deltas."""
        n = len(deltas)
        mat = np.eye(n, dtype=np.float32)
        norms = [d.norm().item() for d in deltas]
        for i in range(n):
            for j in range(i + 1, n):
                if norms[i] < 1e-12 or norms[j] < 1e-12:
                    cos = 0.0
                else:
                    cos = float((deltas[i] @ deltas[j]).item() / (norms[i] * norms[j]))
                mat[i, j] = cos
                mat[j, i] = cos
        return mat

    def _foolsgold_scores(
        self, client_ids: list[int], deltas: list[torch.Tensor]
    ) -> list[float]:
        """Per-client max pairwise cosine using HISTORICAL accumulated updates.

        On the first round (no history), returns zeros — the defender then
        sees an "uninformative" foolsgold signal and should pick another
        method or wait. From round 2 onwards the score is meaningful.
        """
        if not self._history:
            return [0.0] * len(deltas)

        n = len(deltas)
        # Build the historical vector for each current-round client. If a
        # client has no history yet (e.g. newly arrived), fall back to its
        # current-round delta so the matrix is well-defined.
        hist_vecs: list[torch.Tensor] = []
        for cid, d in zip(client_ids, deltas):
            hist_vecs.append(self._history.get(cid, d))

        norms = [v.norm().item() for v in hist_vecs]
        scores = [0.0] * n
        for i in range(n):
            max_cos = 0.0
            for j in range(n):
                if i == j:
                    continue
                if norms[i] < 1e-12 or norms[j] < 1e-12:
                    continue
                cos = float(
                    (hist_vecs[i] @ hist_vecs[j]).item() / (norms[i] * norms[j])
                )
                if cos > max_cos:
                    max_cos = cos
            scores[i] = max_cos
        return scores

    def _update_history(self, client_ids: list[int], deltas: list[torch.Tensor]):
        """Accumulate this round's deltas into per-client history (FoolsGold)."""
        for cid, d in zip(client_ids, deltas):
            if cid in self._history:
                self._history[cid] = self._history[cid] + d.detach().clone()
            else:
                self._history[cid] = d.detach().clone()

    @staticmethod
    def _flame_cluster(cosine_matrix: np.ndarray) -> list[int]:
        """HDBSCAN clustering on the cosine-distance matrix (FLAME).

        Falls back to a single-cluster labelling if `hdbscan` is unavailable
        or if the number of clients is below the minimum cluster size.
        """
        n = cosine_matrix.shape[0]
        if n < 2:
            return [0] * n

        # FLAME paper: min_cluster_size = floor(n/2) + 1, so the algorithm
        # is forced to find a SINGLE majority cluster (or none).
        min_cluster_size = n // 2 + 1

        # cosine distance in [0, 2]
        distance = np.clip(1.0 - cosine_matrix, 0.0, 2.0).astype(np.float64)
        np.fill_diagonal(distance, 0.0)

        try:
            import hdbscan  # type: ignore

            clusterer = hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=min_cluster_size,
                allow_single_cluster=True,
            )
            labels = clusterer.fit_predict(distance)
            return [int(l) for l in labels]
        except ImportError:
            logger.warning(
                "FLAME: hdbscan not installed — falling back to median-cosine "
                "majority labelling. Install with `pip install hdbscan` for "
                "the published algorithm."
            )
            return AnomalyDetector._flame_fallback(cosine_matrix, min_cluster_size)
        except Exception as e:
            logger.error(f"FLAME: HDBSCAN failed ({e}) — using fallback")
            return AnomalyDetector._flame_fallback(cosine_matrix, min_cluster_size)

    @staticmethod
    def _flame_fallback(cosine_matrix: np.ndarray, min_cluster_size: int) -> list[int]:
        """Simple fallback when HDBSCAN is unavailable.

        Each client whose median cosine similarity to peers is above the
        global median is placed in cluster 0 (majority); the rest are
        labelled -1 (noise). This is a coarse approximation of FLAME's
        density-based behaviour but preserves the API.
        """
        n = cosine_matrix.shape[0]
        median_per_client = []
        for i in range(n):
            peers = np.delete(cosine_matrix[i], i)
            median_per_client.append(float(np.median(peers)))
        global_med = float(np.median(median_per_client))
        labels = [0 if m >= global_med else -1 for m in median_per_client]
        # Guard: if too few in the majority cluster, mark all 0 (trust everyone).
        if sum(1 for l in labels if l == 0) < min_cluster_size:
            return [0] * n
        return labels
