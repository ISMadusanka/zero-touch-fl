import torch
import numpy as np
from core.types import ModelUpdate
import logging

logger = logging.getLogger(__name__)

class LayeredDetector:
    """
    A 4-layer statistical defense pipeline:
    1. FLTrust (Cosine similarity with root update)
    2. PCA + K-Means (Anomaly clustering)
    3. L2 Norm Clipping (Influence bounding)
    4. Trimmed Mean (Extreme filtering)
    """

    def __init__(self, root_loader=None, device="cpu"):
        self.root_loader = root_loader
        self.device = device

    def get_features(self, updates: list[ModelUpdate], global_weights: dict) -> dict:
        """Computes features for each client update. Used for XGBoost and LLM analysis."""
        # 1. Flatten updates into vectors
        deltas = []
        for u in updates:
            deltas.append(self._flatten_update(u.weights, global_weights))
        
        deltas_stack = torch.stack(deltas) # (N, D)
        
        # --- Layer 1: FLTrust ---
        # Note: In a real scenario, root_update would come from a clean dataset.
        # If root_loader is missing, we use a neutral reference.
        root_update = self._compute_root_update_from_weights(global_weights)
        fl_trust_scores = self._compute_fl_trust(deltas_stack, root_update)

        # --- Layer 2: PCA + K-Means ---
        cluster_scores = self._compute_clusters(deltas_stack)

        # --- Layer 3: L2 Clipping Score ---
        clipping_scores, raw_norms = self._compute_clipping(deltas_stack)

        # --- Layer 4: Trimmed Mean (Z-Score) ---
        trim_scores = self._compute_trimmed_status(deltas_stack)

        # Compile evidence
        evidence = {}
        for i, u in enumerate(updates):
            evidence[f"client_{u.client_id}"] = {
                "layer_1_fl_trust": round(float(fl_trust_scores[i]), 4),
                "layer_2_cluster": round(float(cluster_scores[i]), 4),
                "layer_3_clipping": round(float(clipping_scores[i]), 4),
                "layer_4_is_trimmed": round(float(trim_scores[i]), 4),
                "raw_norm": round(float(raw_norms[i]), 4)
            }
        
        return evidence

    def analyze(self, updates: list[ModelUpdate], global_weights: dict, strategy: dict) -> list:
        """
        Actually this logic is now handled in main.py by passing features 
        to ExplainabilityEngine and then letting the LLM decide.
        This method is kept for interface compatibility if needed.
        """
        return []

    def _flatten_update(self, update_weights: dict, global_weights: dict) -> torch.Tensor:
        flat_parts = []
        for k in sorted(global_weights.keys()):
            delta = (update_weights[k].to(self.device) - global_weights[k].to(self.device))
            flat_parts.append(delta.flatten().float())
        return torch.cat(flat_parts)

    def _compute_root_update_from_weights(self, global_weights: dict) -> torch.Tensor | None:
        """Placeholder for root update logic."""
        if self.root_loader is None:
            return None
        # In a real impl, we'd train on root_loader here. 
        # For now, we return None and FLTrust will return 1.0 (neutral).
        return None

    def _compute_fl_trust(self, deltas: torch.Tensor, root_update: torch.Tensor | None) -> torch.Tensor:
        if root_update is None:
            return torch.ones(len(deltas))
        
        cos = torch.nn.functional.cosine_similarity(deltas, root_update.unsqueeze(0))
        # Sharpened curve for FLTrust
        trust = torch.where(cos > 0, torch.sigmoid(12 * (cos - 0.15)), torch.zeros_like(cos))
        return trust

    def _compute_clusters(self, deltas: torch.Tensor) -> np.ndarray:
        if len(deltas) < 3:
            return np.zeros(len(deltas))
        
        centered = deltas - deltas.mean(dim=0)
        try:
            # Simple PCA via SVD
            U, S, V = torch.pca_lowrank(centered, q=2)
            projected = torch.matmul(centered, V[:, :2])
        except:
            return np.zeros(len(deltas))

        X = projected
        mu = X[torch.randperm(X.size(0))[:2]]
        
        for _ in range(10):
            dist = torch.cdist(X, mu)
            labels = torch.argmin(dist, dim=1)
            for k in range(2):
                if (labels == k).any():
                    mu[k] = X[labels == k].mean(dim=0)
        
        counts = torch.bincount(labels, minlength=2)
        benign_idx = torch.argmax(counts).item()
        benign_centroid = mu[benign_idx]

        distances = torch.norm(X - benign_centroid, dim=1)
        median_dist = torch.median(distances) + 1e-9
        scores = distances / median_dist
            
        return scores.cpu().numpy()

    def _compute_clipping(self, deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norms = torch.norm(deltas, dim=1)
        median_norm = torch.median(norms) + 1e-9
        scores = norms / median_norm
        return scores, norms

    def _compute_trimmed_status(self, deltas: torch.Tensor) -> np.ndarray:
        norms = torch.norm(deltas, dim=1).cpu().numpy()
        mean = np.mean(norms)
        std = np.std(norms) + 1e-9
        z_scores = np.abs(norms - mean) / std
        return z_scores
