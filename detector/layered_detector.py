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

    def analyze(self, updates: list[ModelUpdate], global_model: torch.nn.Module) -> dict:
        """
        Runs the 4-layer pipeline and returns a structured evidence JSON.
        """
        # 1. Flatten updates into vectors
        deltas = []
        global_params = {k: v.to(self.device) for k, v in global_model.state_dict().items()}
        
        for u in updates:
            flat = torch.cat([
                (u.weights[k].to(self.device) - global_params[k]).flatten().float()
                for k in global_params
            ])
            deltas.append(flat)
        
        deltas_stack = torch.stack(deltas) # (N, D)
        
        # --- Layer 1: FLTrust ---
        root_update = self._compute_root_update(global_model)
        fl_trust_scores = self._compute_fl_trust(deltas_stack, root_update)

        # --- Layer 2: PCA + K-Means Anomaly Scoring ---
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
                "layer_2_cluster_score": round(float(cluster_scores[i]), 4),
                "layer_3_clipping_score": round(float(clipping_scores[i]), 4),
                "layer_4_trim_score": round(float(trim_scores[i]), 4),
                "raw_norm": round(float(raw_norms[i]), 4)
            }
        
        return evidence

    def _compute_root_update(self, global_model: torch.nn.Module) -> torch.Tensor:
        """Calculates a clean update using the server's root dataset."""
        if self.root_loader is None:
            # Fallback: if no root data, use mean of all updates (less secure but functional)
            return None
        
        # Simple one-epoch train on root data
        model_copy = type(global_model)()
        model_copy.load_state_dict(global_model.state_dict())
        model_copy.to(self.device)
        model_copy.train()
        
        optimizer = torch.optim.SGD(model_copy.parameters(), lr=0.05) # Increased LR
        criterion = torch.nn.CrossEntropyLoss()
        
        # Train for 2 epochs on root data for a stronger reference signal
        for _ in range(2):
            for data, target in self.root_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model_copy(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Calculate delta
        root_delta = torch.cat([
            (p.data - global_model.state_dict()[k].to(self.device)).flatten()
            for k, p in model_copy.named_parameters()
        ])
        return root_delta

    def _compute_fl_trust(self, deltas: torch.Tensor, root_update: torch.Tensor) -> torch.Tensor:
        """
        Layer 1: Relative FLTrust scoring. 
        Instead of absolute thresholds (which fail in Non-IID), we score clients based on 
        how their directional similarity compares to the rest of the group.
        
        Outliers (either too low OR suspiciously high compared to the group) are penalized.
        """
        if root_update is None:
            return torch.ones(len(deltas))
        
        # 1. Compute raw cosine similarity with root update
        cos = torch.nn.functional.cosine_similarity(deltas, root_update.unsqueeze(0))
        
        if len(cos) < 2:
            return torch.ones(len(deltas))

        # 2. Compute group statistics for similarities
        mean_cos = torch.mean(cos)
        std_cos = torch.std(cos) + 1e-9
        
        # 3. Calculate Z-scores (How many standard deviations from the group mean?)
        z_scores = (cos - mean_cos) / std_cos
        
        # 4. Convert Z-scores to a 0-1 Trust Score
        # We use a Gaussian kernel (RBF): exp(-0.5 * z^2)
        # Clients near the group average get ~1.0 trust.
        # Clients that are outliers (in either direction) get scores closer to 0.0.
        trust = torch.exp(-0.5 * (z_scores**2))
        
        # 5. Optional: Additionally penalize anything with negative raw cosine similarity
        # (Negative similarity is almost always a sign of a strong attack like sign-flip)
        trust = torch.where(cos < 0, trust * 0.1, trust)
        
        return trust

    def _compute_clusters(self, deltas: torch.Tensor) -> np.ndarray:
        """Layer 2: PCA + K-Means Anomaly Scoring. Returns distance from benign centroid."""
        if len(deltas) < 3:
            return np.zeros(len(deltas))
        
        # PCA via SVD
        centered = deltas - deltas.mean(dim=0)
        try:
            U, S, V = torch.pca_lowrank(centered, q=2)
            projected = torch.matmul(centered, V[:, :2]) # (N, 2)
        except:
            return np.zeros(len(deltas))

        # Simple K-Means (K=2)
        X = projected
        mu = X[torch.randperm(X.size(0))[:2]]
        
        for _ in range(10):
            dist = torch.cdist(X, mu) # (N, 2)
            labels = torch.argmin(dist, dim=1)
            for k in range(2):
                if (labels == k).any():
                    mu[k] = X[labels == k].mean(dim=0)
        
        # Identify the "Benign" centroid (the one with the most members)
        counts = torch.bincount(labels, minlength=2)
        benign_idx = torch.argmax(counts).item()
        benign_centroid = mu[benign_idx]

        # Score = Euclidean distance from the benign centroid
        distances = torch.norm(X - benign_centroid, dim=1)
        # Normalize by median distance to make it a relative "Anomaly Score"
        median_dist = torch.median(distances) + 1e-9
        scores = distances / median_dist
            
        return scores.cpu().numpy()

    def _compute_clipping(self, deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer 3: L2 norm ratio based on median norm (Higher = More Suspicious)."""
        norms = torch.norm(deltas, dim=1)
        median_norm = torch.median(norms) + 1e-9
        # Score = current_norm / median_norm (1.0 is baseline, >1.0 is suspicious)
        scores = norms / median_norm
        return scores, norms

    def _compute_trimmed_status(self, deltas: torch.Tensor) -> np.ndarray:
        """Layer 4: Statistical Z-Score (Distance from mean in standard deviations)."""
        norms = torch.norm(deltas, dim=1).cpu().numpy()
        mean = np.mean(norms)
        std = np.std(norms) + 1e-9
        
        # Z-Score = |x - mean| / std
        z_scores = np.abs(norms - mean) / std
        return z_scores