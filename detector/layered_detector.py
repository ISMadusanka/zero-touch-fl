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

        # --- Layer 2: PCA + K-Means ---
        cluster_labels = self._compute_clusters(deltas_stack)

        # --- Layer 3: L2 Clipping ---
        clipping_ratios, raw_norms = self._compute_clipping(deltas_stack)

        # --- Layer 4: Trimmed Mean ---
        trimmed_status = self._compute_trimmed_status(deltas_stack)

        # Compile evidence
        evidence = {}
        for i, u in enumerate(updates):
            evidence[f"client_{u.client_id}"] = {
                "layer_1_fl_trust": round(float(fl_trust_scores[i]), 4),
                "layer_2_cluster": int(cluster_labels[i]),
                "layer_3_clipping": round(float(clipping_ratios[i]), 4),
                "layer_4_is_trimmed": bool(trimmed_status[i]),
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
        
        optimizer = torch.optim.SGD(model_copy.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
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
        """Layer 1: FLTrust score = ReLU(CosineSimilarity(update, root_update))"""
        if root_update is None:
            return torch.ones(len(deltas))
        
        cos = torch.nn.functional.cosine_similarity(deltas, root_update.unsqueeze(0))
        return torch.clamp(cos, min=0.0)

    def _compute_clusters(self, deltas: torch.Tensor) -> np.ndarray:
        """Layer 2: PCA (2D) + K-Means (K=2) using pure Torch."""
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
        # Initialize centroids randomly
        mu = X[torch.randperm(X.size(0))[:2]]
        
        for _ in range(10): # 10 iterations
            # Compute distances
            dist = torch.cdist(X, mu) # (N, 2)
            labels = torch.argmin(dist, dim=1)
            # Update centroids
            for k in range(2):
                if (labels == k).any():
                    mu[k] = X[labels == k].mean(dim=0)
        
        labels_np = labels.cpu().numpy()
        # Ensure label 0 is the larger cluster (heuristic for "benign")
        if np.sum(labels_np) > len(labels_np) / 2:
            labels_np = 1 - labels_np
            
        return labels_np

    def _compute_clipping(self, deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer 3: L2 norm clipping ratio based on median norm."""
        norms = torch.norm(deltas, dim=1)
        median_norm = torch.median(norms)
        # Ratio = min(1, median_norm / current_norm)
        ratios = torch.clamp(median_norm / (norms + 1e-9), max=1.0)
        return ratios, norms

    def _compute_trimmed_status(self, deltas: torch.Tensor, alpha=0.2) -> np.ndarray:
        """Layer 4: Trimmed Mean (Flags updates in top/bottom alpha percentiles of L2 norm)."""
        norms = torch.norm(deltas, dim=1).cpu().numpy()
        n = len(norms)
        k = int(n * alpha)
        if k == 0: return np.zeros(n, dtype=bool)
        
        sorted_indices = np.argsort(norms)
        trimmed_indices = np.concatenate([sorted_indices[:k], sorted_indices[-k:]])
        
        status = np.zeros(n, dtype=bool)
        status[trimmed_indices] = True
        return status
