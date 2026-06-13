"""PyTorch Dataset for loading FL client weight JSON files.

Loads weight JSON files from `logs/training_client_weights/`, flattens all
weight tensors into a single vector, and optionally normalizes them.
"""

import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class WeightDataset(Dataset):
    """Dataset of flattened FL client weight vectors from training rounds.

    Each sample is a 1-D tensor of all model parameters concatenated in
    a consistent order (net.2.weight, net.2.bias, net.4.weight, net.4.bias).

    Args:
        weights_dir: Path to directory containing round JSON files.
        normalize: If True, standardize to zero-mean unit-variance.
    """

    # Fixed key order to ensure consistent flattening across all samples
    WEIGHT_KEYS = ["net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"]

    def __init__(self, weights_dir: str, normalize: bool = True):
        self.weights_dir = weights_dir
        self.normalize = normalize

        # Load all JSON files sorted by round number
        self.files = sorted(
            [f for f in os.listdir(weights_dir) if f.endswith(".json")],
            key=lambda x: int(x.split("_")[1]),  # sort by round number
        )

        if not self.files:
            raise FileNotFoundError(f"No JSON weight files found in {weights_dir}")

        logger.info(f"WeightDataset: found {len(self.files)} weight files in {weights_dir}")

        # Load and flatten all weights
        self.data = self._load_all()

        # Compute and apply normalization
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0)
        # Avoid division by zero for constant features
        self.std[self.std < 1e-8] = 1.0

        if self.normalize:
            self.data = (self.data - self.mean) / self.std
            logger.info("WeightDataset: normalization applied (zero-mean, unit-variance)")

        logger.info(
            f"WeightDataset: loaded {len(self.data)} samples, "
            f"each with {self.data.shape[1]} features"
        )

    def _load_all(self) -> torch.Tensor:
        """Load all JSON files and flatten weights into tensors."""
        all_vectors = []

        for fname in self.files:
            path = os.path.join(self.weights_dir, fname)
            with open(path, "r") as f:
                data = json.load(f)

            weights = data["weights"]
            # Flatten in consistent key order
            flat_parts = []
            for key in self.WEIGHT_KEYS:
                tensor = torch.tensor(weights[key], dtype=torch.float32)
                flat_parts.append(tensor.flatten())

            vector = torch.cat(flat_parts)
            all_vectors.append(vector)

        return torch.stack(all_vectors)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def get_normalization_stats(self) -> dict:
        """Return normalization stats for use during inference.

        Save these alongside the VAE model so new weights can be
        normalized consistently before being fed to the VAE.
        """
        return {"mean": self.mean, "std": self.std}

    def get_input_dim(self) -> int:
        """Return the dimension of each flattened weight vector."""
        return self.data.shape[1]
