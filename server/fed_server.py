"""Central FL server — orchestrates training and simulation phases."""

import logging
import torch
import torch.nn as nn

from core.types import ModelUpdate
from model.mnist_net import MnistNet, count_parameters

logger = logging.getLogger(__name__)


class FedServer:
    """Holds the global model and provides evaluation."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = MnistNet().to(device)
        logger.info(f"Global model initialized — {count_parameters(self.model)} params")

    def get_global_weights(self) -> dict:
        """Return a CPU copy of the global model state dict."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_global_weights(self, state_dict: dict):
        """Load weights into the global model."""
        self.model.load_state_dict(state_dict)

    def evaluate(self, test_loader) -> float:
        """Evaluate global model on test data. Returns accuracy."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Global model test accuracy: {accuracy:.4f}")
        return accuracy
