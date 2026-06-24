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
        # Optional GPU-resident cache of the fixed test set (see build_eval_cache).
        self._eval_X = None
        self._eval_y = None
        logger.info(f"Global model initialized — {count_parameters(self.model)} params")

    def get_global_weights(self) -> dict:
        """Return a CPU copy of the global model state dict."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_global_weights(self, state_dict: dict):
        """Load weights into the global model."""
        self.model.load_state_dict(state_dict)

    def build_eval_cache(self, test_loader):
        """Cache the entire (fixed) test set as GPU tensors for fast eval.

        The Phase-2 reward oracle calls ``evaluate`` G+1 times per round; the
        test set is fixed and tiny, so materializing it once on-device and doing
        a single forward is bit-identical to the per-batch DataLoader loop but
        removes ~40 host<->device transfers + syncs per evaluation.
        """
        xs, ys = [], []
        for data, target in test_loader:
            xs.append(data)
            ys.append(target)
        self._eval_X = torch.cat(xs).to(self.device)
        self._eval_y = torch.cat(ys).to(self.device)
        logger.info(f"Eval cache built — {self._eval_X.shape[0]} samples on {self.device}")

    def evaluate(self, test_loader=None) -> float:
        """Evaluate global model on test data. Returns accuracy.

        Uses the GPU-resident cache when available (one forward, integer
        correct/total — bit-identical to the batched loop), else falls back to
        iterating ``test_loader`` (used in Phase 1 before the cache is built).
        """
        self.model.eval()
        if self._eval_X is not None:
            with torch.no_grad():
                pred = self.model(self._eval_X).argmax(dim=1)
                correct = int(pred.eq(self._eval_y).sum().item())
                total = int(self._eval_y.numel())
            return correct / total if total > 0 else 0.0

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
