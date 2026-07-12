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
        # Optional preloaded test set (one contiguous device tensor pair). When
        # set, evaluate() runs a single (chunked) forward instead of re-iterating
        # a DataLoader — see preload_test_set().
        self._eval_x = None
        self._eval_y = None
        logger.info(f"Global model initialized — {count_parameters(self.model)} params")

    def get_global_weights(self) -> dict:
        """Return a CPU copy of the global model state dict."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_global_weights(self, state_dict: dict):
        """Load weights into the global model."""
        self.model.load_state_dict(state_dict)

    def preload_test_set(self, test_loader) -> None:
        """Materialize the whole test set into two device tensors ONCE.

        Evaluation during Phase 2 is run up to ``G + 1`` times per RL round (once
        per scored rollout plus the committed round). Re-iterating a DataLoader
        every call pays per-batch host→device copies and Python loop overhead each
        time. Concatenating the test set into a single ``(N, ...)`` tensor pair on
        the device lets ``evaluate()`` run as one (chunked) forward pass, which for
        the tiny MLP is essentially free. Idempotent — safe to call again on reset.
        """
        xs, ys = [], []
        for data, target in test_loader:
            xs.append(data)
            ys.append(target)
        if not xs:
            self._eval_x = self._eval_y = None
            return
        self._eval_x = torch.cat(xs, dim=0).to(self.device)
        self._eval_y = torch.cat(ys, dim=0).to(self.device)
        logger.info(
            f"Preloaded test set for fast eval: {self._eval_x.shape[0]} samples on {self.device}"
        )

    def evaluate(self, test_loader=None, n_samples: int | None = None) -> float:
        """Global-model test accuracy.

        Fast path: when the test set has been preloaded (``preload_test_set``) and
        no explicit ``test_loader`` is passed, evaluate directly on the cached
        device tensors. ``n_samples`` restricts evaluation to the first
        ``n_samples`` cached examples — the test loader is unshuffled, so this is a
        deterministic subset. Used for the per-rollout REWARD signal, which only
        needs a *relative* accuracy drop and does not require the exact 10k-image
        test number (a constant subsample bias cancels in GRPO's group-relative
        advantage).

        Legacy path: when an explicit ``test_loader`` is given (e.g. Phase 1), or
        nothing has been preloaded, iterate the loader batch by batch (unchanged).
        """
        self.model.eval()
        if test_loader is None and self._eval_x is not None:
            return self._evaluate_preloaded(n_samples)
        if test_loader is None:
            raise ValueError(
                "FedServer.evaluate(): no test_loader given and no preloaded test set. "
                "Call preload_test_set() first, or pass a loader."
            )
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

    def _evaluate_preloaded(self, n_samples: int | None) -> float:
        """Accuracy over the preloaded tensors (optionally the first n_samples)."""
        x, y = self._eval_x, self._eval_y
        if n_samples is not None and 0 < n_samples < x.shape[0]:
            x, y = x[:n_samples], y[:n_samples]
        total = x.shape[0]
        if total == 0:
            return 0.0
        correct = 0
        chunk = 8192  # tiny MLP; one or two chunks covers the full 10k test set
        with torch.no_grad():
            for i in range(0, total, chunk):
                output = self.model(x[i:i + chunk])
                correct += output.argmax(dim=1).eq(y[i:i + chunk]).sum().item()
        return correct / total
