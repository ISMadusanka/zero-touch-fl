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

    def evaluate(self, test_loader, return_per_class: bool = False):
        """Evaluate global model on test data.

        Returns accuracy (float) normally.  When ``return_per_class`` is True,
        returns ``(accuracy, {class_id: class_accuracy})``."""
        self.model.eval()
        correct, total = 0, 0
        class_correct: dict[int, int] = {}
        class_total: dict[int, int] = {}
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                if return_per_class:
                    for t, p in zip(target.cpu().tolist(), pred.cpu().tolist()):
                        class_total[t] = class_total.get(t, 0) + 1
                        if t == p:
                            class_correct[t] = class_correct.get(t, 0) + 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Global model test accuracy: {accuracy:.4f}")

        if return_per_class:
            class_acc = {
                c: (class_correct.get(c, 0) / class_total[c])
                for c in sorted(class_total)
            }
            logger.info(f"Per-class accuracies: {{{', '.join(f'{c}: {a:.4f}' for c, a in class_acc.items())}}}")
            return accuracy, class_acc
        return accuracy
