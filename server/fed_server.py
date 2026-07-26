"""Central FL server — orchestrates training and simulation phases."""

import logging
import torch
import torch.nn as nn

from core.types import ClassEval, ModelUpdate
from model.mnist_net import MnistNet, count_parameters

logger = logging.getLogger(__name__)

# MNIST. Only used as the default breakdown width for ``evaluate_per_class``.
N_CLASSES = 10


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

    def _confusion(self, test_loader, n_classes: int):
        """ONE pass over the test set.

        Returns ``(correct, total, per_class_correct, per_class_total)``. Both the
        overall accuracy and the per-class recalls come out of this single pass,
        so the targeted experiment's class breakdown costs **nothing extra** over
        the untargeted path's plain accuracy.
        """
        self.model.eval()
        correct = total = 0
        cls_correct = torch.zeros(n_classes, dtype=torch.long)
        cls_total = torch.zeros(n_classes, dtype=torch.long)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                hit = pred.eq(target)
                correct += hit.sum().item()
                total += target.size(0)
                # Bin the hits/samples by TRUE label -> per-class recall.
                t = target.detach().cpu()
                h = hit.detach().cpu()
                valid = (t >= 0) & (t < n_classes)
                cls_total += torch.bincount(t[valid], minlength=n_classes)
                cls_correct += torch.bincount(t[valid & h], minlength=n_classes)
        return correct, total, cls_correct.tolist(), cls_total.tolist()

    def evaluate(self, test_loader) -> float:
        """Evaluate global model on test data. Returns accuracy."""
        correct, total, _, _ = self._confusion(test_loader, N_CLASSES)
        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Global model test accuracy: {accuracy:.4f}")
        return accuracy

    def evaluate_per_class(self, test_loader, n_classes: int = N_CLASSES) -> ClassEval:
        """Overall accuracy PLUS per-class recall, from one pass over the test set.

        This is the evaluation the targeted experiment is scored on: a
        "misclassify label L" attack must push ``per_class[L]`` down while leaving
        the other entries where the clean model had them. Classes with no test
        samples get recall 0.0 (they cannot be attacked or damaged either way).
        """
        correct, total, cls_correct, cls_total = self._confusion(test_loader, n_classes)
        overall = correct / total if total > 0 else 0.0
        per_class = [(c / t if t > 0 else 0.0) for c, t in zip(cls_correct, cls_total)]
        logger.debug(
            "Global model per-class recall: "
            + " ".join(f"{i}={v:.3f}" for i, v in enumerate(per_class))
        )
        return ClassEval(overall=overall, per_class=per_class, support=cls_total)
