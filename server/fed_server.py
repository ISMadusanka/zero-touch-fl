"""Central FL server — orchestrates training and simulation phases."""

import copy
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

    def compute_root_delta(
        self,
        root_loader,
        lr: float,
        local_epochs: int,
    ) -> torch.Tensor:
        """Train the server's reference model on the root dataset and return
        the flat weight delta (FLTrust, Cao et al., NDSS 2021).

        The server keeps a small clean root dataset and runs the same
        local-training procedure as the clients. The resulting flat delta
        is the trust anchor against which client updates are scored
        (cosine similarity). The global model itself is NOT modified.
        """
        global_state = self.get_global_weights()
        ref_model = copy.deepcopy(self.model).to(self.device)
        ref_model.train()
        optimizer = torch.optim.SGD(ref_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(local_epochs):
            for data, target in root_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = ref_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        new_state = {k: v.cpu() for k, v in ref_model.state_dict().items()}
        flat = torch.cat([
            (new_state[k] - global_state[k]).flatten().float()
            for k in global_state
        ])
        return flat
