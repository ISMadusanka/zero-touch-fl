"""Benign client — honest local training."""

import copy
import torch
import torch.nn as nn
from core.types import ModelUpdate


class BenignClient:
    """Trains locally on its data shard and returns honest weight updates."""

    def __init__(self, client_id: int, data_loader, lr: float, local_epochs: int, device: str):
        self.client_id = client_id
        self.data_loader = data_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device

    def train(self, global_model: nn.Module) -> ModelUpdate:
        """Train on local data starting from global model weights."""
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()

        return ModelUpdate(
            client_id=self.client_id,
            weights={k: v.cpu() for k, v in model.state_dict().items()},
        )
