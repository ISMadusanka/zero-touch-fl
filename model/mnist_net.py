"""Tiny MLP for MNIST — ~805 trainable parameters."""

import torch.nn as nn


class MnistNet(nn.Module):
    """
    Linear(784, 1) → ReLU → Linear(1, 10)
    Total params: 784 + 1 + 10 + 10 = 805
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1),
            nn.ReLU(),
            nn.Linear(1, 10),
        )

    def forward(self, x):
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
