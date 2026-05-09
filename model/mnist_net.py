"""Tiny MLP for MNIST — ~970 trainable parameters.

Architecture:
    AvgPool2d(4) reduces 28×28 → 7×7 = 49 features (no params)
    Linear(49, 16) → ReLU → Linear(16, 10)
    Params: (49*16 + 16) + (16*10 + 10) = 800 + 170 = 970
"""

import torch.nn as nn


class MnistNet(nn.Module):
    """
    AvgPool2d(4) → Flatten → Linear(49, 16) → ReLU → Linear(16, 10)
    Total trainable params: 970
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),  # 28×28 → 7×7 (no learnable params)
            nn.Flatten(),                  # 7×7 → 49
            nn.Linear(49, 16),             # 49*16 + 16 = 800
            nn.ReLU(),
            nn.Linear(16, 10),             # 16*10 + 10 = 170
        )

    def forward(self, x):
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
