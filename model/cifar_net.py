"""Compact CNN for CIFAR-10 — ~33.8k trainable parameters.

Architecture (mirrors ``MnistNet``'s single ``nn.Sequential`` named ``net``, so
state_dict keys stay ``net.<index>.<weight|bias>`` and every layer-aware
component — DeFL's per-layer grouping, ``detector.features.layer_groups``, the
attacker's ``target: "net.6"`` operator addressing — works unchanged):

    Conv2d(3,16,3,pad=1)  ReLU  MaxPool2d(2)   32x32 -> 16x16      448 params
    Conv2d(16,32,3,pad=1) ReLU  MaxPool2d(2)   16x16 -> 8x8      4,640 params
    Conv2d(32,64,3,pad=1) ReLU  MaxPool2d(2)     8x8 -> 4x4     18,496 params
    Flatten  Linear(1024, 10)                                   10,250 params
    -------------------------------------------------------------------------
    total                                                       33,834 params

Why a CNN and not the MLP shape used for MNIST: a 49-feature average-pooled MLP
gets ~30% on CIFAR-10, which is too close to the 10% chance floor for an accuracy
-degradation attack to mean anything. 34k params is still small enough that a
full FL round (20 clients + a test-set pass) stays cheap and that every defense's
flatten-the-whole-model math (FLTrust, Multi-Krum, DnC) is unchanged in cost.

**No BatchNorm, deliberately.** BN running statistics live in the state_dict, so
they would be FedAvg-averaged, fed to the attacker's operators, and clamped like
weights — and ``num_batches_tracked`` is an int64 counter that has no meaningful
"scale by 1.5". Keeping the model BN-free keeps every tensor in the state_dict a
genuine learnable parameter, which is the contract the attack DSL and the
defenses assume.
"""

import torch.nn as nn


class Cifar10Net(nn.Module):
    """3-block conv net for 3x32x32 inputs. Total trainable params: 33,834."""

    def __init__(self, n_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # 0
            nn.ReLU(),                                             # 1
            nn.MaxPool2d(2),                                       # 2  -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),           # 3
            nn.ReLU(),                                             # 4
            nn.MaxPool2d(2),                                       # 5  -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # 6
            nn.ReLU(),                                             # 7
            nn.MaxPool2d(2),                                       # 8  -> 4x4
            nn.Flatten(),                                          # 9  -> 1024
            nn.Linear(64 * 4 * 4, n_classes),                      # 10
        )

    def forward(self, x):
        return self.net(x)
