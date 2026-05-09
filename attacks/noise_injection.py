"""Noise injection model poisoning: adds Gaussian noise to weights."""

import torch
from core.interfaces import BaseAttack
from attacks.registry import register


@register("noise_injection")
class NoiseInjectionAttack(BaseAttack):

    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Add Gaussian noise scaled by `scale` to client weights."""
        scale = params.get("scale", 1.0)
        poisoned = {}
        for key in weights:
            noise = torch.randn_like(weights[key]) * scale
            poisoned[key] = weights[key] + noise
        return poisoned
