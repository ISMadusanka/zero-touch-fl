"""Gaussian noise model poisoning: sends a random weight matrix sampled from N(0, σ²I)."""

import torch
from core.interfaces import BaseAttack
from attacks.registry import register


@register("gaussian_noise")
class GaussianNoiseAttack(BaseAttack):

    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Return a malicious update sampled purely from N(0, σ²I).

        The attacker ignores local data entirely. The malicious gradient is:
            g_mal ~ N(0, σ²I)
        i.e. each parameter tensor is replaced by a zero-mean Gaussian sample
        with variance σ².
        """
        sigma = params.get("sigma", 1.0)          # σ (std dev); variance = σ²
        poisoned = {}
        for key in weights:
            # Sample directly from N(0, σ²I) — local weights are ignored
            poisoned[key] = torch.randn_like(weights[key]) * sigma
        return poisoned
