"""Fang — directed-deviation attack (Fang, Cao, Jia & Gong, USENIX Security 2020).

Fang's "Local Model Poisoning" crafts each malicious coordinate to sit on the
opposite side of the honest values from where they are heading: it deviates the
benign mean AGAINST its own sign. The original method tailors the deviation
magnitude to a *known* aggregation rule (Krum / trimmed-mean / median), halving
it until the malicious update survives that rule. This project's benchmark holds
one attack fixed across every defense, so we implement the **AGR-agnostic**
variant chosen for this comparison: a fixed directed deviation, scaled per
coordinate by the honest spread, with no knowledge of the defender.

``Delta_mal = mu - lambda * sign(mu) * scale``, where ``scale`` is the
coordinate-wise honest std ``sigma`` when it is informative, and falls back to
``|mu|`` when only one client is controlled (so the deviation is still nonzero).
All malicious clients send this same update. ``lambda`` controls how far past the
honest values the deviation lands; larger is stronger but easier to flag.
"""
from __future__ import annotations

import torch

from benchmark.attacks.base import Attack, BenignStats

_DEFAULT_LAMBDA = 3.0


class FangAttack(Attack):
    name = "fang"
    colludes = True

    def __init__(self, lam: float = _DEFAULT_LAMBDA):
        self.lam = float(lam)

    def craft(self, pool_benign, global_sd, budget, gen):
        stats = BenignStats(pool_benign, global_sd)
        delta = {}
        for k in stats.keys:
            mu = stats.mean[k]
            sigma = stats.std[k]
            # Per-coordinate deviation scale: honest std where it exists, else |mu|
            # so a single-client pool still produces a real (nonzero) deviation.
            scale = torch.where(sigma > 0, sigma, mu.abs())
            delta[k] = mu - self.lam * torch.sign(mu) * scale
        w_mal = stats.weights_from_delta(delta)
        return {cid: {k: v.clone() for k, v in w_mal.items()}
                for cid in self.choose_ids(pool_benign, budget)}
