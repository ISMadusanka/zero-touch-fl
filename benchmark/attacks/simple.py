"""Trivial Byzantine baselines: Gaussian noise, sign-flip, and scaling.

These are the weak reference points every FL-robustness paper reports first
(the Byzantine baseline from Blanchard et al., "Machine Learning with
Adversaries" / Krum, NeurIPS 2017, and the naive boosting/sign-flip attacks).
Unlike the optimization attacks, each malicious client manipulates **its own**
honest update independently, so the poisoned updates are diverse (they are not a
single colluding vector) — which is both the faithful form of these attacks and
harder for a Sybil/pairwise-cosine check to flag than identical clones.
"""
from __future__ import annotations

import torch

from benchmark.attacks.base import Attack, BenignStats


class NoiseAttack(Attack):
    """Add zero-mean Gaussian noise to each malicious client's honest update.

    ``Delta_mal_i = Delta_i + N(0, sigma^2)`` (equivalently the noise is added to
    the client's weights). A large ``sigma`` relative to the honest per-weight
    step is a classic Byzantine disruption; a small one is a stealth probe.
    """

    name = "noise"
    colludes = False

    def __init__(self, sigma: float = 1.0):
        self.sigma = float(sigma)

    def craft(self, pool_benign, global_sd, budget, gen):
        stats = BenignStats(pool_benign, global_sd)
        out = {}
        for cid in self.choose_ids(pool_benign, budget):
            delta = {k: stats.delta_by_key[k][stats.cids.index(cid)].clone()
                     for k in stats.keys}
            for k in stats.keys:
                noise = torch.randn(delta[k].shape, generator=gen) * self.sigma
                delta[k] = delta[k] + noise
            out[cid] = stats.weights_from_delta(delta)
        return out


class SignFlipAttack(Attack):
    """Negate each malicious client's honest update: ``Delta_mal_i = -factor * Delta_i``.

    ``factor = 1`` is a pure sign flip; ``factor > 1`` also boosts the flipped
    update so it pulls the aggregate harder (at the cost of a larger, more
    detectable norm).
    """

    name = "sign_flip"
    colludes = False

    def __init__(self, factor: float = 1.0):
        self.factor = float(factor)

    def craft(self, pool_benign, global_sd, budget, gen):
        stats = BenignStats(pool_benign, global_sd)
        out = {}
        for cid in self.choose_ids(pool_benign, budget):
            i = stats.cids.index(cid)
            delta = {k: -self.factor * stats.delta_by_key[k][i] for k in stats.keys}
            out[cid] = stats.weights_from_delta(delta)
        return out


class ScalingAttack(Attack):
    """Boost each malicious client's honest update: ``Delta_mal_i = factor * Delta_i``.

    The simplest "model boosting" attack — amplify the honest direction so the
    poisoned client dominates the FedAvg mean. Large factors are easy to flag by
    norm, which is exactly what makes this a baseline rather than a serious
    attack.
    """

    name = "scaling"
    colludes = False

    def __init__(self, factor: float = 10.0):
        self.factor = float(factor)

    def craft(self, pool_benign, global_sd, budget, gen):
        stats = BenignStats(pool_benign, global_sd)
        out = {}
        for cid in self.choose_ids(pool_benign, budget):
            i = stats.cids.index(cid)
            delta = {k: self.factor * stats.delta_by_key[k][i] for k in stats.keys}
            out[cid] = stats.weights_from_delta(delta)
        return out
