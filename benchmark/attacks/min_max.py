"""Min-Max and Min-Sum — "Manipulating the Byzantine" (Shejwalkar & Houmansadr,
NDSS 2021).

AGR-agnostic optimization attacks: perturb the benign mean along a fixed
malicious direction by the LARGEST magnitude ``gamma`` that still keeps the
crafted update statistically indistinguishable from the honest ones. Because the
constraint is defined purely by the honest updates (not by any specific
aggregation rule), one crafted update is strong against the whole panel — which
is exactly why they are the cleanest cross-defense baselines.

All malicious clients send the same ``Delta_mal = mu + gamma * d``:

* **Min-Max** bounds the crafted update's *maximum* distance to any honest update
  by the maximum honest-to-honest distance:
  ``max_i || Delta_mal - Delta_i ||  <=  max_{i,j} || Delta_i - Delta_j ||``.
* **Min-Sum** bounds its *sum of squared* distances to the honest updates by the
  maximum such sum over the honest updates:
  ``sum_i || Delta_mal - Delta_i ||^2  <=  max_i sum_j || Delta_i - Delta_j ||^2``.

``gamma`` is found by binary search (monotone: the constraint is violated for
large enough ``gamma``). The perturbation direction ``d`` is one of
``-sign(mu)`` (default; always well-defined), ``-sigma`` (the "std/deviation"
direction from the paper), or ``-mu/||mu||`` (unit). These attacks need at least
two controllable clients to estimate the honest spread; with one, the honest
distances are all zero and the attack reduces to the honest mean.
"""
from __future__ import annotations

import torch

from benchmark.attacks.base import Attack, BenignStats, unit


def perturbation_direction(stats: BenignStats, kind: str) -> torch.Tensor:
    """The (flat) malicious direction ``d``, moving AGAINST the honest mean."""
    mean = stats.flat_mean()
    if kind == "std":
        return -stats.flat_std()
    if kind == "unit":
        return -unit(mean)
    if kind == "sign":
        return -torch.sign(mean)
    raise ValueError(f"unknown perturbation direction {kind!r} "
                     f"(use 'sign', 'std', or 'unit')")


def solve_gamma(deltas: torch.Tensor, mean: torch.Tensor, direction: torch.Tensor,
                mode: str, iters: int = 40) -> float:
    """Largest ``gamma`` in ``mu + gamma*d`` satisfying the Min-Max/Min-Sum bound.

    ``deltas`` is ``(f, d)`` honest updates; ``mode`` is ``"max"`` or ``"sum"``.
    Returns 0.0 when the honest set carries no spread (e.g. a single client).
    """
    if direction.norm().item() == 0.0:
        return 0.0
    pdist = torch.cdist(deltas, deltas)                 # (f, f) honest-to-honest
    if mode == "max":
        threshold = float(pdist.max())

        def violation(mal):
            return float((mal.unsqueeze(0) - deltas).norm(dim=1).max())
    elif mode == "sum":
        threshold = float((pdist ** 2).sum(dim=1).max())

        def violation(mal):
            return float(((mal.unsqueeze(0) - deltas).norm(dim=1) ** 2).sum())
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if threshold <= 0.0:
        return 0.0

    lo, hi = 0.0, 1.0
    # Expand hi until the constraint is violated (bounded, to avoid a runaway).
    for _ in range(60):
        if violation(mean + hi * direction) > threshold:
            break
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if violation(mean + mid * direction) <= threshold:
            lo = mid
        else:
            hi = mid
    return lo


class _AGRAgnosticAttack(Attack):
    colludes = True
    _mode = "max"

    def __init__(self, direction: str = "sign"):
        self.direction = direction

    def craft(self, pool_benign, global_sd, budget, gen):
        stats = BenignStats(pool_benign, global_sd)
        mean = stats.flat_mean()
        d = perturbation_direction(stats, self.direction)
        gamma = solve_gamma(stats.flat_deltas(), mean, d, self._mode)
        mal = mean + gamma * d
        w_mal = stats.weights_from_flat_delta(mal)
        return {cid: {k: v.clone() for k, v in w_mal.items()}
                for cid in self.choose_ids(pool_benign, budget)}


class MinMaxAttack(_AGRAgnosticAttack):
    name = "min_max"
    _mode = "max"


class MinSumAttack(_AGRAgnosticAttack):
    name = "min_sum"
    _mode = "sum"
