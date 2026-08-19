"""LIE — "A Little Is Enough" (Baruch, Baruch & Goldberg, NeurIPS 2019).

The malicious clients all send the same update ``Delta_mal = mu - z * sigma``,
where ``mu`` and ``sigma`` are the coordinate-wise mean and std of the honest
updates and ``z`` is a small multiplier. Perturbing every coordinate by a
fraction of its own honest standard deviation shifts the aggregate while keeping
each malicious update *inside the honest variance*, so distance/statistics-based
defenses cannot separate it from a benign client. It is the canonical stealthy
untargeted baseline.

``z`` is bounded by how many workers the aggregation can be assumed to tolerate.
The paper sets ``z_max = Phi^{-1}((n - m - s) / (n - m))`` with
``s = floor(n/2 + 1) - m`` (``n`` total workers, ``m`` malicious). We compute
that from the known federation size and the poison quota, but fall back to a
fixed default when it is degenerate (e.g. a single compromised client, where
``sigma = 0`` and no variance can be estimated). Override with ``z=`` to pin it.
"""
from __future__ import annotations

import logging
import math

import torch

from benchmark.attacks.base import Attack, BenignStats

logger = logging.getLogger("benchmark.attacks")

_DEFAULT_Z = 1.5   # common practical LIE strength when the (n, m) formula degenerates


def lie_z_from_counts(n_clients: int, n_malicious: int) -> float | None:
    """Paper's ``z_max`` from the worker counts, or None if it is degenerate."""
    n, m = int(n_clients), max(1, int(n_malicious))
    if n - m <= 0:
        return None
    s = math.floor(n / 2 + 1) - m
    p = (n - m - s) / (n - m)
    if not (0.0 < p < 1.0):
        return None
    # Phi^{-1}(p) via the error-function inverse: sqrt(2) * erfinv(2p - 1).
    z = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * p - 1.0)).item()
    if not math.isfinite(z) or z <= 0.0:
        return None
    return z


class LIEAttack(Attack):
    name = "lie"
    colludes = True

    def __init__(self, z: float | None = None, n_clients: int | None = None):
        """``z`` pins the multiplier; ``None`` computes it from ``(n_clients,
        budget)`` per round (needs ``n_clients`` — the known federation size)."""
        self.z = None if z is None else float(z)
        self.n_clients = n_clients
        self._warned = False

    def _resolve_z(self, budget: int) -> float:
        if self.z is not None:
            return self.z
        z = lie_z_from_counts(self.n_clients, budget) if self.n_clients else None
        if z is None:
            if not self._warned:
                logger.warning(
                    "LIE: could not derive z from (n=%s, m=%s) — using default z=%.2f. "
                    "Pass an explicit z to override.",
                    self.n_clients, budget, _DEFAULT_Z)
                self._warned = True
            return _DEFAULT_Z
        return z

    def craft(self, pool_benign, global_sd, budget, gen):
        stats = BenignStats(pool_benign, global_sd)
        z = self._resolve_z(budget)
        # Δ_mal = μ - z·σ, coordinate-wise (identical across the colluding clients).
        delta = {k: stats.mean[k] - z * stats.std[k] for k in stats.keys}
        w_mal = stats.weights_from_delta(delta)
        return {cid: {k: v.clone() for k, v in w_mal.items()}
                for cid in self.choose_ids(pool_benign, budget)}
