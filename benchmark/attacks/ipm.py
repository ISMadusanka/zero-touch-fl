"""IPM — Inner Product Manipulation (Xie, Koyejo & Gupta, UAI 2019).

Every malicious client sends ``Delta_mal = -epsilon * mu`` where ``mu`` is the
mean honest update. The attack's aim is to make the aggregated update's inner
product with the true (honest) update *negative*, so the global model steps in
the wrong direction and accuracy falls. It specifically defeats defenses that
reason about direction / inner products rather than magnitude.

With ``f`` malicious and ``n`` total clients under FedAvg, the aggregate is
``((n - f) * mu + f * (-epsilon * mu)) / n``, which flips sign once
``epsilon > (n - f) / f``. Under the partial-insider model the attacker knows
``f`` (its quota) and the structural federation size ``n``, so by default we set
``epsilon`` just past that flip point; pass an explicit ``epsilon`` to pin it.
"""
from __future__ import annotations

import logging

from benchmark.attacks.base import Attack, BenignStats

logger = logging.getLogger("benchmark.attacks")

_FLIP_MARGIN = 1.1     # push epsilon just past the sign-flip point (n-f)/f
_DEFAULT_EPS = 1.0     # used when the federation size is unknown


class IPMAttack(Attack):
    name = "ipm"
    colludes = True

    def __init__(self, epsilon: float | None = None, n_clients: int | None = None):
        """``epsilon`` pins the scale; ``None`` auto-sizes it from ``(n_clients,
        budget)`` so the aggregate's sign flips."""
        self.epsilon = None if epsilon is None else float(epsilon)
        self.n_clients = n_clients
        self._warned = False

    def _resolve_eps(self, budget: int) -> float:
        if self.epsilon is not None:
            return self.epsilon
        f = max(1, int(budget))
        if self.n_clients and self.n_clients > f:
            return _FLIP_MARGIN * (self.n_clients - f) / f
        if not self._warned:
            logger.warning("IPM: federation size unknown — using epsilon=%.2f (may be "
                           "too weak to flip the aggregate). Pass epsilon= to override.",
                           _DEFAULT_EPS)
            self._warned = True
        return _DEFAULT_EPS

    def craft(self, pool_benign, global_sd, budget, gen):
        stats = BenignStats(pool_benign, global_sd)
        eps = self._resolve_eps(budget)
        delta = {k: -eps * stats.mean[k] for k in stats.keys}
        w_mal = stats.weights_from_delta(delta)
        return {cid: {k: v.clone() for k, v in w_mal.items()}
                for cid in self.choose_ids(pool_benign, budget)}
