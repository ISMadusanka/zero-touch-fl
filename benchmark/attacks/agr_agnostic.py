"""Min-Max and Min-Sum — the AGR-agnostic attacks of Shejwalkar & Houmansadr,
"Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for
Federated Learning" (NDSS 2021).

These are the strongest *generic* untargeted model-poisoning attacks in the
literature: unlike Fang's they need no knowledge of which aggregation rule the
server runs, and unlike LIE they do not fix the perturbation size in advance —
they SOLVE for the largest perturbation that still passes for honest.

Both share one shape. Take the honest mean ``mu`` and a malicious direction
``p``, then submit

    delta_mal = mu + gamma * p

with ``gamma`` the largest scalar satisfying a constraint that every honest
update already satisfies:

* **Min-Max** — the malicious update's distance to its FURTHEST honest neighbour
  is no larger than the largest distance between two honest updates::

      max_i ||d_mal - d_i||^2  <=  max_{i,j} ||d_i - d_j||^2

  So it sits inside the honest cloud's own diameter. This is what defeats
  distance-based rules (Krum, Multi-Krum) and, empirically, most of the rest.

* **Min-Sum** — the SUM of squared distances to all honest updates is no larger
  than the largest such sum any honest update has::

      sum_i ||d_mal - d_i||^2  <=  max_j sum_i ||d_j - d_i||^2

  A looser, more permissive constraint in some geometries and tighter in others;
  the paper reports both because neither dominates.

``gamma`` is found by the paper's halving search: start at ``gamma0``, keep the
last feasible value, and move by half the previous step either way. It is a 1-D
search on a monotone feasibility predicate, so ~30 iterations pin it to
``gamma0 * 2^-30``.

The perturbation direction (``--minmax-perturbation``) is one of the paper's
three, all pointing AGAINST the honest consensus:

    ``std``       -sigma  (coordinate-wise std of the honest updates)   [default]
    ``unit_vec``  -mu/||mu||
    ``sign``      -sign(mu)

The paper evaluates all three and takes the best per setting; ``std`` is the one
that wins most often, so it is the default here. Nothing about the choice depends
on the defense, which is the point of "AGR-agnostic".
"""
import logging

from benchmark.attacks.base import DeltaAttack, broadcast

logger = logging.getLogger("benchmark")

PERTURBATIONS = ("std", "unit_vec", "sign")


def perturbation(known, mu, kind: str):
    """The paper's perturbation direction: one that opposes the honest consensus.

    Returned UNNEGATED (the caller subtracts it, as the reference implementation
    does) so ``std`` stays the plain coordinate-wise standard deviation.
    """
    import torch
    if kind == "std":
        if known.shape[0] < 2:
            return torch.sign(mu)                  # no spread to measure: fall back
        return known.std(dim=0, unbiased=False)
    if kind == "unit_vec":
        n = float(mu.norm())
        return mu / n if n > 0 else torch.sign(mu)
    if kind == "sign":
        return torch.sign(mu)
    raise ValueError(f"unknown perturbation {kind!r}; use one of {PERTURBATIONS}")


def pairwise_sq_dists(mat):
    """``k x k`` matrix of squared L2 distances between the rows of ``mat``."""
    sq = (mat * mat).sum(dim=1)
    d2 = sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * (mat @ mat.t())
    return d2.clamp_min(0.0)


def solve_gamma(known, mu, direction, value_fn, limit: float,
                gamma0: float, iters: int = 30):
    """Largest feasible ``gamma`` for ``delta = mu - gamma*direction``, by halving.

    ``value_fn(sq_dists_to_known) -> float`` is the quantity the constraint bounds
    (a max for Min-Max, a sum for Min-Sum) and ``limit`` is the bound. Mirrors the
    reference implementation: probe ``gamma``, keep it as the best feasible value
    if it passes, and move by half the previous step either way, so the interval
    collapses geometrically around the feasibility boundary.

    Returns ``(gamma, mal_delta)``. ``gamma = 0`` (submit the honest mean) is
    always feasible, so a search that never finds a feasible probe degrades to the
    honest mean rather than to an out-of-bounds update.
    """
    gamma = float(gamma0)
    step = float(gamma0)
    best = 0.0
    for _ in range(max(1, int(iters))):
        cand = mu - gamma * direction
        d2 = ((known - cand) ** 2).sum(dim=1)
        if value_fn(d2) <= limit:
            best = gamma
            gamma = gamma + step / 2.0
        else:
            gamma = max(0.0, gamma - step / 2.0)
        step = step / 2.0
    return best, mu - best * direction


class _AGRAgnostic(DeltaAttack):
    """Shared machinery; subclasses supply the feasibility budget."""

    def __init__(self, perturbation_type: str = "std", gamma0: float = 10.0,
                 iters: int = 30):
        if perturbation_type not in PERTURBATIONS:
            raise ValueError(f"unknown perturbation {perturbation_type!r}; "
                             f"use one of {PERTURBATIONS}")
        self.perturbation_type = perturbation_type
        self.gamma0 = float(gamma0)
        self.iters = int(iters)
        self._logged = False

    def _limit(self, d2_pairwise) -> float:
        raise NotImplementedError

    @staticmethod
    def _value(d2) -> float:
        raise NotImplementedError

    def craft_deltas(self, ctx) -> dict:
        known = ctx.known_deltas()
        mu = known.mean(dim=0)
        if known.shape[0] < 2:
            # One visible update: no honest cloud to hide inside, so every gamma > 0
            # is infeasible and the search returns the honest mean. Report it once
            # instead of silently emitting a no-op attack.
            if not self._logged:
                logger.warning(
                    "%s: only %d honest update(s) visible — the feasibility "
                    "constraint has no honest spread to calibrate against and the "
                    "attack degenerates to the honest mean. Raise "
                    "--max-poison-clients or use --baseline-knowledge full.",
                    self.name, known.shape[0])
                self._logged = True
            return broadcast(ctx.poisoned_ids, mu)
        direction = perturbation(known, mu, self.perturbation_type)
        limit = self._limit(pairwise_sq_dists(known))
        gamma, mal = solve_gamma(known, mu, direction, self._value, limit,
                                 self.gamma0, self.iters)
        if not self._logged:
            logger.info("%s: perturbation=%s gamma0=%g -> gamma=%.6g "
                        "(%d honest update(s) visible)", self.name,
                        self.perturbation_type, self.gamma0, gamma, known.shape[0])
            self._logged = True
        return broadcast(ctx.poisoned_ids, mal)


class MinMax(_AGRAgnostic):
    name = "min_max"
    citation = "Shejwalkar & Houmansadr, NDSS 2021"

    def _limit(self, d2_pairwise) -> float:
        """Largest squared distance between two honest updates (the cloud's diameter)."""
        return float(d2_pairwise.max())

    @staticmethod
    def _value(d2) -> float:
        return float(d2.max())


class MinSum(_AGRAgnostic):
    name = "min_sum"
    citation = "Shejwalkar & Houmansadr, NDSS 2021"

    def __init__(self, perturbation_type: str = "std", gamma0: float = 10.0,
                 iters: int = 30, bound: str = "max"):
        """``bound`` selects which honest row the constraint is calibrated to.

        ``max`` (default) reads the paper's inequality literally: the malicious
        sum may be as large as the largest sum an honest update already has, which
        keeps every honest update itself feasible — the property the constraint is
        supposed to express. ``min`` reproduces the reference implementation, whose
        bound is the smallest honest sum and is therefore strictly more
        conservative (a weaker, stealthier attack). Both appear in the literature,
        so the knob is explicit rather than silently picking one.
        """
        super().__init__(perturbation_type, gamma0, iters)
        if bound not in ("max", "min"):
            raise ValueError(f"unknown min_sum bound {bound!r}; use 'max' or 'min'")
        self.bound = bound

    def _limit(self, d2_pairwise) -> float:
        sums = d2_pairwise.sum(dim=1)
        return float(sums.max() if self.bound == "max" else sums.min())

    @staticmethod
    def _value(d2) -> float:
        return float(d2.sum())
