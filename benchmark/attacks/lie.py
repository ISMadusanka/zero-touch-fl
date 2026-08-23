"""LIE — "A Little Is Enough" (Baruch, Baruch & Goldberg, NeurIPS 2019).

The canonical *stealthy* untargeted model-poisoning attack, and the one every
later paper benchmarks against. The insight: a robust aggregator can only reject
what looks like an outlier, so instead of sending a large malicious update the
colluding clients agree on

    Δ_mal = μ + z · σ

where μ and σ are the COORDINATE-WISE mean and standard deviation of the honest
updates the adversary can see, and ``z`` is chosen so that the perturbed value
still falls inside the range the population's own variance makes plausible. Every
coordinate moves by a fraction of a standard deviation — individually
unremarkable, collectively enough to stop or reverse convergence.

``z`` (the paper's ``z^max``) comes from the population, not from a knob::

    s      = floor(n/2 + 1) − m        # honest workers the attacker must out-vote
    z^max  = Φ⁻¹( (n − m − s) / (n − m) )

with ``n`` the federation size and ``m`` the number of compromised clients: it is
the largest number of standard deviations that keeps the malicious value within
the ``n − m − s`` "most normal" honest values. ``--lie-z`` pins it manually.

Direction. The paper perturbs AWAY from the honest mean; sign conventions differ
between reimplementations (``μ + zσ`` and ``μ − zσ`` are the same attack mirrored,
since σ is symmetric) so ``sign`` is explicit here and defaults to −1, matching the
form used in the robust-aggregation literature that benchmarks against LIE.

Knowledge. μ/σ are estimated over ``ctx.known_ids`` — the compromised clients
under partial knowledge (the paper's practical setting: an attacker always knows
its own clients' honest updates), every client under ``--baseline-knowledge full``.
"""
import logging
import math

from benchmark.attacks.base import DeltaAttack, broadcast

logger = logging.getLogger("benchmark")


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF, Acklam's rational approximation (|err| < 1.2e-9).

    Hand-rolled so the attack has no SciPy dependency (the repo's requirements do
    pull SciPy in through scikit-learn, but a benchmark baseline should not fail to
    import on a stripped environment). Accurate far past the precision ``z`` needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf needs 0 < p < 1, got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


#: ``z`` used when the formula is unbounded (colluders at/over half the federation).
#: 4 standard deviations, i.e. past the most extreme honest value the population is
#: likely to contain, which is as far as "look like an honest outlier" can go.
_Z_UNCONSTRAINED = 4.0


def lie_z(n_clients: int, n_malicious: int) -> float:
    """The paper's ``z^max`` for ``n`` workers of which ``m`` are compromised.

    ``s = floor(n/2 + 1) − m`` is how many honest workers the colluders still need
    on their side of the median; the malicious value may therefore sit at the
    ``(n − m − s)/(n − m)`` quantile of the honest distribution. Degenerate inputs
    (``m`` at or past half the federation, so the colluders need no honest support
    at all) leave the quantile at or above 1 — the formula then has no finite
    answer and the attack is un-constrained, so we clamp to a large-but-finite
    ``z`` rather than returning ``inf`` and poisoning every downstream tensor.
    """
    n, m = int(n_clients), int(n_malicious)
    honest = n - m
    if honest <= 0:
        return _Z_UNCONSTRAINED
    s = math.floor(n / 2 + 1) - m
    q = (honest - s) / honest
    if q >= 1.0:
        # No honest support needed: the aggregate is the colluders' to set.
        return _Z_UNCONSTRAINED
    if q <= 0.0:
        # The colluders need EVERY honest worker on side: no perturbation is safe.
        return 0.0
    return _norm_ppf(q)



class LIE(DeltaAttack):
    name = "lie"
    citation = "Baruch et al., NeurIPS 2019"

    def __init__(self, z=None, sign: float = -1.0):
        self.z = None if z is None else float(z)
        self.sign = float(sign)
        self._logged = False

    def craft_deltas(self, ctx) -> dict:
        known = ctx.known_deltas()                      # k x d
        mu = known.mean(dim=0)
        if known.shape[0] < 2:
            # σ over a single sample is 0, so μ + zσ = μ: the attack collapses to
            # "submit the honest mean", i.e. no attack at all. Say so once — a flat
            # 0.00 acc_drop row would otherwise read as "LIE is ineffective here"
            # rather than "LIE was never actually applied".
            if not self._logged:
                logger.warning(
                    "lie: only %d honest update(s) visible, so the coordinate-wise "
                    "sigma is zero and the attack degenerates to submitting the honest "
                    "mean. "
                    "LIE needs >= 2 known updates — raise --max-poison-clients or use "
                    "--baseline-knowledge full.", known.shape[0])
                self._logged = True
            return broadcast(ctx.poisoned_ids, mu)
        # Population (biased) σ, matching the paper's estimator over the observed
        # workers; the unbiased correction would inflate z's effect at small k.
        sigma = known.std(dim=0, unbiased=False)
        z = self.z if self.z is not None else lie_z(ctx.n_clients, ctx.n_malicious)
        if not self._logged:
            logger.info("lie: z=%.4f (n=%d, m=%d, %d update(s) visible), sign=%+.0f",
                        z, ctx.n_clients, ctx.n_malicious, known.shape[0], self.sign)
            self._logged = True
        return broadcast(ctx.poisoned_ids, mu + self.sign * z * sigma)
