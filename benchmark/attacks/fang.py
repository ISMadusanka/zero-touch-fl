"""Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated
Learning" (USENIX Security 2020) — the AGR-TAILORED untargeted attacks.

Where Min-Max/Min-Sum deliberately know nothing about the server, Fang's attacks
are written against one specific aggregation rule and exploit its decision
procedure directly. Two variants are implemented, matching the two rule families
in this benchmark's defense panel:

``fang``       vs coordinate-wise TRIMMED MEAN / MEDIAN (paper Sec. 5).
``fang_krum``  vs KRUM / MULTI-KRUM (paper Sec. 4) — the rule ``multikrum`` runs.

Both share the paper's objective: push the global model in the direction OPPOSITE
to the one it would have moved in, as far as the rule will let them. The honest
mean's sign vector ``s = sign(mu)`` is that direction, and both attacks aim at
``-s``.

Both are also anchored at the honest aggregate ``mu`` rather than at the previous
global — the paper writes the crafted model as ``w_Re - lambda*s`` with ``w_Re``
the before-attack aggregate, and its bound on ``lambda`` is stated as a distance
FROM ``w_Re``, so the two only agree with that anchor. Dropping it (as some
reimplementations do, crafting ``-lambda*s`` outright) moves the crafted model a
full honest update away from where the bound assumes it sits, which both weakens
the stealth the attack is designed around and makes the resulting damage a
function of where the previous global happened to be.

Trimmed-mean / median variant
-----------------------------
Per coordinate ``j``, the aggregate is decided by order statistics, so the way to
drag it is to place the compromised values just outside the honest range on the
far side of ``s_j``::

    s_j > 0  (honest mean rises)  ->  draw from the low side of  min_j
    s_j < 0  (honest mean falls)  ->  draw from the high side of max_j

with the "outside" interval defined multiplicatively by the paper's ``b``
(default 2): for a positive bound the interval is ``[max_j, b*max_j]``, for a
negative one ``[max_j, max_j/b]`` — always *away from zero* on a positive value
and *toward zero* on a negative one, which is what keeps the crafted value on the
correct side of the honest range whatever the sign of the bound. Each compromised
client draws independently and uniformly inside its interval, so the malicious
updates are not identical — that is deliberate in the paper (identical values are
trivially clustered) and it is what makes this attack stochastic across rounds.

Krum variant
------------
Krum selects the single update with the smallest sum of squared distances to its
``n-f-2`` nearest neighbours, so the attack must land a malicious update in that
position. All compromised clients submit the same ``mu - lambda*s``, and ``lambda``
is halved from the paper's derived upper bound (:func:`krum_lambda_bound`) until a
SIMULATED Krum selects one of them (paper Alg. 1). The simulation reuses the
benchmark's own Krum scoring
(``benchmark.defenses.multikrum``), so the attack is solved against exactly the
code the ``multikrum`` column runs rather than a second implementation that might
disagree.

Knowledge. The honest updates used for ``mu``, the per-coordinate ranges and the
Krum simulation are ``ctx.known_ids``: the compromised clients under partial
knowledge (the paper's partial-knowledge setting, where those updates are the
adversary's sample of the population) or the whole federation under
``--baseline-knowledge full`` (the paper's primary setting). Under partial
knowledge the Krum simulation is over a cohort of ``m`` crafted plus the visible
honest updates, which is an approximation of the real cohort — inherent to the
threat model, not to this implementation.
"""
import logging

from benchmark.attacks.base import DeltaAttack, broadcast

logger = logging.getLogger("benchmark")


def outside_range(bound, b: float, above: bool):
    """The paper's interval just OUTSIDE an honest per-coordinate bound.

    ``above=True`` builds the interval beyond ``max_j``; ``above=False`` the one
    below ``min_j``. The multiplicative construction is sign-aware, which is the
    subtle part: multiplying a NEGATIVE maximum by ``b > 1`` moves it *down*, i.e.
    back inside the honest range, so the roles of ``b`` and ``1/b`` swap with the
    sign of the bound. Returns ``(lo, hi)`` with ``lo <= hi``.
    """
    import torch
    positive = bound > 0
    if above:
        # Beyond max_j: b*max for a positive max (further up), max/b for a negative
        # one (toward zero, i.e. still further up).
        other = torch.where(positive, bound * b, bound / b)
    else:
        # Below min_j: min/b for a positive min (toward zero, i.e. further down),
        # b*min for a negative one (further down).
        other = torch.where(positive, bound / b, bound * b)
    return torch.minimum(bound, other), torch.maximum(bound, other)


class FangTrimmedMean(DeltaAttack):
    """Fang's attack tailored to coordinate-wise trimmed mean / median."""

    name = "fang"
    citation = "Fang et al., USENIX Security 2020 (trimmed-mean/median)"

    def __init__(self, b: float = 2.0, seed: int = 0):
        self.b = float(b)
        self.seed = int(seed)
        self._gen = None
        self._logged = False

    def reset(self) -> None:
        self._gen = None

    def _generator(self):
        """Lazily built CPU generator, so the per-round draws are reproducible.

        Deliberately CPU-bound: every state_dict in this codebase is kept on the
        CPU (``BenignClient.train`` and ``FedServer.get_global_weights`` both
        return CPU copies), and a CPU generator is the only kind that works for
        any device once the sample is moved.
        """
        import torch
        if self._gen is None:
            self._gen = torch.Generator(device="cpu")
            self._gen.manual_seed(self.seed)
        return self._gen

    def craft_deltas(self, ctx) -> dict:
        import torch
        known = ctx.known_deltas()
        mu = known.mean(dim=0)
        s = torch.sign(mu)
        lo_hi_above = outside_range(known.max(dim=0).values, self.b, above=True)
        lo_hi_below = outside_range(known.min(dim=0).values, self.b, above=False)
        gen = self._generator()
        if not self._logged:
            logger.info("fang: b=%g over %d honest update(s) visible",
                        self.b, known.shape[0])
            self._logged = True

        out = {}
        for cid in ctx.poisoned_ids:
            u = torch.rand(mu.shape, generator=gen, dtype=mu.dtype).to(mu.device)
            below = lo_hi_below[0] + u * (lo_hi_below[1] - lo_hi_below[0])
            above = lo_hi_above[0] + u * (lo_hi_above[1] - lo_hi_above[0])
            # Oppose the honest direction: drag DOWN where the mean rises, UP where
            # it falls. A coordinate with s_j == 0 carries no direction to oppose,
            # so it is left at the honest mean.
            out[int(cid)] = torch.where(s > 0, below, torch.where(s < 0, above, mu))
        return out


def krum_lambda_bound(known, mu, n: int, c: int) -> float:
    """The paper's upper bound on ``lambda`` (Fang et al. 2020, Sec. 4.2)::

        lambda <=  1/(n-2c-1) * sqrt(1/d) * min_i  sum_{l in Gamma_i} D(w_i, w_l)
                 + sqrt(1/d) * max_i D(w_i, w_Re)

    ``Gamma_i`` is the ``n-c-2`` nearest benign neighbours of benign model ``i``,
    ``D`` is Euclidean distance (not squared), ``d`` the parameter count and
    ``w_Re`` the before-attack aggregate — here the honest mean, since everything
    is in delta space and the shared reference cancels.

    Deriving ``lambda0`` this way rather than from an arbitrary constant is what
    makes the attack self-scaling: the crafted model has to land inside the honest
    cloud's own geometry to be the one Krum picks, and that geometry is exactly
    what these two terms measure.

    The FIRST term is only defined for a MINORITY adversary (``n - 2c - 1 > 0``) —
    it is the slack Krum's resilience guarantee leaves. At or past half the
    federation that guarantee is gone and the denominator turns non-positive, so
    the term is dropped and the bound is the second, always-defined one. That is
    the right degradation: with the colluders in the majority they no longer need
    slack, since their identical updates are each other's nearest neighbours and
    Krum selects them on distance 0 whatever ``lambda`` is.
    """
    d = known.shape[1]
    n_benign = known.shape[0]
    inv_sqrt_d = (1.0 / d) ** 0.5
    dists = (known.unsqueeze(0) - known.unsqueeze(1)).norm(dim=2)   # benign x benign
    bound = inv_sqrt_d * float((known - mu).norm(dim=1).max())
    denom = n - 2 * c - 1
    n_neighbours = max(0, min(n - c - 2, n_benign - 1))
    if denom > 0 and n_neighbours > 0:
        nearest_sums = []
        for i in range(n_benign):
            others = sorted(float(dists[i][j]) for j in range(n_benign) if j != i)
            nearest_sums.append(sum(others[:n_neighbours]))
        bound += inv_sqrt_d * min(nearest_sums) / denom
    return bound


class FangKrum(DeltaAttack):
    """Fang's attack tailored to Krum / Multi-Krum."""

    name = "fang_krum"
    citation = "Fang et al., USENIX Security 2020 (Krum)"

    def __init__(self, num_byzantine=None, lambda_mult: float = 1.0,
                 max_halvings: int = 40):
        """``num_byzantine`` is the ``f`` the attack assumes the server's Krum uses.

        Knowing the rule's parameters is part of Fang's threat model (the attack is
        AGR-tailored by definition), so the benchmark passes the same ``f`` the
        ``multikrum`` column is configured with. ``None`` falls back to the number
        of clients the adversary actually controls.

        ``lambda_mult`` scales the paper's derived ``lambda0``
        (:func:`krum_lambda_bound`); 1.0 is the paper's own starting point and
        larger values trade stealth for damage.
        """
        self.num_byzantine = None if num_byzantine is None else int(num_byzantine)
        self.lambda_mult = float(lambda_mult)
        self.max_halvings = int(max_halvings)
        self._logged = False

    def craft_deltas(self, ctx) -> dict:
        import torch
        from benchmark.defenses.multikrum import (
            k_closest_count, krum_scores, pairwise_sq_dists, select_lowest,
        )

        known = ctx.known_deltas()
        mu = known.mean(dim=0)
        s = torch.sign(mu)
        m = ctx.n_malicious
        if m <= 0:                                   # nothing to craft
            return {}
        # The simulated cohort: m crafted rows first, then every honest update the
        # adversary can see. Krum picking an index < m means the attack lands.
        n_sim = m + known.shape[0]
        f = self.num_byzantine if self.num_byzantine is not None else m
        k = k_closest_count(n_sim, f)

        lam = self.lambda_mult * krum_lambda_bound(known, mu, n_sim, m)
        if not (lam > 0) or lam != lam:              # degenerate/NaN geometry
            lam = self.lambda_mult * max(float(known.abs().mean()), 1e-12)
        floor = lam * 1e-6
        chosen = None
        for _ in range(self.max_halvings):
            mal = mu - lam * s
            cohort = torch.cat([mal.unsqueeze(0).expand(m, -1), known], dim=0)
            scores = krum_scores(pairwise_sq_dists(cohort), k)
            if min(select_lowest(scores, 1)) < m:      # Krum picked a crafted row
                chosen = lam
                break
            lam *= 0.5
            if lam < floor:
                break

        if chosen is None:
            # No feasible lambda: fall back to the smallest one tried, which is the
            # stealthiest crafted update the search reached. The attack is then not
            # guaranteed to be selected by Krum, exactly as in the paper's
            # unsuccessful case.
            chosen = lam
            if not self._logged:
                logger.warning(
                    "fang_krum: no lambda made the simulated Krum (n=%d, f=%d, "
                    "k=%d) select a crafted update after %d halvings; falling back "
                    "to lambda=%.3g", n_sim, f, k, self.max_halvings, chosen)
        elif not self._logged:
            logger.info("fang_krum: lambda=%.6g (paper bound x %g; simulated Krum "
                        "n=%d f=%d k=%d) selects a crafted update",
                        chosen, self.lambda_mult, n_sim, f, k)
        self._logged = True
        return broadcast(ctx.poisoned_ids, mu - chosen * s)
