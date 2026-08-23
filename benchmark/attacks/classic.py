"""The classic Byzantine baselines every robust-aggregation paper reports.

None of these is subtle — that is the point. They bound the panel from the naive
end, so a defense that cannot stop *these* has nothing to say about the optimized
attacks, and an optimized attack that does no better than these has not earned its
complexity.

``sign_flip``  Each compromised client negates its OWN honest update:
               ``delta_mal = -c * delta_i`` (Damaskinos et al.; the "sign-flipping"
               / "reversed gradient" baseline). At ``c = 1`` a compromised client
               exactly cancels an honest one, so with ``m`` of ``n`` clients the
               aggregate keeps only ``(n - 2m)/n`` of the honest progress — and at
               ``m >= n/2`` it stalls the model entirely rather than degrading it.
               Raising ``c`` overshoots into active damage at the cost of an
               obviously oversized update.

``noise``     Each compromised client submits Gaussian noise, the random-Byzantine
              baseline from Blanchard et al. (NeurIPS 2017). The paper's absolute
              ``sigma = 200`` is meaningless for an arbitrary model, so the scale
              is expressed RELATIVE to the honest updates: the per-coordinate
              standard deviation is ``sigma`` times the honest population's own,
              giving a scale-free knob (``sigma = 1`` blends into the honest
              spread; ``sigma = 10`` is blatant). Draws are independent per client
              and per round, and seeded.

``scaling``   Each compromised client boosts its own honest update:
              ``delta_mal = gamma * delta_i`` — the untargeted reading of the
              model-replacement / boosting attack (Bagdasaryan et al., AISTATS
              2020), where the aggregate is dominated by whoever sends the largest
              vector. Included because norm-clipping and distance filters are
              specifically designed against it, so it is the natural "does the
              defense do the easy thing?" control.
"""
import logging

from benchmark.attacks.base import DeltaAttack

logger = logging.getLogger("benchmark")


class SignFlip(DeltaAttack):
    name = "sign_flip"
    citation = "classic Byzantine baseline"

    def __init__(self, c: float = 1.0):
        self.c = float(c)
        self._logged = False

    def craft_deltas(self, ctx) -> dict:
        deltas = ctx.deltas_for(ctx.poisoned_ids)
        if not self._logged:
            n, m = ctx.n_clients, ctx.n_malicious
            logger.info("sign_flip: c=%g, m=%d of n=%d -> undefended aggregate keeps "
                        "%.3f of the honest progress", self.c, m, n,
                        ((n - m) - self.c * m) / max(1, n))
            self._logged = True
        return {int(cid): -self.c * deltas[i]
                for i, cid in enumerate(ctx.poisoned_ids)}


class GaussianNoise(DeltaAttack):
    name = "noise"
    citation = "Blanchard et al., NeurIPS 2017 (random Byzantine)"

    def __init__(self, sigma: float = 10.0, seed: int = 0):
        self.sigma = float(sigma)
        self.seed = int(seed)
        self._gen = None
        self._logged = False

    def reset(self) -> None:
        self._gen = None

    def _generator(self):
        import torch
        if self._gen is None:
            self._gen = torch.Generator(device="cpu")
            self._gen.manual_seed(self.seed)
        return self._gen

    def craft_deltas(self, ctx) -> dict:
        import torch
        known = ctx.known_deltas()
        # Scale-free sigma: a multiple of the honest population's own per-coordinate
        # spread. With a single visible update there is no spread to measure, so
        # fall back to that update's own RMS coordinate magnitude.
        if known.shape[0] >= 2:
            scale = known.std(dim=0, unbiased=False)
        else:
            scale = known.abs().mean().expand(known.shape[1]).clone()
        gen = self._generator()
        if not self._logged:
            logger.info("noise: sigma=%g x honest per-coordinate std "
                        "(mean honest std %.3g)", self.sigma, float(scale.mean()))
            self._logged = True
        out = {}
        for cid in ctx.poisoned_ids:
            eps = torch.randn(scale.shape, generator=gen, dtype=scale.dtype).to(scale.device)
            out[int(cid)] = self.sigma * scale * eps
        return out


class Scaling(DeltaAttack):
    name = "scaling"
    citation = "boosting / model replacement (Bagdasaryan et al., AISTATS 2020)"

    def __init__(self, gamma: float = 10.0):
        self.gamma = float(gamma)
        self._logged = False

    def craft_deltas(self, ctx) -> dict:
        deltas = ctx.deltas_for(ctx.poisoned_ids)
        if not self._logged:
            logger.info("scaling: gamma=%g", self.gamma)
            self._logged = True
        return {int(cid): self.gamma * deltas[i]
                for i, cid in enumerate(ctx.poisoned_ids)}
