"""Mimic — Karimireddy, He & Jaggi, "Byzantine-Robust Learning on Heterogeneous
Datasets via Bucketing" (ICLR 2022), Sec. 3.

The attack designed to defeat robust aggregators *without ever sending an
abnormal update*. Every compromised client copies — mimics — the honest update of
one carefully chosen benign client:

    delta_mal = delta_{i*}

Nothing is out of distribution, because the malicious update IS an honest update:
no norm filter, no distance filter and no spectral filter can flag it, and an
oracle that could would also have to flag client ``i*``. What it does instead is
OVER-WEIGHT one client's data distribution in the aggregate. Under IID data that
is nearly harmless, which is why the paper introduces it alongside heterogeneity;
under the non-IID partition this benchmark runs by default (``data.iid: false``,
FLTrust-style bias ``q=0.5``) it drags the global model toward one client's class
mixture and away from the population's.

Choosing ``i*``. The paper picks the client whose update is most extreme along the
direction of greatest variance of the honest population, and tracks that direction
across rounds by power iteration rather than recomputing an eigendecomposition::

    z <- normalize( sum_i <d_i - mu, z> (d_i - mu) )     # one power step on the covariance
    i* = argmax_i <d_i - mu, z>

``z`` persists between rounds (warm start), so the estimate sharpens as the run
goes on — the paper's ``mimic`` variant with online estimation.
"""
import logging

from benchmark.attacks.base import DeltaAttack, broadcast

logger = logging.getLogger("benchmark")


class Mimic(DeltaAttack):
    name = "mimic"
    citation = "Karimireddy et al., ICLR 2022"

    def __init__(self, warmup_iters: int = 10, iters_per_round: int = 1, seed: int = 0):
        """``warmup_iters`` power steps on the first round, ``iters_per_round`` after.

        The first round has no warm-started direction, so it does the bulk of the
        work there and then tracks incrementally, which is what makes the online
        variant cheap.
        """
        self.warmup_iters = int(warmup_iters)
        self.iters_per_round = int(iters_per_round)
        self.seed = int(seed)
        self._z = None
        self._logged = False

    def reset(self) -> None:
        self._z = None

    def _power_iterate(self, centered, steps: int):
        """``steps`` power iterations of the honest covariance, updating ``self._z``."""
        import torch
        if self._z is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed)
            z = torch.randn(centered.shape[1], generator=gen,
                            dtype=centered.dtype).to(centered.device)
            n = float(z.norm())
            self._z = z / n if n > 0 else z
        for _ in range(max(0, steps)):
            # cov @ z, without ever forming the d x d covariance matrix.
            nxt = centered.t() @ (centered @ self._z)
            n = float(nxt.norm())
            if n <= 0:
                break                      # degenerate population: keep the old z
            self._z = nxt / n

    def craft_deltas(self, ctx) -> dict:
        known = ctx.known_deltas()
        ids = list(ctx.known_ids)
        if known.shape[0] < 2:
            # Only one visible update, and under partial knowledge that update
            # belongs to a compromised client — mimicking it is a no-op.
            if not self._logged:
                logger.warning(
                    "mimic: only %d honest update(s) visible, so there is no other "
                    "client to mimic and the attack is a no-op. Raise "
                    "--max-poison-clients or use --baseline-knowledge full.",
                    known.shape[0])
                self._logged = True
            return broadcast(ctx.poisoned_ids, known.mean(dim=0))

        centered = known - known.mean(dim=0, keepdim=True)
        first = self._z is None
        self._power_iterate(centered, self.warmup_iters if first else self.iters_per_round)
        proj = centered @ self._z
        target = ids[int(proj.argmax())]
        if not self._logged:
            logger.info("mimic: mimicking client %d (%d honest update(s) visible, "
                        "%d warm-up power step(s))", target, known.shape[0],
                        self.warmup_iters)
            self._logged = True
        return broadcast(ctx.poisoned_ids, known[ids.index(target)])
