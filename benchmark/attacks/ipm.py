"""IPM — Inner Product Manipulation (Xie, Koyejo & Gupta, "Fall of Empires:
Breaking Byzantine-tolerant SGD by Inner Product Manipulation", UAI 2019).

The observation that broke the first generation of Byzantine-robust aggregators:
convergence needs only that the aggregate has a POSITIVE inner product with the
true gradient. An attack does not have to look like an outlier, it only has to
flip that sign. So the colluding clients submit

    delta_mal = -epsilon * mu

with ``mu`` the mean of the honest updates the adversary can see. Every malicious
update then points exactly backwards along the honest consensus, and with
``m`` compromised of ``n`` clients the resulting FedAvg aggregate is

    ((n - m) - epsilon*m) / n  *  mu

which reverses direction once ``epsilon > (n - m)/m``.

``epsilon`` is the whole attack:

* **small** (the paper's headline setting, 0.1) makes each malicious update tiny
  and perfectly collinear with the honest mean — invisible to norm- and
  distance-based filters, which is the point of the paper — while still cancelling
  a large share of the honest progress;
* **large** (>= 1) turns it into a blatant reversal that damages an undefended
  FedAvg badly but is easy to filter.

Defaults to 0.1, the stealthy setting the paper is known for.
"""
import logging

from benchmark.attacks.base import DeltaAttack, broadcast

logger = logging.getLogger("benchmark")


class IPM(DeltaAttack):
    name = "ipm"
    citation = "Xie et al., UAI 2019"

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = float(epsilon)
        self._logged = False

    def craft_deltas(self, ctx) -> dict:
        mu = ctx.known_deltas().mean(dim=0)
        if not self._logged:
            m, n = ctx.n_malicious, ctx.n_clients
            flips = self.epsilon * m > (n - m)
            logger.info(
                "ipm: epsilon=%g, m=%d of n=%d -> undefended aggregate is %.3f*mu "
                "(%s the honest direction)", self.epsilon, m, n,
                ((n - m) - self.epsilon * m) / max(1, n),
                "REVERSES" if flips else "preserves")
            self._logged = True
        return broadcast(ctx.poisoned_ids, -self.epsilon * mu)
