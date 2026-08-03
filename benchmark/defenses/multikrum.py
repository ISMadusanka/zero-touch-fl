"""Multi-Krum (Blanchard et al., "Machine Learning with Adversaries: Byzantine
Tolerant Gradient Descent", NeurIPS 2017) as a benchmark defense, in the
non-iterative form stated by Damaskinos et al. ("AGGREGATHOR", SysML 2019, Eq. 5).

Multi-Krum is a DISTANCE-based robust aggregator. It scores each client by how far
it sits from the bulk of the others and keeps the ``m`` most-central updates:

  score(i) = sum_{i->j} || w_i - w_j ||^2                            (Eq. 5)

where ``i->j`` ranges over the ``n - f - 2`` CLOSEST other updates to client i
(closest by L2 distance), and ``f`` is the assumed number of Byzantine clients.
Multi-Krum then selects the ``m`` clients with the SMALLEST scores and averages
them; Krum is the special case ``m = 1``. We use the common practical setting
``m = n - f`` (drop the ``f`` worst-scoring clients). The paper's strong-resilience
bound is ``m <= n - f - 2`` (and it assumes ``n >= 2f + 3``); set ``--multikrum-m``
to enforce that if desired. The SCORE always uses the ``n - f - 2`` closest
neighbours per Eq. 5, independent of ``m``.

Codebase specifics (mirrors FLTrust / DnC):
  * Client submissions are ABSOLUTE weights, not gradients — but the pairwise
    distances ``||w_i - w_j||`` are invariant to the shared global reference (it
    cancels in the difference), so Multi-Krum runs on absolute weights unchanged and
    the "average of the selected" is exactly FedAvg over the kept clients (reuses
    ``FedAvgAggregator``).
  * Detection read-out: ``is_suspicious=True`` iff the client was NOT selected
    (dropped from the aggregate) — a DERIVED flag of an aggregator, like FLTrust/DnC.
    By construction it drops a fixed ``n - m`` clients/round, so its TPR/FPR are
    pinned to that budget; read ``acc_drop`` as the primary metric (benchmark/README.md).
  * ``f`` (num_byzantine) is an assumed adversary budget (a hyperparameter, defaulted
    to the configured poison count), NOT per-round ground truth.
  * Robustness hardening: a client with NaN/Inf weights is forced out (its score is
    set to +inf) and the matrix is sanitised so its distances cannot poison the
    others' scores; ``select_lowest`` also treats NaN as the worst score.

The selection logic (``k_closest_count``, ``num_selected``, ``krum_scores``,
``select_lowest``) is torch-free and unit-tested in tests/test_multikrum_logic.py;
torch is imported lazily (flatten + pairwise distances only). The end-to-end torch
path is tested in tests/test_multikrum.py.
"""
from core.types import DetectionVerdict

from benchmark.defenses.base import Defense, StepResult, rank_normalized_scores


def k_closest_count(n: int, f: int) -> int:
    """Number of nearest neighbours summed in the score (paper's n - f - 2), clamped
    to [1, n-1] so the score is always computable."""
    if n <= 1:
        return 0
    return max(1, min(n - f - 2, n - 1))


def num_selected(n: int, f: int, m_override) -> int:
    """How many updates Multi-Krum keeps and averages. Defaults to n - f (drop the f
    assumed-Byzantine); clamped to [1, n]."""
    m = m_override if m_override is not None else (n - f)
    return max(1, min(int(m), n))


def krum_scores(dist: list, k: int) -> list:
    """Per-client Multi-Krum score: the sum of the ``k`` smallest squared distances
    to OTHER clients. ``dist`` is an n x n matrix of squared L2 distances (list of
    list of floats)."""
    n = len(dist)
    scores = []
    for i in range(n):
        others = sorted(dist[i][j] for j in range(n) if j != i)
        scores.append(float(sum(others[:max(0, k)])))
    return scores


def select_lowest(scores: list, m: int) -> set:
    """Indices of the ``m`` lowest-scoring clients (ties broken by index). A NaN score
    is treated as +inf (worst) so an undefined-score client is never selected."""
    inf = float("inf")
    order = sorted(range(len(scores)),
                   key=lambda i: (inf if scores[i] != scores[i] else scores[i], i))
    return set(order[:max(0, min(m, len(scores)))])


def pairwise_sq_dists(mat) -> list:
    """n x n matrix of squared L2 distances between the rows of ``mat`` (n x d tensor),
    returned as a Python list of lists. Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b."""
    import torch
    sq = (mat * mat).sum(dim=1)
    gram = mat @ mat.t()
    d2 = sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * gram
    return d2.clamp_min(0.0).tolist()       # clamp tiny negatives from round-off


class MultiKrum(Defense):
    name = "multikrum"

    def __init__(self, device: str = "cpu", num_byzantine: int = 1, m=None):
        super().__init__(device)
        self.num_byzantine = int(num_byzantine)   # assumed #Byzantine f
        self.m = None if m is None else int(m)    # #selected (default n - f)
        from server.aggregation import FedAvgAggregator   # lazy: keeps module torch-free
        self._agg = FedAvgAggregator()

    def step(self, updates, poisoned_ids) -> StepResult:
        import torch
        gw = self._global
        keys = list(gw.keys())
        n = len(updates)

        mat = torch.stack([
            torch.cat([u.weights[k].reshape(-1).float() for k in keys])
            for u in updates
        ])                                         # n x d (absolute weights)

        f = self.num_byzantine
        m = num_selected(n, f, self.m)
        k = k_closest_count(n, f)

        if m >= n or n <= 1:
            # Select everyone -> plain FedAvg (nothing to drop). Nobody is under
            # suspicion, so p_malicious is 0 for everyone.
            verdicts = [DetectionVerdict(u.client_id, False, 0.0, "multikrum select-all",
                                        p_malicious=0.0)
                        for u in updates]
            selected = set(range(n))
        else:
            # A non-finite client would poison every score; sanitise + force it out.
            finite_row = torch.isfinite(mat).all(dim=1).tolist()
            if not all(finite_row):
                mat = torch.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            scores = krum_scores(pairwise_sq_dists(mat), k)
            if not all(finite_row):
                scores = [float("inf") if not finite_row[i] else scores[i]
                          for i in range(n)]
            selected = select_lowest(scores, m)
            # The Krum score is an unbounded sum of squared distances (and +inf for a
            # non-finite client), so it is not a probability: reporting it as one both
            # saturated the attacker's stealth reward and ran BACKWARDS over the
            # selected clients, and `inf` is not valid JSON in the round logs.
            # Rank-normalize it; `select_lowest` slices the same ordering, so the hard
            # flag and the soft score agree. See base.rank_normalized_scores.
            p_mal = rank_normalized_scores(scores)
            verdicts = [
                DetectionVerdict(
                    u.client_id, i not in selected, abs(2.0 * p_mal[i] - 1.0),
                    f"multikrum {'selected' if i in selected else 'dropped'} score={scores[i]:.3g}",
                    p_malicious=p_mal[i],
                )
                for i, u in enumerate(updates)
            ]

        new_global = self._agg.aggregate(updates, verdicts)   # mean over selected
        if new_global is not None:
            self._global = new_global
        return StepResult(new_global, verdicts,
                          info={"n_dropped": int(n - len(selected)), "m": int(m),
                                "k_closest": int(k)})
