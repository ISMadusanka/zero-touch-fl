"""DnC (Shejwalkar & Houmansadr, "Manipulating the Byzantine: Optimizing Model
Poisoning Attacks and Defenses for Federated Learning", NDSS 2021) as a benchmark
defense — their Divide-and-Conquer robust aggregator (paper Algorithm 2).

DnC is a SPECTRAL outlier filter. Unlike distance/coordinate pruning (Krum,
Trimmed-Mean), it projects the client updates onto their top principal direction
and removes the gradients that stick out furthest along it — the direction a model
poisoner is forced to move the aggregate. Each round (paper Algorithm 2):

  for it in 1..niters:
    r        = b random coordinates of the d-dim update      (dimensionality reduction)
    sub      = each client's update restricted to r           (n x b)
    mu       = mean over clients of sub                       (dimension-wise mean)
    Dc       = sub - mu                                       (n x b, centered)
    v        = top RIGHT singular vector of Dc                (SVD-based spectral method)
    s_i      = ( <Dc[i], v> )^2                               (outlier score, paper line 9)
    keep_it  = the (n - c*m) clients with the LOWEST scores   (drop c*m highest)
  I_final    = intersection of all keep_it sets               (survive every subsample)
  aggregate  = mean of the FULL updates of the clients in I_final

Parameters (paper's iid defaults): ``niters=1``, ``c=1``, ``b=10000``. ``m`` is the
assumed number of malicious clients — a HYPERPARAMETER (an upper bound on the
adversary's budget, exactly as Krum/Trimmed-Mean assume), NOT per-round ground
truth; we default it to the configured poison count. ``c*m`` clients are removed
each iteration (rounded to an int, clamped to keep >= 1 client).

Codebase specifics:
  * Client submissions are ABSOLUTE weights, not gradients. DnC centers by the
    client-mean, and that mean subtraction CANCELS the global reference, so running
    DnC on absolute weights yields the SAME scores / kept set as on deltas — and
    step 3's "mean of the good set" is then exactly FedAvg over the kept clients. We
    therefore reuse ``FedAvgAggregator`` for the aggregate (averaging the kept
    clients' absolute weights).
  * Detection read-out: ``is_suspicious=True`` iff the client was FILTERED OUT (not
    in I_final) — i.e. exactly the clients dropped from the aggregate. Like FLTrust,
    this is a DERIVED flag of an aggregator (not a trained classifier); compare its
    TPR/FPR loosely and treat acc_drop as the primary metric — see benchmark/README.md.
  * Our model is tiny (d=681 << b=10000), so b clamps to d and no subsampling
    happens; with niters=1 DnC is then deterministic. Subsampling (d > b) uses a
    seeded RNG for reproducibility.

The selection logic (``num_to_remove``, ``keep_lowest``, ``finalize_keep``,
``subsample_indices``) is torch-free and unit-tested in tests/test_dnc_logic.py;
torch is imported lazily, only for the flatten + SVD + scores. The end-to-end torch
path is tested in tests/test_dnc.py.
"""
import random

from core.types import DetectionVerdict

from benchmark.defenses.base import (
    Defense, StepResult, boundary_calibrated_p, selection_boundary,
)


def num_to_remove(c: float, m: int, n: int) -> int:
    """c*m clients to drop, rounded to an int and clamped to [0, n-1] so the kept
    set always has at least one client."""
    k = int(round(float(c) * int(m)))
    return max(0, min(k, n - 1))


def keep_lowest(scores: list, keep_count: int) -> set:
    """Indices of the ``keep_count`` lowest-scoring clients (ties broken by index).

    A NaN score is treated as +inf (the worst possible outlier) so a client whose
    score is undefined is always dropped rather than silently kept — NaN compares
    False to everything, which would otherwise leave it in a position-dependent slot.
    """
    inf = float("inf")
    order = sorted(range(len(scores)),
                   key=lambda i: (inf if scores[i] != scores[i] else scores[i], i))
    return set(order[:max(0, keep_count)])


def finalize_keep(kept_sets: list, mean_scores: list, keep_count: int) -> set:
    """Algorithm 2 line 14: intersection of the per-iteration kept sets. If the
    intersection is empty (only possible with niters>1 and disjoint subsample
    verdicts), fall back to the ``keep_count`` clients with the lowest MEAN score so
    the result is always non-empty and deterministic."""
    if not kept_sets:
        return set(range(len(mean_scores)))
    inter = set(kept_sets[0])
    for s in kept_sets[1:]:
        inter &= s
    if inter:
        return inter
    return keep_lowest(mean_scores, keep_count)


def subsample_indices(d: int, b: int, rng: random.Random) -> list:
    """A sorted set of ``min(b, d)`` distinct coordinates in ``[0, d)``. When b >= d
    every coordinate is used (no subsampling)."""
    if b >= d:
        return list(range(d))
    return sorted(rng.sample(range(d), b))


def outlier_scores(sub) -> list:
    """Paper lines 6-9 on a subsampled matrix ``sub`` (n x b tensor): center by the
    client-mean, take the top right singular vector v, and return each client's
    squared projection ( <centered_i, v> )^2 as a list[float]."""
    import torch
    n = sub.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    centered = sub - sub.mean(dim=0, keepdim=True)
    # Top right singular vector (first row of Vh). full_matrices=False keeps it cheap
    # for the short-fat (n << b) centered matrix.
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    v = vh[0]
    proj = centered @ v
    return [float(x) for x in (proj * proj).tolist()]


class DnC(Defense):
    name = "dnc"

    def __init__(self, device: str = "cpu", num_byzantine: int = 1,
                 c: float = 1.0, niters: int = 1, sub_dim: int = 10000,
                 seed: int = 0):
        super().__init__(device)
        self.num_byzantine = int(num_byzantine)   # assumed adversary budget m
        self.c = float(c)                         # filtering fraction
        self.niters = max(1, int(niters))
        self.sub_dim = max(1, int(sub_dim))       # b (>=1; b<=0 is meaningless)
        self._rng = random.Random(int(seed))
        # Imported lazily so importing this module stays torch-free (server.aggregation
        # pulls in torch); DnC is only ever instantiated where torch is present.
        from server.aggregation import FedAvgAggregator
        self._agg = FedAvgAggregator()

    # The coordinate subsampler is the only thing ``step`` carries across rounds
    # (and only when d > sub_dim). Snapshot/restore keeps a *scored* (uncommitted)
    # round from advancing the stream — see ``benchmark.defenses.base.Defense``.
    def state_snapshot(self) -> dict:
        return {"rng": self._rng.getstate()}

    def state_restore(self, snapshot: dict) -> None:
        if "rng" in snapshot:
            self._rng.setstate(snapshot["rng"])

    def step(self, updates, poisoned_ids) -> StepResult:
        import torch
        gw = self._global
        keys = list(gw.keys())
        n = len(updates)

        mat = torch.stack([
            torch.cat([u.weights[k].reshape(-1).float() for k in keys])
            for u in updates
        ])                                         # n x d (absolute weights)
        d = mat.shape[1]
        keep_count = n - num_to_remove(self.c, self.num_byzantine, n)

        # A client with NaN/Inf weights would otherwise poison the SVD (NaN scores
        # for EVERYONE) and slip through. Sanitise the matrix for a stable SVD and
        # force any non-finite client to be a maximal outlier so it is always dropped.
        finite_row = torch.isfinite(mat).all(dim=1).tolist()
        if not all(finite_row):
            mat = torch.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

        if keep_count >= n or n <= 1:
            # Nothing to remove (m=0 or a single client) -> plain FedAvg over all.
            # No client is under suspicion, so p_malicious is 0 for everyone.
            verdicts = [DetectionVerdict(u.client_id, False, 0.0, "dnc keep-all",
                                        p_malicious=0.0)
                        for u in updates]
            kept = set(range(n))
            mean_scores = [0.0] * n
        else:
            kept_sets, score_iters = [], []
            for _ in range(self.niters):
                r = subsample_indices(d, self.sub_dim, self._rng)
                sub = mat[:, r] if len(r) < d else mat
                scores = outlier_scores(sub)
                if not all(finite_row):
                    scores = [float("inf") if not finite_row[i] else scores[i]
                              for i in range(n)]
                kept_sets.append(keep_lowest(scores, keep_count))
                score_iters.append(scores)
            mean_scores = [sum(it[i] for it in score_iters) / len(score_iters)
                           for i in range(n)]
            kept = finalize_keep(kept_sets, mean_scores, keep_count)
            # The raw spectral score is unbounded (a squared projection) and can be
            # +inf for a non-finite client, so it is NOT a probability — reporting it
            # as one saturated the attacker's stealth reward into a binary and ran
            # backwards over the kept clients.
            #
            # Rank-normalizing it (the previous fix) was bounded and monotone but
            # purely RELATIVE: the ranks are a fixed 0..1 spread every round, so a
            # client's p moved whenever OTHER clients moved and said nothing about
            # whether it was detected. Calibrate against the real cut instead — the
            # midpoint between the worst kept and the best removed score — so
            # p >= 0.5 means exactly "DnC removed me". See base.boundary_calibrated_p.
            flags = [i not in kept for i in range(n)]
            p_mal = boundary_calibrated_p(
                mean_scores, selection_boundary(mean_scores, kept), flags=flags)
            verdicts = [
                DetectionVerdict(
                    u.client_id, flags[i],
                    abs(2.0 * p_mal[i] - 1.0),
                    f"dnc {'removed' if i not in kept else 'kept'} score={mean_scores[i]:.3g}",
                    p_malicious=p_mal[i],
                )
                for i, u in enumerate(updates)
            ]

        new_global = self._agg.aggregate(updates, verdicts)   # mean over kept clients
        if new_global is not None:
            self._global = new_global
        return StepResult(new_global, verdicts,
                          info={"n_removed": int(n - len(kept)),
                                "keep_count": int(keep_count),
                                "dims": int(d)})
