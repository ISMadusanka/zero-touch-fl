"""DeFL (Yan et al., "DeFL: Defending against Model Poisoning Attacks in Federated
Learning via Critical Learning Periods Awareness", AAAI-23) as a benchmark defense.

DeFL differs from the Byzantine-robust aggregators it is compared against: instead
of treating the model as a black box, it inspects the DNN *layer by layer* and ties
its defense to the **critical learning period (CLP)** — the first few rounds whose
corruption is, per the paper, irreversible. It needs NEITHER a clean validation set
(unlike FLTrust) NOR historical client records (unlike FLDetector).

Three pieces, all driven by one cheap per-layer statistic:

1. Federated Gradient Norm Vector (FGNV).  For client ``i`` and layer ``j`` the
   per-layer training-loss change is approximated (Taylor, paper Eq. 2) by the
   squared gradient norm. Client submissions here are ABSOLUTE weights, so the
   per-layer "gradient" is the update delta vs the CURRENT global model:

       FGNV^j_i = || w^j_i - w^j_global ||^2                       (paper Eq. 2)

   and FGNV_i = (FGNV^1_i, ..., FGNV^L_i). A "layer" is a module (its weight+bias
   grouped), matching the paper's notion (e.g. the FC net {784,512,10} = 2 layers).
   The data-weighted across-client mean is the round's per-layer FGNV (Eq. 3):

       FGNV^j(t) = sum_i (|D_i| / sum_k |D_k|) FGNV^j_i(t)

   (the benchmark's replayed updates carry no sample counts, so |D_i| defaults to
   uniform; an update's ``metadata["train_samples"]`` is honoured when present).

2. CLP detection (paper Eq. 4).  With S(t) = sum_j FGNV^j(t), round t is in a CLP iff
   the total per-layer FGNV ROSE by at least ``delta`` relative to the previous round:

       ( S(t) - S(t-1) ) / S(t-1)  >=  delta        (delta default 0.05)

   The first round (no S(t-1)) is treated as a CLP — the initial phase is critical.

3. MOUD-Vote malicious-client detection.  The paper casts detection as per-layer
   statistical outlier detection (MOUD) followed by a vote across layers with an
   ADAPTIVE threshold. Each client's per-layer FGNV is compared to a reference over
   all participating clients; a layer that deviates significantly votes "malicious".
   Start the vote threshold at L (flag a client only if ALL L layers call it an
   outlier); if that flags nobody, lower it to L-1, and so on, until at least one
   client is flagged. We implement the per-layer outlier test as a robust
   median/MAD z-score, which is the realizable form of "compare each client's layer
   value to a reference observed from all clients" — the paper's pairwise-regression
   MOUD index is under-specified for the scalar-per-(client,layer) inputs it feeds.
   Deviation from the paper: we do NOT force a flag on a round with zero per-layer
   outliers (avoids a guaranteed false positive on a genuinely clean round).

CLP + detector are combined exactly as in DeFL's Algorithm 1, augmented with a
per-client Bayesian (Beta) "good-update" probability that makes the defense robust
to detection errors (FPR):

  * Each round, every client's Beta counts are bumped by its vote: benign -> alpha+1,
    malicious -> beta+1 (init alpha=beta=1). The good-update prob is p_i = a/(a+b)
    (Eq. 5), computed AFTER this round's bump (so a fresh detection lowers p_i now).
  * During a CLP, detected-malicious clients are REMOVED (weight 0); everyone else
    keeps weight p_i. Outside a CLP, all clients keep p_i (so a detected-malicious
    client is soft-down-weighted, not dropped — this is what makes DeFL tolerant of
    false positives rather than discarding honest clients forever).
  * The new global is the trust+data-weighted average of the clients' ABSOLUTE
    weights, renormalised so the coefficients sum to 1 (the absolute-weight analogue
    of Alg. 1's aggregation; without renormalisation a sum < 1 would shrink the model).

Detection read-out: ``is_suspicious=True`` iff MOUD-Vote flagged the client this
round — INDEPENDENT of the CLP gating (matches the paper's MOUD-Vote TPR/FPR in
Table 2). The CLP gate only changes a flagged client's aggregation WEIGHT. As with
FLTrust this is a detector whose flag and aggregate can disagree post-CLP (flagged
but kept at small weight); treat ROBUSTNESS (acc_drop) as the primary cross-defense
metric — see benchmark/README.md.

The pure helpers (``group_layers``, ``is_clp``, ``per_layer_zscores``,
``per_layer_votes``, ``moud_vote``,
``BetaTracker``) are torch-free and unit-tested in isolation (tests/test_defl_logic.py);
torch is imported lazily only where tensors are actually touched, so importing this
module stays cheap. The end-to-end torch math is tested in tests/test_defl.py.
"""
import math

from core.types import DetectionVerdict
from benchmark.defenses.base import Defense, StepResult, boundary_calibrated_p


# A "layer" groups a module's parameter tensors (weight + bias + any buffers).
_PARAM_SUFFIXES = (
    ".weight", ".bias", ".running_mean", ".running_var",
    ".num_batches_tracked", ".gamma", ".beta",
)


def group_layers(keys: list) -> dict:
    """Group state_dict keys into layers (a module = one layer), preserving order.

    ``net.2.weight`` and ``net.2.bias`` -> layer ``net.2``. Keys without a known
    parameter suffix become their own single-key layer. Returns an insertion-ordered
    ``{layer_name: [keys]}``.
    """
    groups: dict = {}
    for k in keys:
        layer = k
        for suf in _PARAM_SUFFIXES:
            if k.endswith(suf):
                layer = k[: -len(suf)]
                break
        groups.setdefault(layer, []).append(k)
    return groups


def fgnv_for_update(weights: dict, global_weights: dict, layer_groups: dict) -> list:
    """FGNV_i: per-layer squared L2 norm of the client's update delta vs the global.

    Returns ``list[float]`` aligned with ``layer_groups`` order (one entry per layer).
    """
    import torch
    out = []
    for keys in layer_groups.values():
        sq = 0.0
        for k in keys:
            d = weights[k].reshape(-1).float() - global_weights[k].reshape(-1).float()
            sq += float(torch.dot(d, d).item())
        out.append(sq)
    return out


def aggregate_fgnv(fgnv_matrix: list, data_weights: list) -> list:
    """Across-client data-weighted mean per layer (paper Eq. 3).

    ``fgnv_matrix``: per client, a list of per-layer FGNV floats. ``data_weights``:
    per-client weight (|D_i|; uniform if unknown). Returns the per-layer mean.
    """
    n = len(fgnv_matrix)
    if n == 0:
        return []
    L = len(fgnv_matrix[0])
    sw = sum(data_weights) or float(n)
    agg = [0.0] * L
    for i in range(n):
        w = data_weights[i] / sw
        row = fgnv_matrix[i]
        for j in range(L):
            agg[j] += w * row[j]
    return agg


def is_clp(total_now: float, total_prev, delta: float) -> bool:
    """Paper Eq. 4: round is in a CLP iff total FGNV rose by >= ``delta`` (relative).

    ``total_prev is None`` (first round) -> CLP. A non-positive previous total is
    treated as CLP too (degenerate base; the very early phase is critical).
    """
    if total_prev is None:
        return True
    if total_prev <= 0.0:
        return True
    return (total_now - total_prev) / total_prev >= delta


def _median(xs: list) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def per_layer_zscores(fgnv_matrix: list, eps: float = 1e-12) -> list:
    """Per-(client, layer) robust z-score against the across-client median.

    ``z = |v - median| / (1.4826 * MAD)``. When the spread (MAD) collapses to ~0 the
    z-score is undefined, so a value that differs from the median at all is reported
    as ``inf`` and one that does not as ``0.0`` — which is what the ``dev > eps``
    branch of the vote test means. Returns an ``n x L`` list of floats.

    Split out of :func:`per_layer_votes` so the continuous magnitudes survive the
    thresholding: the vote COUNT is what the paper's cross-layer threshold compares,
    but with a two-layer model that count only takes three values, which is far too
    coarse to be the attacker's stealth gradient on its own (see
    :meth:`DeFL.step`).
    """
    n = len(fgnv_matrix)
    if n == 0:
        return []
    L = len(fgnv_matrix[0])
    inf = float("inf")
    z = [[0.0] * L for _ in range(n)]
    for j in range(L):
        col = [fgnv_matrix[i][j] for i in range(n)]
        med = _median(col)
        scale = 1.4826 * _median([abs(v - med) for v in col])
        for i in range(n):
            dev = abs(col[i] - med)
            if scale <= eps:
                z[i][j] = inf if dev > eps else 0.0
            else:
                z[i][j] = dev / scale
    return z


def per_layer_votes(fgnv_matrix: list, tau: float, eps: float = 1e-12) -> list:
    """Per-client count of layers on which it is a robust statistical outlier.

    A client whose robust z-score (see :func:`per_layer_zscores`) exceeds ``tau`` on
    a layer casts one outlier vote there. Returns ``list[int]`` (votes per client,
    0..L).
    """
    if not fgnv_matrix:
        return []
    z = per_layer_zscores(fgnv_matrix, eps)
    return [sum(1 for zij in row if zij > tau) for row in z]


#: Hard bound on the sub-vote tiebreak (see :func:`_subvote_tiebreak`). Must stay
#: strictly below 0.5 so the tiebreak can never carry a client across the half-vote
#: gap that separates ``votes >= threshold`` from ``votes <= threshold - 1``.
_SUBVOTE_BOUND = 0.45


def _subvote_tiebreak(zrow: list, tau: float) -> float:
    """A continuous refinement within one vote count, in ``[0, _SUBVOTE_BOUND)``.

    Two clients with the same number of outlier votes are not equally suspicious —
    one may sit just over ``tau`` on each flagged layer and the other far past it.
    This orders them by their mean robust z-score, squashed so the value can never
    reach 0.5. ``inf`` (a collapsed-MAD layer) saturates to the bound.
    """
    if not zrow:
        return 0.0
    inf = float("inf")
    if any(zij == inf for zij in zrow):
        return _SUBVOTE_BOUND
    mean_z = sum(zrow) / len(zrow)
    scale = float(tau) if tau > 0.0 else 1.0
    return _SUBVOTE_BOUND * math.tanh(mean_z / scale)


def moud_vote(fgnv_matrix: list, tau: float) -> tuple:
    """MOUD-Vote with the paper's adaptive cross-layer threshold.

    Returns ``(flagged, votes, threshold)`` where ``flagged`` is a per-client
    list[bool] and ``threshold`` is the vote count the flag test actually used. The
    threshold starts at L (outlier on every layer) and is lowered until at least one
    client is flagged; if no client is an outlier on any layer, nobody is flagged and
    the reported threshold is 1 (the lowest the loop reaches), which is still the
    correct boundary — every client has 0 votes there.

    ``threshold`` is part of the return value because the flag condition is
    ``votes >= threshold`` and that boundary MOVES between rounds. Without it a
    consumer cannot tell a flagged client from an accepted one by vote count alone:
    on a two-layer model the adaptive rule routinely settles at ``threshold = 1``, so
    one vote out of two is a rejection — which is exactly how a calibrated
    ``p_malicious`` of 0.5 came to be reported for clients DeFL had just caught.
    """
    n = len(fgnv_matrix)
    if n == 0:
        return [], [], 1
    L = len(fgnv_matrix[0])
    votes = per_layer_votes(fgnv_matrix, tau)
    flagged = [False] * n
    used = 1
    for thr in range(L, 0, -1):
        cand = [v >= thr for v in votes]
        if any(cand):
            flagged = cand
            used = thr
            break
    return flagged, votes, used


class BetaTracker:
    """Per-client Beta(alpha, beta) good-update model (paper Eq. 5).

    ``alpha``/``beta`` start at 1 and persist across rounds. A benign vote bumps
    alpha; a malicious vote bumps beta. ``prob`` is the Beta mean alpha/(alpha+beta).
    """

    def __init__(self):
        self.alpha: dict = {}
        self.beta: dict = {}

    def update(self, client_id: int, benign: bool):
        a = self.alpha.get(client_id, 1.0)
        b = self.beta.get(client_id, 1.0)
        if benign:
            a += 1.0
        else:
            b += 1.0
        self.alpha[client_id] = a
        self.beta[client_id] = b

    def prob(self, client_id: int) -> float:
        a = self.alpha.get(client_id, 1.0)
        b = self.beta.get(client_id, 1.0)
        return a / (a + b)


class DeFL(Defense):
    name = "defl"

    def __init__(self, device: str = "cpu", delta: float = 0.05, tau: float = 2.5):
        super().__init__(device)
        self.delta = float(delta)          # CLP relative-rise threshold (Eq. 4)
        self.tau = float(tau)              # MOUD per-layer outlier z-threshold
        self._beta = BetaTracker()
        self._prev_total_fgnv = None       # S(t-1)
        self._layer_groups: dict | None = None

    def reset(self, init_global):
        super().reset(init_global)         # clones into self._global
        self._beta = BetaTracker()
        self._prev_total_fgnv = None
        self._layer_groups = group_layers(list(init_global.keys()))

    # DeFL is the only stateful defense: ``step`` bumps every client's Beta counts
    # and advances S(t-1). Snapshot/restore keeps a *scored* (uncommitted) round
    # from polluting that memory — see ``benchmark.defenses.base.Defense``.
    def state_snapshot(self) -> dict:
        return {"alpha": dict(self._beta.alpha), "beta": dict(self._beta.beta),
                "prev_total_fgnv": self._prev_total_fgnv}

    def state_restore(self, snapshot: dict) -> None:
        self._beta.alpha = dict(snapshot.get("alpha", {}))
        self._beta.beta = dict(snapshot.get("beta", {}))
        self._prev_total_fgnv = snapshot.get("prev_total_fgnv")

    def step(self, updates, poisoned_ids) -> StepResult:
        import torch
        gw = self._global
        if self._layer_groups is None:
            self._layer_groups = group_layers(list(gw.keys()))
        groups = self._layer_groups
        L = len(groups)

        # 1) FGNV per client + this round's CLP decision.
        fgnv = [fgnv_for_update(u.weights, gw, groups) for u in updates]
        data_w = [float(u.metadata.get("train_samples", 1.0)) for u in updates]
        agg_fgnv = aggregate_fgnv(fgnv, data_w)
        total_now = sum(agg_fgnv)
        in_clp = is_clp(total_now, self._prev_total_fgnv, self.delta)
        self._prev_total_fgnv = total_now

        # 2) MOUD-Vote detection, then 3) bump each client's Beta counts.
        flagged, votes, vote_threshold = moud_vote(fgnv, self.tau)
        for i, u in enumerate(updates):
            self._beta.update(u.client_id, benign=not flagged[i])

        # 4) Aggregation weights: p_i (Beta mean), forced to 0 for detected-malicious
        #    clients DURING a CLP (hard removal); soft p_i otherwise.
        coeffs = []
        for i, u in enumerate(updates):
            p = self._beta.prob(u.client_id)
            if in_clp and flagged[i]:
                p = 0.0
            coeffs.append(p * data_w[i])

        # ``votes/L`` was reported as P(malicious), and it is NOT one: the flag test
        # is ``votes >= vote_threshold`` with an ADAPTIVE threshold, so on this
        # codebase's two-layer model (one group per nn.Linear) the rule settles at
        # ``votes >= 1`` and a client DeFL just caught reported ``p = 1/2 = 0.5``.
        # The attacker collected half of its full stealth bonus for being detected.
        #
        # The score is therefore the vote count measured against the round's own
        # threshold, plus a strictly bounded sub-vote tiebreak built from the raw
        # z-score magnitudes. The tiebreak restores continuity that a 0..L integer
        # cannot carry (with L=2 the vote count alone takes three values, so the
        # "continuous" stealth signal the reward is built on was nearly binary), and
        # because |tiebreak| < 0.5 it can never move a client across the half-vote
        # gap that separates flagged from accepted.
        #
        # ``confidence`` is certainty in the verdict (see core.types) = |2p - 1|.
        z = per_layer_zscores(fgnv)
        scores = [votes[i] + _subvote_tiebreak(z[i], self.tau) for i in range(len(updates))]
        p_mals = boundary_calibrated_p(scores, vote_threshold - 0.5, flags=flagged)
        verdicts = [
            DetectionVerdict(
                u.client_id, bool(flagged[i]),
                abs(2.0 * p_mals[i] - 1.0),
                f"votes={votes[i]}/{L} thr={vote_threshold} clp={int(in_clp)}",
                p_malicious=p_mals[i],
            )
            for i, u in enumerate(updates)
        ]

        # 5) Trust+data-weighted average of ABSOLUTE weights (renormalised).
        wsum = sum(coeffs)
        if wsum <= 0.0:
            # Everyone removed (CLP + all flagged) -> keep the previous global.
            return StepResult(None, verdicts,
                              info={"in_clp": in_clp, "total_fgnv": total_now,
                                    "n_flagged": int(sum(flagged))})
        coeffs = [c / wsum for c in coeffs]
        new_global = {}
        for k in gw.keys():
            acc = None
            for i, u in enumerate(updates):
                if coeffs[i] == 0.0:
                    continue
                t = u.weights[k].float() * coeffs[i]
                acc = t if acc is None else acc + t
            new_global[k] = acc.to(gw[k].dtype)
        self._global = new_global

        return StepResult(new_global, verdicts,
                          info={"in_clp": in_clp, "total_fgnv": total_now,
                                "n_flagged": int(sum(flagged)),
                                "agg_fgnv": agg_fgnv})
