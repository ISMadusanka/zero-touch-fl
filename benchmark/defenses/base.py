"""Defense interface for the benchmark.

A benchmark ``Defense`` maintains its OWN global model and, each round, decides
which clients to trust and produces a new global model from the SAME set of
client updates every other defense sees. This lets us compare defenses head to
head: same attacker, same updates, different defense.

Unlike ``core.interfaces.BaseAggregator`` (which takes externally-supplied
verdicts and returns only a state_dict), a ``Defense`` (a) owns its current
global model — needed by delta/feature-based defenses like FLTrust and the LLM
defender — and (b) RETURNS its own per-client accept/reject verdicts so we can
score detection quality against the ground-truth poisoned set.
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.types import DetectionVerdict, ModelUpdate

#: The largest ``p_malicious`` an UN-flagged client may report. The calibration
#: invariant below is ``p >= 0.5 <=> is_suspicious``, so an accepted client that
#: cannot be separated from the boundary by its score alone is pinned just under
#: 0.5 rather than being allowed to cross it.
JUST_BELOW_HALF = 0.5 - 1e-9


def _snap_to_flag(p: float, flagged: bool) -> float:
    """Enforce ``p >= 0.5 <=> flagged`` for one client."""
    return max(p, 0.5) if flagged else min(p, JUST_BELOW_HALF)


def _median_sorted(values: list) -> float:
    """Median of an ALREADY SORTED list; 0.0 for an empty one."""
    n = len(values)
    if n == 0:
        return 0.0
    mid = n // 2
    return values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])


def boundary_calibrated_p(scores: list, threshold: float, *,
                          higher_is_suspicious: bool = True,
                          flags: list | None = None) -> list:
    """Map raw per-client suspicion scores to P(malicious) in [0, 1].

    Every defense here decides by comparing its own score to its own boundary
    (FLTrust: ``trust <= 0``; DeFL: ``votes >= thr``; Multi-Krum / DnC: "not in the
    lowest-scoring keep set"). Those scores are on wildly different, unbounded,
    round-dependent scales, so they are not probabilities. This turns one into a
    probability with the **invariant**

        p_i >= 0.5   if and only if   the defense flagged client i

    while staying continuous and monotone in the client's signed distance PAST that
    boundary. Concretely, with ``m_i`` = how far client ``i`` is on the suspicious
    side of ``threshold``::

        p_i = 0.5 * (1 + tanh(m_i / s)),    s = MEDIAN |m| over the finite scores

    The **sign** of ``m_i`` is absolute — it carries the accept/reject decision, and
    it is invariant to ``s`` — so the hard flag and the soft score can never
    disagree. The **magnitude** is normalized by the round's own spread, so the map
    neither saturates on a defense whose scores live at 1e-3 nor collapses on one
    whose scores live at 1e9. That split is the whole point: the attacker's
    ``stealth`` reward needs a usable gradient (magnitude) that still means
    "undetected" (sign).

    ``s`` is the **median**, not the mean, because a blatant attack is exactly the
    case that produces a handful of enormous margins: Multi-Krum and DnC score by
    squared distance, so four scaled-up poisoned clients out of twenty inflated a
    mean-based ``s`` until every remaining client landed within 0.002 of 0.5. The
    caught clients were still reported correctly, but all gradient among the
    *accepted* ones — the only clients whose stealth the attacker can still improve
    — was gone. A median ignores the outliers, so the near-boundary population keeps
    its spread and the blatant ones simply saturate toward 1.0, which is right.

    Two earlier attempts at this field were both wrong, in opposite directions, and
    both silently inverted the attacker's stealth gradient:

    * **Raw suspicion score as a probability** (``1 - ReLU(cos)`` for FLTrust,
      ``votes/L`` for DeFL). The decision boundary is not at 0.5, so an ACCEPTED
      client routinely reported ``p > 0.5`` (FLTrust cosines are ~0.05 in a small
      model, so ``p ~ 0.95`` for every honest client) and a FLAGGED one routinely
      reported ``p < 0.5`` (DeFL's adaptive threshold flags on as little as one
      layer-vote out of two, i.e. ``p = 0.5``). The attacker was paid for being
      caught and punished for evading.
    * **Cohort rank** (``rank_normalized_scores``, used by Multi-Krum / DnC). Bounded
      and monotone, but purely relative: the mean is ~0.5 every round by
      construction, so ``p`` moved when OTHER clients moved and carried no
      information about whether this client was detected.

    ``higher_is_suspicious=False`` flips the comparison for a TRUST score (FLTrust),
    where a *low* value is the suspicious one.

    ``flags`` — the defense's own boolean decisions — makes the invariant
    unconditional. It is needed because a keep-the-lowest-``k`` rule breaks ties by
    index, so two clients with an identical score can land on opposite sides of the
    cut and no threshold can separate them. Any client whose margin disagrees with
    its flag is snapped to its side of 0.5 (see :func:`_snap_to_flag`) instead of
    being reported with the wrong sign.

    NaN is ordered as ``+inf`` (the worst score), matching ``keep_lowest`` /
    ``select_lowest``, so an undefined-score client is never scored as trusted.
    """
    n = len(scores)
    if n == 0:
        return []
    inf = float("inf")
    sign = 1.0 if higher_is_suspicious else -1.0
    thr = float(threshold)

    # Signed distance past the boundary: > 0 = the defense's rule rejects this
    # client, < 0 = it accepts, 0 = exactly on the cut.
    margins: list[float] = []
    for s in scores:
        v = inf if s != s else float(s)                 # NaN -> +inf (worst)
        if v == inf or v == -inf:
            margins.append(v if higher_is_suspicious else -v)
        else:
            margins.append(sign * (v - thr))

    finite = sorted(abs(m) for m in margins if m not in (inf, -inf))
    scale = _median_sorted(finite)
    if scale <= 0.0 and finite:
        # More than half the cohort sits exactly on the boundary, so the median
        # carries no spread — fall back to the mean before giving up.
        scale = sum(finite) / len(finite)
    if scale <= 0.0:
        # Every finite score sits exactly on the boundary: no spread to report.
        scale = 1.0

    out: list[float] = []
    for i, m in enumerate(margins):
        if m == inf:
            p = 1.0
        elif m == -inf:
            p = 0.0
        else:
            p = 0.5 * (1.0 + math.tanh(m / scale))
        if flags is not None and i < len(flags):
            p = _snap_to_flag(p, bool(flags[i]))
        out.append(p)
    return out


def selection_boundary(scores: list, kept) -> float:
    """The score threshold implied by a keep-the-lowest-``k`` selection.

    Multi-Krum and DnC do not compare against a fixed number — they keep a fixed
    COUNT. The equivalent boundary is therefore the midpoint between the worst kept
    score and the best dropped one, which is what :func:`boundary_calibrated_p`
    needs. Degenerate cases (nothing dropped, nothing kept, ``+inf`` on either side)
    return a threshold that puts every client on the correct side; exact ties
    straddling the cut are resolved by ``boundary_calibrated_p``'s ``flags``.
    """
    inf = float("inf")
    vals = [inf if s != s else float(s) for s in scores]
    keep = [v for i, v in enumerate(vals) if i in kept]
    drop = [v for i, v in enumerate(vals) if i not in kept]
    if not drop:                                  # everyone kept -> boundary above all
        hi = max(keep) if keep else 0.0
        return 0.0 if hi == inf else hi + max(1.0, abs(hi))
    if not keep:                                  # everyone dropped -> boundary below all
        lo = min(drop)
        return 0.0 if lo == inf else lo - max(1.0, abs(lo))
    hi, lo = max(keep), min(drop)
    if lo == inf:                                 # dropped only non-finite clients
        return 0.0 if hi == inf else hi + max(1.0, abs(hi))
    if hi == inf:                                 # a non-finite client survived: degenerate
        return lo
    return 0.5 * (hi + lo)


@dataclass
class StepResult:
    """What a defense produces for one round."""
    new_global: dict | None             # aggregated state_dict, or None = keep previous global
    verdicts: list[DetectionVerdict]    # per-client decision; is_suspicious=True means REJECTED
    info: dict = field(default_factory=dict)


class Defense(ABC):
    """One defense maintaining its own global model across rounds."""

    name: str = "defense"
    requires_llm: bool = False          # True if it needs the LLM policy (e.g. the LLM defender)

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._global: dict | None = None

    def reset(self, init_global: dict):
        """Initialise this defense's global model. Clone so the worlds stay
        independent (each defense evolves its own copy)."""
        self._global = {k: v.clone() for k, v in init_global.items()}

    def global_weights(self) -> dict | None:
        return self._global

    # ------------------------------------------------------------------
    # Hooks for running a defense against an EXTERNALLY owned global model
    # (the Phase-2 arms race — see ``server.algo_defender``). The benchmark
    # itself never needs them: there each defense owns its own world.
    # ------------------------------------------------------------------
    def sync_global(self, weights: dict) -> None:
        """Re-base this defense on an externally owned global model.

        Unlike :meth:`reset` this KEEPS whatever cross-round memory the defense
        has accumulated (DeFL's Beta counts, for example) and only changes the
        model the next :meth:`step` measures its deltas against. The reference is
        adopted as-is (no clone): the caller owns it, and ``step`` replaces
        ``self._global`` with its own aggregate anyway.
        """
        self._global = weights

    def state_snapshot(self) -> dict:
        """Copy of the cross-round state :meth:`step` mutates.

        Empty for the stateless aggregators (Multi-Krum, FLTrust, FedAvg,
        Oracle). Paired with :meth:`state_restore` it lets a caller SCORE a
        candidate round without the defense's memory absorbing it — needed when
        several candidate attacks are graded against one identical defense.
        """
        return {}

    def state_restore(self, snapshot: dict) -> None:
        """Undo every mutation made since :meth:`state_snapshot`. No-op by default."""
        return None

    @abstractmethod
    def step(self, updates: list[ModelUpdate], poisoned_ids: set[int]) -> StepResult:
        """Process one round of client updates against this defense's current
        global model. Implementations update ``self._global`` and return a
        ``StepResult`` (new global + per-client verdicts)."""
        raise NotImplementedError
