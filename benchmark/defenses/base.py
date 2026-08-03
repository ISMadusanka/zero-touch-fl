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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.types import DetectionVerdict, ModelUpdate


def rank_normalized_scores(scores: list) -> list:
    """Turn per-client suspicion scores into calibrated P(malicious) in [0, 1].

    Multi-Krum and DnC score clients on scales that are unbounded and
    round-dependent (sums of squared distances, squared spectral projections),
    with the convention "lower = more trusted". Those numbers cannot be used as a
    probability: they routinely exceed 1, so clipping them collapses the signal to
    a binary, and ``+inf`` (the sentinel both use for a non-finite client) is not
    even serializable. See ``core.types.DetectionVerdict.p_malicious``.

    We therefore report the client's normalized RANK: the fraction of OTHER
    clients that scored strictly lower.

        p_i = |{j : score_j < score_i}| / (n - 1)

    This is bounded, scale-free, comparable across rounds and defenses, monotone
    in the defense's own suspicion, ties-aware (equal scores get equal p), and it
    agrees with the keep/drop decision by construction — both defenses keep the
    lowest-scoring clients, so survivors land below the dropped ones. Crucially it
    stays CONTINUOUS inside the surviving set, which is what gives the attacker's
    stealth reward a usable gradient.

    NaN is ordered as ``+inf`` (the worst score), matching ``keep_lowest`` /
    ``select_lowest``, so an undefined-score client is never scored as trusted.
    """
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    inf = float("inf")
    ordered = [inf if s != s else float(s) for s in scores]   # NaN -> +inf
    return [
        sum(1 for other in ordered if other < s) / (n - 1)
        for s in ordered
    ]


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
