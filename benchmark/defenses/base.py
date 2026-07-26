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
        """Initialise this defense's global model AND clear any cross-round state.

        Clones the weights so the worlds stay independent (each defense evolves its
        own copy). Subclasses that carry history across rounds (e.g. DeFL) override
        this to also reset that history.
        """
        self.set_global_weights(init_global)

    def set_global_weights(self, global_weights: dict):
        """Point this defense at ``global_weights`` WITHOUT touching its history.

        ``reset`` is "start a new run"; this is "judge this round against that
        global model". The ensemble (``server/defense_ensemble.py``) runs several
        defenses as pure detectors over ONE shared global model, so it re-points
        them every round instead of letting each evolve its own — but DeFL's CLP /
        trust history must survive that, hence the split.
        """
        self._global = {k: v.clone() for k, v in global_weights.items()}

    def global_weights(self) -> dict | None:
        return self._global

    # -- cross-round state (only defenses with history need to override) --------
    def state_dict(self) -> dict:
        """Serializable snapshot of this defense's cross-round state.

        Defaults to empty: most defenses are memoryless (their verdict depends only
        on this round's updates + the current global). Used by the ensemble to run a
        defense as a pure scoring function during GRPO rollouts — the rollouts must
        not advance a defense's history, only the COMMITTED round may.
        """
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore a snapshot from :meth:`state_dict` (no-op when memoryless)."""
        return None

    @abstractmethod
    def step(self, updates: list[ModelUpdate], poisoned_ids: set[int]) -> StepResult:
        """Process one round of client updates against this defense's current
        global model. Implementations update ``self._global`` and return a
        ``StepResult`` (new global + per-client verdicts)."""
        raise NotImplementedError
