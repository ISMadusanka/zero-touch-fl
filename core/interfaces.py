"""Abstract base classes defining component contracts.

Only the aggregation contract survives the redesign. The attacker no longer
implements a fixed ``BaseAttack`` plugin (the LLM emits an attack plan)
and the defender no longer implements a fixed ``BaseDetector`` rule (the LLM
classifies clients directly), so those interfaces were removed.
"""

from abc import ABC, abstractmethod
from core.types import ModelUpdate, DetectionVerdict


class BaseAggregator(ABC):
    """Interface for aggregation strategies."""

    @abstractmethod
    def aggregate(
        self, updates: list[ModelUpdate], verdicts: list[DetectionVerdict],
        strategy: dict | None = None,
    ) -> dict:
        """Aggregate non-suspicious updates into a new global state_dict."""
        ...
