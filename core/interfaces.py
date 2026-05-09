"""Abstract base classes defining component contracts."""

from abc import ABC, abstractmethod
from core.types import ModelUpdate, DetectionVerdict


class BaseAttack(ABC):
    """Interface for model poisoning attacks."""

    @abstractmethod
    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Apply poisoning to client weights. Returns poisoned state_dict."""
        ...


class BaseDetector(ABC):
    """Interface for anomaly detection."""

    @abstractmethod
    def analyze(
        self, updates: list[ModelUpdate], global_weights: dict
    ) -> list[DetectionVerdict]:
        """Analyze all client updates. Returns one verdict per client."""
        ...


class BaseAggregator(ABC):
    """Interface for aggregation strategies."""

    @abstractmethod
    def aggregate(
        self, updates: list[ModelUpdate], verdicts: list[DetectionVerdict]
    ) -> dict:
        """Aggregate non-suspicious updates into a new global state_dict."""
        ...
