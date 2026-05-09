"""Shared data types used across all components."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelUpdate:
    """A single client's model weights submission."""
    client_id: int
    weights: dict  # state_dict tensors
    metadata: dict = field(default_factory=dict)


@dataclass
class DetectionVerdict:
    """Anomaly detector's verdict for one client."""
    client_id: int
    is_suspicious: bool
    confidence: float
    reason: str


@dataclass
class RoundLog:
    """Complete record of a single simulation round."""
    round_num: int
    attack_strategy: dict
    defend_strategy: dict
    verdicts: list[dict]
    test_accuracy: float
    baseline_accuracy: float
    attack_detected: bool
    attacker_adapted: bool
    defender_adapted: bool
