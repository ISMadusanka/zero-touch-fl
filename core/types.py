"""Shared data types used across all components."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelUpdate:
    """A single client's model weights submission.

    For benign clients ``weights`` is the locally trained state_dict. For
    poisoned clients it is the raw poisoned state_dict emitted by the attacker
    LLM (after validation/clamping by ``agents.weight_codec``).
    """
    client_id: int
    weights: dict  # state_dict tensors
    metadata: dict = field(default_factory=dict)


@dataclass
class DetectionVerdict:
    """The defender LLM's classification for one client.

    ``is_suspicious=True`` means the defender labelled the client malicious.
    Reused unchanged from the old detector contract so aggregation/metrics keep
    working — only the producer changed (LLM classification instead of a
    hardcoded statistical rule).
    """
    client_id: int
    is_suspicious: bool
    confidence: float
    reason: str


@dataclass
class RoundLog:
    """Complete record of a single simulation/training round.

    The old adaptation/skip bookkeeping (attacker_adapted, defender_adapted,
    all_clients_flagged, round_skipped) is gone — adaptation is now RL, not a
    hand-rolled feedback loop. The new fields capture the per-round ground
    truth (``poisoned_client_ids``), the defender's predictions, and the
    verifiable rewards used for the GRPO updates.
    """
    round_num: int
    attack_goal: dict
    poisoned_client_ids: list[int]
    predicted_labels: list[dict]          # [{client_id, is_suspicious, confidence, reason}]
    test_accuracy: float
    baseline_accuracy: float
    attacker_reward: float
    defender_reward: float
    learning_agent: str                   # "attacker" | "defender" | "none"
    attack_metadata: dict = field(default_factory=dict)
