"""Shared data types used across all components."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelUpdate:
    """A single client's model weights submission.

    For benign clients ``weights`` is the locally trained state_dict. For
    poisoned clients it is produced by applying the attacker LLM's attack plan
    to the benign weights (see ``agents.attack_ops.apply_plan``).
    """
    client_id: int
    weights: dict  # state_dict tensors
    metadata: dict = field(default_factory=dict)


@dataclass
class ClassEval:
    """A test-set evaluation broken down PER CLASS.

    ``overall`` is plain top-1 accuracy — exactly what ``FedServer.evaluate``
    returns, so the untargeted path is unaffected. ``per_class[c]`` is class
    ``c``'s **recall**: of all test samples truly labelled ``c``, the fraction the
    model classified correctly. That is the quantity a targeted
    "make label ``c`` be misclassified" attack is scored on — a successful attack
    drives ``per_class[label]`` toward 0 while leaving every other entry alone.

    ``support[c]`` is how many test samples carry label ``c`` (MNIST's test split
    is mildly unbalanced: 892–1135 per class), so a reader can tell a real
    collapse from small-sample noise.
    """
    overall: float
    per_class: list[float]
    support: list[int]

    def recall(self, c: int) -> float:
        """Recall of class ``c`` (0.0 if ``c`` is out of range)."""
        return self.per_class[c] if 0 <= c < len(self.per_class) else 0.0

    def others_mean(self, c: int) -> float:
        """Mean recall over every class EXCEPT ``c``.

        Unweighted on purpose: a targeted attack must leave *each* other class
        working, and support-weighting would let a big class mask a small one.
        """
        vals = [v for i, v in enumerate(self.per_class) if i != c]
        return sum(vals) / len(vals) if vals else 0.0


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
