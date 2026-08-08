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
class DetectionVerdict:
    """One detector's classification for one client.

    ``is_suspicious=True`` means the detector labelled the client malicious.
    Reused unchanged from the old detector contract so aggregation/metrics keep
    working — only the producer changed (LLM classification instead of a
    hardcoded statistical rule).

    **``confidence`` is certainty in THIS verdict**, not a suspicion score: 1.0 =
    "I am sure of the label I just gave", 0.0 = "coin flip". That is the LLM
    defender's contract (it is asked for exactly that) and it is what
    ``rl.rewards._soft_malicious_prob`` assumes when it reconstructs a soft
    P(malicious) from ``(is_suspicious, confidence)``.

    ``p_malicious`` is the OPTIONAL, explicitly calibrated P(malicious) in [0, 1],
    monotonically increasing in how suspicious the detector finds the client. It
    exists because ``(is_suspicious, confidence)`` **cannot** represent the
    algorithmic defenses: they are threshold filters whose natural output is a
    suspicion score whose decision boundary is not at p=0.5 (FLTrust drops at
    trust<=0, Multi-Krum/DnC drop a fixed count per round). Squeezing such a
    score into ``confidence`` inverted the soft signal for every *un-flagged*
    client — the attacker's ``stealth`` reward then paid it for sitting ON the
    detection boundary instead of for looking honest. Producers that can compute a
    calibrated probability set this field; consumers must prefer it over the
    ``(is_suspicious, confidence)`` reconstruction. ``None`` = not supplied.

    **Calibration contract.** When ``p_malicious`` is supplied it MUST satisfy

        p_malicious >= 0.5   if and only if   is_suspicious

    Anything else is not a probability, it is a raw suspicion score wearing one, and
    it silently inverts every consumer that reads it — most importantly the
    attacker's ``stealth`` reward, which then pays for being caught. Reporting a raw
    score here (``1 - ReLU(cos)``, ``votes/L``) is exactly the bug that produced an
    attacker reward of 0.44 on a round where every poisoned client was detected and
    no damage was done. ``benchmark.defenses.base.boundary_calibrated_p`` is the
    shared helper that guarantees the invariant while keeping the value continuous;
    ``tests/test_p_malicious_calibration.py`` asserts it for every defense.
    """
    client_id: int
    is_suspicious: bool
    confidence: float
    reason: str
    p_malicious: float | None = None


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
