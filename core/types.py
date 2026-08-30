"""Shared data types used across all components."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelUpdate:
    """A single client's model weights submission.

    ``weights`` is ALWAYS a locally trained state_dict — poisoned clients included.
    The attack is label flipping (see ``data.label_flip``), so a poisoned update is
    the honest output of honest SGD over corrupted labels; nothing edits the
    weights after training. ``metadata['poisoned']`` marks which is which, and for
    a poisoned client ``metadata`` also carries ``n_flipped`` / ``flip_fraction``.
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

    ``poisoned_client_ids`` is the round's ground truth — the clients that shipped
    flipped labels. ``defender_reward`` is the verifiable signal the GRPO update
    used. ``attack_effectiveness`` is NOT a reward: the attack is a fixed adaptive
    schedule with no policy, so this is a pure measurement of how much accuracy
    the round's flipped labels cost, normalized by ``attack_goal``'s target drop
    (see ``rl.rewards.attack_effectiveness``). ``attack_metadata`` carries the
    ladder's level and transition for the round.
    """
    round_num: int
    attack_goal: dict
    poisoned_client_ids: list[int]
    predicted_labels: list[dict]          # [{client_id, is_suspicious, confidence, reason}]
    test_accuracy: float
    baseline_accuracy: float
    attack_effectiveness: float
    defender_reward: float
    learning_agent: str                   # "defender" | "none"
    attack_metadata: dict = field(default_factory=dict)
