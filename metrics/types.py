"""Dataclasses representing per-round and aggregate evaluation metrics."""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class RoundMetrics:
    """Detection / accuracy metrics for a single simulation round.

    Confusion matrix is computed against the ground-truth set of malicious
    client ids:
      - TP: malicious client flagged as suspicious
      - FN: malicious client missed
      - FP: honest client flagged as suspicious
      - TN: honest client correctly not flagged
    """

    round_num: int

    # Confusion matrix
    tp: int
    fn: int
    fp: int
    tn: int

    # Per-round attack outcome. ``attack_success`` is the canonical goal result:
    # the attack itself induced at least the requested accuracy drop. Detection
    # evasion is intentionally separate; passing a filter without damaging the
    # model is not an attack success.
    attack_success: bool
    evasion_success: bool           # at least one malicious client passed detection
    reference_accuracy: float       # clean counterfactual used for attribution
    induced_drop: float             # reference_accuracy - current_accuracy
    target_accuracy_drop: float

    # Per-round detection / preservation rates
    tpr: float                      # recall on malicious clients (TP / (TP + FN))
    fpr: float                      # FP / (FP + TN)
    recall: float                   # alias of TPR — kept explicit for clarity
    accuracy_preservation_rate: float  # current_accuracy / baseline_accuracy

    # Raw accuracies, helpful for downstream analysis
    current_accuracy: float
    baseline_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    """Cumulative metrics over the entire simulation."""

    total_rounds: int

    # Confusion matrix summed across rounds
    tp: int
    fn: int
    fp: int
    tn: int

    # Cumulative rates
    attack_success_rate: float      # rounds whose induced drop reached the target
    evasion_success_rate: float     # rounds where at least one poisoner evaded
    tpr: float                      # sum(TP) / (sum(TP) + sum(FN))
    fpr: float                      # sum(FP) / (sum(FP) + sum(TN))
    recall: float                   # alias of TPR

    # Accuracy preservation uses the final round's accuracy
    accuracy_preservation_rate: float
    baseline_accuracy: float
    final_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
