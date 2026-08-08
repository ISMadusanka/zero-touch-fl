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

    # Per-round derived rates
    attack_success: bool            # the attack caused the damage it was asked for:
                                    # induced_drop >= success_drop. NOT evasion — see
                                    # `evaded`, which is what this field used to mean.
    tpr: float                      # recall on malicious clients (TP / (TP + FN))
    fpr: float                      # FP / (FP + TN)
    recall: float                   # alias of TPR — kept explicit for clarity
    accuracy_preservation_rate: float  # current_accuracy / baseline_accuracy

    # Raw accuracies, helpful for downstream analysis
    current_accuracy: float
    baseline_accuracy: float

    # --- Damage, measured against THIS round's clean counterfactual -------------
    # `accuracy_preservation_rate` divides by the FIXED phase-1 baseline, so it barely
    # moves (0.97-0.99 across a whole recorded run) no matter what the attack did. The
    # counterfactual-relative fields below are the ones that answer "what did this
    # attack cost". All are None when the round's defense produced no clean aggregate,
    # in which case there was nothing to measure — filter those rounds out rather than
    # reading a 0 drop as "the attack achieved nothing".
    evaded: bool = False            # at least one malicious client passed detection
    clean_accuracy: float | None = None      # unpoisoned counterfactual for this round
    induced_drop: float | None = None        # clean_accuracy - current_accuracy
    success_drop: float | None = None        # the damage bar this round had to clear
    accuracy_preservation_vs_clean: float | None = None   # current / clean_accuracy

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
    attack_success_rate: float      # rounds that hit their damage bar / total_rounds
    tpr: float                      # sum(TP) / (sum(TP) + sum(FN))
    fpr: float                      # sum(FP) / (sum(FP) + sum(TN))
    recall: float                   # alias of TPR

    # Accuracy preservation uses the final round's accuracy
    accuracy_preservation_rate: float
    baseline_accuracy: float
    final_accuracy: float

    # Evasion, reported separately from success (see RoundMetrics.attack_success), and
    # the mean damage over the rounds where the counterfactual was actually measurable.
    evasion_rate: float = 0.0       # rounds with a missed malicious client / total
    mean_induced_drop: float = 0.0  # mean(clean_accuracy - accuracy) over measured rounds
    measured_rounds: int = 0        # rounds whose clean counterfactual was measurable

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
