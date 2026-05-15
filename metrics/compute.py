"""Pure functions for computing detection / accuracy metrics.

All functions here are side-effect free: they take primitive inputs and return
metric values. State accumulation is handled by `MetricsTracker`.
"""

from collections.abc import Iterable

from core.types import DetectionVerdict
from metrics.types import RoundMetrics


def _safe_div(num: float, denom: float) -> float:
    """Division that returns 0.0 when the denominator is 0 (no samples)."""
    return num / denom if denom > 0 else 0.0


def confusion_counts(
    verdicts: Iterable[DetectionVerdict],
    malicious_ids: set[int],
) -> tuple[int, int, int, int]:
    """Return (tp, fn, fp, tn) for a single round.

    A client is a positive sample if it is in `malicious_ids`. A verdict's
    `is_suspicious=True` is treated as a positive prediction.
    """
    tp = fn = fp = tn = 0
    for v in verdicts:
        is_malicious = v.client_id in malicious_ids
        if is_malicious and v.is_suspicious:
            tp += 1
        elif is_malicious and not v.is_suspicious:
            fn += 1
        elif not is_malicious and v.is_suspicious:
            fp += 1
        else:
            tn += 1
    return tp, fn, fp, tn


def compute_round_metrics(
    round_num: int,
    verdicts: list[DetectionVerdict],
    malicious_ids: set[int],
    current_accuracy: float,
    baseline_accuracy: float,
) -> RoundMetrics:
    """Compute all per-round metrics from raw verdicts and accuracies."""
    tp, fn, fp, tn = confusion_counts(verdicts, malicious_ids)

    tpr = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    apr = _safe_div(current_accuracy, baseline_accuracy)

    # An attack "succeeds" in a round when at least one malicious client is
    # not flagged. With a single attacker this collapses to `fn > 0`.
    attack_success = fn > 0

    return RoundMetrics(
        round_num=round_num,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        attack_success=attack_success,
        tpr=tpr,
        fpr=fpr,
        recall=tpr,
        accuracy_preservation_rate=apr,
        current_accuracy=current_accuracy,
        baseline_accuracy=baseline_accuracy,
    )
