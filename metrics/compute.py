"""Pure functions for computing detection / accuracy metrics.

All functions here are side-effect free: they take primitive inputs and return
metric values. State accumulation is handled by `MetricsTracker`.
"""

from collections.abc import Iterable

from core.types import DetectionVerdict
from metrics.types import RoundMetrics


DEFAULT_TARGET_ACCURACY_DROP = 0.10


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
    *,
    reference_accuracy: float | None = None,
    target_accuracy_drop: float = DEFAULT_TARGET_ACCURACY_DROP,
) -> RoundMetrics:
    """Compute all per-round metrics from raw verdicts and accuracies.

    ``attack_success`` has one canonical meaning: the attack-induced accuracy
    drop reached ``target_accuracy_drop``. The drop is attributed against this
    round's clean counterfactual when ``reference_accuracy`` is supplied; the
    Phase-1 baseline is a backwards-compatible fallback for older callers.

    Detection evasion is reported independently as ``evasion_success``. This
    prevents an ineffective update that merely passes the detector from being
    counted as a successful attack.
    """
    tp, fn, fp, tn = confusion_counts(verdicts, malicious_ids)

    tpr = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    apr = _safe_div(current_accuracy, baseline_accuracy)

    reference = (float(baseline_accuracy) if reference_accuracy is None
                 else float(reference_accuracy))
    target = float(target_accuracy_drop)
    if target <= 0.0:
        raise ValueError("target_accuracy_drop must be > 0")
    induced_drop = reference - float(current_accuracy)

    # A no-op/malformed round has no attack to credit, even if model noise alone
    # happens to move accuracy by the requested amount.
    attack_success = bool(malicious_ids) and induced_drop >= target - 1e-12
    evasion_success = fn > 0

    return RoundMetrics(
        round_num=round_num,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        attack_success=attack_success,
        evasion_success=evasion_success,
        reference_accuracy=reference,
        induced_drop=induced_drop,
        target_accuracy_drop=target,
        tpr=tpr,
        fpr=fpr,
        recall=tpr,
        accuracy_preservation_rate=apr,
        current_accuracy=current_accuracy,
        baseline_accuracy=baseline_accuracy,
    )
