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
    clean_accuracy: float | None = None,
    success_drop: float | None = None,
) -> RoundMetrics:
    """Compute all per-round metrics from raw verdicts and accuracies.

    ``clean_accuracy`` is THIS round's clean counterfactual — the accuracy the
    aggregate reaches with no poison (``rl.env.clean_reference_accuracy``) — or
    ``None`` when it could not be measured. ``success_drop`` is the damage bar the
    round had to clear, i.e. the schedule's ``win_fraction * target_accuracy_drop``.
    Both are needed to say whether the attack actually SUCCEEDED; without them the
    round is recorded as unsuccessful rather than guessed at.
    """
    tp, fn, fp, tn = confusion_counts(verdicts, malicious_ids)

    tpr = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    apr = _safe_div(current_accuracy, baseline_accuracy)

    # Evasion: at least one malicious client was not flagged. This is what
    # `attack_success` used to mean, and it is NOT attack success — it says nothing
    # about damage. The logs showed the contradiction plainly: a round where accuracy
    # ROSE by 1pp was recorded as attack_success=True (one poisoned client slipped
    # through) while a round where it FELL by 5pp was recorded as False (the poisoned
    # client was caught). Kept as its own field because evasion is still a real,
    # separately interesting quantity — it just is not success.
    evaded = fn > 0

    # Damage relative to the clean counterfactual, and success as "did the attack
    # cause the damage it was asked for". Unmeasurable -> not a success, never a guess.
    induced_drop = (None if clean_accuracy is None
                    else float(clean_accuracy) - float(current_accuracy))
    apr_vs_clean = (None if not clean_accuracy
                    else _safe_div(current_accuracy, clean_accuracy))
    attack_success = (induced_drop is not None and success_drop is not None
                      and induced_drop >= float(success_drop))

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
        evaded=evaded,
        clean_accuracy=(None if clean_accuracy is None else float(clean_accuracy)),
        induced_drop=induced_drop,
        success_drop=(None if success_drop is None else float(success_drop)),
        accuracy_preservation_vs_clean=apr_vs_clean,
    )
