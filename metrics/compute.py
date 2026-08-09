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
    reference_accuracy: float | None = None,
    attack_goal_met: bool | None = None,
) -> RoundMetrics:
    """Compute all per-round metrics from raw verdicts and accuracies.

    ``attack_goal_met`` is the verdict of ``rl.switch.attacker_succeeded`` —
    enough of the goal's damage done, collateral within tolerance, AND the
    poisoned client evaded. Detection metrics alone cannot answer that, so the
    caller supplies it. Passing ``None`` means "could not judge", and
    ``attack_success`` then degrades to plain evasion with ``goal_evaluated``
    set False so a reader can tell the two apart.

    Conflating the two is not academic: under a ``targeted_label`` goal a round
    that leaves the model *better than baseline* still has ``fn=1`` whenever the
    detector stays quiet, and would be counted a successful attack.

    ``reference_accuracy`` is the round's clean counterfactual
    (``FLArmsRaceEnv.clean_reference_accuracy``). It is what accuracy
    preservation should be measured against — ``baseline_accuracy`` is pinned at
    the Phase-1 value for the whole run and goes stale the moment an honest FL
    interlude moves the model. Defaults to ``baseline_accuracy`` when unknown.
    """
    tp, fn, fp, tn = confusion_counts(verdicts, malicious_ids)

    tpr = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    reference = baseline_accuracy if reference_accuracy is None else reference_accuracy
    apr = _safe_div(current_accuracy, reference)
    bpr = _safe_div(current_accuracy, baseline_accuracy)

    # Evasion: at least one malicious client was not flagged. With a single
    # attacker this collapses to `fn > 0`. This is a DETECTION outcome — it says
    # nothing about whether the attack achieved what it set out to do.
    attack_evaded = fn > 0
    goal_evaluated = attack_goal_met is not None
    attack_success = bool(attack_goal_met) if goal_evaluated else attack_evaded

    return RoundMetrics(
        round_num=round_num,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        attack_success=attack_success,
        attack_evaded=attack_evaded,
        goal_evaluated=goal_evaluated,
        tpr=tpr,
        fpr=fpr,
        recall=tpr,
        accuracy_preservation_rate=apr,
        baseline_preservation_rate=bpr,
        current_accuracy=current_accuracy,
        baseline_accuracy=baseline_accuracy,
        reference_accuracy=reference,
    )
