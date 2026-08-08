"""Stateful tracker that accumulates per-round research metrics.

Now that a *random subset* of clients is poisoned each round, the ground-truth
malicious set is passed to ``update()`` per round (no longer a fixed set in the
constructor). The old ``get_windowed_metrics`` helper — which only existed to
feed the deleted adapt-when-caught/bypassed loop — has been removed. These
metrics are for researcher evaluation; the RL reward is computed separately in
``rl/rewards.py``.
"""

import json
import logging
import os
from collections import deque

from core.types import DetectionVerdict
from metrics.compute import (
    DEFAULT_TARGET_ACCURACY_DROP,
    compute_round_metrics,
    _safe_div,
)
from metrics.types import AggregateMetrics, RoundMetrics

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Accumulates round-level metrics and exposes aggregate statistics.

    Long runs (``fl.simulation_rounds`` is a very large budget) made the original
    design unbounded in two ways: one ``round_NNN.json`` file per round, and every
    ``RoundMetrics`` retained in memory and re-serialised in full into
    ``summary.json``. Both are now bounded:

    * every round is APPENDED to ``rounds.jsonl`` (one line per round) — O(1) per
      round, one file instead of millions;
    * the running confusion/accuracy totals are accumulated incrementally, and
      only the last ``keep_rounds`` ``RoundMetrics`` are held in memory for the
      ``per_round`` tail of ``summary.json``. ``aggregate()`` is computed from the
      running totals, so it still covers EVERY round.
    """

    def __init__(self, baseline_accuracy: float, output_dir: str = "logs/metrics",
                 keep_rounds: int = 2000):
        self.baseline_accuracy: float = float(baseline_accuracy)
        self.output_dir: str = output_dir
        self.keep_rounds: int = max(1, int(keep_rounds))
        # Bounded tail of recent rounds (for summary.json's `per_round` block).
        self.rounds: deque[RoundMetrics] = deque(maxlen=self.keep_rounds)
        self.jsonl_path: str = os.path.join(self.output_dir, "rounds.jsonl")

        # Running totals over ALL rounds (not just the retained tail).
        self._total_rounds = 0
        self._tp = self._fn = self._fp = self._tn = 0
        self._n_attack_successes = 0
        self._n_evasion_successes = 0
        self._final_accuracy = 0.0

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            "MetricsTracker initialized — "
            f"baseline_accuracy={self.baseline_accuracy:.4f}, output_dir={self.output_dir}, "
            f"keep_rounds={self.keep_rounds}"
        )

    # ------------------------------------------------------------------
    def update(
        self,
        round_num: int,
        verdicts: list[DetectionVerdict],
        current_accuracy: float,
        malicious_ids: set[int],
        *,
        reference_accuracy: float | None = None,
        target_accuracy_drop: float = DEFAULT_TARGET_ACCURACY_DROP,
    ) -> RoundMetrics:
        """Compute and store metrics for a single round. Returns them.

        ``malicious_ids`` is this round's ground-truth poisoned set.
        """
        metrics = compute_round_metrics(
            round_num=round_num,
            verdicts=verdicts,
            malicious_ids=set(malicious_ids),
            current_accuracy=current_accuracy,
            baseline_accuracy=self.baseline_accuracy,
            reference_accuracy=reference_accuracy,
            target_accuracy_drop=target_accuracy_drop,
        )
        self.rounds.append(metrics)
        self._total_rounds += 1
        self._tp += metrics.tp
        self._fn += metrics.fn
        self._fp += metrics.fp
        self._tn += metrics.tn
        self._n_attack_successes += int(metrics.attack_success)
        self._n_evasion_successes += int(metrics.evasion_success)
        self._final_accuracy = metrics.current_accuracy
        self._log_round(metrics)
        self._save_round(metrics)
        return metrics

    # ------------------------------------------------------------------
    def aggregate(self) -> AggregateMetrics:
        """Build the cumulative summary across ALL recorded rounds.

        Uses the running totals, not ``self.rounds`` (which only retains the last
        ``keep_rounds`` entries), so the summary stays correct on long runs.
        """
        total_rounds = self._total_rounds
        if total_rounds == 0:
            logger.warning("MetricsTracker.aggregate() called with no rounds recorded")
            return AggregateMetrics(
                total_rounds=0, tp=0, fn=0, fp=0, tn=0,
                attack_success_rate=0.0, evasion_success_rate=0.0,
                tpr=0.0, fpr=0.0, recall=0.0,
                accuracy_preservation_rate=0.0,
                baseline_accuracy=self.baseline_accuracy, final_accuracy=0.0,
            )

        tp, fn, fp, tn = self._tp, self._fn, self._fp, self._tn
        n_attack_successes = self._n_attack_successes
        n_evasion_successes = self._n_evasion_successes
        final_accuracy = self._final_accuracy

        return AggregateMetrics(
            total_rounds=total_rounds,
            tp=tp, fn=fn, fp=fp, tn=tn,
            attack_success_rate=_safe_div(n_attack_successes, total_rounds),
            evasion_success_rate=_safe_div(n_evasion_successes, total_rounds),
            tpr=_safe_div(tp, tp + fn),
            fpr=_safe_div(fp, fp + tn),
            recall=_safe_div(tp, tp + fn),
            accuracy_preservation_rate=_safe_div(final_accuracy, self.baseline_accuracy),
            baseline_accuracy=self.baseline_accuracy,
            final_accuracy=final_accuracy,
        )

    # ------------------------------------------------------------------
    def save_summary(self, path: str | None = None) -> str:
        """Write the aggregate summary to JSON and log a human-readable block.

        ``aggregate`` covers every round; ``per_round`` is the retained tail (the
        full history lives in ``rounds.jsonl``).
        """
        summary = self.aggregate()
        out_path = path or os.path.join(self.output_dir, "summary.json")
        payload = {
            "aggregate": summary.to_dict(),
            "per_round_is_tail": self._total_rounds > len(self.rounds),
            "per_round_path": self.jsonl_path,
            "per_round": [r.to_dict() for r in self.rounds],
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        self._log_summary(summary, out_path)
        return out_path

    # ------------------------------------------------------------------
    def _log_round(self, m: RoundMetrics) -> None:
        logger.info(
            "Metrics [round=%d] tp=%d fn=%d fp=%d tn=%d | "
            "attack_success=%s evasion_success=%s induced_drop=%+.4f target=%.4f "
            "tpr=%.3f fpr=%.3f apr=%.3f "
            "(acc=%.4f / reference=%.4f / baseline=%.4f)",
            m.round_num, m.tp, m.fn, m.fp, m.tn,
            m.attack_success, m.evasion_success, m.induced_drop,
            m.target_accuracy_drop, m.tpr, m.fpr, m.accuracy_preservation_rate,
            m.current_accuracy, m.reference_accuracy, m.baseline_accuracy,
        )

    def _log_summary(self, agg: AggregateMetrics, out_path: str) -> None:
        logger.info("=" * 60)
        logger.info("AGGREGATE METRICS (over %d round(s))", agg.total_rounds)
        logger.info("  Confusion: TP=%d FN=%d FP=%d TN=%d", agg.tp, agg.fn, agg.fp, agg.tn)
        logger.info("  Attack Goal Success Rate:      %.4f", agg.attack_success_rate)
        logger.info("  Detection Evasion Rate:        %.4f", agg.evasion_success_rate)
        logger.info("  True Positive Rate (TPR):      %.4f", agg.tpr)
        logger.info("  False Positive Rate (FPR):     %.4f", agg.fpr)
        logger.info("  Accuracy Preservation Rate:    %.4f (final=%.4f / baseline=%.4f)",
                    agg.accuracy_preservation_rate, agg.final_accuracy, agg.baseline_accuracy)
        logger.info("  Summary saved to %s", out_path)
        logger.info("=" * 60)

    def _save_round(self, m: RoundMetrics) -> None:
        """Append one round to ``rounds.jsonl`` (one JSON object per line)."""
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(m.to_dict()) + "\n")
        logger.debug("Round metrics appended to %s", self.jsonl_path)
