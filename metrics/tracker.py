"""Stateful tracker that accumulates per-round metrics into an aggregate."""

import json
import logging
import os
from typing import Iterable

from core.types import DetectionVerdict
from metrics.compute import compute_round_metrics, _safe_div
from metrics.types import AggregateMetrics, RoundMetrics

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Accumulates round-level metrics and exposes aggregate statistics.

    Typical usage:
        tracker = MetricsTracker(
            malicious_ids={0},
            baseline_accuracy=0.95,
            output_dir="logs/metrics",
        )
        for round_num in ...:
            ...
            tracker.update(round_num, verdicts, current_accuracy)
        tracker.save_summary()
    """

    def __init__(
        self,
        malicious_ids: Iterable[int],
        baseline_accuracy: float,
        output_dir: str = "logs/metrics",
    ):
        self.malicious_ids: set[int] = set(malicious_ids)
        self.baseline_accuracy: float = float(baseline_accuracy)
        self.output_dir: str = output_dir
        self.rounds: list[RoundMetrics] = []

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            "MetricsTracker initialized — "
            f"malicious_ids={sorted(self.malicious_ids)}, "
            f"baseline_accuracy={self.baseline_accuracy:.4f}, "
            f"output_dir={self.output_dir}"
        )

    # ------------------------------------------------------------------
    # Recording rounds
    # ------------------------------------------------------------------

    def update(
        self,
        round_num: int,
        verdicts: list[DetectionVerdict],
        current_accuracy: float,
    ) -> RoundMetrics:
        """Compute and store metrics for a single round. Returns them."""
        metrics = compute_round_metrics(
            round_num=round_num,
            verdicts=verdicts,
            malicious_ids=self.malicious_ids,
            current_accuracy=current_accuracy,
            baseline_accuracy=self.baseline_accuracy,
        )
        self.rounds.append(metrics)
        self._log_round(metrics)
        self._save_round(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> AggregateMetrics:
        """Build the cumulative summary across all recorded rounds."""
        total_rounds = len(self.rounds)
        if total_rounds == 0:
            logger.warning("MetricsTracker.aggregate() called with no rounds recorded")
            return AggregateMetrics(
                total_rounds=0,
                tp=0, fn=0, fp=0, tn=0,
                attack_success_rate=0.0,
                tpr=0.0, fpr=0.0, recall=0.0,
                accuracy_preservation_rate=0.0,
                baseline_accuracy=self.baseline_accuracy,
                final_accuracy=0.0,
            )

        tp = sum(r.tp for r in self.rounds)
        fn = sum(r.fn for r in self.rounds)
        fp = sum(r.fp for r in self.rounds)
        tn = sum(r.tn for r in self.rounds)
        n_attack_successes = sum(1 for r in self.rounds if r.attack_success)
        final_accuracy = self.rounds[-1].current_accuracy

        tpr = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn)
        apr = _safe_div(final_accuracy, self.baseline_accuracy)
        asr = _safe_div(n_attack_successes, total_rounds)

        return AggregateMetrics(
            total_rounds=total_rounds,
            tp=tp, fn=fn, fp=fp, tn=tn,
            attack_success_rate=asr,
            tpr=tpr, fpr=fpr, recall=tpr,
            accuracy_preservation_rate=apr,
            baseline_accuracy=self.baseline_accuracy,
            final_accuracy=final_accuracy,
        )

    # ------------------------------------------------------------------
    # Windowed metrics for agent feedback
    # ------------------------------------------------------------------

    def get_windowed_metrics(self, window: int = 5) -> dict:
        """Compute ASR, TPR, FPR, APR over the last `window` rounds.

        Returns a dict suitable for injecting into agent contexts.
        With a single attacker per round, per-round TPR is binary (0 or 1),
        so a trailing window gives the LLM a meaningful trajectory.
        """
        recent = self.rounds[-window:] if self.rounds else []
        if not recent:
            return {
                "attack_success_rate": 0.0,
                "tpr": 0.0,
                "fpr": 0.0,
                "accuracy_preservation_rate": 0.0,
                "window_size": 0,
            }

        tp = sum(r.tp for r in recent)
        fn = sum(r.fn for r in recent)
        fp = sum(r.fp for r in recent)
        tn = sum(r.tn for r in recent)
        n_successes = sum(1 for r in recent if r.attack_success)

        return {
            "attack_success_rate": _safe_div(n_successes, len(recent)),
            "tpr": _safe_div(tp, tp + fn),
            "fpr": _safe_div(fp, fp + tn),
            "accuracy_preservation_rate": _safe_div(
                recent[-1].current_accuracy, self.baseline_accuracy
            ),
            "window_size": len(recent),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_summary(self, path: str | None = None) -> str:
        """Write the aggregate summary to JSON and log a human-readable block."""
        summary = self.aggregate()
        out_path = path or os.path.join(self.output_dir, "summary.json")
        payload = {
            "aggregate": summary.to_dict(),
            "per_round": [r.to_dict() for r in self.rounds],
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        self._log_summary(summary, out_path)
        return out_path

    # ------------------------------------------------------------------
    # Internal logging helpers
    # ------------------------------------------------------------------

    def _log_round(self, m: RoundMetrics) -> None:
        logger.info(
            "Metrics [round=%d] tp=%d fn=%d fp=%d tn=%d | "
            "attack_success=%s tpr=%.3f fpr=%.3f recall=%.3f apr=%.3f "
            "(acc=%.4f / baseline=%.4f)",
            m.round_num, m.tp, m.fn, m.fp, m.tn,
            m.attack_success, m.tpr, m.fpr, m.recall, m.accuracy_preservation_rate,
            m.current_accuracy, m.baseline_accuracy,
        )

    def _log_summary(self, agg: AggregateMetrics, out_path: str) -> None:
        logger.info("=" * 60)
        logger.info("AGGREGATE METRICS (over %d round(s))", agg.total_rounds)
        logger.info("  Confusion: TP=%d FN=%d FP=%d TN=%d",
                    agg.tp, agg.fn, agg.fp, agg.tn)
        logger.info("  Attack Success Rate (ASR):     %.4f", agg.attack_success_rate)
        logger.info("  True Positive Rate (TPR):      %.4f", agg.tpr)
        logger.info("  False Positive Rate (FPR):     %.4f", agg.fpr)
        logger.info("  Recall:                        %.4f", agg.recall)
        logger.info("  Accuracy Preservation Rate:    %.4f "
                    "(final=%.4f / baseline=%.4f)",
                    agg.accuracy_preservation_rate,
                    agg.final_accuracy, agg.baseline_accuracy)
        logger.info("  Summary saved to %s", out_path)
        logger.info("=" * 60)

    def _save_round(self, m: RoundMetrics) -> None:
        """Persist a single round's metrics for downstream analysis."""
        path = os.path.join(self.output_dir, f"round_{m.round_num:03d}.json")
        with open(path, "w") as f:
            json.dump(m.to_dict(), f, indent=2)
        logger.debug("Round metrics saved to %s", path)
