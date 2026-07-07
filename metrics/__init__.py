"""Evaluation metrics for the attack/defend simulation.

Exposes:
  - RoundMetrics, AggregateMetrics  (data containers)
  - compute_round_metrics           (pure function)
  - MetricsTracker                  (stateful accumulator with logging)
"""

from metrics.types import RoundMetrics, AggregateMetrics
from metrics.compute import compute_round_metrics
from metrics.tracker import MetricsTracker

__all__ = [
    "RoundMetrics",
    "AggregateMetrics",
    "compute_round_metrics",
    "MetricsTracker",
]
