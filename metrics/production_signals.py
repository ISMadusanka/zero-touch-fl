"""Production-ready proxy signals for the defender agent.

All functions here compute feedback signals that require ZERO ground truth
about which clients are malicious.  They replace the oracle-dependent metrics
(TPR, FPR, attack_passed_through) that the simulation's MetricsTracker
provides for researcher evaluation.

The MetricsTracker still computes oracle metrics for research papers —
these functions exist so the defender agent can be trained/evaluated with
the same signals it would have in a real deployment.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Accuracy-based signals
# ------------------------------------------------------------------

def compute_accuracy_delta(
    current_accuracy: float,
    previous_accuracy: float,
) -> float:
    """Round-over-round accuracy change.

    Negative values suggest a poisoned update may have passed through.
    This is the primary production replacement for ``attack_passed_through``.
    """
    return current_accuracy - previous_accuracy


def compute_accuracy_trend(
    accuracy_history: Sequence[float],
    window: int = 5,
) -> float:
    """Linear slope of accuracy over the last *window* rounds.

    Positive = improving, negative = degrading, near-zero = stable.
    Uses least-squares linear regression on the window.
    """
    recent = list(accuracy_history[-window:])
    if len(recent) < 2:
        return 0.0
    x = np.arange(len(recent), dtype=np.float64)
    y = np.array(recent, dtype=np.float64)
    # slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    if den < 1e-12:
        return 0.0
    return float(num / den)


def compute_accuracy_volatility(
    accuracy_history: Sequence[float],
    window: int = 5,
) -> float:
    """Standard deviation of accuracy over the last *window* rounds.

    High volatility suggests the system is oscillating — the defender
    keeps switching strategies poorly or attacks are intermittent.
    """
    recent = list(accuracy_history[-window:])
    if len(recent) < 2:
        return 0.0
    return float(np.std(recent, ddof=1))


# ------------------------------------------------------------------
# Defender self-assessment signals
# ------------------------------------------------------------------

def compute_flag_rate(n_flagged: int, n_total: int) -> float:
    """Fraction of clients flagged this round.

    Combined with accuracy_delta this is a strong proxy for FPR:
      - High flag_rate + dropping accuracy → likely over-flagging (high FPR)
      - High flag_rate + stable accuracy → probably catching real attackers
      - Low flag_rate + dropping accuracy → too lenient (low TPR)
      - Low flag_rate + stable accuracy → all is well
    """
    if n_total <= 0:
        return 0.0
    return n_flagged / n_total


def compute_rounds_skipped(
    defender_history: Sequence[dict],
    window: int = 5,
) -> int:
    """Count how many of the last *window* rounds were skipped (all clients flagged).

    Persistent round-skipping means the defender is chronically too aggressive.
    """
    recent = list(defender_history[-window:])
    return sum(1 for entry in recent if entry.get("all_clients_flagged", False))


# ------------------------------------------------------------------
# Client-level temporal signals
# ------------------------------------------------------------------

def compute_client_flag_history(
    defender_history: Sequence[dict],
    window: int = 10,
) -> dict[int, int]:
    """Per-client flag count over the last *window* rounds.

    A client flagged in 8/10 rounds is far more likely to be malicious
    than one flagged 1/10.  The LLM can reason about persistent offenders.
    """
    recent = list(defender_history[-window:])
    counts: dict[int, int] = {}
    for entry in recent:
        for verdict in entry.get("verdicts", []):
            cid = verdict.get("client_id")
            if cid is None:
                continue
            if verdict.get("suspicious", False):
                counts[cid] = counts.get(cid, 0) + 1
            else:
                counts.setdefault(cid, 0)
    return counts
