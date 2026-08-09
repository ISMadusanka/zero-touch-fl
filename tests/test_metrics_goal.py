"""Regression tests for the metrics layer's attack-success semantics.

Locks in two fixes, both reproduced from a real ``targeted_label`` run:

1. ``attack_success`` used to be ``fn > 0`` — pure evasion. Under a targeted
   goal that made every quiet-detector round a "successful attack", including
   rounds that left the model BETTER than baseline, and it disagreed with the
   schedule's own win-gate on the same round.
2. ``accuracy_preservation_rate`` divided by a ``baseline_accuracy`` pinned at
   the Phase-1 value for the whole run, so after an honest FL interlude moved
   the model the tracker and ``rl.env`` were quoting two different denominators
   for the same round.

Torch-free:
    python tests/test_metrics_goal.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import DetectionVerdict  # noqa: E402
from metrics.compute import compute_round_metrics  # noqa: E402
from metrics.tracker import MetricsTracker  # noqa: E402

BASELINE = 0.7823


def _verdicts(flagged: set, n_clients=20):
    return [DetectionVerdict(i, i in flagged, 1.0, "") for i in range(n_clients)]


# --- 1. goal vs. evasion -----------------------------------------------------

def test_evasion_alone_is_no_longer_reported_as_attack_success():
    """Round 47 of the logged run: the detector stayed quiet (fn=1) but the
    target class only lost 0.122 of the 0.300 it needed. Evasion yes, goal no."""
    m = compute_round_metrics(47, _verdicts(set(range(1, 6))), {0},
                              current_accuracy=0.7782, baseline_accuracy=BASELINE,
                              reference_accuracy=0.7823, attack_goal_met=False)
    assert m.fn == 1 and m.attack_evaded is True
    assert m.attack_success is False
    assert m.goal_evaluated is True


def test_a_round_that_improved_the_model_is_not_a_successful_attack():
    """Round 49: acc 0.7876 vs a 0.7823 baseline — apr > 1 — and it was logged
    as a successful attack purely because fn=1."""
    m = compute_round_metrics(49, _verdicts({1, 2}), {0},
                              current_accuracy=0.7876, baseline_accuracy=BASELINE,
                              reference_accuracy=0.7823, attack_goal_met=False)
    assert m.accuracy_preservation_rate > 1.0
    assert m.attack_evaded is True
    assert m.attack_success is False


def test_goal_met_is_reported_even_though_the_numbers_look_alike():
    """Round 48: same fn=1 as round 47, but the class was actually destroyed."""
    m = compute_round_metrics(48, _verdicts({1, 2, 3}), {0},
                              current_accuracy=0.7005, baseline_accuracy=BASELINE,
                              reference_accuracy=0.7823, attack_goal_met=True)
    assert m.attack_success is True and m.attack_evaded is True


def test_unjudged_rounds_fall_back_to_evasion_and_say_so():
    """A caller with no goal information must not silently claim to have judged
    the goal — that is exactly how the two definitions got confused."""
    m = compute_round_metrics(1, _verdicts(set()), {0},
                              current_accuracy=0.70, baseline_accuracy=BASELINE)
    assert m.goal_evaluated is False
    assert m.attack_success == m.attack_evaded is True


def test_caught_attacker_is_never_a_success():
    m = compute_round_metrics(68, _verdicts({0}), {0},
                              current_accuracy=0.7754, baseline_accuracy=BASELINE,
                              reference_accuracy=0.7782, attack_goal_met=False)
    assert m.tp == 1 and m.fn == 0
    assert m.attack_evaded is False and m.attack_success is False


# --- 2. the reference the round is actually scored against -------------------

def test_preservation_tracks_the_clean_reference_not_the_stale_baseline():
    """After the honest FL interlude the clean reference moved to 0.7782 while
    `baseline_accuracy` stayed pinned at the Phase-1 0.7823. Both are reported,
    and `apr` uses the one the round was actually scored against."""
    m = compute_round_metrics(66, _verdicts({1, 2}), {0},
                              current_accuracy=0.7416, baseline_accuracy=BASELINE,
                              reference_accuracy=0.7782, attack_goal_met=False)
    assert abs(m.accuracy_preservation_rate - 0.7416 / 0.7782) < 1e-9
    assert abs(m.baseline_preservation_rate - 0.7416 / BASELINE) < 1e-9
    assert m.reference_accuracy == 0.7782
    assert m.baseline_accuracy == BASELINE


def test_tracker_carries_the_reference_forward_between_rounds():
    out = tempfile.mkdtemp()
    try:
        t = MetricsTracker(BASELINE, output_dir=out)
        # Round 66 hands over the post-interlude reference...
        t.update(66, _verdicts({1}), 0.7416, {0},
                 reference_accuracy=0.7782, attack_goal_met=False)
        # ...round 67 omits it, and must NOT rewind to the Phase-1 baseline.
        m = t.update(67, _verdicts(set()), 0.7425, {0}, attack_goal_met=False)
        assert m.reference_accuracy == 0.7782

        agg = t.aggregate()
        assert agg.total_rounds == 2
        assert agg.goal_evaluated_rounds == 2
        assert agg.attack_success_rate == 0.0     # neither round met the goal
        assert agg.attack_evasion_rate == 1.0     # both slipped past the detector
        assert abs(agg.accuracy_preservation_rate - 0.7425 / 0.7782) < 1e-9
        assert abs(agg.baseline_preservation_rate - 0.7425 / BASELINE) < 1e-9
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_aggregate_separates_the_two_rates():
    """The headline number the run reports must be the goal, with evasion kept
    alongside it rather than standing in for it."""
    out = tempfile.mkdtemp()
    try:
        t = MetricsTracker(BASELINE, output_dir=out)
        for rn, met in ((1, False), (2, False), (3, True), (4, False)):
            t.update(rn, _verdicts(set()), 0.75, {0},
                     reference_accuracy=0.78, attack_goal_met=met)
        agg = t.aggregate()
        assert agg.attack_success_rate == 0.25    # 1 of 4 rounds hit the goal
        assert agg.attack_evasion_rate == 1.0     # all 4 evaded detection
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} metrics-goal tests passed.")


if __name__ == "__main__":
    _run()
