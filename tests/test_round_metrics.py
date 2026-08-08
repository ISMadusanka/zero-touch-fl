"""``attack_success`` must mean the attack SUCCEEDED, not that it evaded detection.

``metrics.compute.compute_round_metrics`` defined ``attack_success = fn > 0`` — "at
least one poisoned client was not flagged". That is evasion, and it is decoupled from
damage, which produced flatly contradictory records in a real run:

    round 47: accuracy 0.7617 -> 0.7711 (UP 1pp)    attack_success=True
    round 48: accuracy 0.7711 -> 0.7223 (DOWN 5pp)  attack_success=False

so the headline ASR measured stealth while claiming to measure success. The same
function's ``accuracy_preservation_rate`` divides by the FIXED phase-1 baseline, which
barely moves whatever the attack does (0.947-1.002 across the whole recorded sample).

Run:  python tests/test_round_metrics.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import DetectionVerdict  # noqa: E402
from metrics.compute import compute_round_metrics  # noqa: E402
from metrics.tracker import MetricsTracker  # noqa: E402
from rl.switch import SwitchConfig, success_drop_bar  # noqa: E402

BASELINE = 0.7823
GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.02}


def _verdicts(flagged, n=20):
    """``n`` clients; ids in ``flagged`` are marked suspicious."""
    return [DetectionVerdict(i, i in flagged, 1.0, "",
                            p_malicious=1.0 if i in flagged else 0.0)
            for i in range(n)]


def _round(*, poisoned, flagged, accuracy, clean=None, bar=0.012):
    return compute_round_metrics(
        round_num=1, verdicts=_verdicts(flagged), malicious_ids=set(poisoned),
        current_accuracy=accuracy, baseline_accuracy=BASELINE,
        clean_accuracy=clean, success_drop=bar,
    )


# --- success is damage, evasion is evasion -----------------------------------

def test_evasion_without_damage_is_not_success():
    """Recorded round 47: one poisoned client slipped through and accuracy went UP."""
    m = _round(poisoned={0}, flagged=set(), accuracy=0.7711, clean=0.7693)
    assert m.fn == 1 and m.evaded is True          # it did evade...
    assert m.induced_drop < 0                      # ...and IMPROVED the model
    assert m.attack_success is False               # so it did not succeed


def test_damage_while_detected_is_still_not_success():
    """The mirror case (recorded round 48): the poisoned client was caught, so the
    damage came from somewhere else. Evasion is False and success is False, but for
    different reasons — and the two fields now say so separately."""
    m = _round(poisoned={0}, flagged={0}, accuracy=0.7223, clean=0.7223)
    assert m.tp == 1 and m.evaded is False
    assert m.induced_drop == 0.0
    assert m.attack_success is False


def test_success_requires_clearing_the_damage_bar():
    below = _round(poisoned={0}, flagged=set(), accuracy=0.79, clean=0.80, bar=0.012)
    at = _round(poisoned={0}, flagged=set(), accuracy=0.788, clean=0.80, bar=0.012)
    above = _round(poisoned={0}, flagged=set(), accuracy=0.77, clean=0.80, bar=0.012)
    assert abs(below.induced_drop - 0.01) < 1e-9 and below.attack_success is False
    assert abs(at.induced_drop - 0.012) < 1e-9 and at.attack_success is True
    assert above.attack_success is True


def test_unmeasurable_round_is_never_recorded_as_a_success():
    """No counterfactual -> no damage measurement -> not a success, and not a
    measured zero either (the damage fields are None, so callers can filter)."""
    m = _round(poisoned={0}, flagged=set(), accuracy=0.7684, clean=None)
    assert m.induced_drop is None
    assert m.accuracy_preservation_vs_clean is None
    assert m.attack_success is False
    assert m.evaded is True                        # detection still measured fine


def test_no_bar_supplied_is_not_a_guess():
    m = compute_round_metrics(1, _verdicts(set()), {0}, 0.70, BASELINE,
                              clean_accuracy=0.80, success_drop=None)
    assert m.induced_drop is not None              # damage is still reported...
    assert m.attack_success is False               # ...but success is not invented


def test_the_metric_bar_is_the_schedule_bar():
    """The tracker's ``attack_success`` and the phase gate must not drift apart."""
    cfg = SwitchConfig(win_fraction=0.6)
    assert abs(success_drop_bar(GOAL, cfg) - 0.012) < 1e-12
    # No goal -> the absolute fallback the win gate also uses.
    assert success_drop_bar(None, cfg) == cfg.attacker_min_drop


# --- the counterfactual-relative rate ----------------------------------------

def test_clean_relative_preservation_moves_when_the_baseline_rate_does_not():
    """apr vs the fixed phase-1 baseline hid the attack; apr vs the clean
    counterfactual shows it."""
    m = _round(poisoned={0}, flagged=set(), accuracy=0.75, clean=0.80)
    assert abs(m.accuracy_preservation_rate - 0.75 / BASELINE) < 1e-12   # ~0.959
    assert abs(m.accuracy_preservation_vs_clean - 0.75 / 0.80) < 1e-12   # 0.9375
    assert m.accuracy_preservation_vs_clean < m.accuracy_preservation_rate


# --- aggregation --------------------------------------------------------------

def test_aggregate_separates_success_from_evasion_and_skips_unmeasured_rounds():
    out = tempfile.mkdtemp()
    try:
        t = MetricsTracker(BASELINE, output_dir=out)
        # 1) evaded, no damage       -> evasion only
        t.update(1, _verdicts(set()), 0.80, {0}, clean_accuracy=0.80, success_drop=0.012)
        # 2) evaded and damaging     -> both
        t.update(2, _verdicts(set()), 0.78, {0}, clean_accuracy=0.80, success_drop=0.012)
        # 3) caught, no damage       -> neither
        t.update(3, _verdicts({0}), 0.80, {0}, clean_accuracy=0.80, success_drop=0.012)
        # 4) unmeasurable            -> neither, and excluded from the damage mean
        t.update(4, _verdicts(set()), 0.80, {0}, clean_accuracy=None, success_drop=0.012)

        agg = t.aggregate()
        assert agg.total_rounds == 4
        assert abs(agg.attack_success_rate - 0.25) < 1e-12      # only round 2
        assert abs(agg.evasion_rate - 0.75) < 1e-12             # rounds 1, 2, 4
        assert agg.measured_rounds == 3                         # round 4 excluded
        # mean over rounds 1-3 only: (0.00 + 0.02 + 0.00) / 3
        assert abs(agg.mean_induced_drop - 0.02 / 3) < 1e-12
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_aggregate_on_an_empty_tracker_is_well_formed():
    out = tempfile.mkdtemp()
    try:
        agg = MetricsTracker(BASELINE, output_dir=out).aggregate()
        assert agg.total_rounds == 0 and agg.measured_rounds == 0
        assert agg.attack_success_rate == 0.0 and agg.evasion_rate == 0.0
        assert "evasion_rate" in agg.to_dict()
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_round_metrics_serialize_the_new_fields():
    d = _round(poisoned={0}, flagged=set(), accuracy=0.75, clean=0.80).to_dict()
    for key in ("attack_success", "evaded", "clean_accuracy", "induced_drop",
                "success_drop", "accuracy_preservation_vs_clean"):
        assert key in d, key


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} round-metrics tests passed.")


if __name__ == "__main__":
    _run()
