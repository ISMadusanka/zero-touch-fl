"""Torch-free tests for the benchmark metrics + report + panel resolution.

Runs anywhere:  python tests/test_benchmark.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import DetectionVerdict          # noqa: E402
from benchmark.metrics import DefenseMetrics      # noqa: E402
from benchmark import report                      # noqa: E402
from benchmark.run_benchmark import (                       # noqa: E402
    _resolve_llm_defender, resolve_poison_clients,
)


def _verdicts(flagged_ids, all_ids):
    return [DetectionVerdict(c, c in flagged_ids, 1.0, "") for c in all_ids]


def test_perfect_detection_metrics():
    m = DefenseMetrics("oracle", baseline_accuracy=0.8)
    # 3 rounds, client 0 poisoned each round, flagged exactly.
    for r in range(3):
        m.record(r, _verdicts({0}, [0, 1, 2, 3, 4]), {0}, accuracy=0.8)
    s = m.summary()
    assert s["detection_rate"] == 1.0 and s["recall"] == 1.0
    assert s["fpr"] == 0.0 and s["precision"] == 1.0 and s["f1"] == 1.0
    assert s["attack_success_rate"] == 0.0
    assert s["false_alarms"] == 0
    assert abs(s["mean_acc_drop"] - 0.0) < 1e-9


def test_no_defense_metrics():
    m = DefenseMetrics("fedavg", baseline_accuracy=0.8)
    # flags nobody -> never catches the poisoned client -> attack always succeeds.
    for r in range(4):
        m.record(r, _verdicts(set(), [0, 1, 2, 3, 4]), {1}, accuracy=0.6)
    s = m.summary()
    assert s["detection_rate"] == 0.0
    assert s["attack_success_rate"] == 1.0      # poisoned client slips through every round
    assert abs(s["mean_acc_drop"] - 0.2) < 1e-9  # 0.8 baseline - 0.6 mean
    assert s["final_accuracy"] == 0.6


def test_partial_detection_and_false_positives():
    m = DefenseMetrics("d", baseline_accuracy=1.0)
    # round 0: catches poisoned 0 (tp); round 1: misses poisoned 0 (fn) but flags honest 2 (fp)
    m.record(0, _verdicts({0}, [0, 1, 2]), {0}, 0.9)
    m.record(1, _verdicts({2}, [0, 1, 2]), {0}, 0.5)
    s = m.summary()
    # tp=1, fn=1 -> recall 0.5 ; fp=1, tn(=honest not flagged): r0 honest{1,2} not flagged=2,
    # r1 honest{1,2}: 1 not flagged, 2 flagged -> tn r0=2,r1=1 =3; fp=1 -> fpr=1/4
    assert abs(s["recall"] - 0.5) < 1e-9
    assert s["false_alarms"] == 1
    assert abs(s["fpr"] - (1 / 4)) < 1e-9
    assert s["attack_success_rate"] == 0.5      # 1 of 2 rounds had a miss


def test_goal_success_rate_is_weighted_by_how_much_of_the_drop_landed():
    # baseline 0.8, target drop 0.1. atk_succ is WEIGHTED: each round scores
    # min(1, acc_drop / 0.1), so a half-strength attack is 50% successful — it is not
    # written off as a failure just because it fell short of the full target.
    m = DefenseMetrics("d", baseline_accuracy=0.8, target_drop=0.1)
    m.record(0, _verdicts(set(), [0, 1]), {0}, accuracy=0.65)   # drop 0.15 -> capped 1.0
    m.record(1, _verdicts(set(), [0, 1]), {0}, accuracy=0.70)   # drop 0.10 -> exactly 1.0
    m.record(2, _verdicts(set(), [0, 1]), {0}, accuracy=0.75)   # drop 0.05 -> 0.5
    s = m.summary()
    assert abs(s["goal_success_rate"] - (1.0 + 1.0 + 0.5) / 3) < 1e-9
    # The all-or-nothing view is still available (2 of 3 rounds hit the full target).
    assert abs(s["goal_full_success_rate"] - (2 / 3)) < 1e-9
    assert abs(s["goal_threshold"] - 0.7) < 1e-9 and s["target_drop"] == 0.1
    # Per-round weights are recorded too, so the history shows the partial credit.
    assert [round(h["goal_success"], 6) for h in m.history] == [1.0, 1.0, 0.5]
    assert [h["goal_hit"] for h in m.history] == [True, True, False]
    # Evasion (atk_thru) is independent: nobody was flagged, so it's 1.0 every round.
    assert s["attack_success_rate"] == 1.0


def test_goal_success_partial_credit_edges():
    # A drop of 0.02 against a 0.1 target is 20% success; a round that leaves the model
    # no worse (or better) than the clean baseline scores 0, never negative.
    m = DefenseMetrics("d", baseline_accuracy=0.8, target_drop=0.1)
    assert abs(m.goal_score(0.78) - 0.2) < 1e-9
    assert m.goal_score(0.80) == 0.0
    assert m.goal_score(0.85) == 0.0        # the attack HELPED -> not negative success
    assert m.goal_score(0.20) == 1.0        # gross overshoot stays capped at 100%
    # Averaged: 20% + 0% over two rounds -> 10%.
    m.record(0, _verdicts(set(), [0, 1]), {0}, accuracy=0.78)
    m.record(1, _verdicts(set(), [0, 1]), {0}, accuracy=0.85)
    s = m.summary()
    assert abs(s["goal_success_rate"] - 0.1) < 1e-9
    assert s["goal_full_success_rate"] == 0.0


def test_goal_success_rate_none_without_target():
    # No target_drop -> goal-success is n/a (None); the table must render it, not crash.
    m = DefenseMetrics("d", baseline_accuracy=0.8)
    m.record(0, _verdicts(set(), [0, 1]), {0}, accuracy=0.1)
    s = m.summary()
    assert s["goal_success_rate"] is None and s["goal_threshold"] is None
    assert s["goal_full_success_rate"] is None
    assert m.history[0]["goal_success"] is None and m.history[0]["goal_hit"] is False
    assert "n/a" in report.format_table([s]) and "atk_succ" in report.format_table([s])


def test_report_legend_describes_the_weighted_metric():
    s = [DefenseMetrics("fedavg", 0.8, target_drop=0.1).summary()]
    text = report.render(s, n_rounds=10, baseline_accuracy=0.8,
                         goal={"type": "untargeted_degrade", "target_accuracy_drop": 0.1})
    assert "WEIGHTED attack success" in text
    assert "0.100" in text and "0.700" in text      # the target and the full-credit acc


def test_report_table_has_all_defenses():
    summaries = [
        DefenseMetrics("fedavg", 0.8).summary(),
        DefenseMetrics("fltrust", 0.8).summary(),
    ]
    table = report.format_table(summaries)
    assert "fedavg" in table and "fltrust" in table and "detect%" in table
    text = report.render(summaries, n_rounds=10, baseline_accuracy=0.8, out_dir=None)
    assert "DEFENSE BENCHMARK" in text and "Legend" in text


def test_report_header_shows_poisoner_count():
    s = [DefenseMetrics("fedavg", 0.8).summary()]
    text = report.render(s, n_rounds=10, baseline_accuracy=0.8, n_poisoners=3,
                         goal={"type": "untargeted_degrade", "target_accuracy_drop": 0.1})
    assert "Num of poisoners=3" in text
    # The '=' bar under the title spans the whole (now longer) title line.
    lines = text.splitlines()
    title = next(ln for ln in lines if ln.startswith("DEFENSE BENCHMARK"))
    assert set(lines[lines.index(title) + 1]) == {"="} and len(lines[lines.index(title) + 1]) == len(title)
    # Omitted entirely when not provided (backward compatible).
    assert "Num of poisoners" not in report.render(s, n_rounds=10, baseline_accuracy=0.8)


def test_rolling_rate():
    from benchmark.plot import _rolling_rate
    # window=2 over tp=[1,0,1], den=[1,1,1]: r0=1/1, r1=1/2, r2=1/2
    assert _rolling_rate([1, 0, 1], [1, 1, 1], window=2) == [1.0, 0.5, 0.5]
    # zero denominator -> NaN (x != x)
    r = _rolling_rate([0], [0], window=1)[0]
    assert r != r


def test_plot_skips_gracefully_without_matplotlib():
    # On a box without matplotlib, plot_history must warn + return None (not crash).
    from benchmark.plot import plot_history
    hist = {"fedavg": [{"round": 1, "tp": 0, "fn": 1, "fp": 0, "tn": 4, "accuracy": 0.8}]}
    try:
        import matplotlib  # noqa: F401
        return  # matplotlib present here -> nothing to assert about the skip path
    except Exception:
        assert plot_history(hist, 0.8, "logs/benchmark/benchmark.png") is None


# --- panel resolution: a missing defender adapter must not kill the run ------

_PATHS = {"defender": "checkpoints/defender_adapter"}
_ALGORITHMIC = {"defense": {"mode": "algorithmic"}}
_FULL_PANEL = ["fedavg", "oracle", "llm_defender", "fltrust", "defl", "dnc", "multikrum"]


def test_missing_defender_adapter_skips_only_that_column():
    """`llm_defender` is in the DEFAULT --defenses list and needs a trained defender
    adapter, which a box that has never run training does not have. A hard exit there
    used to discard all six other defenses because one optional column was
    unavailable."""
    names, skipped = _resolve_llm_defender(
        _FULL_PANEL, _PATHS, _ALGORITHMIC, exists=lambda p: False)
    assert skipped is True
    assert names == ["fedavg", "oracle", "fltrust", "defl", "dnc", "multikrum"]
    assert "llm_defender" not in names
    # Order of the surviving columns is preserved (the report renders in panel order).
    assert names == [n for n in _FULL_PANEL if n != "llm_defender"]


def test_present_defender_adapter_keeps_the_column():
    names, skipped = _resolve_llm_defender(
        _FULL_PANEL, _PATHS, _ALGORITHMIC, exists=lambda p: True)
    assert skipped is False and names == _FULL_PANEL


def test_panel_without_llm_defender_is_untouched():
    """No adapter lookup should even happen when the column was not requested."""
    panel = ["fedavg", "fltrust", "dnc"]

    def _boom(path):
        raise AssertionError("should not probe the defender adapter")

    names, skipped = _resolve_llm_defender(panel, _PATHS, _ALGORITHMIC, exists=_boom)
    assert names == panel and skipped is False


def test_llm_defender_alone_is_a_clear_error_not_a_silent_fedavg_run():
    """If it was the ONLY defense asked for there is nothing to compare against, so
    failing is right — but the message must name the flag that fixes it. `fedavg` is
    force-added as the no-defense reference and does not count as a comparison."""
    try:
        _resolve_llm_defender(["fedavg", "llm_defender"], _PATHS, _ALGORITHMIC,
                             exists=lambda p: False)
    except SystemExit as e:
        msg = str(e)
        assert "--defenses" in msg and "--defender-adapter" in msg
        assert "no defender adapter has been trained yet" in msg
    else:
        raise AssertionError("expected SystemExit when no comparable defense remains")


def test_skip_warning_names_the_path_and_the_fix():
    """The warning has to be actionable: which checkpoint was missing, and how to
    get the column back."""
    import io
    import logging

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger = logging.getLogger("benchmark")
    logger.addHandler(handler)
    try:
        _resolve_llm_defender(_FULL_PANEL, _PATHS, {"defense": {"mode": "llm"}},
                             exists=lambda p: False)
        out = buf.getvalue()
        assert _PATHS["defender"] in out
        assert "python main.py" in out and "--defender-adapter" in out
    finally:
        logger.removeHandler(handler)


def test_missing_defense_config_still_resolves():
    """A config with no `defense:` block must still resolve, not KeyError."""
    names, skipped = _resolve_llm_defender(_FULL_PANEL, _PATHS, {}, exists=lambda p: False)
    assert skipped is True and "llm_defender" not in names


# --- which clients flip labels in an evaluation ------------------------------
# The real resolution used by the benchmark — not a re-implementation, so these
# tests cannot drift away from what actually runs.

def test_poison_clients_default_to_the_attack_config():
    assert resolve_poison_clients({"poison_client_ids": [0, 3]}, None, 20) == [0, 3]
    assert resolve_poison_clients({}, None, 20) == [0]
    # A bare int is accepted as a one-element set.
    assert resolve_poison_clients({"poison_client_ids": 2}, None, 20) == [2]


def test_cli_overrides_the_configured_poison_set():
    assert resolve_poison_clients({"poison_client_ids": [0]}, "0,1,2", 20) == [0, 1, 2]
    assert resolve_poison_clients({"poison_client_ids": [0]}, " 4 , 2 ", 20) == [2, 4]


def test_evaluation_can_poison_every_client():
    """The benchmark's ceiling is fl.n_clients, so the standard 'attack success vs
    fraction malicious' sweep is expressible — that is what evaluation is for."""
    ids = resolve_poison_clients({}, ",".join(str(i) for i in range(20)), 20)
    assert ids == list(range(20))


def test_out_of_range_ids_are_dropped_not_clamped():
    """Clamping [0, 25] would silently produce two attacks on whichever id the clamp
    landed on, which is not what the config asked for."""
    assert resolve_poison_clients({}, "0,25,1,1,-3", 20) == [0, 1]
    try:
        resolve_poison_clients({}, "99", 20)
    except SystemExit:
        pass
    else:
        raise AssertionError("an entirely invalid poison set must abort")


def test_honest_majority_warnings_fire_at_the_right_thresholds():
    import io
    import logging

    from benchmark.run_benchmark import _warn_about_adversary_share

    def _capture(budget, n):
        buf = io.StringIO()
        h = logging.StreamHandler(buf)
        log = logging.getLogger("bench-share-test")
        log.setLevel(logging.INFO)
        log.addHandler(h)
        try:
            _warn_about_adversary_share(budget, n, log)
            return buf.getvalue()
        finally:
            log.removeHandler(h)

    assert _capture(5, 20) == ""                          # 25%: nothing to say
    assert "1/3 point" in _capture(8, 20)                 # 40%: informational
    assert "NO honest majority" in _capture(10, 20)       # 50%: guarantees void
    assert "EVERY client flips labels" in _capture(20, 20)
    assert _capture(1, 0) == ""                           # degenerate, no crash


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} benchmark tests passed.")


if __name__ == "__main__":
    _run()
