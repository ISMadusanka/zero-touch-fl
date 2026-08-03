"""Torch-free tests for the benchmark metrics + report + panel resolution.

Runs anywhere:  python tests/test_benchmark.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import DetectionVerdict          # noqa: E402
from benchmark.metrics import DefenseMetrics      # noqa: E402
from benchmark import report                      # noqa: E402
from benchmark.run_benchmark import _resolve_llm_defender   # noqa: E402


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


def test_goal_success_rate():
    # baseline 0.8, target drop 0.1 -> goal threshold 0.7: acc <= 0.7 == attack succeeded.
    m = DefenseMetrics("d", baseline_accuracy=0.8, target_drop=0.1)
    m.record(0, _verdicts(set(), [0, 1]), {0}, accuracy=0.65)   # below threshold -> goal met
    m.record(1, _verdicts(set(), [0, 1]), {0}, accuracy=0.70)   # exactly at threshold -> met
    m.record(2, _verdicts(set(), [0, 1]), {0}, accuracy=0.75)   # above threshold -> not met
    s = m.summary()
    assert abs(s["goal_success_rate"] - (2 / 3)) < 1e-9
    assert abs(s["goal_threshold"] - 0.7) < 1e-9 and s["target_drop"] == 0.1
    # Evasion (atk_thru) is independent: nobody was flagged, so it's 1.0 every round.
    assert s["attack_success_rate"] == 1.0


def test_goal_success_rate_none_without_target():
    # No target_drop -> goal-success is n/a (None); the table must render it, not crash.
    m = DefenseMetrics("d", baseline_accuracy=0.8)
    m.record(0, _verdicts(set(), [0, 1]), {0}, accuracy=0.1)
    s = m.summary()
    assert s["goal_success_rate"] is None and s["goal_threshold"] is None
    assert "n/a" in report.format_table([s]) and "atk_succ" in report.format_table([s])


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

_PATHS = {"attacker": "checkpoints/attacker_adapter",
          "defender": "checkpoints/defender_adapter"}
_ALGORITHMIC = {"defense": {"mode": "algorithmic"}}
_FULL_PANEL = ["fedavg", "oracle", "llm_defender", "fltrust", "defl", "dnc", "multikrum"]


def test_missing_defender_adapter_skips_only_that_column():
    """The reported bug. `llm_defender` is in the DEFAULT --defenses list, and with
    `defense.mode: algorithmic` (the shipped config) the defender adapter is never
    trained — so a plain `run_benchmark` used to sys.exit before measuring anything,
    discarding all six other defenses because one optional column was unavailable."""
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
        assert "defense.mode: algorithmic" in msg
    else:
        raise AssertionError("expected SystemExit when no comparable defense remains")


def test_skip_reason_distinguishes_disabled_from_untrained():
    """`defense.mode: llm` means a defender WAS supposed to be trained, so the message
    should not blame the config."""
    import io
    import logging

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger = logging.getLogger("benchmark")
    logger.addHandler(handler)
    try:
        _resolve_llm_defender(_FULL_PANEL, _PATHS, {"defense": {"mode": "llm"}},
                             exists=lambda p: False)
        assert "no defender adapter has been trained yet" in buf.getvalue()
        buf.truncate(0), buf.seek(0)
        _resolve_llm_defender(_FULL_PANEL, _PATHS, _ALGORITHMIC, exists=lambda p: False)
        assert "defender LLM is disabled" in buf.getvalue()
    finally:
        logger.removeHandler(handler)


def test_missing_defense_config_defaults_to_algorithmic():
    """An older config with no `defense:` block must still resolve, not KeyError."""
    names, skipped = _resolve_llm_defender(_FULL_PANEL, _PATHS, {}, exists=lambda p: False)
    assert skipped is True and "llm_defender" not in names


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} benchmark tests passed.")


if __name__ == "__main__":
    _run()
