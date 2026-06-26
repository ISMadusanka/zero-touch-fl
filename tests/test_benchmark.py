"""Torch-free tests for the benchmark metrics + report.

Runs anywhere:  python tests/test_benchmark.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import DetectionVerdict          # noqa: E402
from benchmark.metrics import DefenseMetrics      # noqa: E402
from benchmark import report                      # noqa: E402


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


def test_report_table_has_all_defenses():
    summaries = [
        DefenseMetrics("fedavg", 0.8).summary(),
        DefenseMetrics("fltrust", 0.8).summary(),
    ]
    table = report.format_table(summaries)
    assert "fedavg" in table and "fltrust" in table and "detect%" in table
    text = report.render(summaries, n_rounds=10, baseline_accuracy=0.8, out_dir=None)
    assert "DEFENSE BENCHMARK" in text and "Legend" in text


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


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} benchmark tests passed.")


if __name__ == "__main__":
    _run()
