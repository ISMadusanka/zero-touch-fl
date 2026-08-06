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
    _resolve_eval_budget, _resolve_llm_defender,
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


# --- eval poison budget: settable up to n_clients, pool widened to match -----

class _FakeEnv:
    """Just the attributes the benchmark's budget/pool resolution touches."""

    def __init__(self, n_clients=20, n_compromisable=5):
        self.n_clients = n_clients
        self.n_compromisable = n_compromisable
        self.budget_cap = 1
        self.sample_budget = True
        self.sample_target = True

    def pool_ids(self):
        return list(range(self.n_compromisable))


# The real resolution used by the benchmark — not a re-implementation, so these
# tests cannot drift away from what actually runs.
_resolve_budget = _resolve_eval_budget


def test_eval_budget_can_exceed_the_trained_pool_up_to_n_clients():
    """The reported bug: --max-poison-clients was clamped to fl.n_compromisable (5),
    so asking for 10 silently behaved exactly like 5."""
    env = _FakeEnv(n_clients=20, n_compromisable=5)
    assert _resolve_budget(env, 10) == 10
    assert env.budget_cap == 10
    # Raising the cap is useless unless the POOL grows too: the attacker may only
    # choose from range(n_compromisable), and select_and_apply clamps to the pool size.
    assert env.n_compromisable == 10
    assert env.pool_ids() == list(range(10))
    assert env.sample_budget is False       # eval never randomises the budget


def test_eval_budget_reaches_every_client():
    env = _FakeEnv(n_clients=20, n_compromisable=5)
    assert _resolve_budget(env, 20) == 20
    assert env.n_compromisable == 20 and env.pool_ids() == list(range(20))


def test_eval_budget_is_capped_at_n_clients():
    env = _FakeEnv(n_clients=20, n_compromisable=5)
    assert _resolve_budget(env, 999) == 20     # cannot poison more clients than exist
    assert _resolve_budget(_FakeEnv(), 0) == 1  # ...and at least one
    assert _resolve_budget(_FakeEnv(), -5) == 1


def test_small_eval_budget_leaves_the_pool_alone():
    """Asking for fewer than the trained pool must NOT shrink it — the attacker still
    chooses which of its 5 insiders to use."""
    env = _FakeEnv(n_clients=20, n_compromisable=5)
    assert _resolve_budget(env, 2) == 2
    assert env.n_compromisable == 5 and env.budget_cap == 2


def test_attacker_can_actually_select_ten_of_twenty():
    """End-to-end on the parsing path: with a widened pool the agent really does return
    10 poisoned clients, rather than being truncated to the old pool of 5."""
    import torch
    from agents.attacker_agent import AttackerAgent

    env = _FakeEnv(n_clients=20, n_compromisable=5)
    budget = _resolve_budget(env, 10)
    pool = {cid: {"w": torch.ones(4)} for cid in env.pool_ids()}
    assert len(pool) == 10

    plan = {"clients": [{"id": cid, "operations": [{"op": "scale", "target": "all",
                                                    "factor": 1.5 + 0.1 * cid}]}
                        for cid in range(10)]}
    poisoned, chosen, n_malformed = AttackerAgent().select_and_apply(
        __import__("json").dumps(plan), pool, budget)
    assert chosen == list(range(10)), chosen
    assert len(poisoned) == 10 and n_malformed == 0
    # Every chosen client's weights really changed (not silently a no-op).
    assert all(not torch.equal(poisoned[c]["w"], pool[c]["w"]) for c in chosen)


def test_selection_is_still_truncated_to_the_budget():
    """Widening the pool must not let the attacker exceed the budget it was given."""
    import json

    import torch
    from agents.attacker_agent import AttackerAgent

    env = _FakeEnv(n_clients=20, n_compromisable=5)
    budget = _resolve_budget(env, 10)
    pool = {cid: {"w": torch.ones(3)} for cid in env.pool_ids()}
    plan = {"clients": [{"id": cid, "operations": [{"op": "scale", "target": "all",
                                                    "factor": 2.0}]}
                        for cid in range(10)]}
    plan["clients"] += [{"id": 0, "operations": [{"op": "scale", "target": "all",
                                                  "factor": 3.0}]}]      # duplicate
    _p, chosen, _m = AttackerAgent().select_and_apply(json.dumps(plan), pool, budget)
    assert len(chosen) == budget == 10
    assert len(set(chosen)) == len(chosen)          # deduped


def test_benchmark_sized_pool_underselection_is_filled_to_exact_budget():
    """The benchmark must not silently turn a quota of 10 into one poisoner."""
    import json

    import torch
    from agents.attacker_agent import AttackerAgent

    env = _FakeEnv(n_clients=20, n_compromisable=5)
    budget = _resolve_budget(env, 10)
    pool = {cid: {"w": torch.ones(3)} for cid in env.pool_ids()}
    one_client_plan = {"clients": [{"id": 7, "operations": [
        {"op": "scale", "target": "all", "factor": 2.0}]}]}
    poisoned, chosen, malformed = AttackerAgent().select_and_apply(
        json.dumps(one_client_plan), pool, budget)
    assert len(chosen) == len(poisoned) == budget == 10
    assert chosen[0] == 7 and len(set(chosen)) == budget
    assert malformed == 0


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
    assert "EVERY client is a poisoner" in _capture(20, 20)
    assert _capture(1, 0) == ""                           # degenerate, no crash


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} benchmark tests passed.")


if __name__ == "__main__":
    _run()
