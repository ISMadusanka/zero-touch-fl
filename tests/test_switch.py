"""Unit tests for the success-gated best-response controller (rl/switch.py).

Torch-free, so it runs anywhere (no GPU needed):  python tests/test_switch.py
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.switch import (  # noqa: E402
    PhaseController, SwitchConfig, attacker_succeeded, committed_success, defender_succeeded,
)


@dataclass
class V:
    """Minimal DetectionVerdict stand-in."""
    client_id: int
    is_suspicious: bool


def _cfg(**kw):
    base = dict(min_phase_rounds=3, max_phase_rounds=10, success_streak=2,
                attacker_min_drop=0.02, attacker_min_evaded=1.0,
                defender_min_tpr=0.99, defender_max_fpr=0.10)
    base.update(kw)
    return SwitchConfig(**base)


def test_attacker_success_predicate():
    cfg = _cfg()
    # Poisoned client 0 evades (not flagged) AND drop above threshold -> win.
    assert attacker_succeeded(0.05, [V(0, False), V(1, True)], [0], cfg)
    # Evades but no damage -> not a win.
    assert not attacker_succeeded(0.0, [V(0, False)], [0], cfg)
    # Caught -> not a win even if (impossibly) some drop reported.
    assert not attacker_succeeded(0.05, [V(0, True)], [0], cfg)
    # No poisoned clients -> never a win.
    assert not attacker_succeeded(0.5, [V(0, False)], [], cfg)


def test_defender_success_predicate():
    cfg = _cfg()
    # Catches the 1 poisoned client, no false positives -> win.
    assert defender_succeeded([V(0, True), V(1, False), V(2, False)], [0], cfg)
    # Misses the poisoned client -> TPR 0 -> not a win.
    assert not defender_succeeded([V(0, False), V(1, False)], [0], cfg)
    # Catches it but over-flags an honest client (FPR 0.5 > 0.10) -> not a win.
    assert not defender_succeeded([V(0, True), V(1, True), V(2, False)], [0], cfg)


def test_defender_wins_a_clean_round_by_staying_quiet():
    """The frozen attacker can produce a round with NO effective poison (every
    selected client's plan was a no-op). TPR is undefined there; treating it as 0
    made a flawless round a loss and could stall the defender's phase whenever its
    opponent degenerated. Staying quiet is the win condition on a clean round."""
    cfg = _cfg()
    assert defender_succeeded([V(0, False), V(1, False), V(2, False)], [], cfg)
    # ...but over-flagging honest clients is still a loss.
    assert not defender_succeeded([V(0, True), V(1, True), V(2, False)], [], cfg)


def test_committed_success_dispatch():
    cfg = _cfg()
    assert committed_success("attacker", 0.05, [V(0, False)], [0], cfg)
    assert committed_success("defender", 0.0, [V(0, True), V(1, False)], [0], cfg)


def test_attacker_relative_win_gate():
    """With a per-round goal, the damage bar is win_fraction * target_accuracy_drop."""
    cfg = _cfg(win_fraction=0.6)
    small = {"type": "untargeted_degrade", "target_accuracy_drop": 0.05}   # bar = 0.03
    big = {"type": "untargeted_degrade", "target_accuracy_drop": 0.30}     # bar = 0.18
    evaded = [V(0, False)]
    # Small target: 0.04 clears 0.03; 0.02 does not.
    assert attacker_succeeded(0.04, evaded, [0], cfg, small)
    assert not attacker_succeeded(0.02, evaded, [0], cfg, small)
    # Big target: a 0.05 drop that would win ABSOLUTELY (>=0.02) now fails (< 0.18);
    # 0.20 clears the proportional bar.
    assert not attacker_succeeded(0.05, evaded, [0], cfg, big)
    assert attacker_succeeded(0.20, evaded, [0], cfg, big)
    # committed_success threads the goal through to the attacker gate.
    assert not committed_success("attacker", 0.05, evaded, [0], cfg, big)
    assert committed_success("attacker", 0.20, evaded, [0], cfg, big)
    # slow_degrade uses per_round_drop as the target (bar = 0.6 * 0.02 = 0.012).
    slow = {"type": "slow_degrade", "per_round_drop": 0.02}
    assert attacker_succeeded(0.015, evaded, [0], cfg, slow)
    # No goal -> absolute fallback (attacker_min_drop = 0.02) is preserved.
    assert attacker_succeeded(0.03, evaded, [0], cfg)
    assert not attacker_succeeded(0.01, evaded, [0], cfg)


def test_controller_state_roundtrip():
    """state_dict/load_state_dict preserve the schedule state (resume support)."""
    cfg = _cfg()
    ctrl = PhaseController(cfg, first_learner="attacker")
    ctrl.record(True)                 # phase_round=1, streak=1
    ctrl.next_phase("cap")            # -> defender, phase_index=1, capped=True, streak=0
    ctrl.record(True)                 # phase_round=1, streak=1
    snap = ctrl.state_dict()
    assert snap == {"learner": "defender", "phase_index": 1, "phase_round": 1,
                    "streak": 1, "capped": True}
    # Restore into a fresh (default-attacker) controller — it becomes an exact copy.
    restored = PhaseController(cfg, first_learner="attacker")
    restored.load_state_dict(snap)
    assert restored.state_dict() == snap
    assert restored.learner == "defender" and restored.opponent == "attacker"
    # And it continues consistently from the restored streak.
    restored.record(True)
    assert restored.streak == 2
    # Missing keys keep current values; a bad learner is rejected.
    restored.load_state_dict({"streak": 5})
    assert restored.streak == 5 and restored.learner == "defender"
    try:
        restored.load_state_dict({"learner": "bogus"})
        assert False, "should reject an invalid learner"
    except ValueError:
        pass


def test_switch_requires_sustained_win_after_min_rounds():
    cfg = _cfg(min_phase_rounds=3, success_streak=2)
    ctrl = PhaseController(cfg, first_learner="attacker")
    # Win on rounds 1,2 — but min_phase_rounds=3 not yet reached, so no switch.
    assert ctrl.record(True) == (False, None)   # round 1
    assert ctrl.record(True) == (False, None)   # round 2 (streak 2 but round<min)
    # Round 3: min reached and streak>=2 -> switch on success.
    assert ctrl.record(True) == (True, "success")


def test_streak_resets_on_failure():
    cfg = _cfg(min_phase_rounds=2, success_streak=2)
    ctrl = PhaseController(cfg, first_learner="attacker")
    ctrl.record(True)            # round 1, streak 1
    assert ctrl.record(False) == (False, None)   # round 2, streak reset to 0
    assert ctrl.record(True) == (False, None)    # round 3, streak 1 (<2)
    assert ctrl.record(True) == (True, "success")  # round 4, streak 2 -> switch


def test_cap_forces_switch_without_win():
    cfg = _cfg(min_phase_rounds=2, max_phase_rounds=4, success_streak=2)
    ctrl = PhaseController(cfg, first_learner="attacker")
    assert ctrl.record(False) == (False, None)   # 1
    assert ctrl.record(False) == (False, None)   # 2
    assert ctrl.record(False) == (False, None)   # 3
    assert ctrl.record(False) == (True, "cap")   # 4 -> cap


def test_next_phase_flips_learner_and_sets_capped_flag():
    cfg = _cfg()
    ctrl = PhaseController(cfg, first_learner="attacker")
    assert ctrl.learner == "attacker" and ctrl.opponent == "defender"
    ctrl.next_phase("success")
    assert ctrl.learner == "defender" and ctrl.capped is False
    assert ctrl.phase_index == 1 and ctrl.phase_round == 0 and ctrl.streak == 0
    ctrl.next_phase("cap")
    assert ctrl.learner == "attacker" and ctrl.capped is True


def test_from_cfg_reads_yaml_dict():
    cfg = SwitchConfig.from_cfg({"min_phase_rounds": 5, "success_streak": 4})
    assert cfg.min_phase_rounds == 5 and cfg.success_streak == 4
    assert cfg.max_phase_rounds == 200   # default preserved
    assert cfg.win_fraction == 0.6       # relative-gate default preserved
    assert SwitchConfig.from_cfg({"win_fraction": 0.8}).win_fraction == 0.8


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} switch-controller tests passed.")


if __name__ == "__main__":
    _run()
