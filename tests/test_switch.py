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


def test_committed_success_dispatch():
    cfg = _cfg()
    assert committed_success("attacker", 0.05, [V(0, False)], [0], cfg)
    assert committed_success("defender", 0.0, [V(0, True), V(1, False)], [0], cfg)


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


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} switch-controller tests passed.")


if __name__ == "__main__":
    _run()
