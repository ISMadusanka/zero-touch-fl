"""Unit tests for the defender phase controller (rl/switch.py).

Torch-free, so it runs anywhere (no GPU needed):  python tests/test_switch.py
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.switch import (  # noqa: E402
    PhaseController, SwitchConfig, attack_was_damaging, defender_succeeded,
    success_drop_bar,
)


@dataclass
class V:
    """Minimal DetectionVerdict stand-in."""
    client_id: int
    is_suspicious: bool


def _cfg(**kw):
    base = dict(min_phase_rounds=3, max_phase_rounds=10, success_streak=2,
                attacker_min_drop=0.02, defender_min_tpr=0.99, defender_max_fpr=0.10)
    base.update(kw)
    return SwitchConfig(**base)


def test_defender_success_predicate():
    cfg = _cfg()
    # Catches the 1 poisoned client, no false positives -> win.
    assert defender_succeeded([V(0, True), V(1, False), V(2, False)], [0], cfg)
    # Misses the poisoned client -> TPR 0 -> not a win.
    assert not defender_succeeded([V(0, False), V(1, False)], [0], cfg)
    # Catches it but over-flags an honest client (FPR 0.5 > 0.10) -> not a win.
    assert not defender_succeeded([V(0, True), V(1, True), V(2, False)], [0], cfg)


def test_defender_must_catch_every_poisoned_client():
    """defender_min_tpr defaults to 0.99, so a partial catch is a loss — which is
    the same bar the attack ladder's default `caught_rule: all` uses, so a
    defender 'win' and a ladder step-down mean the same thing."""
    cfg = _cfg()
    assert defender_succeeded([V(0, True), V(1, True), V(2, False)], [0, 1], cfg)
    assert not defender_succeeded([V(0, True), V(1, False), V(2, False)], [0, 1], cfg)


def test_defender_wins_a_clean_round_by_staying_quiet():
    """A ladder level that rounds to zero flipped labels leaves every update
    honest. TPR is undefined there; treating it as 0 would make a flawless round a
    loss. Staying quiet is the win condition on a clean round."""
    cfg = _cfg()
    assert defender_succeeded([V(0, False), V(1, False), V(2, False)], [], cfg)
    # ...but over-flagging honest clients is still a loss.
    assert not defender_succeeded([V(0, True), V(1, True), V(2, False)], [], cfg)


def test_damage_bar_is_relative_to_the_rounds_target():
    """The reported `attack_damaging` bar is win_fraction * target_accuracy_drop."""
    cfg = _cfg(win_fraction=0.6)
    small = {"type": "untargeted_degrade", "target_accuracy_drop": 0.05}   # bar = 0.03
    big = {"type": "untargeted_degrade", "target_accuracy_drop": 0.30}     # bar = 0.18
    assert attack_was_damaging(0.04, [0], cfg, small)
    assert not attack_was_damaging(0.02, [0], cfg, small)
    # A 0.05 drop clears the small target's bar but not the big one's.
    assert not attack_was_damaging(0.05, [0], cfg, big)
    assert attack_was_damaging(0.20, [0], cfg, big)
    # slow_degrade uses per_round_drop as the target (bar = 0.6 * 0.02 = 0.012).
    assert attack_was_damaging(0.015, [0], cfg, {"type": "slow_degrade",
                                                 "per_round_drop": 0.02})
    # No goal -> absolute fallback (attacker_min_drop = 0.02).
    assert attack_was_damaging(0.03, [0], cfg)
    assert not attack_was_damaging(0.01, [0], cfg)
    # A round with no poison was never an attack, whatever the accuracy did.
    assert not attack_was_damaging(0.9, [], cfg, small)


def test_success_drop_bar_is_the_shared_definition():
    """The metrics tracker and the round log both read this, so they cannot drift
    apart about what 'the attack was damaging' means."""
    cfg = _cfg(win_fraction=0.6)
    goal = {"type": "untargeted_degrade", "target_accuracy_drop": 0.10}
    assert abs(success_drop_bar(goal, cfg) - 0.06) < 1e-9
    assert success_drop_bar(None, cfg) == cfg.attacker_min_drop
    # cfg=None uses the dataclass defaults (the eval/baseline paths have no schedule).
    assert abs(success_drop_bar(goal) - 0.06) < 1e-9


def test_controller_state_roundtrip():
    """state_dict/load_state_dict preserve the schedule state (resume support)."""
    cfg = _cfg()
    ctrl = PhaseController(cfg)
    ctrl.record(True)                 # phase_round=1, streak=1
    ctrl.next_phase("cap")            # -> phase_index=1, capped=True, streak=0
    ctrl.record(True)                 # phase_round=1, streak=1
    snap = ctrl.state_dict()
    assert snap == {"learner": "defender", "phase_index": 1, "phase_round": 1,
                    "streak": 1, "capped": True}
    restored = PhaseController(cfg)
    restored.load_state_dict(snap)
    assert restored.state_dict() == snap
    # And it continues consistently from the restored streak.
    restored.record(True)
    assert restored.streak == 2
    # Missing keys keep current values.
    restored.load_state_dict({"streak": 5})
    assert restored.streak == 5 and restored.learner == "defender"


def test_resuming_a_pre_label_flip_checkpoint_falls_back_to_the_defender():
    """Checkpoints written before the attacker LLM was removed say
    learner="attacker". That phase no longer exists, so the controller continues
    as the defender rather than refusing to resume."""
    ctrl = PhaseController(_cfg())
    ctrl.load_state_dict({"learner": "attacker", "phase_index": 4, "streak": 1})
    assert ctrl.learner == "defender" and ctrl.phase_index == 4


def test_controller_rejects_a_non_defender_rotation():
    try:
        PhaseController(_cfg(), first_learner="defender", learners=("attacker",))
    except ValueError:
        return
    raise AssertionError("only the defender is trainable — 'attacker' must be rejected")


def test_switch_requires_sustained_win_after_min_rounds():
    cfg = _cfg(min_phase_rounds=3, success_streak=2)
    ctrl = PhaseController(cfg)
    # Win on rounds 1,2 — but min_phase_rounds=3 not yet reached, so no switch.
    assert ctrl.record(True) == (False, None)   # round 1
    assert ctrl.record(True) == (False, None)   # round 2 (streak 2 but round<min)
    # Round 3: min reached and streak>=2 -> switch on success.
    assert ctrl.record(True) == (True, "success")


def test_streak_resets_on_failure():
    cfg = _cfg(min_phase_rounds=2, success_streak=2)
    ctrl = PhaseController(cfg)
    ctrl.record(True)            # round 1, streak 1
    assert ctrl.record(False) == (False, None)   # round 2, streak reset to 0
    assert ctrl.record(True) == (False, None)    # round 3, streak 1 (<2)
    assert ctrl.record(True) == (True, "success")  # round 4, streak 2 -> switch


def test_cap_forces_switch_without_win():
    cfg = _cfg(min_phase_rounds=2, max_phase_rounds=4, success_streak=2)
    ctrl = PhaseController(cfg)
    assert ctrl.record(False) == (False, None)   # 1
    assert ctrl.record(False) == (False, None)   # 2
    assert ctrl.record(False) == (False, None)   # 3
    assert ctrl.record(False) == (True, "cap")   # 4 -> cap


def test_next_phase_starts_a_fresh_phase_and_sets_capped_flag():
    cfg = _cfg()
    ctrl = PhaseController(cfg)
    assert ctrl.learner == "defender"
    ctrl.next_phase("success")
    assert ctrl.learner == "defender" and ctrl.capped is False
    assert ctrl.phase_index == 1 and ctrl.phase_round == 0 and ctrl.streak == 0
    ctrl.next_phase("cap")
    assert ctrl.learner == "defender" and ctrl.capped is True and ctrl.phase_index == 2


def test_from_cfg_reads_yaml_dict():
    cfg = SwitchConfig.from_cfg({"min_phase_rounds": 5, "success_streak": 4})
    assert cfg.min_phase_rounds == 5 and cfg.success_streak == 4
    assert cfg.max_phase_rounds == 200   # default preserved
    assert cfg.win_fraction == 0.6       # relative-bar default preserved
    assert SwitchConfig.from_cfg({"win_fraction": 0.8}).win_fraction == 0.8


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} phase-controller tests passed.")


if __name__ == "__main__":
    _run()
