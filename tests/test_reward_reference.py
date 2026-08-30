"""Regression tests for the reward semantics (rl/rewards.py).

The defender's reward is the only trained signal; ``attack_effectiveness`` is a
reported measurement of how much the round's flipped labels cost the model.
Locked in here:

1. Damage is measured against the round's CLEAN counterfactual, so an attack of a
   given strength scores the same every time it is sent — a "previous round's
   accuracy" reference would make a repeated, equally damaging attack read as ~0
   from the second round on.
2. The damage term is strictly monotonic at BOTH ends instead of flat, so it
   still separates rounds that all overshoot (or all backfire).
3. A round with no effective poison — a ladder level that rounded to zero flips —
   is scored sanely: the defender is rewarded for staying quiet instead of being
   handed an undefined F1 of 0.

Torch-free:  python tests/test_reward_reference.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import DetectionVerdict  # noqa: E402
from rl.rewards import (  # noqa: E402
    attack_effectiveness, defender_reward, drop_term, goal_target, group_advantages,
)

GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}


# --- 1. clean-reference semantics -------------------------------------------

def test_identical_attack_scores_identically_every_round():
    """Each round rebuilds its own counterfactual from the same anchor, so the
    same attack strength must measure the same every round."""
    clean, post = 0.90, 0.60          # this round's unpoisoned acc vs achieved acc
    r = [attack_effectiveness(clean, post, GOAL) for _ in range(3)]
    assert r[0] == r[1] == r[2]
    assert r[0] > 1.0                 # a 0.30 drop against a 0.20 target


def test_effectiveness_tracks_goal_attainment():
    half = attack_effectiveness(0.90, 0.80, GOAL)
    full = attack_effectiveness(0.90, 0.70, GOAL)
    assert full > half
    # drop == target -> exactly 1.0, by construction.
    assert abs(full - 1.0) < 1e-9
    # No damage -> 0.0; a backfire (the model got BETTER) -> negative.
    assert abs(attack_effectiveness(0.90, 0.90, GOAL)) < 1e-9
    assert attack_effectiveness(0.90, 0.95, GOAL) < 0.0


def test_goal_target_reads_both_goal_shapes():
    assert abs(goal_target(GOAL) - 0.20) < 1e-9
    assert abs(goal_target({"type": "slow_degrade", "per_round_drop": 0.02}) - 0.02) < 1e-9
    assert goal_target({}) > 0.0      # never zero, so the division is always safe


# --- 2. no flat region at EITHER end ----------------------------------------

def test_drop_term_is_linear_through_the_whole_original_range():
    """The linear region and the value at the goal are unchanged."""
    for x in (-0.5, 0.0, 0.25, 0.5, 1.0):
        assert abs(drop_term(x * 0.2, 0.2) - x) < 1e-9


def test_drop_term_is_bounded_and_strictly_increasing_past_the_goal():
    xs = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
    vals = [drop_term(x * 0.2, 0.2) for x in xs]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    assert vals[0] == 1.0 and max(vals) < 1.5


def test_drop_term_is_bounded_and_strictly_decreasing_past_a_backfire():
    """The mirror of the overshoot property. ``x < -0.5`` means "the round made the
    model more than 0.5*target BETTER than the clean counterfactual", which a weak
    ladder level on a small target reaches routinely — a hard floor there would
    make every such round report exactly -0.5 and hide the differences."""
    xs = [-0.5, -0.75, -1.0, -2.0, -5.0, -20.0]
    vals = [drop_term(x * 0.02, 0.02) for x in xs]
    assert all(b < a for a, b in zip(vals, vals[1:])), vals
    assert vals[0] == -0.5                     # the knee is exactly the old floor
    assert min(vals) > -0.75                   # still bounded-bad, not unbounded


def test_overshoot_is_worth_much_less_than_reaching_the_goal():
    """Saturation stays fast, so 1.0 keeps meaning 'hit the target'."""
    at_goal = drop_term(0.20, 0.20)
    quadruple = drop_term(0.80, 0.20)
    assert at_goal == 1.0
    assert quadruple - at_goal < 0.4          # 4x the target buys < 0.4 extra


# --- 3. the defender's reward -----------------------------------------------

def test_defender_rewarded_for_silence_on_a_clean_round():
    """A ladder level that rounds to zero flips leaves every update honest. F1 is
    undefined there and scoring a flawless defender 0 would train it to invent
    detections; staying quiet is the win condition instead."""
    quiet = defender_reward([DetectionVerdict(i, False, 1.0, "") for i in range(20)], [])
    unsure = defender_reward([DetectionVerdict(i, False, 0.0, "") for i in range(20)], [])
    noisy = defender_reward([DetectionVerdict(i, True, 1.0, "") for i in range(20)], [])
    assert abs(quiet - 1.0) < 1e-9
    assert abs(unsure - 0.5) < 1e-9
    assert abs(noisy - 0.0) < 1e-9


def test_defender_scored_on_soft_f1_when_poison_exists():
    verdicts = [DetectionVerdict(0, True, 1.0, "")] + [
        DetectionVerdict(i, False, 1.0, "") for i in range(1, 20)]
    assert abs(defender_reward(verdicts, [0]) - 1.0) < 1e-6      # perfect catch
    missed = [DetectionVerdict(i, False, 1.0, "") for i in range(20)]
    assert defender_reward(missed, [0]) < 0.01                   # missed it entirely


def test_defender_reward_is_continuous_in_confidence():
    """The soft F1 is what gives GRPO a gradient when several rollouts agree on the
    hard flags but differ in how sure they are."""
    def r(conf):
        return defender_reward(
            [DetectionVerdict(0, True, conf, "")]
            + [DetectionVerdict(i, False, 1.0, "") for i in range(1, 20)], [0])
    vals = [r(c) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


def test_defender_group_keeps_a_learning_signal():
    """Four verdict sets of differing quality must separate, or the advantages
    collapse and grpo_step skips the update."""
    def r(n_caught):
        poisoned = [0, 1, 2, 3]
        v = [DetectionVerdict(i, i < n_caught, 1.0, "") for i in range(4)]
        v += [DetectionVerdict(i, False, 1.0, "") for i in range(4, 20)]
        return defender_reward(v, poisoned)
    rs = [r(n) for n in (0, 1, 2, 4)]
    assert all(b > a for a, b in zip(rs, rs[1:])), rs
    _adv, zero_frac = group_advantages(rs)
    assert zero_frac == 0.0


def test_tpr_minus_fpr_mode_penalises_false_positives():
    poisoned = [0]
    perfect = [DetectionVerdict(0, True, 1.0, "")] + [
        DetectionVerdict(i, False, 1.0, "") for i in range(1, 11)]
    over = [DetectionVerdict(i, True, 1.0, "") for i in range(6)] + [
        DetectionVerdict(i, False, 1.0, "") for i in range(6, 11)]
    assert defender_reward(perfect, poisoned, mode="tpr_minus_fpr") == 1.0
    assert defender_reward(over, poisoned, mode="tpr_minus_fpr") < 1.0


def test_calibrated_p_malicious_is_preferred_over_confidence():
    """Algorithmic defenses report a boundary-calibrated p_malicious; the
    (is_suspicious, confidence) reconstruction is only the fallback."""
    poisoned = [0]
    others = [DetectionVerdict(i, False, 1.0, "", p_malicious=0.0) for i in range(1, 11)]
    strong = defender_reward(
        [DetectionVerdict(0, True, 0.0, "", p_malicious=1.0)] + others, poisoned)
    weak = defender_reward(
        [DetectionVerdict(0, True, 0.0, "", p_malicious=0.55)] + others, poisoned)
    assert strong > weak, "p_malicious must drive the reward, not `confidence`"


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} reward-reference tests passed.")


if __name__ == "__main__":
    _run()
