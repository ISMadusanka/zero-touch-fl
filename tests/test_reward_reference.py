"""Regression tests for the attacker/defender reward semantics (rl/rewards.py).

Locks in three fixes:

1. Damage is measured against the round's CLEAN counterfactual, so an attack that
   hits its target scores the same every time it hits it — the previous
   "prev round's post-attack accuracy" reference made a repeated, equally
   devastating attack score ~0 from the second round on, which put the
   schedule's ``success_streak`` gate permanently out of reach.
2. The damage term is strictly monotonic above the goal instead of flat, so a
   group of competent rollouts no longer collapses to zero advantage (which made
   ``grpo_step`` skip the update exactly when the attacker got good).
3. A round with no effective poison is scored sanely for BOTH agents: the
   defender is rewarded for staying quiet instead of being handed an undefined
   F1 of 0, and the attacker is charged the full waste penalty.

Torch-free except for ``perturbation_diversity`` (not used here):
    python tests/test_reward_reference.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import DetectionVerdict  # noqa: E402
from rl.rewards import (  # noqa: E402
    attacker_reward, defender_reward, drop_term, group_advantages,
)

GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}


def _evaded(ids, n_clients=20):
    """Every client passes as benign (confidently) -> full stealth for poisoners."""
    return [DetectionVerdict(cid, False, 1.0, "") for cid in range(n_clients)]


# --- 1. clean-reference semantics -------------------------------------------

def test_identical_attack_scores_identically_every_round():
    """The memoryless env rebuilds the same aggregate each round, so the same
    attack must earn the same reward each round."""
    clean, post = 0.90, 0.60          # this round's unpoisoned acc vs achieved acc
    r = [attacker_reward(clean, post, GOAL, [0], _evaded([0]), 0, pool_size=5)
         for _ in range(3)]
    assert r[0] == r[1] == r[2]
    assert r[0] > 1.0                 # a 0.30 drop against a 0.20 target


def test_reward_tracks_goal_attainment_not_round_over_round_change():
    half = attacker_reward(0.90, 0.80, GOAL, [0], _evaded([0]), 0, pool_size=5)
    full = attacker_reward(0.90, 0.70, GOAL, [0], _evaded([0]), 0, pool_size=5)
    assert full > half
    # drop == target -> damage term is exactly 1.0 (plus the 0.5 stealth bonus).
    assert abs(full - (1.0 + 0.5)) < 1e-9


# --- 2. no flat region above the goal ---------------------------------------

def test_drop_term_matches_old_shape_up_to_the_goal():
    for x in (-3.0, -0.5, 0.0, 0.25, 0.5, 1.0):
        assert abs(drop_term(x * 0.2, 0.2) - max(-0.5, x)) < 1e-9


def test_drop_term_is_bounded_and_strictly_increasing_past_the_goal():
    xs = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
    vals = [drop_term(x * 0.2, 0.2) for x in xs]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    assert vals[0] == 1.0 and max(vals) < 1.5


def test_competent_group_keeps_a_learning_signal():
    """Four rollouts that all overshoot the target used to tie at the 1.5 clip →
    zero advantage → grpo_step skipped the step. They must now separate."""
    posts = [0.45, 0.30, 0.18, 0.09]          # drops 0.45 / 0.60 / 0.72 / 0.81
    rs = [attacker_reward(0.90, p, GOAL, [0], _evaded([0]), 0, pool_size=5)
          for p in posts]
    assert len(set(rs)) == len(rs), rs
    _adv, zero_frac = group_advantages(rs)
    assert zero_frac == 0.0
    assert all(b > a for a, b in zip(rs, rs[1:]))     # more damage -> more reward


def test_overshoot_is_worth_much_less_than_reaching_the_goal():
    """Saturation must stay fast: the objective is 'hit the target', not
    'destroy the model'."""
    at_goal = drop_term(0.20, 0.20)
    quadruple = drop_term(0.80, 0.20)
    assert at_goal == 1.0
    assert quadruple - at_goal < 0.4          # 4x the target buys < 0.4 extra


# --- 3. clean rounds (no effective poison) ----------------------------------

def test_defender_rewarded_for_silence_on_a_clean_round():
    quiet = defender_reward([DetectionVerdict(i, False, 1.0, "") for i in range(20)], [])
    unsure = defender_reward([DetectionVerdict(i, False, 0.0, "") for i in range(20)], [])
    noisy = defender_reward([DetectionVerdict(i, True, 1.0, "") for i in range(20)], [])
    assert abs(quiet - 1.0) < 1e-9
    assert abs(unsure - 0.5) < 1e-9
    assert abs(noisy - 0.0) < 1e-9


def test_defender_still_scored_on_f1_when_poison_exists():
    verdicts = [DetectionVerdict(0, True, 1.0, "")] + [
        DetectionVerdict(i, False, 1.0, "") for i in range(1, 20)]
    assert abs(defender_reward(verdicts, [0]) - 1.0) < 1e-6      # perfect catch
    missed = [DetectionVerdict(i, False, 1.0, "") for i in range(20)]
    assert defender_reward(missed, [0]) < 0.01                   # missed it entirely


def test_attacker_punished_for_a_wasted_round():
    """No effective poison: no damage, no stealth, full waste penalty."""
    r = attacker_reward(0.90, 0.90, GOAL, [], _evaded([]), 1, pool_size=5)
    assert abs(r - (-1.0)) < 1e-9
    # Selecting three clients and wasting all three is no worse per-client...
    assert abs(attacker_reward(0.90, 0.90, GOAL, [], _evaded([]), 3, pool_size=5)
               - (-1.0)) < 1e-9
    # ...but wasting 1 of 2 selected is only half the penalty.
    half = attacker_reward(0.90, 0.90, GOAL, [1], _evaded([1]), 1, pool_size=5)
    assert abs(half - (0.0 + 0.5 * 1.0 - 1.0 * 0.5)) < 1e-9


def test_doing_nothing_scores_worse_than_a_real_attack():
    nothing = attacker_reward(0.90, 0.90, GOAL, [], _evaded([]), 1, pool_size=5)
    real = attacker_reward(0.90, 0.75, GOAL, [0], _evaded([0]), 0, pool_size=5)
    assert real > nothing + 1.0


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} reward-reference tests passed.")


if __name__ == "__main__":
    _run()
