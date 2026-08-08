"""Regression tests for the attacker/defender reward semantics (rl/rewards.py).

Locks in the reward invariants needed by attacker training:

1. Damage is measured against the round's CLEAN counterfactual, so an attack that
   hits its target scores the same every time it hits it — the previous
   "prev round's post-attack accuracy" reference made a repeated, equally
   devastating attack score ~0 from the second round on, which put the
   schedule's ``success_streak`` gate permanently out of reach.
2. Damage stays linear up to the requested threshold and saturates there, so the
   objective means "at least the target" rather than rewarding gratuitous harm.
3. Stealth and collaboration are paid only in proportion to positive damage and
   hard survival, with their positive contribution capped below damage.
4. A round with no effective poison is scored sanely for BOTH agents: the
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
    attacker_reward, attacker_reward_breakdown, defender_reward, drop_term,
    group_advantages, soft_malicious_probability,
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
    r = [attacker_reward(clean, post, GOAL, [0], _evaded([0]), 0)
         for _ in range(3)]
    assert r[0] == r[1] == r[2]
    assert r[0] > 1.0                 # a 0.30 drop against a 0.20 target


def test_reward_tracks_goal_attainment_not_round_over_round_change():
    half = attacker_reward(0.90, 0.80, GOAL, [0], _evaded([0]), 0)
    full = attacker_reward(0.90, 0.70, GOAL, [0], _evaded([0]), 0)
    assert full > half
    # drop == target -> damage term is exactly 1.0 (plus the 0.5 stealth bonus).
    assert abs(full - (1.0 + 0.5)) < 1e-9


# --- 2. at-least target semantics -------------------------------------------

def test_drop_term_matches_old_shape_up_to_the_goal():
    for x in (-3.0, -0.5, 0.0, 0.25, 0.5, 1.0):
        assert abs(drop_term(x * 0.2, 0.2) - max(-0.5, x)) < 1e-9


def test_drop_term_saturates_once_the_target_is_reached():
    xs = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
    vals = [drop_term(x * 0.2, 0.2) for x in xs]
    assert vals == [1.0] * len(xs)


def test_point_one_target_has_linear_progress_then_saturates():
    target = 0.10
    assert abs(drop_term(0.05, target) - 0.5) < 1e-12
    assert drop_term(0.10, target) == 1.0
    assert drop_term(0.30, target) == 1.0


def test_subtarget_group_keeps_a_considerable_learning_signal():
    """Sub-target damage stays linear and comfortably clears the noise gate."""
    posts = [0.88, 0.84, 0.80, 0.76]          # drops .02 / .06 / .10 / .14
    rs = [attacker_reward(0.90, p, GOAL, [0], _evaded([0]), 0)
          for p in posts]
    assert len(set(rs)) == len(rs), rs
    _adv, zero_frac = group_advantages(rs)
    assert zero_frac == 0.0
    assert all(b > a for a, b in zip(rs, rs[1:]))     # more damage -> more reward


def test_overshoot_is_not_worth_more_than_reaching_the_goal():
    at_goal = drop_term(0.20, 0.20)
    quadruple = drop_term(0.80, 0.20)
    assert at_goal == quadruple == 1.0


def test_above_target_ties_are_an_intentional_terminal_plateau():
    posts = [0.70, 0.60, 0.50, 0.40]
    rs = [attacker_reward(0.90, p, GOAL, [0], _evaded([0]), 0)
          for p in posts]
    assert len(set(rs)) == 1
    _adv, zero_frac = group_advantages(rs)
    assert zero_frac == 1.0


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
    r = attacker_reward(0.90, 0.90, GOAL, [], _evaded([]), 1)
    assert abs(r - (-1.0)) < 1e-9
    # Selecting three clients and wasting all three is no worse per-client...
    assert abs(attacker_reward(0.90, 0.90, GOAL, [], _evaded([]), 3)
               - (-1.0)) < 1e-9
    # ...but wasting 1 of 2 selected is only half the penalty.
    half = attacker_reward(0.90, 0.90, GOAL, [1], _evaded([1]), 1)
    assert abs(half - (-0.5)) < 1e-9


def test_doing_nothing_scores_worse_than_a_real_attack():
    nothing = attacker_reward(0.90, 0.90, GOAL, [], _evaded([]), 1)
    real = attacker_reward(0.90, 0.75, GOAL, [0], _evaded([0]), 0)
    assert real > nothing + 1.0


# --- 4. stealth has to be earned --------------------------------------------
#
# Ungated, `beta * stealth` is a guaranteed 0.5 for submitting a rounding error:
# undetectable by construction, and better than attacking, because a robust
# aggregator rescales magnitudes and drops outliers so the damage term is hard to
# move. Policies collapse into exactly that. The gate scales each client's stealth
# by min(1, ||poisoned-benign|| / ||benign-global|| / stealth_floor).

def test_stealth_is_not_paid_for_a_negligible_perturbation():
    """At equal positive damage, a real edit out-earns a token one."""
    inert = attacker_reward(0.90, 0.80, GOAL, [0], _evaded([0]), 0,
                            perturbation_ratios={0: 0.001}, stealth_floor=1.0)
    real = attacker_reward(0.90, 0.80, GOAL, [0], _evaded([0]), 0,
                           perturbation_ratios={0: 1.0}, stealth_floor=1.0)
    assert abs(inert - 0.50025) < 1e-9
    assert abs(real - 0.75) < 1e-9
    assert real > inert


def test_stealth_credit_is_continuous_in_the_attack_size():
    """The gradient stealth exists to provide must survive the gate."""
    rs = [attacker_reward(0.90, 0.80, GOAL, [0], _evaded([0]), 0,
                          perturbation_ratios={0: x}, stealth_floor=1.0)
          for x in (0.1, 0.25, 0.5, 0.75)]
    assert rs == sorted(rs) and rs[0] < rs[-1]


def test_stealth_credit_saturates_at_the_floor():
    at = attacker_reward(0.90, 0.80, GOAL, [0], _evaded([0]), 0,
                         perturbation_ratios={0: 1.0}, stealth_floor=1.0)
    over = attacker_reward(0.90, 0.80, GOAL, [0], _evaded([0]), 0,
                           perturbation_ratios={0: 25.0}, stealth_floor=1.0)
    assert abs(at - over) < 1e-9               # no reward for gratuitous size


def test_a_caught_client_earns_no_stealth_however_big_its_edit():
    caught = [DetectionVerdict(0, True, 1.0, "")]
    r = attacker_reward(0.90, 0.80, GOAL, [0], caught, 0,
                        perturbation_ratios={0: 5.0}, stealth_floor=1.0)
    assert abs(r - 0.5) < 1e-9                 # damage only; no stealth payout


def test_the_gate_is_off_by_default_and_switchable():
    """Callers that cannot measure the perturbation, and stealth_floor<=0, keep the
    original perturbation-size behaviour exactly."""
    ungated = attacker_reward(0.90, 0.80, GOAL, [0], _evaded([0]), 0)
    assert abs(ungated - 0.75) < 1e-9
    disabled = attacker_reward(0.90, 0.80, GOAL, [0], _evaded([0]), 0,
                               perturbation_ratios={0: 0.0}, stealth_floor=0.0)
    assert abs(disabled - 0.75) < 1e-9
    unmeasured = attacker_reward(0.90, 0.80, GOAL, [0, 1], _evaded([0, 1]), 0,
                                 perturbation_ratios={0: 1.0}, stealth_floor=1.0)
    assert abs(unmeasured - 0.75) < 1e-9       # missing ratio -> full credit


def test_an_infinite_ratio_earns_full_credit():
    """||benign - global|| == 0 (the honest update was zero), so ANY edit is
    infinitely larger than it — that is a real attack, not an undefined one."""
    r = attacker_reward(0.90, 0.80, GOAL, [0], _evaded([0]), 0,
                        perturbation_ratios={0: float("inf")}, stealth_floor=1.0)
    assert abs(r - 0.75) < 1e-9


def test_no_positive_auxiliary_reward_without_positive_damage():
    kwargs = dict(alpha=1.0, beta=100.0, gamma=0.0, zeta=100.0,
                  diversity=1.0)
    verdicts = _evaded([0, 1])
    zero = attacker_reward(0.90, 0.90, GOAL, [0, 1], verdicts, 0, **kwargs)
    negative = attacker_reward(0.90, 0.95, GOAL, [0, 1], verdicts, 0, **kwargs)
    assert zero == 0.0
    assert abs(negative - (-0.25)) < 1e-9


def test_hard_rejection_zeroes_stealth_despite_low_soft_probability():
    contradictory = [DetectionVerdict(
        0, True, 0.0, "hard reject", p_malicious=0.01)]
    shaped = attacker_reward(0.90, 0.80, GOAL, [0], contradictory, 0,
                             beta=100.0)
    damage_only = attacker_reward(0.90, 0.80, GOAL, [0], contradictory, 0,
                                  beta=0.0)
    assert abs(shaped - damage_only) < 1e-12
    assert abs(damage_only - 0.5) < 1e-12


def test_positive_auxiliary_is_capped_below_damage():
    breakdown = attacker_reward_breakdown(
        0.90, 0.80, GOAL, [0, 1], _evaded([0, 1]), 0,
        beta=100.0, zeta=100.0, diversity=1.0)
    assert abs(breakdown["damage_component"] - 0.5) < 1e-9
    assert abs(breakdown["auxiliary_component"] - 0.375) < 1e-9
    assert breakdown["auxiliary_component"] < breakdown["damage_component"]
    assert abs(breakdown["total"] - 0.875) < 1e-9


def test_reward_breakdown_reconciles_with_scalar_api():
    kwargs = dict(alpha=1.2, beta=0.5, gamma=2.0, zeta=0.2,
                  diversity=1.0, perturbation_ratios={0: 1.0, 1: 0.5})
    breakdown = attacker_reward_breakdown(
        0.90, 0.80, GOAL, [0, 1], _evaded([0, 1]), 1, **kwargs)
    scalar = attacker_reward(
        0.90, 0.80, GOAL, [0, 1], _evaded([0, 1]), 1, **kwargs)
    components = (breakdown["damage_component"]
                  + breakdown["stealth_component"]
                  + breakdown["collaboration_component"]
                  + breakdown["malformed_component"])
    assert abs(breakdown["total"] - components) < 1e-12
    assert scalar == breakdown["total"]
    assert abs(breakdown["induced_drop"] - 0.10) < 1e-12
    assert breakdown["target_drop"] == 0.20
    assert abs(breakdown["damage_progress"] - 0.5) < 1e-9


def test_public_soft_probability_matches_calibrated_verdict():
    verdict = DetectionVerdict(0, False, 0.1, "", p_malicious=0.73)
    assert soft_malicious_probability(verdict) == 0.73


def test_damage_still_dominates_the_gated_stealth():
    """The gate must not let a stealthy no-op beat a caught, effective attack."""
    inert_but_hidden = attacker_reward(0.90, 0.90, GOAL, [0], _evaded([0]), 0,
                                       perturbation_ratios={0: 0.001},
                                       stealth_floor=1.0)
    caught_but_effective = attacker_reward(
        0.90, 0.70, GOAL, [0], [DetectionVerdict(0, True, 1.0, "")], 0,
        perturbation_ratios={0: 2.0}, stealth_floor=1.0)
    assert caught_but_effective > inert_but_hidden


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} reward-reference tests passed.")


if __name__ == "__main__":
    _run()
