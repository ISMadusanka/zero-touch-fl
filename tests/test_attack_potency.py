"""Regression tests for the stealth gate (rl.rewards.attack_potency) and for the
defense-malfunction guard that voids a round's gradient (rl.env).

Both lock in fixes for the same class of bug: the reward paying for something
other than attacking.

1. STEALTH GATE. ``beta * stealth`` used to be an unconditional payment for not
   being caught, and the cheapest way not to be caught is not to attack. A poison
   one part in a thousand of the honest update is byte-different (so it clears the
   ``n_malformed`` no-op check) and collects full stealth for zero damage. Over a
   recorded 262-round run that term supplied ~89% of the attacker's reward at a
   median induced drop of 0.0000, while the global model GAINED 0.097 accuracy.
   ``attack_potency`` gates the payment on how much poison was actually shipped.

2. DEFENSE-MALFUNCTION GUARD. When the round's defense rejects the honest majority
   of an entirely UNPOISONED cohort, every accuracy it produces measures its own
   false-positive rate. 45 of those same 262 rounds were in that state and all of
   them contributed a gradient.

Needs torch (attack_potency is a tensor computation):
    python tests/test_attack_potency.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import yaml  # noqa: E402

from core.types import DetectionVerdict  # noqa: E402
from rl.env import _accepts_honest_majority  # noqa: E402
from rl.rewards import (  # noqa: E402
    MIN_REAL_DROP, attack_potency, attacker_reward, attacker_reward_terms,
    check_reward_balance,
)

_CFG = yaml.safe_load(open(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "configs", "base.yaml"), encoding="utf-8"))
# Read the SHIPPED weights rather than restating them, so these tests fail when
# configs/base.yaml drifts out of the balance they assert — which is exactly how
# the balance was lost last time.
W = _CFG["rl"]["reward"]["attacker"]
TARGET = float(_CFG["attack"]["goal"]["target_accuracy_drop"])
ALPHA, BETA, GAMMA, ZETA = (float(W["alpha"]), float(W["beta"]),
                            float(W["gamma"]), float(W["zeta"]))
GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": TARGET}
# MIN_REAL_DROP (the smallest drop worth calling an attack) comes from rl.rewards,
# so the tests and the runtime check_reward_balance warning cannot disagree.


def _sd(*vals):
    return {"w": torch.tensor(list(vals), dtype=torch.float32)}


def _add(sd, eps):
    return {k: v + eps for k, v in sd.items()}


def _all_benign(n=20):
    """Every client passes, confidently -> full ungated stealth for poisoners."""
    return [DetectionVerdict(cid, False, 1.0, "", p_malicious=0.0) for cid in range(n)]


def _all_flagged(n=20):
    return [DetectionVerdict(cid, True, 1.0, "", p_malicious=1.0) for cid in range(n)]


# --- 1. attack_potency measures poison against the honest update -------------

GLOBAL = _sd(1.0, 1.0, 1.0, 1.0)
BENIGN = _sd(1.1, 1.1, 1.1, 1.1)        # honest update: ||delta|| = 0.2


def test_no_poison_scores_zero_potency():
    assert attack_potency({}, {}, GLOBAL) == 0.0
    # An exactly-unchanged client is also zero, not "a tiny bit potent".
    assert attack_potency({0: BENIGN}, {0: BENIGN}, GLOBAL) == 0.0


def test_potency_is_the_poison_relative_to_the_honest_update():
    # Poison of the SAME size as the honest update -> s = 1 -> s/(s+1) = 0.5.
    poisoned = {0: _add(BENIGN, 0.1)}    # ||poison|| = 0.2 == ||honest delta||
    p = attack_potency(poisoned, {0: BENIGN}, GLOBAL)
    assert abs(p - 0.5) < 1e-5
    # Three times the honest update -> s = 3 -> 0.75.
    p3 = attack_potency({0: _add(BENIGN, 0.3)}, {0: BENIGN}, GLOBAL)
    assert abs(p3 - 0.75) < 1e-5


def test_potency_is_strictly_increasing_and_bounded():
    """No flat region at either end — the same requirement drop_term is built
    around, because a flat reward region collapses GRPO's within-group spread."""
    prev = -1.0
    for eps in (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0):
        p = attack_potency({0: _add(BENIGN, eps)}, {0: BENIGN}, GLOBAL)
        assert p > prev, f"potency not increasing at eps={eps}"
        assert 0.0 <= p < 1.0
        prev = p


def test_potency_averages_over_the_poisoned_clients():
    refs = {0: BENIGN, 1: BENIGN}
    poisoned = {0: _add(BENIGN, 0.1), 1: BENIGN}    # one real, one untouched
    assert abs(attack_potency(poisoned, refs, GLOBAL) - 0.25) < 1e-5


# --- 2. the gate closes the "don't attack" optimum ---------------------------

def _terms(poisoned, refs, post, verdicts, n_malformed=0):
    return attacker_reward_terms(
        0.90, post, GOAL, sorted(poisoned), verdicts, n_malformed,
        alpha=ALPHA, beta=BETA, gamma=GAMMA, zeta=ZETA,
        potency=attack_potency(poisoned, refs, GLOBAL),
    )


def _reward(poisoned, refs, post, verdicts, n_malformed=0):
    return _terms(poisoned, refs, post, verdicts, n_malformed)["total"]


def test_a_non_attack_no_longer_outscores_a_real_attack():
    """THE regression. A near-honest update that evades everything used to earn
    the full stealth payment while doing no damage; a real attack that lands a
    genuine drop earned less. That ordering was what the policy learned."""
    refs = {0: BENIGN}
    # Nearly-honest poison, undetected, zero damage — the old reward's optimum.
    sneaky = _reward({0: _add(BENIGN, 1e-4)}, refs, 0.90, _all_benign())
    # A real attack: poison the size of the honest update, undetected, 2pp drop.
    real = _reward({0: _add(BENIGN, 0.1)}, refs, 0.88, _all_benign())
    assert real > sneaky
    # And the non-attack must earn almost nothing at all, not ~0.5.
    assert sneaky < 0.01


def test_gate_does_not_reward_being_loud_either():
    """The gate must not create a new degenerate optimum: a huge perturbation
    maxes potency but is flagged, so stealth -> 0 and the term collapses."""
    refs = {0: BENIGN}
    caught = _reward({0: _add(BENIGN, 100.0)}, refs, 0.90, _all_flagged())
    assert caught <= 0.0
    balanced = _reward({0: _add(BENIGN, 0.1)}, refs, 0.88, _all_benign())
    assert balanced > caught


def test_shipped_weights_satisfy_the_shaping_budget_invariant():
    """THE invariant. Damage is the objective; stealth and collaboration are shaping
    and must never be able to outbid it:

        beta + zeta  <  alpha * MIN_REAL_DROP / target

    Both previous settings violated this. At beta 0.5 / zeta 0.2 the shaping budget
    was 0.7 against the 0.5 a 1pp drop paid, so the reward-maximizing move was to
    look busy and harmless — which is what the policy did for 262 rounds."""
    budget = BETA + ZETA
    real_damage = ALPHA * MIN_REAL_DROP / TARGET
    assert budget < real_damage, (
        f"shaping budget {budget} >= what a {MIN_REAL_DROP} drop earns "
        f"({real_damage}); the reward pays more for hiding than for damaging"
    )
    # The runtime guard that every run mode calls must agree with this test, so a
    # misbalanced config is caught in the log even when nobody ran the tests.
    assert check_reward_balance(W, GOAL, context="test") is True
    assert check_reward_balance({}, GOAL, context="test") is False, (
        "the bare attacker_reward defaults must NOT pass at the shipped target — "
        "that combination is what --baseline/--dry-run silently used"
    )


def test_a_harmless_evading_attack_cannot_outbid_real_damage():
    """The gate alone did NOT fix this and the weights alone would not either.

    A large perturbation that evades but does no damage has potency ~0.5 and full
    stealth — it passes the potency gate honestly. Only the budget cap stops it
    paying like a real attack. At the old beta 0.5 it collected 0.245 against the
    0.5 a 1pp drop earned: a 2x gap that noise closes."""
    refs = {0: BENIGN}
    big = {0: _add(BENIGN, 0.2)}                   # potency ~0.5, fully evading
    harmless = _terms(big, refs, 0.90, _all_benign())
    real = _terms(big, refs, 0.90 - MIN_REAL_DROP, _all_benign())
    assert harmless["damage"] == 0.0
    assert harmless["stealth"] > 0.0               # it IS a real, evading attack...
    assert real["total"] > 3 * harmless["total"], (
        f"a harmless evading attack scores {harmless['total']:.3f} against "
        f"{real['total']:.3f} for a {MIN_REAL_DROP} drop — too close"
    )


def test_damage_is_the_dominant_term_whenever_the_attack_worked():
    refs = {0: BENIGN}
    poisoned = {0: _add(BENIGN, 0.1)}
    for drop in (0.005, 0.01, 0.05, TARGET):
        t = _terms(poisoned, refs, 0.90 - drop, _all_benign())
        others = abs(t["stealth"]) + abs(t["collab"]) + abs(t["malformed"])
        assert t["damage"] > others, (
            f"at drop={drop} damage {t['damage']:.3f} does not dominate "
            f"the other terms {others:.3f}"
        )


def test_terms_sum_to_the_total_reward():
    """attacker_reward must stay exactly the sum of the decomposition it reports,
    or the logged breakdown is decoration rather than evidence."""
    refs = {0: BENIGN}
    poisoned = {0: _add(BENIGN, 0.1), 1: _add(BENIGN, 0.2)}
    refs = {0: BENIGN, 1: BENIGN}
    t = attacker_reward_terms(0.90, 0.85, GOAL, [0, 1], _all_benign(), 1,
                              alpha=ALPHA, beta=BETA, gamma=GAMMA, zeta=ZETA,
                              diversity=0.7,
                              potency=attack_potency(poisoned, refs, GLOBAL))
    assert abs(t["total"] - (t["damage"] + t["stealth"] + t["malformed"]
                             + t["collab"])) < 1e-12
    total = attacker_reward(0.90, 0.85, GOAL, [0, 1], _all_benign(), 1,
                            alpha=ALPHA, beta=BETA, gamma=GAMMA, zeta=ZETA,
                            diversity=0.7,
                            potency=attack_potency(poisoned, refs, GLOBAL))
    assert abs(total - t["total"]) < 1e-12
    assert t["malformed"] < 0.0                    # a wasted slot must cost


def test_ungated_when_no_potency_is_supplied():
    """potency=None keeps the old behaviour for external callers / older tests."""
    ungated = attacker_reward(0.90, 0.90, GOAL, [0], _all_benign(), 0,
                              alpha=ALPHA, beta=BETA, gamma=GAMMA)
    assert abs(ungated - BETA) < 1e-9      # full stealth, no gate applied


# --- 3. the defense-malfunction guard ----------------------------------------

def test_honest_majority_check():
    """Applied to the CLEAN counterfactual, where every update is honest, so any
    flag is a false positive."""
    assert _accepts_honest_majority([]) is True
    assert _accepts_honest_majority(_all_benign(20)) is True
    assert _accepts_honest_majority(_all_flagged(20)) is False
    # 10 of 19 rejected — the exact FLTrust case from the recorded run.
    v = ([DetectionVerdict(c, True, 1.0, "") for c in range(10)]
         + [DetectionVerdict(c, False, 1.0, "") for c in range(10, 19)])
    assert _accepts_honest_majority(v) is False
    # Exactly half rejected is still a malfunction (no strict majority kept).
    half = ([DetectionVerdict(c, True, 1.0, "") for c in range(10)]
            + [DetectionVerdict(c, False, 1.0, "") for c in range(10, 20)])
    assert _accepts_honest_majority(half) is False
    # One flag short of half is fine.
    ok = ([DetectionVerdict(c, True, 1.0, "") for c in range(9)]
          + [DetectionVerdict(c, False, 1.0, "") for c in range(9, 20)])
    assert _accepts_honest_majority(ok) is True


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} attack-potency tests passed.")


if __name__ == "__main__":
    _run()
