"""Verifiable, continuous reward functions for the RL policy.

The DEFENDER is the only learner: the attack is a deterministic, adaptive
label-flip schedule (:mod:`agents.label_flip_attacker`), not a policy. Its
reward is computed from ground truth (the per-round poisoned set and the
defender's verdicts), so it is exactly verifiable. It is deliberately
**continuous** (not 0/1): GRPO's advantage is the within-group reward spread,
which collapses to zero — and produces no gradient — if every sampled rollout
earns the same binary reward.

The defender reward uses ground truth, but only as a *training signal*: the
defender's policy input (its prompt) never contains it.

:func:`drop_term` and :func:`goal_target` remain for the ATTACK-EFFECTIVENESS
score — a reported measurement of how much accuracy the round's flipped labels
actually cost, normalized by the configured target. Nothing is trained on it;
it exists so the round logs, metrics and monitor can say how hard the current
ladder level is hitting the model.
"""

import logging

from core.types import DetectionVerdict

logger = logging.getLogger(__name__)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# The smallest accuracy drop worth calling an attack. On a 10k-example test set
# accuracy is quantized to 1e-4, so 1pp is ~100x the measurement quantum: it is
# unambiguously signal.
MIN_REAL_DROP = 0.01


# Upper bound of the drop term. Reaching the goal exactly scores 1.0; overshoot
# is worth at most another 0.5 (see :func:`drop_term`).
_OVERSHOOT_BONUS = 0.5
# Ratio (drop/target beyond 1.0) at which HALF the overshoot bonus is earned.
# Small = the bonus saturates fast, so the reward stays "hit the target", not
# "maximize damage"; strictly positive so the term never goes flat.
_OVERSHOOT_HALF = 1.0

# Where the linear region ends on the BACKFIRE side: an attack that improves the
# model by ``0.5 * target`` scores -0.5, as it always has.
_BACKFIRE_KNEE = -0.5
# How much worse than the knee a catastrophic backfire can score, and how fast it
# gets there. Mirrors the overshoot construction above (see :func:`drop_term`).
_BACKFIRE_EXTRA = 0.25
_BACKFIRE_HALF = 1.0


def attack_effectiveness(clean_accuracy: float, post_accuracy: float,
                         goal: dict) -> float:
    """How hard this round's flipped labels hit the model, on the goal's scale.

    ``drop_term(clean - post, goal_target(goal))``: 1.0 means the round achieved
    exactly the requested ``target_accuracy_drop``, 0.0 means it changed nothing,
    negative means the poisoned aggregate came out BETTER than the clean one.

    Reported only. Nothing is trained on this — the attack has no policy — but it
    is the single number that says whether the current ladder level is a real
    attack or a formality, so the round log, the metrics tracker and ``monitor.py``
    all read it. A run whose effectiveness sits near 0 at every ladder level is
    training the defender to detect an attack that does no damage.
    """
    return drop_term(clean_accuracy - post_accuracy, goal_target(goal))


def drop_term(drop: float, target: float) -> float:
    """Shape an achieved accuracy drop onto the goal's scale.

    ``x = drop / target``:

    * ``-0.5 <= x <= 1`` → ``x``: linear, and hitting the goal scores exactly 1.0.
    * ``x > 1``          → ``1 + 0.5·(x-1)/(x-1+1)``: strictly increasing, asymptotic
      to ``1.5``.
    * ``x < -0.5``       → ``-0.5 - 0.25·u/(u+1)`` with ``u = -0.5 - x``: strictly
      DEcreasing, asymptotic to ``-0.75``.

    Strictly monotonic everywhere rather than clipped flat at both ends, so two
    rounds whose damage differs always differ here too — the property that made it
    usable as a training signal, and that still makes it a readable measurement.
    """
    x = drop / target
    if x > 1.0:
        over = x - 1.0
        return 1.0 + _OVERSHOOT_BONUS * over / (over + _OVERSHOOT_HALF)
    if x >= _BACKFIRE_KNEE:
        return x
    under = _BACKFIRE_KNEE - x
    return _BACKFIRE_KNEE - _BACKFIRE_EXTRA * under / (under + _BACKFIRE_HALF)


def goal_target(goal: dict) -> float:
    """The target accuracy drop this goal asks for (>0).

    Single source of truth shared by :func:`attack_effectiveness` (which normalizes
    the drop by it) and the damage bar in ``rl/switch.py``, so the two never
    disagree about what the round's target is. ``slow_degrade`` uses
    ``per_round_drop``; ``untargeted_degrade`` (and, for now, ``targeted_label``,
    which falls back to overall accuracy until per-class eval is wired in) uses
    ``target_accuracy_drop``.
    """
    gtype = goal.get("type", "untargeted_degrade")
    if gtype == "slow_degrade":
        target = float(goal.get("per_round_drop", 0.02))
    else:
        target = float(goal.get("target_accuracy_drop", 0.20))
    return max(target, 1e-6)


def _soft_malicious_prob(v: DetectionVerdict) -> float:
    """Map a verdict to a soft P(malicious) in [0, 1].

    Prefers the verdict's explicitly calibrated ``p_malicious`` when the producer
    supplied one (every algorithmic defense does — see ``core.types``). Only when
    it is absent do we reconstruct the probability from ``(is_suspicious,
    confidence)``: confident flag → ~1, confident pass → ~0, unsure → ~0.5. That
    reconstruction is correct for the LLM defender, which is asked for its
    certainty in the label it just assigned.

    It is NOT correct for a threshold filter: the algorithmic defenses report a
    *suspicion score* whose decision boundary is not at 0.5, so ``0.5 - 0.5 * c``
    runs BACKWARDS over their un-flagged clients (a barely-trusted survivor scores
    as "confidently benign", a well-trusted one as "unsure"), and unbounded scores
    (Multi-Krum/DnC) clip to 1.0 and collapse the soft signal into a binary.

    ``p_malicious`` therefore carries a contract — ``p >= 0.5`` if and only if
    ``is_suspicious`` — enforced at the producers by
    ``benchmark.defenses.base.boundary_calibrated_p`` and asserted for every
    defense in ``tests/test_p_malicious_calibration.py``. Without it FLTrust
    reported ``p ~ 0.95`` for clients it had ACCEPTED (a cosine of 0.05 is normal on
    a small model) and DeFL reported ``p = 0.5`` for clients it had FLAGGED (its
    adaptive threshold fires at one layer-vote out of two) — both of which invert
    the soft-F1 the defender is trained on.
    """
    p = getattr(v, "p_malicious", None)
    if p is not None:
        return _clip(float(p), 0.0, 1.0)
    c = _clip(float(v.confidence), 0.0, 1.0)
    return 0.5 + 0.5 * c if v.is_suspicious else 0.5 - 0.5 * c


def defender_reward(
    verdicts: list[DetectionVerdict],
    poisoned_ids: list[int],
    *,
    mode: str = "soft_f1",
    fpr_penalty: float = 1.0,
) -> float:
    """Reward the defender for correctly identifying the poisoned clients.

    ``soft_f1`` (default): confidence-weighted soft F1 in [0, 1].
    ``tpr_minus_fpr``:     clip(TPR - fpr_penalty * FPR, 0, 1) using hard flags.

    **Clean rounds.** ``poisoned_ids`` can legitimately be empty — a ladder level
    that rounds to zero flipped labels (a tiny client shard at a low fraction)
    leaves every update the server receives genuinely honest, and the env
    correctly reports no poison (see ``FLArmsRaceEnv.begin_round``). F1 is
    undefined there and would score a perfectly-behaved defender 0, training it
    to invent detections. Instead we
    score the only thing that matters on a clean round: staying quiet. The result
    is ``1 - mean soft P(malicious)`` — 1.0 for a confident all-benign verdict,
    0.5 for maximal uncertainty, 0.0 for confidently flagging everyone — which
    stays continuous, stays in [0, 1], and agrees with soft-F1's direction.
    """
    poisoned = set(poisoned_ids)
    if not poisoned:
        if not verdicts:
            return 1.0
        mean_p = sum(_soft_malicious_prob(v) for v in verdicts) / len(verdicts)
        return _clip(1.0 - mean_p, 0.0, 1.0)

    if mode == "tpr_minus_fpr":
        tp = sum(1 for v in verdicts if v.client_id in poisoned and v.is_suspicious)
        fn = sum(1 for v in verdicts if v.client_id in poisoned and not v.is_suspicious)
        fp = sum(1 for v in verdicts if v.client_id not in poisoned and v.is_suspicious)
        tn = sum(1 for v in verdicts if v.client_id not in poisoned and not v.is_suspicious)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return _clip(tpr - fpr_penalty * fpr, 0.0, 1.0)

    # soft_f1
    eps = 1e-8
    tp = fp = fn = 0.0
    for v in verdicts:
        p = _soft_malicious_prob(v)
        if v.client_id in poisoned:
            tp += p
            fn += 1.0 - p
        else:
            fp += p
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


# Default noise floor for a group's reward spread. The defender reward is a soft
# F1 over ~20 clients, so two verdict sets that differ only in one client's
# confidence by a hundredth separate by well under this; anything below the floor
# is sampling wobble, not a real difference between the verdicts. See
# group_advantages.
DEFAULT_MIN_REWARD_SPREAD = 0.02
# Floor on the z-score denominator, in reward units. Stops a barely-separated
# group from being rescaled up to full-magnitude advantages. See group_advantages.
DEFAULT_ADVANTAGE_STD_FLOOR = 0.05


def group_advantages(
    rewards: list[float],
    *,
    min_spread: float = DEFAULT_MIN_REWARD_SPREAD,
    std_floor: float = DEFAULT_ADVANTAGE_STD_FLOOR,
) -> tuple[list[float], float]:
    """Group-relative normalized advantages plus the zero-advantage fraction.

    ``A_i = (r_i - mean) / max(std, std_floor)``.

    Two guards separate "the rollouts really differed" from "the measurement wobbled":

    * ``min_spread`` — the group is declared DEGENERATE (advantages all 0,
      zero-fraction 1.0, so ``grpo_step`` resamples or skips) unless
      ``max(r) - min(r) >= min_spread``. The previous test was ``std < 1e-6``,
      which is ~1000x smaller than the reward noise floor: two behaviourally
      identical rollouts that happened to differ by a rounding step produced a
      ~1e-3 reward gap, passed the gate, and were then z-scored up to ``A = ±1.2``
      and applied at full strength. GRPO was confidently reinforcing coin flips.
      A meaningful difference clears this bar easily (flagging one more poisoned
      client out of a handful moves soft F1 by far more).

    * ``std_floor`` — floors the denominator instead of dividing by the raw std.
      Plain z-scoring is scale-free: a group spread over 0.02 of reward and a group
      spread over 1.0 both come out at ``A = ±1.2``, so the update size carries no
      information about how much better the winning rollout actually was. With the
      floor, advantages stay proportional to real reward differences until the
      spread is genuinely large (std > std_floor), where this reduces to standard
      GRPO z-scoring. The defender reward spans [0, 1], so a healthy group — one
      whose verdicts genuinely disagree about who is poisoned — is unaffected.

    Pass ``min_spread=0.0, std_floor=0.0`` for textbook GRPO behaviour.
    """
    n = len(rewards)
    if n == 0:
        return [], 1.0
    eps = 1e-6
    spread = max(rewards) - min(rewards)
    if spread < max(float(min_spread), eps):
        return [0.0] * n, 1.0
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    denom = max(std, float(std_floor), eps)
    adv = [(r - mean) / denom for r in rewards]
    return adv, 0.0
