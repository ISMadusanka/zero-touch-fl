"""Verifiable, continuous reward functions for the two RL policies.

Both rewards are computed from ground truth (the per-round poisoned set and the
measured post-aggregation accuracy), so they are exactly verifiable. They are
deliberately **continuous** (not 0/1): GRPO's advantage is the within-group
reward spread, which collapses to zero — and produces no gradient — if every
sampled rollout earns the same binary reward. The tiny action space here makes
that failure mode acute, so we shape both rewards smoothly.

The defender reward uses ground truth, but only as a *training signal*: the
defender's policy input (its prompt) never contains it.
"""

from core.types import DetectionVerdict


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# Upper bound of the drop term. Reaching the goal exactly scores 1.0; overshoot
# is worth at most another 0.5 (see :func:`drop_term`).
_OVERSHOOT_BONUS = 0.5
# Ratio (drop/target beyond 1.0) at which HALF the overshoot bonus is earned.
# Small = the bonus saturates fast, so the reward stays "hit the target", not
# "maximize damage"; strictly positive so the term never goes flat.
_OVERSHOOT_HALF = 1.0


def drop_term(drop: float, target: float) -> float:
    """Shape the achieved accuracy drop into the reward's damage term.

    ``x = drop / target``:

    * ``x <= 1``  → ``x`` (linear up to the goal), floored at ``-0.5`` so a
      round that *improves* the model is bounded-bad.
    * ``x > 1``   → ``1 + 0.5·(x-1)/(x-1+1)``: strictly increasing, asymptotic to
      ``1.5``.

    The range and the value at the target are exactly what the previous hard
    ``clip(x, -0.5, 1.5)`` gave, so the objective ("cut accuracy by about
    ``target``") is unchanged. What changes is that the term is now **strictly
    monotonic above the goal instead of flat**. The flat region was a real
    training failure: once the policy reliably overshot ``1.5·target``, every
    rollout in a GRPO group scored an identical reward, the group's advantage
    spread collapsed to zero, and ``grpo_step`` skipped the update — the
    attacker stopped learning exactly when it got good. The saturation is
    deliberately fast (half the bonus by ``x = 2``) so gross overshoot is worth
    very little: enough to break ties, not enough to turn the goal into
    "destroy the model".
    """
    x = drop / target
    if x <= 1.0:
        return max(-0.5, x)
    over = x - 1.0
    return 1.0 + _OVERSHOOT_BONUS * over / (over + _OVERSHOOT_HALF)


def goal_target(goal: dict) -> float:
    """The target accuracy drop this goal asks for (>0).

    Single source of truth shared by the attacker reward (which normalizes the
    drop by it) and the schedule's relative win-gate (``rl/switch.py``) so the two
    never disagree about what the round's target is. ``slow_degrade`` uses
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


def attacker_reward(
    reference_accuracy: float,
    post_accuracy: float,
    goal: dict,
    poisoned_ids: list[int],
    verdicts: list[DetectionVerdict],
    n_malformed: int,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 1.0,
    zeta: float = 0.0,
    diversity: float | None = None,
) -> float:
    """Reward the attacker for degrading accuracy while staying stealthy and,
    when the exact quota contains several clients, coordinating their plans.

    reward = alpha * drop_term(drop, target)
           + beta  * stealth
           - gamma * malformed_fraction
           + zeta  * collab_bonus

    ``drop = reference_accuracy - post_accuracy``.

    ``reference_accuracy`` is the **clean counterfactual for THIS round**: the
    accuracy the aggregate would have had if nobody had poisoned it (see
    ``FLArmsRaceEnv.clean_reference_accuracy``). It is deliberately NOT the
    previous round's post-attack accuracy. With ``benign_retrain_each_round:
    false`` the environment is memoryless — each round's global is rebuilt from
    frozen benign weights plus this round's poison — so measuring against the
    previous round made ``drop`` a round-over-round *difference* rather than the
    damage this attack caused: repeating an identical, devastating attack scored
    high once and ~0 forever after, and the schedule's ``success_streak`` gate
    became unreachable. Against the clean counterfactual, an attack that hits its
    target scores the same every time it hits it, in either retrain mode.

    ``stealth`` is a CONTINUOUS evasion signal in [0, 1]: the mean over poisoned
    clients of ``1 - soft P(malicious)`` derived from the defender's *confidence*
    (not just the binary flag). Confidently caught -> ~0; confidently passed as
    benign -> ~1; unsure -> ~0.5. Because it moves smoothly with the defender's
    confidence, it gives GRPO a gradient even when every sampled plan ends up
    with the SAME hard flag. It is 0 when the attacker poisoned nobody — there
    was nothing to sneak past the defender.

    ``malformed_fraction`` is over the clients the attacker SELECTED
    (``n_used + n_malformed``), not over the ones that landed, so selecting three
    clients and wasting two is penalized twice as hard as wasting one.

    ``collab_bonus``
    = ``diversity`` in [0, 1] (only when >1 client) rewards distinct, coordinated
    per-client perturbations over identical clones — see ``perturbation_diversity``.
    """
    target = goal_target(goal)   # shared with the schedule's relative win-gate

    drop = reference_accuracy - post_accuracy
    damage = drop_term(drop, target)

    verdict_by_id = {v.client_id: v for v in verdicts}
    n_used = len(poisoned_ids)
    stealth = 0.0
    for cid in poisoned_ids:
        v = verdict_by_id.get(cid)
        # No verdict for a poisoned client => treat as undetected (passed).
        stealth += 1.0 if v is None else (1.0 - _soft_malicious_prob(v))
    stealth = stealth / n_used if n_used else 0.0

    # Normalize the waste penalty by how many clients were SELECTED — a client
    # whose plan did nothing still consumed budget, so it belongs in the
    # denominator even though it is not in ``poisoned_ids``.
    n_selected = n_used + max(0, int(n_malformed))
    malformed_fraction = n_malformed / n_selected if n_selected else 0.0

    # Collaboration bonus: reward diverse (coordinated) multi-client attacks. Only
    # meaningful with >1 client; `diversity` in [0, 1].
    collab_bonus = 0.0
    if zeta and n_used > 1 and diversity is not None:
        collab_bonus = _clip(float(diversity), 0.0, 1.0)

    return (alpha * damage + beta * stealth - gamma * malformed_fraction
            + zeta * collab_bonus)


def perturbation_diversity(poisoned_by_client: dict, references: dict) -> float:
    """1 − mean pairwise cosine of the chosen clients' perturbation vectors.

    Each client's perturbation is ``(poisoned - benign)`` flattened over all
    layers. Returns 0.0 for fewer than 2 clients (no collaboration to reward).
    Higher = the clients push the model in MORE different directions (distinct,
    coordinated roles rather than identical clones — which also look like
    colluding Sybils to the defender's ``max_pairwise_cos`` feature).
    """
    import torch

    cids = [cid for cid in poisoned_by_client if cid in references]
    if len(cids) < 2:
        return 0.0
    eps = 1e-8
    normed = []
    for cid in cids:
        pw, bw = poisoned_by_client[cid], references[cid]
        delta = torch.cat([
            (pw[k].flatten().float() - bw[k].flatten().float()) for k in bw
        ])
        normed.append(delta / (delta.norm() + eps))
    total, pairs = 0.0, 0
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            total += float(torch.dot(normed[i], normed[j]))
            pairs += 1
    mean_cos = total / pairs if pairs else 0.0
    return _clip(1.0 - mean_cos, 0.0, 1.0)


def _soft_malicious_prob(v: DetectionVerdict) -> float:
    """Map a verdict to a soft P(malicious) in [0, 1].

    Prefers the verdict's explicitly calibrated ``p_malicious`` when the producer
    supplied one (every algorithmic defense does — see ``core.types``). Only when
    it is absent do we reconstruct the probability from ``(is_suspicious,
    confidence)``: confident flag → ~1, confident pass → ~0, unsure → ~0.5. That
    reconstruction is correct for the LLM defender, which is asked for its
    certainty in the label it just assigned.

    It is NOT correct for a threshold filter, and relying on it was a bug: the
    algorithmic defenses report a *suspicion score* whose decision boundary is not
    at 0.5, so ``0.5 - 0.5 * c`` ran BACKWARDS over their un-flagged clients (a
    barely-trusted survivor scored as "confidently benign", a well-trusted one as
    "unsure"). Since the attacker's ``stealth`` term is ``1 - p`` averaged over
    the clients it poisoned, that paid the attacker for creeping up to the
    detection boundary rather than for looking honest — the exact opposite of the
    intended gradient. Unbounded scores (Multi-Krum/DnC) additionally clipped to
    1.0 and collapsed ``stealth`` to a binary, destroying the within-group spread
    the continuous reward exists to provide.
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

    **Clean rounds.** ``poisoned_ids`` can legitimately be empty — the attacker
    may select clients whose plans turn out to be no-ops, in which case every
    update the server receives really is honest (see
    ``AttackerAgent.select_and_apply``). F1 is undefined there and would score a
    perfectly-behaved defender 0, training it to invent detections. Instead we
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


# Default noise floor for a group's reward spread. The rewards are built from a
# test-set accuracy measured on 10k MNIST examples, so accuracy is quantized to
# 1e-4; divided by the smallest sampled ``target_accuracy_drop`` (0.05) that is
# ~2e-3 of reward per SINGLE flipped test example. Anything below this floor is
# measurement noise, not a difference between the plans. See group_advantages.
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

    Two guards separate "the plans really differed" from "the measurement wobbled":

    * ``min_spread`` — the group is declared DEGENERATE (advantages all 0,
      zero-fraction 1.0, so ``grpo_step`` resamples or skips) unless
      ``max(r) - min(r) >= min_spread``. The previous test was ``std < 1e-6``,
      which is ~1000x smaller than the reward noise floor: two behaviourally
      identical rollouts that happened to classify one test example differently
      produced a 2e-3 reward gap, passed the gate, and were then z-scored up to
      ``A = ±1.2`` and applied at full strength. GRPO was confidently reinforcing
      coin flips. A meaningful difference clears this bar easily (a 1% accuracy
      gap at the smallest target is 0.2 of reward).

    * ``std_floor`` — floors the denominator instead of dividing by the raw std.
      Plain z-scoring is scale-free: a group spread over 0.02 of reward and a group
      spread over 1.0 both come out at ``A = ±1.2``, so the update size carries no
      information about how much better the winning rollout actually was. With the
      floor, advantages stay proportional to real reward differences until the
      spread is genuinely large (std > std_floor), where this reduces to standard
      GRPO z-scoring. The attacker reward spans about [-1.8, 2.2], so a healthy
      group is unaffected.

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
