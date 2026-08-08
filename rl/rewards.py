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


# Positive auxiliary shaping (stealth and collaboration) is capped to this
# fraction of the achieved damage component. Damage therefore remains the
# largest positive category even with aggressive auxiliary weights.
_MAX_AUXILIARY_TO_DAMAGE = 0.75


def drop_term(drop: float, target: float) -> float:
    """Shape the achieved accuracy drop into the reward's damage term.

    ``x = drop / target``:

    The term is linear until the requested drop is reached and then saturates:
    ``clip(drop / target, -0.5, 1.0)``. This implements an *at least* target:
    there remains a considerable gradient throughout the sub-target region,
    while damage beyond the requested threshold is neither discouraged nor
    rewarded as an invitation to destroy the model further.
    """
    return _clip(drop / target, -0.5, 1.0)


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


def attacker_reward_breakdown(
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
    perturbation_ratios: dict | None = None,
    stealth_floor: float = 1.0,
) -> dict[str, float]:
    """Return the attacker's reward and its verifiable scalar components.

    The ``total`` entry is the reward used for training. The remaining entries
    make the damage, gates, cap, and final weighted contributions observable to
    logs and tests without duplicating this shaping logic.

    reward = alpha * drop_term(drop, target)
           + capped(progress * (beta * stealth + zeta * collab_bonus))
           - gamma * malformed_fraction

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

    ``stealth`` is a CONTINUOUS evasion signal in [0, 1], but only for updates
    that actually pass the defender's hard decision. A suspicious verdict gives
    that client exactly zero stealth even if its soft probability is inconsistent
    with the hard flag. Among hard survivors, ``1 - soft P(malicious)`` preserves
    a useful confidence gradient. Missing verdicts still mean the update was not
    rejected and are treated as hard survivors.

    **Stealth has to be earned.** Passing as benign is only an achievement if
    there was something to hide. Left ungated, ``beta * stealth`` is a guaranteed
    payout for submitting an update that is a rounding error away from the honest
    one — no risk, no detection, and (because ``drop_term`` is near 0 and hard to
    move against a robust aggregator that rescales magnitudes and drops outliers)
    strictly better than trying. That is a genuine local optimum and policies fall
    into it: they converge on perturbations SMALLER than the honest client-to-client
    spread, which every robust aggregator then ranks as more central than a real
    client, so the attack survives every filter and costs the model nothing. The
    zero-advantage guard then freezes them there, because once every rollout is
    equally harmless the group spread collapses and ``grpo_step`` stops stepping.

    ``perturbation_ratios`` closes it: ``{client_id: ‖poisoned − benign‖ / ‖benign
    − global‖}`` (see ``agents.attack_ops.perturbation_size``), i.e. each edit
    measured against the honest update it replaces. A client's stealth is scaled by
    ``min(1, ratio / stealth_floor)``, so full credit needs an edit at least
    ``stealth_floor`` times the honest update — the same scale the attacker's own
    prompt is calibrated in. The term stays continuous in BOTH inputs (detector
    confidence and attack size), so the gradient it exists to provide survives; what
    disappears is the payout for doing nothing. Omit ``perturbation_ratios`` (or set
    ``stealth_floor <= 0``) to restore the ungated behaviour.

    ``malformed_fraction`` is over the clients the attacker SELECTED
    (``n_used + n_malformed``), not over the ones that landed, so selecting three
    clients and wasting two is penalized twice as hard as wasting one.

    ``collab_bonus`` is ``diversity`` in [0, 1] only when at least two poisoned
    clients hard-survive, scaled by the surviving fraction. See
    ``perturbation_diversity``.

    Both positive auxiliary terms are multiplied by damage progress
    ``clip(drop / target, 0, 1)``. Thus zero or negative induced damage earns no
    positive payout for merely being benign-looking or diverse. Their combined
    positive contribution is additionally capped below the positive damage
    component, keeping the optimization damage-dominant even when ``beta`` or
    ``zeta`` is configured aggressively. Negative auxiliary weights, if supplied,
    remain penalties rather than being clipped away.
    """
    target = goal_target(goal)   # shared with the schedule's relative win-gate

    drop = reference_accuracy - post_accuracy
    damage = drop_term(drop, target)
    damage_progress = _clip(drop / target, 0.0, 1.0)
    damage_component = alpha * damage

    verdict_by_id = {v.client_id: v for v in verdicts}
    n_used = len(poisoned_ids)
    stealth = 0.0
    n_hard_survivors = 0
    for cid in poisoned_ids:
        v = verdict_by_id.get(cid)
        # No verdict means the update was not rejected. A hard rejection is an
        # absolute gate: a contradictory low p_malicious cannot earn stealth.
        hard_survived = v is None or not v.is_suspicious
        if not hard_survived:
            continue
        n_hard_survivors += 1
        soft_evasion = (1.0 if v is None else
                        1.0 - soft_malicious_probability(v))
        stealth += soft_evasion * _stealth_credit(
            cid, perturbation_ratios, stealth_floor)
    stealth = stealth / n_used if n_used else 0.0
    hard_survival_fraction = n_hard_survivors / n_used if n_used else 0.0

    # Normalize the waste penalty by how many clients were SELECTED — a client
    # whose plan did nothing still consumed budget, so it belongs in the
    # denominator even though it is not in ``poisoned_ids``.
    n_selected = n_used + max(0, int(n_malformed))
    malformed_fraction = n_malformed / n_selected if n_selected else 0.0

    # Collaboration only counts when at least two edited updates really survive.
    # The supplied diversity covers all poisoners, so scale it by their hard
    # survival rate instead of paying in full for a mostly rejected coalition.
    collab_bonus = 0.0
    if zeta and n_hard_survivors > 1 and diversity is not None:
        collab_bonus = (_clip(float(diversity), 0.0, 1.0)
                        * hard_survival_fraction)

    raw_stealth_component = damage_progress * beta * stealth
    raw_collaboration_component = damage_progress * zeta * collab_bonus
    positive_auxiliary = (max(0.0, raw_stealth_component)
                          + max(0.0, raw_collaboration_component))
    auxiliary_cap = _MAX_AUXILIARY_TO_DAMAGE * max(0.0, damage_component)
    auxiliary_scale = (min(1.0, auxiliary_cap / positive_auxiliary)
                       if positive_auxiliary > 0.0 else 1.0)

    # Only positive shaping is capped. A caller-supplied negative beta/zeta is a
    # deliberate penalty and must not be weakened by the positive-reward cap.
    stealth_component = (raw_stealth_component * auxiliary_scale
                         if raw_stealth_component > 0.0
                         else raw_stealth_component)
    collaboration_component = (raw_collaboration_component * auxiliary_scale
                                if raw_collaboration_component > 0.0
                                else raw_collaboration_component)
    malformed_component = -gamma * malformed_fraction
    auxiliary_component = stealth_component + collaboration_component
    total = damage_component + auxiliary_component + malformed_component

    return {
        "induced_drop": float(drop),
        "target_drop": float(target),
        "damage": float(damage),
        "damage_progress": float(damage_progress),
        "hard_survival_fraction": float(hard_survival_fraction),
        "stealth": float(stealth),
        "malformed_fraction": float(malformed_fraction),
        "collaboration": float(collab_bonus),
        "damage_component": float(damage_component),
        "stealth_component_uncapped": float(raw_stealth_component),
        "collaboration_component_uncapped": float(raw_collaboration_component),
        "positive_auxiliary_cap": float(auxiliary_cap),
        "positive_auxiliary_scale": float(auxiliary_scale),
        "stealth_component": float(stealth_component),
        "malformed_component": float(malformed_component),
        "collaboration_component": float(collaboration_component),
        "auxiliary_component": float(auxiliary_component),
        "total": float(total),
    }


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
    perturbation_ratios: dict | None = None,
    stealth_floor: float = 1.0,
) -> float:
    """Return the scalar attacker reward (see :func:`attacker_reward_breakdown`)."""
    return attacker_reward_breakdown(
        reference_accuracy,
        post_accuracy,
        goal,
        poisoned_ids,
        verdicts,
        n_malformed,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        zeta=zeta,
        diversity=diversity,
        perturbation_ratios=perturbation_ratios,
        stealth_floor=stealth_floor,
    )["total"]


def _stealth_credit(client_id, perturbation_ratios: dict | None,
                    stealth_floor: float) -> float:
    """How much of this client's evasion counts, in [0, 1].

    ``1.0`` (ungated) when no ratio is available or the gate is switched off, so
    callers that cannot measure the perturbation — and the legacy call signature —
    behave exactly as before. Otherwise ``min(1, ratio / stealth_floor)``: an edit
    at or above ``stealth_floor`` times the honest update earns full credit, a
    tenth of that earns a tenth. A non-finite ratio (the honest update was zero, so
    ANY edit is infinitely larger than it) earns full credit.
    """
    if not perturbation_ratios or stealth_floor <= 0.0:
        return 1.0
    ratio = perturbation_ratios.get(client_id)
    if ratio is None:
        return 1.0
    ratio = float(ratio)
    if ratio != ratio:                       # NaN: undefined, do not pay for it
        return 0.0
    if ratio == float("inf"):
        return 1.0
    return _clip(ratio / float(stealth_floor), 0.0, 1.0)


def perturbation_diversity(poisoned_by_client: dict, references: dict) -> float:
    """Mean pairwise ``1 - abs(cosine)`` of non-zero perturbations.

    Each client's perturbation is ``(poisoned - benign)`` flattened over all
    layers. Zero edits are excluded and fewer than two real perturbations return
    0.0 (there is no collaboration to reward). Orthogonal roles score 1.0;
    identical vectors score 0.0. Crucially, anti-parallel vectors also score
    0.0: equal-and-opposite updates cancel in aggregation and must not receive
    the maximum diversity reward merely because their signed cosine is -1.
    """
    import torch

    eps = 1e-8
    normed = []
    for cid in poisoned_by_client:
        if cid not in references:
            continue
        pw, bw = poisoned_by_client[cid], references[cid]
        delta = torch.cat([
            (pw[k].flatten().float() - bw[k].flatten().float()) for k in bw
        ])
        norm = delta.norm()
        if not bool(torch.isfinite(norm)) or float(norm) <= eps:
            continue
        normed.append(delta / norm)
    if len(normed) < 2:
        return 0.0

    total, pairs = 0.0, 0
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            cosine = float(torch.dot(normed[i], normed[j]))
            total += _clip(1.0 - abs(cosine), 0.0, 1.0)
            pairs += 1
    return total / pairs if pairs else 0.0


def soft_malicious_probability(v: DetectionVerdict) -> float:
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


def _soft_malicious_prob(v: DetectionVerdict) -> float:
    """Backward-compatible private alias for :func:`soft_malicious_probability`."""
    return soft_malicious_probability(v)


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
        mean_p = sum(soft_malicious_probability(v) for v in verdicts) / len(verdicts)
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
        p = soft_malicious_probability(v)
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
