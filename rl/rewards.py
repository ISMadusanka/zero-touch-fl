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


# --- targeted_label defaults ------------------------------------------------
# How much of the target class's recall the attack is asked to destroy. 0.80 is
# effectively "wipe this class out": it is clamped to the class's CLEAN recall
# (see ``targeted_terms``), so on a class the honest model only gets 0.62 right,
# hitting the goal means driving it to ~0 — not achieving an impossible 0.80.
DEFAULT_TARGET_CLASS_DROP = 0.80
# How much mean recall the OTHER classes may lose before the attack stops being
# "targeted". 0.05 = five points of average collateral damage.
DEFAULT_MAX_COLLATERAL = 0.05
# Floor on the effective per-class target, so a class that is already broken in
# the clean model cannot make the normalizer explode.
_MIN_CLASS_TARGET = 0.05
# Cap on the collateral penalty in reward units. Without it, an indiscriminate
# "destroy everything" rollout produces an unbounded negative that swamps the
# GRPO group's reward spread and kills the gradient for every other rollout.
_MAX_COLLATERAL_COST = 3.0


def goal_target(goal: dict) -> float:
    """The target drop this goal asks for (>0).

    Single source of truth shared by the attacker reward (which normalizes the
    drop by it) and the schedule's relative win-gate (``rl/switch.py``) so the two
    never disagree about what the round's target is.

    * ``slow_degrade``        → ``per_round_drop``   (overall accuracy)
    * ``untargeted_degrade``  → ``target_accuracy_drop`` (overall accuracy)
    * ``targeted_label``      → ``target_class_drop`` (recall of ONE class)

    For ``targeted_label`` this is the *requested* drop; the drop actually used to
    normalize is clamped to the class's clean recall by :func:`targeted_terms`.
    """
    gtype = goal.get("type", "untargeted_degrade")
    if gtype == "slow_degrade":
        target = float(goal.get("per_round_drop", 0.02))
    elif gtype == "targeted_label":
        target = float(goal.get("target_class_drop", DEFAULT_TARGET_CLASS_DROP))
    else:
        target = float(goal.get("target_accuracy_drop", 0.20))
    return max(target, 1e-6)


def goal_label(goal: dict | None) -> int | None:
    """The class a ``targeted_label`` goal attacks, or ``None`` for other goals."""
    if not goal or goal.get("type") != "targeted_label":
        return None
    try:
        return int(goal.get("label", 0))
    except (TypeError, ValueError):
        return None


def targeted_terms(goal: dict | None, clean_eval, post_eval) -> dict | None:
    """Decompose a targeted round into the numbers everything downstream needs.

    Returns ``None`` unless this is a ``targeted_label`` goal AND both per-class
    evaluations are available (so every non-targeted caller keeps its old
    behaviour untouched). Otherwise a dict:

    ``label``            the class under attack
    ``clean_recall``     its recall in the round's CLEAN counterfactual
    ``post_recall``      its recall after the attack
    ``target_drop``      ``clean_recall - post_recall`` — what the attack achieved
    ``effective_target`` the drop that counts as "goal met" (see below)
    ``collateral``       mean recall LOST across the other classes (>= 0)
    ``others_clean`` / ``others_post``  mean recall of the other classes, for logs

    **effective_target** is ``min(requested, clean_recall)`` floored at
    ``_MIN_CLASS_TARGET``. A class the honest model only gets 62% right cannot
    lose 80 points of recall, so without the clamp that label would be unwinnable
    and the reward would be incomparable between labels — exactly the failure mode
    that matters when one run trains over several labels. With the clamp, "hit the
    target" means the same thing ("this class is destroyed") for every label.

    **collateral** sums only per-class LOSSES (``max(0, clean-post)``) and averages
    over the other classes. Clipping at zero stops a class that happened to improve
    from cancelling out one that was destroyed.
    """
    label = goal_label(goal)
    if label is None or clean_eval is None or post_eval is None:
        return None
    clean_recall = clean_eval.recall(label)
    post_recall = post_eval.recall(label)
    n = min(len(clean_eval.per_class), len(post_eval.per_class))
    losses = [max(0.0, clean_eval.per_class[c] - post_eval.per_class[c])
              for c in range(n) if c != label]
    collateral = sum(losses) / len(losses) if losses else 0.0
    return {
        "label": label,
        "clean_recall": clean_recall,
        "post_recall": post_recall,
        "target_drop": clean_recall - post_recall,
        "effective_target": max(min(goal_target(goal), clean_recall), _MIN_CLASS_TARGET),
        "collateral": collateral,
        "max_collateral": max(float(goal.get("max_collateral", DEFAULT_MAX_COLLATERAL)), 1e-6),
        "others_clean": clean_eval.others_mean(label),
        "others_post": post_eval.others_mean(label),
    }


def goal_drop(goal: dict | None, reference_accuracy: float, post_accuracy: float,
              clean_eval=None, post_eval=None) -> float:
    """The drop the GOAL is judged on — the single number the win-gate compares.

    Targeted rounds are judged on the TARGET CLASS's recall drop; every other goal
    on the overall accuracy drop. Keeping this in one place stops the reward and
    the schedule from measuring different things.
    """
    terms = targeted_terms(goal, clean_eval, post_eval)
    if terms is not None:
        return terms["target_drop"]
    return reference_accuracy - post_accuracy


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
    delta: float = 0.0,
    zeta: float = 0.0,
    eta: float = 1.0,
    pool_size: int | None = None,
    diversity: float | None = None,
    clean_eval=None,
    post_eval=None,
) -> float:
    """Reward the attacker for degrading accuracy while staying stealthy, using
    the FEWEST clients, and (when it uses several) collaborating with them.

    reward = alpha * drop_term(drop, target)
           + beta  * stealth
           - gamma * malformed_fraction
           - delta * client_cost
           + zeta  * collab_bonus
           - eta   * collateral_cost      (targeted_label only)

    ``drop = reference_accuracy - post_accuracy``.

    **Targeted mode.** When the goal is ``targeted_label`` AND per-class
    evaluations are supplied (``clean_eval`` / ``post_eval`` —
    ``core.types.ClassEval`` from ``FLArmsRaceEnv.clean_reference_eval`` and
    ``evaluate_updates_full``), two things change and nothing else does:

    * ``drop`` becomes the TARGET CLASS's recall drop instead of the overall
      accuracy drop, normalized by that class's clamped target
      (:func:`targeted_terms`), and
    * a new ``collateral_cost`` term charges the mean recall the OTHER classes
      lost, in units of the goal's ``max_collateral`` tolerance, capped at
      ``_MAX_COLLATERAL_COST``.

    That second term is what makes the attack *targeted* rather than merely
    destructive. Without it the cheapest way to crush class 2's recall is to crush
    every class, which scores the same on the first term — so the policy would
    simply rediscover the untargeted attack. With it, an indiscriminate rollout
    pays roughly ``-eta * _MAX_COLLATERAL_COST`` while a surgical one pays almost
    nothing, and GRPO's group-relative advantage does the rest.

    When the goal is not ``targeted_label`` (or the evaluations are absent, e.g.
    the CPU baseline harness) the terms are computed exactly as before, so the
    untargeted experiment is bit-for-bit unchanged.

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

    ``client_cost`` = ``(n_used - 1) / (pool_size - 1)`` in [0, 1] penalizes using
    more of the controllable pool than necessary (0 for a single client), so the
    attacker learns to achieve the goal with the fewest clients. ``collab_bonus``
    = ``diversity`` in [0, 1] (only when >1 client) rewards distinct, coordinated
    per-client perturbations over identical clones — see ``perturbation_diversity``.
    """
    # Targeted goals score the target CLASS's recall drop and pay for collateral
    # damage; everything else scores the overall accuracy drop and pays nothing.
    terms = targeted_terms(goal, clean_eval, post_eval)
    if terms is not None:
        damage = drop_term(terms["target_drop"], terms["effective_target"])
        collateral_cost = _clip(terms["collateral"] / terms["max_collateral"],
                                0.0, _MAX_COLLATERAL_COST)
    else:
        # shared with the schedule's relative win-gate
        damage = drop_term(reference_accuracy - post_accuracy, goal_target(goal))
        collateral_cost = 0.0

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

    # Minimal-clients penalty: using more of the controllable pool than needed is
    # costly. Normalized to [0, 1] by the pool size so it is budget-independent.
    client_cost = 0.0
    if pool_size and pool_size > 1 and n_used > 1:
        client_cost = _clip((n_used - 1) / (pool_size - 1), 0.0, 1.0)

    # Collaboration bonus: reward diverse (coordinated) multi-client attacks. Only
    # meaningful with >1 client; `diversity` in [0, 1].
    collab_bonus = 0.0
    if zeta and n_used > 1 and diversity is not None:
        collab_bonus = _clip(float(diversity), 0.0, 1.0)

    return (alpha * damage + beta * stealth - gamma * malformed_fraction
            - delta * client_cost + zeta * collab_bonus - eta * collateral_cost)


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
    """Map a verdict to a soft P(malicious) in [0, 1] using its confidence.

    Confident flag → ~1, confident pass → ~0, unsure → ~0.5.
    """
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


def group_advantages(rewards: list[float]) -> tuple[list[float], float]:
    """Group-relative normalized advantages plus the zero-advantage fraction.

    A_i = (r_i - mean) / (std + eps). When the group has (near-)zero spread the
    advantages are ~0 → no learning signal; we report that fraction so the
    schedule can surface a stalled-gradient warning.
    """
    n = len(rewards)
    if n == 0:
        return [], 1.0
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    eps = 1e-6
    if std < eps:
        return [0.0] * n, 1.0
    adv = [(r - mean) / (std + eps) for r in rewards]
    return adv, 0.0
