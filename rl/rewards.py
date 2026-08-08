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

import logging

from core.types import DetectionVerdict

logger = logging.getLogger(__name__)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# The smallest accuracy drop worth calling an attack. On a 10k-example test set
# accuracy is quantized to 1e-4, so 1pp is ~100x the measurement quantum: it is
# unambiguously signal. Used as the anchor for the shaping-budget invariant below.
MIN_REAL_DROP = 0.01


def check_reward_balance(reward_cfg: dict, goal: dict, *, context: str = "") -> bool:
    """Log the attacker's reward weights and verify the shaping-budget invariant.

        beta + zeta  <  alpha * MIN_REAL_DROP / target

    Damage is the objective; stealth and collaboration are shaping that exists only
    to keep the gradient dense while damage is near zero. If their combined ceiling
    reaches what a real attack earns, the reward-maximizing policy is to look busy
    and harmless — which is what a recorded 262-round run learned to do, ending with
    a median induced drop of 0.0000 and a global model 0.097 accuracy BETTER than it
    started.

    Called at the start of every run mode, because the invariant couples four values
    across two config blocks (``attack.goal.target_accuracy_drop`` and the three
    ``rl.reward.attacker`` weights) and has been broken twice by changing one of them
    alone. Returns True when it holds; logs a WARNING naming the fix when it does not
    — this is deliberately loud rather than fatal, since an intentionally unbalanced
    reward is a legitimate experiment.
    """
    alpha = float(reward_cfg.get("alpha", 1.0))
    beta = float(reward_cfg.get("beta", 0.5))
    gamma = float(reward_cfg.get("gamma", 1.0))
    zeta = float(reward_cfg.get("zeta", 0.0))
    target = goal_target(goal)
    shaping = beta + zeta
    real_damage = alpha * MIN_REAL_DROP / target
    tag = f"[{context}] " if context else ""
    logger.info(
        "%sAttacker reward: alpha=%g beta=%g gamma=%g zeta=%g vs target=%g "
        "-> a %g drop pays %.3f, shaping (beta+zeta) tops out at %.3f",
        tag, alpha, beta, gamma, zeta, target, MIN_REAL_DROP, real_damage, shaping,
    )
    if shaping < real_damage:
        return True
    logger.warning(
        "%sREWARD IMBALANCE: shaping (beta+zeta=%.3f) is >= what a %g accuracy drop "
        "earns (alpha*drop/target=%.3f), so a rollout that evades without doing any "
        "damage can score as well as one that attacks. This is the failure that made "
        "an earlier run converge on 'perturb as little as possible while still "
        "changing a byte'. Fix by lowering rl.reward.attacker.beta/zeta, or by "
        "raising alpha in step with attack.goal.target_accuracy_drop (keep "
        "alpha/target constant).",
        tag, shaping, MIN_REAL_DROP, real_damage,
    )
    return False


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


def drop_term(drop: float, target: float) -> float:
    """Shape the achieved accuracy drop into the reward's damage term.

    ``x = drop / target``:

    * ``-0.5 <= x <= 1`` → ``x``: linear, and hitting the goal scores exactly 1.0.
    * ``x > 1``          → ``1 + 0.5·(x-1)/(x-1+1)``: strictly increasing, asymptotic
      to ``1.5``.
    * ``x < -0.5``       → ``-0.5 - 0.25·u/(u+1)`` with ``u = -0.5 - x``: strictly
      DEcreasing, asymptotic to ``-0.75``.

    The value at the target and the whole linear region are exactly what the original
    hard ``clip(x, -0.5, 1.5)`` gave, so the objective ("cut accuracy by about
    ``target``") is unchanged. What changes is that the term is **strictly monotonic
    everywhere instead of flat at both ends**.

    Flat regions are a specific, observed training failure, because GRPO's advantage
    *is* the within-group reward spread: once every rollout lands in the same flat
    region they all score identically, the spread collapses, and ``grpo_step`` skips
    the update. The overshoot flat came first — the attacker stopped learning exactly
    when it got good at overshooting ``1.5·target``. The backfire flat matters once
    ``target`` is small: at ``target = 0.02`` (the configured value) ``x < -0.5`` means
    "the attack made the model more than 1pp BETTER than the clean counterfactual",
    which is not exotic — several rounds in a recorded run were well past it, and every
    such rollout in a group used to score exactly ``-0.5``.

    Both saturations are deliberately fast and small, so the extremes are worth just
    enough to break ties: gross overshoot does not turn the goal into "destroy the
    model", and a catastrophic backfire stays bounded-bad rather than dominating.
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
    potency: float | None = None,
) -> float:
    """Total attacker reward — the sum of :func:`attacker_reward_terms`.

    See that function for the per-term decomposition (and use it instead when you
    want to report or assert on *which* term paid, which is the only way to catch
    the failure mode described below before it has cost you a training run).
    """
    return attacker_reward_terms(
        reference_accuracy, post_accuracy, goal, poisoned_ids, verdicts, n_malformed,
        alpha=alpha, beta=beta, gamma=gamma, zeta=zeta,
        diversity=diversity, potency=potency,
    )["total"]


def attacker_reward_terms(
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
    potency: float | None = None,
) -> dict:
    """Reward the attacker for degrading accuracy while staying stealthy and,
    when the exact quota contains several clients, coordinating their plans.

    Returns the decomposition ``{damage, stealth, malformed, collab, total, drop,
    stealth_raw, potency}`` — every weighted term plus the raw inputs — so a caller
    can log WHICH term paid. ``total`` is what :func:`attacker_reward` returns.

    reward = alpha * drop_term(drop, target)      <- the objective
           + beta  * stealth * potency            <- shaping, budget-capped
           - gamma * malformed_fraction           <- penalty
           + zeta  * diversity * stealth * potency <- shaping, budget-capped

    **Damage is the objective; stealth and collaboration are shaping.** Nothing here
    is an end in itself except the drop, so the shaping budget ``beta + zeta`` is
    deliberately held BELOW what a real attack earns: at the shipped
    ``alpha 5.0 / target 0.10``, a 1pp drop pays 0.5 and the entire shaping budget is
    0.15, so any genuine damage outbids everything a non-damaging rollout can collect.

    That bound is the actual fix for the failure below, and it is easy to lose. It
    was lost once by moving ``target`` (which divides damage) without moving
    ``alpha``; and gating stealth on ``potency`` alone did NOT restore it — a large,
    evading, completely harmless perturbation still scored ``beta * 0.5 = 0.245``
    against the 0.5 a real 1pp drop pays, i.e. roughly "half a point of accuracy" for
    doing no damage whatsoever. Keep the invariant explicit:

        beta + zeta  <  alpha * (smallest drop worth calling an attack) / target

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

    ``potency`` in [0, 1] is how much poison the attack actually put on the wire,
    relative to the honest update it hides inside (see :func:`attack_potency`). It
    GATES the stealth payment, and this gate is load-bearing: without it ``beta *
    stealth`` is an unconditional payment for not being caught, and the cheapest way
    to not be caught is to not attack. A poison one part in a thousand of the honest
    update is byte-different (so it escapes the ``n_malformed`` no-op check) and
    scores stealth ~1.0 while doing no damage at all.

    That is not a hypothetical. Over a recorded 262-round run the damage term
    supplied ~11% of the attacker's reward and ``beta * stealth`` supplied the rest
    (mean stealth 0.83 on FLTrust rounds, 0.91 on Multi-Krum); the median induced
    drop was 0.0000, the max 0.0121 against a 0.02 target, and the global model got
    0.097 BETTER over the run it was supposedly being attacked through. The policy
    had correctly maximized the reward it was given — "perturb as little as possible
    while still changing a byte" — which is the exact inverse of the objective.

    Gating stealth on potency removes that optimum without introducing another: a
    huge perturbation drives potency to 1 but is flagged, so stealth (and with it
    the whole term) collapses to ~0, and a flagged client is dropped from the
    aggregate so it earns no damage either. Both degenerate extremes now score ~0
    and only "big enough to hurt, subtle enough to pass" pays. This is the same
    argument that already gates the collaboration bonus on stealth — a term you can
    max out without attacking is a term the policy will farm instead of attacking.

    ``malformed_fraction`` is over the clients the attacker SELECTED
    (``n_used + n_malformed``), not over the ones that landed, so selecting three
    clients and wasting two is penalized twice as hard as wasting one. It is NOT
    gated: unusable output must cost whatever else happened.

    ``collab_bonus``
    = ``diversity * stealth`` in [0, 1] (only when >1 client) rewards distinct,
    coordinated per-client perturbations over identical clones — see
    ``perturbation_diversity``.

    The ``stealth`` factor is what keeps the bonus honest. Diversity is almost free
    to maximize — emit a distinct plan per client and it goes to ~1.0 whether or not
    the attack does anything — so as a standalone additive term it was a flat payment
    for formatting. On one recorded round the attacker collected 0.197 of collaboration
    bonus while every poisoned client was detected and the model got 0.0007 *better*;
    that single term was most of the round's positive reward. Gating it on evasion also
    matches the reason diversity is worth rewarding in the first place: identical clones
    are what the defender's ``max_pairwise_cos`` feature is looking for, so coordination
    only has value in an attack that survives detection.
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

    # Evading a defender is only worth paying for in proportion to how much poison
    # there was to evade WITH. ``potency=None`` (no measurement available) keeps the
    # old ungated behaviour so external callers and unit tests are unaffected.
    gate = 1.0 if potency is None else _clip(float(potency), 0.0, 1.0)
    stealth_raw = stealth
    stealth = stealth * gate

    # Normalize the waste penalty by how many clients were SELECTED — a client
    # whose plan did nothing still consumed budget, so it belongs in the
    # denominator even though it is not in ``poisoned_ids``.
    n_selected = n_used + max(0, int(n_malformed))
    malformed_fraction = n_malformed / n_selected if n_selected else 0.0

    # Collaboration bonus: reward diverse (coordinated) multi-client attacks. Only
    # meaningful with >1 client; `diversity` in [0, 1]. Scaled by the gated stealth
    # so a well-coordinated attack that got caught — or that never really attacked —
    # earns nothing for its coordination. See the docstring.
    collab_bonus = 0.0
    if zeta and n_used > 1 and diversity is not None:
        collab_bonus = _clip(float(diversity), 0.0, 1.0) * stealth

    terms = {
        "damage": alpha * damage,
        "stealth": beta * stealth,
        "malformed": -gamma * malformed_fraction,
        "collab": zeta * collab_bonus,
        # Raw inputs, so a log line can show what drove each weighted term.
        "drop": drop,
        "stealth_raw": stealth_raw,
        "potency": gate,
    }
    terms["total"] = terms["damage"] + terms["stealth"] + terms["malformed"] + terms["collab"]
    return terms


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


def attack_potency(poisoned_by_client: dict, references: dict,
                   global_weights: dict) -> float:
    """How much poison the attack actually put on the wire, in [0, 1).

    Per poisoned client, the poison and the honest update it is hidden inside are
    measured in the same units and divided:

        s_i = ‖poisoned_i − benign_i‖ / ‖benign_i − global‖

    ``s`` is the natural scale for this because it is exactly what the defenses
    look at — every one of them scores a client on its update ``Δ = w_i − global``,
    so ``s`` says how far the submitted Δ was displaced as a fraction of the honest
    Δ. It needs no absolute magnitude and is therefore architecture- and
    round-independent, like the statistics the attacker is shown.

    The per-client score is squashed ``s / (s + 1)``: 0 poison → 0, poison the size
    of the honest update → 0.5, 3× → 0.75, asymptotic to 1. Smooth and strictly
    increasing everywhere, so it never creates a flat region for GRPO to collapse
    into (the same requirement :func:`drop_term` is built around). Returns the mean
    over the poisoned clients, and 0.0 when nothing was poisoned.

    This exists to gate the stealth term in :func:`attacker_reward` — see its
    docstring for the failure it prevents.
    """
    import torch

    cids = [cid for cid in poisoned_by_client if cid in references]
    if not cids:
        return 0.0
    eps = 1e-12
    scores = []
    for cid in cids:
        pw, bw = poisoned_by_client[cid], references[cid]
        keys = [k for k in bw if k in global_weights and k in pw]
        if not keys:
            continue
        poison = torch.cat([(pw[k].flatten().float() - bw[k].flatten().float())
                            for k in keys])
        honest = torch.cat([(bw[k].flatten().float() - global_weights[k].flatten().float())
                            for k in keys])
        s = float(poison.norm()) / (float(honest.norm()) + eps)
        scores.append(s / (s + 1.0))
    return sum(scores) / len(scores) if scores else 0.0


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

    Moving the score into ``p_malicious`` did not by itself fix that: a raw score in
    the right FIELD is still a raw score. ``p_malicious`` therefore carries a
    contract — ``p >= 0.5`` if and only if ``is_suspicious`` — enforced at the
    producers by ``benchmark.defenses.base.boundary_calibrated_p`` and asserted for
    every defense in ``tests/test_p_malicious_calibration.py``. Without it FLTrust
    reported ``p ~ 0.95`` for clients it had ACCEPTED (a cosine of 0.05 is normal on a
    small model) and DeFL reported ``p = 0.5`` for clients it had FLAGGED (its
    adaptive threshold fires at one layer-vote out of two), so ``stealth`` was worth
    ~0.03 of its 0.5 weight on FLTrust rounds and paid 0.25 for being caught on DeFL
    rounds.
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
      GRPO z-scoring. The attacker reward spans about [-1.3, 2.2], so a healthy
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
