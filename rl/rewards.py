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


def attacker_reward(
    prev_accuracy: float,
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
    pool_size: int | None = None,
    diversity: float | None = None,
    # --- Per-class accuracy (targeted attacks) ---
    prev_class_acc: dict | None = None,
    post_class_acc: dict | None = None,
    baseline_class_acc: dict | None = None,
) -> float:
    """Reward the attacker for degrading accuracy while staying stealthy, using
    the FEWEST clients, and (when it uses several) collaborating with them.

    reward = alpha * drop_term
           + beta  * stealth
           - gamma * malformed_fraction
           - delta * client_cost
           + zeta  * collab_bonus

    For ``targeted_label`` with per-class data available, the ``drop_term`` is
    computed from the most-damaged class accuracy (not global accuracy), and a
    **stealth penalty** is applied if the global accuracy drops beyond
    ``max_global_drop``.  This dual-objective formulation forces the LLM to
    learn precision: destroy one class while keeping the model healthy overall.

    ``stealth`` is a CONTINUOUS evasion signal in [0, 1]: the mean over poisoned
    clients of ``1 - soft P(malicious)`` derived from the defender's *confidence*
    (not just the binary flag).

    ``client_cost`` = ``(n_used - 1) / (pool_size - 1)`` in [0, 1] penalizes using
    more of the controllable pool than necessary (0 for a single client), so the
    attacker learns to achieve the goal with the fewest clients. ``collab_bonus``
    = ``diversity`` in [0, 1] (only when >1 client) rewards distinct, coordinated
    per-client perturbations over identical clones — see ``perturbation_diversity``.
    """
    import logging
    _logger = logging.getLogger("rl.rewards")

    gtype = goal.get("type", "untargeted_degrade")

    if gtype == "targeted_label" and post_class_acc is not None and baseline_class_acc is not None:
        # --- Dual-objective: class damage + stealth constraint ---
        target_class_drop = float(goal.get("target_class_drop", 0.80))
        max_global_drop = float(goal.get("max_global_drop", 0.03))
        target_class_drop = max(target_class_drop, 1e-6)
        label_cfg = goal.get("label", "menu")

        # Compute per-class accuracy drops relative to baseline
        drops = {c: baseline_class_acc.get(c, 0.0) - post_class_acc.get(c, 0.0)
                 for c in post_class_acc}

        if str(label_cfg).lower() == "menu":
            # LLM chose dynamically — reward the best class it attacked
            best_class = max(drops, key=drops.get) if drops else 0
            class_drop = drops.get(best_class, 0.0)
        else:
            best_class = int(label_cfg)
            class_drop = drops.get(best_class, 0.0)

        drop_term = _clip(class_drop / target_class_drop, -0.5, 1.5)

        # Stealth penalty: penalize if global accuracy drops too much
        global_drop = prev_accuracy - post_accuracy
        stealth_penalty = 0.0
        if global_drop > max_global_drop:
            stealth_penalty = (global_drop - max_global_drop) * 5.0
            drop_term -= stealth_penalty

        _logger.info(
            f"Targeted reward — best_class={best_class}, class_drop={class_drop:.4f}, "
            f"global_drop={global_drop:.4f}, stealth_penalty={stealth_penalty:.4f}, "
            f"drop_term={drop_term:.4f}"
        )

    elif gtype == "slow_degrade":
        target = float(goal.get("per_round_drop", 0.02))
        target = max(target, 1e-6)
        drop = prev_accuracy - post_accuracy
        drop_term = _clip(drop / target, -0.5, 1.5)
    else:
        # untargeted_degrade (default)
        target = float(goal.get("target_accuracy_drop", 0.20))
        target = max(target, 1e-6)
        drop = prev_accuracy - post_accuracy
        drop_term = _clip(drop / target, -0.5, 1.5)

    verdict_by_id = {v.client_id: v for v in verdicts}
    n_used = len(poisoned_ids)
    n_pois = max(1, n_used)
    stealth = 0.0
    for cid in poisoned_ids:
        v = verdict_by_id.get(cid)
        # No verdict for a poisoned client => treat as undetected (passed).
        stealth += 1.0 if v is None else (1.0 - _soft_malicious_prob(v))
    stealth /= n_pois

    malformed_fraction = n_malformed / n_pois

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

    reward = (alpha * drop_term + beta * stealth - gamma * malformed_fraction
              - delta * client_cost + zeta * collab_bonus)

    _logger.info(
        f"Attacker reward={reward:.4f} (drop_term={drop_term:.4f}, stealth={stealth:.4f}, "
        f"malformed={malformed_fraction:.4f}, client_cost={client_cost:.4f}, collab={collab_bonus:.4f})"
    )

    return reward


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
    damage_weight: float = 0.0,
    accuracy_drop: float | None = None,
) -> float:
    """Reward the defender for correctly identifying the poisoned clients.

    ``soft_f1`` (default): confidence-weighted soft F1 in [0, 1].
    ``tpr_minus_fpr``:     clip(TPR - fpr_penalty * FPR, 0, 1) using hard flags.

    ``damage_weight`` / ``accuracy_drop`` (both optional, OFF by default): the
    base score above treats every miss identically regardless of how much
    damage it actually did. When ``damage_weight > 0`` and ``accuracy_drop``
    (prev_accuracy - post_accuracy for THIS verdict set's own hypothetical
    aggregate) is supplied, an extra ``damage_weight * max(0, accuracy_drop)``
    is subtracted, making a miss that causes real damage strictly worse than
    one that happens to be harmless. Computing ``accuracy_drop`` per scored
    candidate requires an extra FedAvg + eval pass (see ``rl/turns.py``),
    which removes the "no per-attempt model evaluation" speed advantage
    defender-learning rounds otherwise have — left off by default for that
    reason, and enabling it means retraining the defender adapter from
    scratch (it changes the reward the policy was fit to).
    """
    poisoned = set(poisoned_ids)
    if mode == "tpr_minus_fpr":
        tp = sum(1 for v in verdicts if v.client_id in poisoned and v.is_suspicious)
        fn = sum(1 for v in verdicts if v.client_id in poisoned and not v.is_suspicious)
        fp = sum(1 for v in verdicts if v.client_id not in poisoned and v.is_suspicious)
        tn = sum(1 for v in verdicts if v.client_id not in poisoned and not v.is_suspicious)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        reward = _clip(tpr - fpr_penalty * fpr, 0.0, 1.0)
        if damage_weight > 0.0 and accuracy_drop is not None:
            reward -= damage_weight * max(0.0, float(accuracy_drop))
        return reward

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
    reward = 2 * precision * recall / (precision + recall + eps)
    if damage_weight > 0.0 and accuracy_drop is not None:
        reward -= damage_weight * max(0.0, float(accuracy_drop))
    return reward


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
