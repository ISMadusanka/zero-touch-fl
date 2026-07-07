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
) -> float:
    """Reward the attacker for degrading accuracy while staying stealthy, using
    the FEWEST clients, and (when it uses several) collaborating with them.

    reward = alpha * clip(drop / target, -0.5, 1.5)
           + beta  * stealth
           - gamma * malformed_fraction
           - delta * client_cost
           + zeta  * collab_bonus

    ``drop = prev_accuracy - post_accuracy``. ``stealth`` is a CONTINUOUS evasion
    signal in [0, 1]: the mean over poisoned clients of ``1 - soft P(malicious)``
    derived from the defender's *confidence* (not just the binary flag).
    Confidently caught -> ~0; confidently passed as benign -> ~1; unsure -> ~0.5.
    Because it moves smoothly with the defender's confidence, it gives GRPO a
    gradient even when every sampled plan ends up with the SAME hard flag — the
    fix for zero-advantage attacker groups.

    ``client_cost`` = ``(n_used - 1) / (pool_size - 1)`` in [0, 1] penalizes using
    more of the controllable pool than necessary (0 for a single client), so the
    attacker learns to achieve the goal with the fewest clients. ``collab_bonus``
    = ``diversity`` in [0, 1] (only when >1 client) rewards distinct, coordinated
    per-client perturbations over identical clones — see ``perturbation_diversity``.
    """
    gtype = goal.get("type", "untargeted_degrade")
    if gtype == "slow_degrade":
        target = float(goal.get("per_round_drop", 0.02))
    else:
        # untargeted_degrade (and, for now, targeted_label falls back to overall
        # accuracy until per-class evaluation is wired in).
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

    return (alpha * drop_term + beta * stealth - gamma * malformed_fraction
            - delta * client_cost + zeta * collab_bonus)


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
    """
    poisoned = set(poisoned_ids)
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
