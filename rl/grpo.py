"""grpo_step — one Group-Relative Policy Optimization update.

For the current turn's prompt we sample ``G`` completions, score each with the
verifiable reward, and form a group-relative advantage (z-score within the
group — no value/critic network, which is GRPO's whole point vs PPO). The loss
is the single-iteration, length-normalized policy-gradient term plus a per-token
KL penalty to the frozen base model (k3 estimator):

    loss = mean_i [ -A_i * mean_t logπ(o_i,t)  +  β * mean_t KL_t ]
    KL_t = exp(refLP_t - LP_t) - (refLP_t - LP_t) - 1

Single-iteration (we update on freshly sampled data) means the importance ratio
is 1, so no clipping is needed — the PPO-style clip only matters when reusing a
batch for multiple gradient steps.
"""

import logging

import torch

from rl.rewards import group_advantages

logger = logging.getLogger(__name__)


def grpo_step(
    policy,
    adapter: str,
    optimizer,
    turn,
    *,
    G: int = 4,
    kl_beta: float = 0.02,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    grad_clip: float = 1.0,
) -> dict:
    """Run one GRPO update for ``adapter`` against ``turn``. Returns metrics.

    ``turn`` must expose ``messages() -> (system, user)`` and
    ``reward(completion_text) -> float`` (see ``rl/turns.py``).
    """
    system, user = turn.messages()

    # 1. Sample a group of G candidate actions (no grad).
    completions = policy.generate(
        adapter, system, user, n=G, temperature=temperature, max_new_tokens=max_new_tokens,
    )

    # 2. Verifiable reward for each candidate.
    rewards = [float(turn.reward(c)) for c in completions]

    # 3. Group-relative advantages.
    advantages, zero_frac = group_advantages(rewards)

    # 4. Score all G completions in TWO batched forwards (policy grad + reference
    #    no-grad), then one summed backward over the shared graph. Mathematically
    #    identical to per-sample (gradient is linear), but far better GPU use.
    policy_lps = policy.policy_token_logprobs_batch(adapter, system, user, completions)
    ref_lps = policy.reference_token_logprobs_batch(system, user, completions)

    optimizer.zero_grad()
    total_loss = 0.0
    n_used = 0
    loss_accum = None
    for lp, ref, adv in zip(policy_lps, ref_lps, advantages):
        if lp is None or lp.numel() == 0:
            continue
        L = min(lp.shape[0], ref.shape[0])
        lp, ref = lp[-L:], ref[-L:]

        log_ratio = ref - lp
        kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
        pg = -(adv * lp.mean())
        loss_i = (pg + kl_beta * kl) / max(1, G)
        loss_accum = loss_i if loss_accum is None else loss_accum + loss_i
        total_loss += float(loss_i.detach())
        n_used += 1

    if n_used > 0:
        loss_accum.backward()
        torch.nn.utils.clip_grad_norm_(policy.adapter_parameters(adapter), grad_clip)
        optimizer.step()

    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    metrics = {
        "loss": total_loss,
        "mean_reward": mean_r,
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "zero_advantage_fraction": zero_frac,
        "rewards": rewards,
        "completions": completions,
        "advantages": advantages,
    }
    if zero_frac >= 1.0:
        logger.warning(
            f"GRPO[{adapter}]: zero-advantage group (all {G} rewards equal "
            f"≈{mean_r:.3f}) — no policy gradient this step"
        )
    return metrics
