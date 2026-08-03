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

from core.debug import dbg
from rl.rewards import (
    DEFAULT_ADVANTAGE_STD_FLOOR, DEFAULT_MIN_REWARD_SPREAD, group_advantages,
)

logger = logging.getLogger(__name__)


def _completed_flags(policy, n: int) -> list[bool]:
    """Per-rollout "ended on EOS" flags from the last ``policy.generate`` call.

    Falls back to all-False (never train on a stop token we can't confirm) for
    generators that don't expose the attribute or return a mismatched length.
    """
    flags = getattr(policy, "last_generation_completed", None)
    if isinstance(flags, list) and len(flags) == n:
        return [bool(f) for f in flags]
    return [False] * n


def _generation_ids(policy, n: int) -> list:
    """The EXACT completion token ids from the last ``policy.generate`` call.

    These are what the log-prob pass must score: re-tokenizing the decoded text is
    not a guaranteed inverse of sampling, and any mismatch means we differentiate a
    sequence the policy never produced — which breaks the ``ratio == 1`` identity
    this single-iteration loss relies on.

    Falls back to ``[None] * n`` (log-probs then re-tokenize the text, the legacy
    behaviour) for generators that don't expose the attribute — the same
    length-mismatch guard as ``_completed_flags``, since both are overwritten by
    whichever ``generate`` ran last.
    """
    ids = getattr(policy, "last_generation_ids", None)
    if isinstance(ids, list) and len(ids) == n:
        return list(ids)
    return [None] * n


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
    skip_zero_advantage: bool = True,
    resample_on_zero_advantage: bool = False,
    resample_temperature: float = 1.3,
    min_reward_spread: float = DEFAULT_MIN_REWARD_SPREAD,
    advantage_std_floor: float = DEFAULT_ADVANTAGE_STD_FLOOR,
) -> dict:
    """Run one GRPO update for ``adapter`` against ``turn``. Returns metrics.

    ``turn`` must expose ``messages() -> (system, user)`` and
    ``reward(completion_text) -> float`` (see ``rl/turns.py``).

    Zero-advantage handling (the attacker-collapse guard): when every sampled
    rollout earns the same reward the within-group spread is ~0, so the
    policy-gradient term is 0 and the loss reduces to ``kl_beta * KL`` — a pull
    *back toward the base model*, i.e. active un-learning. To avoid that:

    * ``resample_on_zero_advantage`` — re-draw the group ONCE at a higher
      temperature to try to recover reward spread before giving up.
    * ``skip_zero_advantage`` — if the group is still degenerate, skip the
      optimizer step entirely (apply no gradient) instead of stepping on a pure
      KL-to-base signal.

    "Degenerate" is decided by ``min_reward_spread``, not by exact equality: the
    rewards derive from an accuracy measured on a finite test set, so two
    behaviourally identical rollouts routinely differ by a hair. Treating that as
    signal meant z-scoring measurement noise up to full-magnitude advantages — see
    ``rl.rewards.group_advantages``.
    """
    system, user = turn.messages()
    eff_temperature = temperature

    # 1. Sample a group of G candidate actions (no grad).
    completions = policy.generate(
        adapter, system, user, n=G, temperature=temperature, max_new_tokens=max_new_tokens,
    )
    # Capture BEFORE scoring: turn.reward() runs the opponent's generate(), which
    # overwrites both of these. `completion_ids` are the exact sampled tokens (what
    # the log-prob pass must score); `completed` marks which rollouts ended with EOS
    # instead of being cut off at max_new_tokens.
    completion_ids = _generation_ids(policy, len(completions))
    completed = _completed_flags(policy, len(completions))

    # 2. Verifiable reward for each candidate.
    rewards = [float(turn.reward(c)) for c in completions]

    # 3. Group-relative advantages.
    advantages, zero_frac = group_advantages(
        rewards, min_spread=min_reward_spread, std_floor=advantage_std_floor)

    # 3b. A degenerate (zero-spread) group gives no learning signal. Optionally
    #     re-roll once at a higher temperature to recover diversity.
    resampled = False
    if zero_frac >= 1.0 and resample_on_zero_advantage:
        resampled = True
        dbg.resampling()
        # The re-roll samples from a HOTTER distribution, so the log-prob pass below
        # has to score that same distribution — track the temperature actually used.
        eff_temperature = max(temperature, resample_temperature)
        completions = policy.generate(
            adapter, system, user, n=G,
            temperature=eff_temperature,
            max_new_tokens=max_new_tokens,
        )
        completion_ids = _generation_ids(policy, len(completions))
        completed = _completed_flags(policy, len(completions))
        rewards = [float(turn.reward(c)) for c in completions]
        advantages, zero_frac = group_advantages(
            rewards, min_spread=min_reward_spread, std_floor=advantage_std_floor)

    mean_r = sum(rewards) / len(rewards) if rewards else 0.0

    # 3c. Still degenerate → do NOT step (would only pull the adapter to base).
    stepped = not (zero_frac >= 1.0 and skip_zero_advantage)

    # 4. Accumulate the GRPO loss; backward per-sample to bound memory.
    total_loss = 0.0
    n_used = 0
    if stepped:
        optimizer.zero_grad()
        for completion, adv, eos, comp_ids in zip(completions, advantages, completed,
                                                  completion_ids):
            # (L,) grad over the EXACT sampled tokens (comp_ids), scored at the
            # temperature they were sampled at. append_eos only matters on the text
            # fallback, when the generator could not hand back ids.
            lp = policy.policy_token_logprobs(adapter, system, user, completion,
                                              append_eos=eos, completion_ids=comp_ids,
                                              temperature=eff_temperature)
            if lp.numel() == 0:
                continue
            # Same ids/temperature so the KL lines up token-for-token.
            ref = policy.reference_token_logprobs(system, user, completion,
                                                  append_eos=eos, completion_ids=comp_ids,
                                                  temperature=eff_temperature)  # (L,) no grad
            L = min(lp.shape[0], ref.shape[0])
            lp, ref = lp[-L:], ref[-L:]

            log_ratio = ref - lp
            kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
            pg = -(adv * lp.mean())
            loss_i = (pg + kl_beta * kl) / max(1, G)
            loss_i.backward()
            total_loss += float(loss_i.detach())
            n_used += 1

        if n_used > 0:
            torch.nn.utils.clip_grad_norm_(policy.adapter_parameters(adapter), grad_clip)
            optimizer.step()
        else:
            stepped = False

    spread = (max(rewards) - min(rewards)) if rewards else 0.0
    metrics = {
        "loss": total_loss,
        "mean_reward": mean_r,
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "reward_spread": spread,
        "zero_advantage_fraction": zero_frac,
        "rewards": rewards,
        "completions": completions,
        "advantages": advantages,
        "stepped": stepped,
        "resampled": resampled,
        "temperature": eff_temperature,
    }
    if zero_frac >= 1.0 and not stepped:
        logger.warning(
            f"GRPO[{adapter}]: degenerate group - {G} rewards span only "
            f"{spread:.4g} (< min_reward_spread={min_reward_spread:g}) around "
            f"{mean_r:.3f}; skipped step (no gradient applied)"
        )
    dbg.grpo_summary(metrics)
    return metrics
