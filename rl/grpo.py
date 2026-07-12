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
from rl.rewards import group_advantages

logger = logging.getLogger(__name__)


def _grpo_loss_from_logprobs(lps, refs, advantages, G, kl_beta):
    """Sum the GRPO per-completion loss from precomputed log-probs.

    ``lps`` are grad-connected (policy), ``refs`` detached (frozen base). Returns
    ``(loss_tensor_or_None, total_loss_float, n_used)`` and does NOT call backward
    — the caller decides when. The per-completion term is the length-normalized
    policy gradient plus the k3 KL penalty, identical to the sequential loop."""
    loss = None
    total_loss = 0.0
    n_used = 0
    for lp, ref, adv in zip(lps, refs, advantages):
        if lp.numel() == 0:
            continue
        L = min(lp.shape[0], ref.shape[0])
        lp_, ref_ = lp[-L:], ref[-L:]
        log_ratio = ref_ - lp_
        kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
        pg = -(adv * lp_.mean())
        loss_i = (pg + kl_beta * kl) / max(1, G)
        loss = loss_i if loss is None else loss + loss_i
        total_loss += float(loss_i.detach())
        n_used += 1
    return loss, total_loss, n_used


def _grpo_grads_batched(policy, adapter, system, user, completions, advantages, G, kl_beta):
    """Fast path: TWO batched log-prob forwards (one adapter-on grad pass + one
    base no-grad pass) then a single ``backward``. Kept in its own function so that
    if it raises (e.g. OOM), its frame — and the big batched tensors it holds —
    unwinds and becomes collectable before the caller runs the fallback."""
    lps = policy.policy_completion_logprobs_batch(adapter, system, user, completions)
    refs = policy.reference_completion_logprobs_batch(system, user, completions)
    loss, total_loss, n_used = _grpo_loss_from_logprobs(lps, refs, advantages, G, kl_beta)
    if n_used > 0:
        loss.backward()
    return total_loss, n_used


def _grpo_grads_sequential(policy, adapter, system, user, completions, advantages, G, kl_beta):
    """The exact original per-completion loop (one forward per completion). Used as
    the always-correct fallback when the batched path can't run (e.g. OOM)."""
    total_loss = 0.0
    n_used = 0
    for completion, adv in zip(completions, advantages):
        lp = policy.policy_token_logprobs(adapter, system, user, completion)  # (L,) grad
        if lp.numel() == 0:
            continue
        ref = policy.reference_token_logprobs(system, user, completion)       # (L,) no grad
        L = min(lp.shape[0], ref.shape[0])
        lp, ref = lp[-L:], ref[-L:]
        log_ratio = ref - lp
        kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
        pg = -(adv * lp.mean())
        loss_i = (pg + kl_beta * kl) / max(1, G)
        loss_i.backward()
        total_loss += float(loss_i.detach())
        n_used += 1
    return total_loss, n_used


def _accumulate_grpo_grads(policy, adapter, optimizer, system, user, completions,
                           advantages, G, kl_beta):
    """Populate ``adapter`` gradients for the GRPO loss. Returns ``(total_loss, n_used)``.

    Tries the batched fast path; the gradient of the summed loss equals the sum of
    the per-completion gradients, so it is mathematically identical to the original
    per-sample loop — it just parallelizes the forwards. Any failure (typically an
    out-of-memory on an unusually long group) falls back to the exact per-completion
    loop, so the batching can never corrupt or crash a training step — worst case it
    is as slow as before for that one round."""
    err = None
    try:
        return _grpo_grads_batched(policy, adapter, system, user, completions, advantages, G, kl_beta)
    except Exception as e:                       # noqa: BLE001 — any failure must degrade, not crash
        err = f"{type(e).__name__}: {e}"
    # Reached only on failure. We are now OUTSIDE the except block, so the caught
    # exception — and the traceback that was pinning the batched tensors — has been
    # released; empty_cache can therefore actually reclaim that memory, and any
    # partial gradients are cleared, before the per-completion fallback re-runs.
    logger.warning(f"GRPO batched log-prob path failed ({err}); "
                   "falling back to the per-completion loop.")
    optimizer.zero_grad(set_to_none=True)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return _grpo_grads_sequential(policy, adapter, system, user, completions, advantages, G, kl_beta)


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
) -> dict:
    """Run one GRPO update for ``adapter`` against ``turn``. Returns metrics.

    ``turn`` must expose ``messages() -> (system, user)`` and
    ``rewards(list[completion_text]) -> list[float]`` — the group scorer, which for
    the attacker turn batches the frozen defender's G verdicts into one decode (see
    ``rl/turns.py``). ``reward(one_text)`` also still exists for the debug path.

    Zero-advantage handling (the attacker-collapse guard): when every sampled
    rollout earns the same reward the within-group spread is ~0, so the
    policy-gradient term is 0 and the loss reduces to ``kl_beta * KL`` — a pull
    *back toward the base model*, i.e. active un-learning. To avoid that:

    * ``resample_on_zero_advantage`` — re-draw the group ONCE at a higher
      temperature to try to recover reward spread before giving up.
    * ``skip_zero_advantage`` — if the group is still degenerate, skip the
      optimizer step entirely (apply no gradient) instead of stepping on a pure
      KL-to-base signal.
    """
    system, user = turn.messages()

    # 1. Sample a group of G candidate actions (no grad).
    completions = policy.generate(
        adapter, system, user, n=G, temperature=temperature, max_new_tokens=max_new_tokens,
    )

    # 2. Verifiable reward for each candidate. ``turn.rewards`` scores the whole
    #    group at once — for the attacker turn that samples the frozen defender's
    #    verdicts for all G candidates in ONE batched decode (same per-rollout
    #    distribution as scoring them one by one).
    rewards = [float(r) for r in turn.rewards(completions)]

    # 3. Group-relative advantages.
    advantages, zero_frac = group_advantages(rewards)

    # 3b. A degenerate (zero-spread) group gives no learning signal. Optionally
    #     re-roll once at a higher temperature to recover diversity.
    resampled = False
    if zero_frac >= 1.0 and resample_on_zero_advantage:
        resampled = True
        dbg.resampling()
        completions = policy.generate(
            adapter, system, user, n=G,
            temperature=max(temperature, resample_temperature),
            max_new_tokens=max_new_tokens,
        )
        rewards = [float(r) for r in turn.rewards(completions)]
        advantages, zero_frac = group_advantages(rewards)

    mean_r = sum(rewards) / len(rewards) if rewards else 0.0

    # 3c. Still degenerate → do NOT step (would only pull the adapter to base).
    stepped = not (zero_frac >= 1.0 and skip_zero_advantage)

    # 4. Accumulate the GRPO loss and take one optimizer step.
    total_loss = 0.0
    n_used = 0
    if stepped:
        optimizer.zero_grad()
        total_loss, n_used = _accumulate_grpo_grads(
            policy, adapter, optimizer, system, user, completions, advantages, G, kl_beta,
        )
        if n_used > 0:
            torch.nn.utils.clip_grad_norm_(policy.adapter_parameters(adapter), grad_clip)
            optimizer.step()
        else:
            stepped = False

    metrics = {
        "loss": total_loss,
        "mean_reward": mean_r,
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "zero_advantage_fraction": zero_frac,
        "rewards": rewards,
        "completions": completions,
        "advantages": advantages,
        "stepped": stepped,
        "resampled": resampled,
    }
    if zero_frac >= 1.0 and not stepped:
        logger.warning(
            f"GRPO[{adapter}]: zero-advantage group (all {G} rewards equal "
            f"≈{mean_r:.3f}) — skipped step (no gradient applied)"
        )
    dbg.grpo_summary(metrics)
    return metrics
