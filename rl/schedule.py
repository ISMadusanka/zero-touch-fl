"""Stackelberg freeze-and-alternate training driver + opponent league.

Two policies that co-adapt are non-stationary; updating both at once tends to
cycle. So we alternate: train the attacker for ``K_a`` rounds while the
defender is frozen, then the defender for ``K_d`` rounds while the attacker is
frozen, and repeat. To stop the learner over-fitting the *latest* opponent we
keep an opponent **league** (periodic snapshots) and, with probability
``league_prob``, face a random past snapshot for a whole phase.

The frozen opponent always plays greedily (``opponent_temperature``≈0), which
makes each candidate's reward a deterministic best-response score.
"""

import logging

from core.types import RoundLog
from rl.grpo import grpo_step
from rl.policy import PolicyGenerator
from rl.rewards import attacker_reward, defender_reward
from rl.turns import AttackerTurn, DefenderTurn

logger = logging.getLogger(__name__)


class League:
    """In-memory pool of past adapter snapshots (CPU LoRA state dicts)."""

    def __init__(self, rng):
        self.rng = rng
        self.snapshots: dict[str, list[dict]] = {}

    def snapshot(self, policy, names):
        for name in names:
            self.snapshots.setdefault(name, []).append(policy.get_adapter_state(name))
        logger.info(f"League: snapshotted {list(names)} "
                    f"(sizes={ {k: len(v) for k, v in self.snapshots.items()} })")

    def has(self, name) -> bool:
        return bool(self.snapshots.get(name))

    def sample(self, name) -> dict:
        return self.rng.choice(self.snapshots[name])


def _argmax(xs: list[float]) -> int:
    best, bi = xs[0], 0
    for i, x in enumerate(xs):
        if x > best:
            best, bi = x, i
    return bi


def train(
    env,
    policy,
    attacker_agent,
    defender_agent,
    cfg: dict,
    metrics_tracker,
    save_round_log,
    rng,
    progress_cb=None,
    start_round: int = 0,
):
    """Run the alternating GRPO training loop over ``simulation_rounds`` rounds."""
    rl = cfg.get("rl", {})
    G = int(rl.get("G", 4))
    kl_beta = float(rl.get("kl_beta", 0.02))
    lr = float(rl.get("lr", 1e-5))
    K_a = int(rl.get("K_a", 4))
    K_d = int(rl.get("K_d", 4))
    learner_temp = float(rl.get("temperature", 1.0))
    opp_temp = float(rl.get("opponent_temperature", 0.0))
    max_new_tokens = int(rl.get("max_new_tokens", 2048))
    grad_clip = float(rl.get("grad_clip", 1.0))
    save_every = int(rl.get("save_every", 50))
    snap_every = int(rl.get("league_snapshot_every", 100))
    league_prob = float(rl.get("league_prob", 0.0))
    adapter_paths = rl.get("adapter_paths", {
        "attacker": "checkpoints/attacker_adapter",
        "defender": "checkpoints/defender_adapter",
    })
    reward_cfg = rl.get("reward", {})
    reward_att = reward_cfg.get("attacker", {})
    reward_def = reward_cfg.get("defender", {})

    total_rounds = int(cfg["fl"]["simulation_rounds"])

    import torch
    optimizers = {
        "attacker": torch.optim.AdamW(policy.adapter_parameters("attacker"), lr=lr),
        "defender": torch.optim.AdamW(policy.adapter_parameters("defender"), lr=lr),
    }
    league = League(rng)

    done = start_round
    schedule = [("attacker", K_a), ("defender", K_d)]
    si = 0
    while done < total_rounds:
        learner, K = schedule[si % len(schedule)]
        si += 1
        opp = "defender" if learner == "attacker" else "attacker"

        # League: optionally face a snapshot opponent for this whole phase.
        live_opp = policy.get_adapter_state(opp)
        use_snap = league_prob > 0 and league.has(opp) and rng.random() < league_prob
        if use_snap:
            policy.set_adapter_state(opp, league.sample(opp))
            logger.info(f"Phase {learner}: facing a LEAGUE snapshot of {opp}")
        opp_gen = PolicyGenerator(policy, opp, max_new_tokens)

        for _ in range(K):
            if done >= total_rounds:
                break
            ctx = env.begin_round()

            if learner == "attacker":
                turn = AttackerTurn(env, attacker_agent, defender_agent, opp_gen,
                                    reward_cfg=reward_att, opponent_temperature=opp_temp)
            else:
                turn = DefenderTurn(env, attacker_agent, defender_agent, opp_gen,
                                    reward_cfg=reward_def, opponent_temperature=opp_temp)

            stats = grpo_step(
                policy, learner, optimizers[learner], turn,
                G=G, kl_beta=kl_beta, temperature=learner_temp,
                max_new_tokens=max_new_tokens, grad_clip=grad_clip,
            )

            # Advance the env by committing the best-scoring candidate action.
            best = _argmax(stats["rewards"]) if stats["rewards"] else 0
            info = turn.commit(stats["completions"][best])

            _log_round(env, ctx, info, learner, stats, metrics_tracker, save_round_log)
            done += 1

            if snap_every and done % snap_every == 0:
                league.snapshot(policy, policy.adapters)
            # Checkpoint adapters + progress TOGETHER so a resume is always
            # consistent (the saved round count never points past the saved
            # adapter weights). Rounds since the last checkpoint are simply
            # re-trained from it on resume.
            if save_every and done % save_every == 0:
                _checkpoint(policy, adapter_paths, progress_cb, done)

        if use_snap:
            policy.set_adapter_state(opp, live_opp)   # restore the live opponent

    _checkpoint(policy, adapter_paths, progress_cb, done)   # final save
    logger.info(f"Training complete — {done} rounds. Adapters saved to {adapter_paths}")


def _checkpoint(policy, adapter_paths, progress_cb, done):
    """Atomically-ish save: adapters first, then advance the progress counter."""
    _save_adapters(policy, adapter_paths)
    if progress_cb:
        progress_cb(done)


def _save_adapters(policy, adapter_paths: dict):
    for name, path in adapter_paths.items():
        policy.save_adapter(name, path)


def _log_round(env, ctx, info, learner, stats, metrics_tracker, save_round_log):
    verdicts = info["verdicts"]
    post_acc = info["post_accuracy"]
    n_malformed = info["n_malformed"]

    a_rew = attacker_reward(ctx.global_accuracy, post_acc, env.goal,
                            ctx.poisoned_ids, verdicts, n_malformed)
    d_rew = defender_reward(verdicts, ctx.poisoned_ids)

    metrics_tracker.update(ctx.round_num, verdicts, post_acc, set(ctx.poisoned_ids))
    save_round_log(RoundLog(
        round_num=ctx.round_num,
        attack_goal=env.goal,
        poisoned_client_ids=ctx.poisoned_ids,
        predicted_labels=[
            {"client_id": v.client_id, "is_suspicious": v.is_suspicious,
             "confidence": v.confidence, "reason": v.reason}
            for v in verdicts
        ],
        test_accuracy=post_acc,
        baseline_accuracy=env.baseline_accuracy,
        attacker_reward=a_rew,
        defender_reward=d_rew,
        learning_agent=learner,
        attack_metadata={
            "n_malformed": n_malformed,
            "train": {
                "loss": stats["loss"],
                "mean_reward": stats["mean_reward"],
                "max_reward": stats["max_reward"],
                "zero_advantage_fraction": stats["zero_advantage_fraction"],
            },
        },
    ))
    logger.info(
        f"Round {ctx.round_num} [learn={learner}]: acc {ctx.global_accuracy:.4f}->{post_acc:.4f} "
        f"| att_reward={a_rew:.3f} def_reward={d_rew:.3f} "
        f"| grpo_loss={stats['loss']:.4f} mean_r={stats['mean_reward']:.3f} "
        f"zero_adv={stats['zero_advantage_fraction']:.2f}"
    )
