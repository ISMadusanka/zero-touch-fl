"""Freeze-and-alternate GRPO training driver + opponent league.

Two policies that co-adapt are non-stationary; updating both at once tends to
cycle. So we alternate: train ONE agent while the other is frozen, then swap.
Two schedules are supported (``rl.switch_mode``):

* ``best_response`` (default) — **success-gated iterated best response**. Train
  the attacker (defender frozen) until an attack *passes* (a sustained win),
  freeze the attacker at that checkpoint, then train the defender until it
  *reliably catches* that frozen attacker, freeze, swap, repeat. A per-phase
  min/max round band (see ``rl/switch.py``) guards against switching on a fluke
  or running a hopeless phase forever. This is the double-oracle ratchet that
  keeps both agents climbing instead of one running away.

* ``fixed`` — the legacy fixed-clock alternation: ``K_a`` attacker rounds, then
  ``K_d`` defender rounds, repeated.

To stop the learner over-fitting the *latest* opponent we keep an opponent
**league** (snapshots). With probability ``league_prob`` a phase faces a random
past snapshot; and after a phase that *capped without a win*, the next learner
faces an earlier snapshot (curriculum) so it can find a foothold.

When SCORING the G candidate rollouts the frozen opponent is sampled at
``scoring_opponent_temperature`` (usually nonzero) so candidates see varied
opponent responses — this restores the within-group reward spread GRPO needs.
The COMMITTED round uses the greedy ``opponent_temperature`` so each win/loss is
measured against the real, deterministic opponent.
"""

import logging

from core.types import RoundLog
from core.debug import dbg
from rl.grpo import grpo_step
from rl.policy import PolicyGenerator
from rl.rewards import attacker_reward, defender_reward, perturbation_diversity
from rl.switch import PhaseController, SwitchConfig, committed_success
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
    scoring_opp_temp = float(rl.get("scoring_opponent_temperature", opp_temp))
    max_new_tokens = int(rl.get("max_new_tokens", 2048))
    grad_clip = float(rl.get("grad_clip", 1.0))
    save_every = int(rl.get("save_every", 50))
    snap_every = int(rl.get("league_snapshot_every", 100))
    league_prob = float(rl.get("league_prob", 0.0))
    skip_zero_adv = bool(rl.get("skip_zero_advantage", True))
    resample_zero_adv = bool(rl.get("resample_on_zero_advantage", True))
    resample_temp = float(rl.get("resample_temperature", 1.3))
    switch_mode = str(rl.get("switch_mode", "best_response"))
    first_learner = str(rl.get("first_learner", "attacker"))
    curriculum_on_cap = bool(rl.get("curriculum_on_cap", True))
    fl_interlude = bool(rl.get("fl_interlude_between_phases", True))
    adapter_paths = rl.get("adapter_paths", {
        "attacker": "checkpoints/attacker_adapter",
        "defender": "checkpoints/defender_adapter",
    })
    reward_cfg = rl.get("reward", {})
    reward_att = reward_cfg.get("attacker", {})
    reward_def = reward_cfg.get("defender", {})
    switch_cfg = SwitchConfig.from_cfg(rl)

    total_rounds = int(cfg["fl"]["simulation_rounds"])

    import torch
    optimizers = {
        "attacker": torch.optim.AdamW(policy.adapter_parameters("attacker"), lr=lr),
        "defender": torch.optim.AdamW(policy.adapter_parameters("defender"), lr=lr),
    }
    league = League(rng)

    # Per-round knobs bundled so both schedules share one round body.
    knobs = dict(
        G=G, kl_beta=kl_beta, learner_temp=learner_temp, max_new_tokens=max_new_tokens,
        grad_clip=grad_clip, opp_temp=opp_temp, scoring_opp_temp=scoring_opp_temp,
        skip_zero_adv=skip_zero_adv, resample_zero_adv=resample_zero_adv,
        resample_temp=resample_temp, reward_att=reward_att, reward_def=reward_def,
    )

    state = dict(
        env=env, policy=policy, attacker_agent=attacker_agent, defender_agent=defender_agent,
        optimizers=optimizers, league=league, knobs=knobs, switch_cfg=switch_cfg,
        metrics_tracker=metrics_tracker, save_round_log=save_round_log,
        adapter_paths=adapter_paths, progress_cb=progress_cb, total_rounds=total_rounds,
        save_every=save_every, snap_every=snap_every, league_prob=league_prob,
        max_new_tokens=max_new_tokens, rng=rng, curriculum_on_cap=curriculum_on_cap,
        fl_interlude=fl_interlude,
    )

    if switch_mode == "best_response":
        logger.info(
            f"Schedule=best_response: success-gated iterated best response "
            f"(first_learner={first_learner}, streak={switch_cfg.success_streak}, "
            f"min/max phase={switch_cfg.min_phase_rounds}/{switch_cfg.max_phase_rounds})"
        )
        done = _train_best_response(state, first_learner, start_round)
    else:
        logger.info(f"Schedule=fixed: K_a={K_a}, K_d={K_d}")
        done = _train_fixed(state, K_a, K_d, start_round)

    _checkpoint(policy, adapter_paths, progress_cb, done)   # final save
    logger.info(f"Training complete — {done} rounds. Adapters saved to {adapter_paths}")


def _step_round(state, learner, opp, opp_gen, phase_index, phase_round):
    """Run ONE committed GRPO round for ``learner`` vs the frozen ``opp``.

    Returns ``(stats, committed_drop, success)`` where ``committed_drop`` is the
    accuracy lost on the committed round (prev - post) and ``success`` is whether
    the learner won this round (per the success-gate thresholds)."""
    env = state["env"]
    k = state["knobs"]
    ctx = env.begin_round()

    dbg.round_header(ctx.round_num, learner, opp, phase_index, phase_round,
                     ctx.pool_ids, ctx.budget, ctx.global_accuracy, k["G"],
                     k["scoring_opp_temp"], k["opp_temp"])
    # The poison SET is chosen per-rollout by the attacker, so it is unknown at
    # begin_round — show only the federated fine-tuning here (poison at commit).
    dbg.fl_round(ctx.round_num, [], env.honest_updates,
                 env.current_accuracy, env.benign_retrain)

    if learner == "attacker":
        turn = AttackerTurn(
            env, state["attacker_agent"], state["defender_agent"], opp_gen,
            reward_cfg=k["reward_att"], opponent_temperature=k["opp_temp"],
            scoring_opponent_temperature=k["scoring_opp_temp"],
        )
    else:
        # The defender's frozen attacker is sampled at the scoring temperature so
        # it faces a distribution of attacks (not one greedy plan) across rounds.
        turn = DefenderTurn(
            env, state["attacker_agent"], state["defender_agent"], opp_gen,
            reward_cfg=k["reward_def"], opponent_temperature=k["scoring_opp_temp"],
        )

    stats = grpo_step(
        state["policy"], learner, state["optimizers"][learner], turn,
        G=k["G"], kl_beta=k["kl_beta"], temperature=k["learner_temp"],
        max_new_tokens=k["max_new_tokens"], grad_clip=k["grad_clip"],
        skip_zero_advantage=k["skip_zero_adv"],
        resample_on_zero_advantage=k["resample_zero_adv"],
        resample_temperature=k["resample_temp"],
    )

    # Advance the env by committing the best-scoring candidate action.
    best = _argmax(stats["rewards"]) if stats["rewards"] else 0
    info = turn.commit(stats["completions"][best])
    poisoned_ids = info["poisoned_ids"]        # the attacker's committed choice

    drop = ctx.global_accuracy - info["post_accuracy"]

    # For targeted attacks, compute the class-level drop for the phase switch
    class_drop = None
    if env._is_targeted and env.baseline_class_accuracies and info.get("post_class_accuracy"):
        label_cfg = env.goal.get("label", "menu")
        drops = {c: env.baseline_class_accuracies.get(c, 0.0) - info["post_class_accuracy"].get(c, 0.0)
                 for c in info["post_class_accuracy"]}
        if str(label_cfg).lower() == "menu":
            best_class = max(drops, key=drops.get) if drops else 0
            class_drop = drops.get(best_class, 0.0)
        else:
            class_drop = drops.get(int(label_cfg), 0.0)

    success = committed_success(learner, drop, info["verdicts"], poisoned_ids,
                                state["switch_cfg"], class_drop)

    _log_round(env, ctx, info, learner, stats, state["metrics_tracker"],
               state["save_round_log"], reward_att=k["reward_att"], reward_def=k["reward_def"],
               phase_index=phase_index, phase_round=phase_round, success=success,
               best_index=best, poisoned_ids=poisoned_ids)
    return stats, drop, success


def _post_round_bookkeeping(state, done):
    """League snapshots + checkpoints on the configured cadence."""
    if state["snap_every"] and done % state["snap_every"] == 0:
        state["league"].snapshot(state["policy"], state["policy"].adapters)
    # Checkpoint adapters + progress TOGETHER so a resume is always consistent
    # (the saved round count never points past the saved adapter weights).
    if state["save_every"] and done % state["save_every"] == 0:
        _checkpoint(state["policy"], state["adapter_paths"], state["progress_cb"], done)


def _opponent_generator(state, opp, face_snapshot):
    """Return (opp_gen, restore_fn). If ``face_snapshot`` and a snapshot exists,
    temporarily swap the opponent adapter for a past (weaker) snapshot — used for
    the league mix and for the curriculum after a capped phase."""
    policy = state["policy"]
    live_opp = policy.get_adapter_state(opp)
    used = False
    if face_snapshot and state["league"].has(opp):
        policy.set_adapter_state(opp, state["league"].sample(opp))
        used = True
        logger.info(f"Phase: facing a LEAGUE snapshot of {opp}")
    opp_gen = PolicyGenerator(policy, opp, state["max_new_tokens"])

    def restore():
        if used:
            policy.set_adapter_state(opp, live_opp)
    return opp_gen, restore


def _run_fl_interlude(state, next_learner, phase_index):
    """Between two arms-race phases, advance the shared FL state by one honest
    FedAvg round (exactly like a Phase-1 round) and log it.

    The user-facing contract: after a learner wins and we hand off, we do NOT let
    the incoming learner keep training against the same frozen client weights —
    we first run one benign FL round so the refreshed client weights + global are
    what the next learner, the frozen opponent, AND the aggregator now consume
    (see ``FLArmsRaceEnv.run_benign_fl_round``). This runs before EVERY phase
    after the first, whatever caused the switch (sustained win or cap)."""
    env = state["env"]
    info = env.run_benign_fl_round()
    if info is None:
        return
    round_num = info["round_num"]
    updates = info["updates"]

    # Structured debug view (reuse the per-round FL panel; benign_retrain=True so
    # it prints each client's local-training stats).
    dbg.phase_event("FL ROUND (interlude)", round=round_num, next_learner=next_learner,
                    phase=phase_index,
                    prev_acc=round(info["prev_accuracy"], 4),
                    post_acc=round(info["post_accuracy"], 4))
    dbg.fl_round(round_num, [], updates, info["post_accuracy"], benign_retrain=True)
    dbg.flush()

    # Persist a round log so the interlude shows up in logs/round_data/round_NNN.json.
    state["save_round_log"](RoundLog(
        round_num=round_num,
        attack_goal=env.goal,
        poisoned_client_ids=[],
        predicted_labels=[],
        test_accuracy=info["post_accuracy"],
        baseline_accuracy=env.baseline_accuracy,
        attacker_reward=0.0,
        defender_reward=0.0,
        learning_agent="none",
        attack_metadata={
            "event": "benign_fl_round",
            "next_learner": next_learner,
            "phase_index": phase_index,
            "prev_accuracy": info["prev_accuracy"],
            "post_accuracy": info["post_accuracy"],
            "post_class_accuracy": info.get("post_class_accuracy"),
            "n_clients": info["n_clients"],
            "clients": [
                {"client_id": u.client_id, **(u.metadata or {})} for u in updates
            ],
        },
    ))
    logger.info(
        f"[FL round {round_num}] interlude before {next_learner} phase {phase_index}: "
        f"acc {info['prev_accuracy']:.4f} -> {info['post_accuracy']:.4f} "
        f"— new benign client weights now consumed by attacker/defender/aggregator"
    )


def _train_best_response(state, first_learner, start_round):
    """Success-gated iterated best response. Returns the final round count."""
    ctrl = PhaseController(state["switch_cfg"], first_learner=first_learner)
    rng = state["rng"]
    done = start_round

    while done < state["total_rounds"]:
        learner, opp = ctrl.learner, ctrl.opponent
        # Before every phase AFTER the first, run one honest FL round so the
        # incoming learner + frozen opponent + aggregator train against a freshly
        # advanced client state (mirrors a Phase-1 round). Skipped for the very
        # first phase, which uses the Phase-1 checkpoint as-is.
        if ctrl.phase_index > 0 and state.get("fl_interlude", True):
            _run_fl_interlude(state, next_learner=learner, phase_index=ctrl.phase_index)
        # Curriculum: a phase that capped without a win means the opponent is too
        # strong — let this learner face an earlier snapshot of it. Otherwise mix
        # in a league snapshot with probability league_prob (anti-overfit).
        face_snapshot = (
            (state["curriculum_on_cap"] and ctrl.capped)
            or (state["league_prob"] > 0 and rng.random() < state["league_prob"])
        )
        opp_gen, restore = _opponent_generator(state, opp, face_snapshot)
        dbg.phase_event("PHASE START", phase=ctrl.phase_index, learner=learner,
                        opponent=opp, facing_snapshot=face_snapshot)

        reason = None
        while done < state["total_rounds"]:
            _stats, _drop, success = _step_round(
                state, learner, opp, opp_gen, ctrl.phase_index, ctrl.phase_round
            )
            done += 1
            _post_round_bookkeeping(state, done)
            switch, reason = ctrl.record(success)
            if switch:
                break

        restore()
        # Freeze the just-trained learner as a best-response checkpoint: snapshot
        # it into the league and persist its adapter.
        state["league"].snapshot(state["policy"], [learner])
        state["policy"].save_adapter(learner, state["adapter_paths"][learner])
        logger.info(
            f"Phase {ctrl.phase_index} [{learner}] ended ({reason or 'budget'}) "
            f"after {ctrl.phase_round} rounds (streak={ctrl.streak}) — froze {learner}"
        )
        dbg.phase_event("PHASE END", phase=ctrl.phase_index, learner=learner,
                        reason=reason or "budget", rounds=ctrl.phase_round, streak=ctrl.streak)
        if reason is None:   # ran out of total budget mid-phase
            break
        ctrl.next_phase(reason)

    return done


def _train_fixed(state, K_a, K_d, start_round):
    """Legacy fixed-clock alternation. Returns the final round count."""
    rng = state["rng"]
    done = start_round
    schedule = [("attacker", K_a), ("defender", K_d)]
    si = 0
    while done < state["total_rounds"]:
        learner, K = schedule[si % len(schedule)]
        si += 1
        opp = "defender" if learner == "attacker" else "attacker"
        face_snapshot = state["league_prob"] > 0 and rng.random() < state["league_prob"]
        opp_gen, restore = _opponent_generator(state, opp, face_snapshot)

        for _ in range(K):
            if done >= state["total_rounds"]:
                break
            _step_round(state, learner, opp, opp_gen, si, _)
            done += 1
            _post_round_bookkeeping(state, done)

        restore()
    return done


def _checkpoint(policy, adapter_paths, progress_cb, done):
    """Atomically-ish save: adapters first, then advance the progress counter."""
    _save_adapters(policy, adapter_paths)
    if progress_cb:
        progress_cb(done)


def _save_adapters(policy, adapter_paths: dict):
    for name, path in adapter_paths.items():
        policy.save_adapter(name, path)


def _log_round(env, ctx, info, learner, stats, metrics_tracker, save_round_log,
               reward_att=None, reward_def=None, phase_index=0, phase_round=0,
               success=False, best_index=0, poisoned_ids=None):
    verdicts = info["verdicts"]
    post_acc = info["post_accuracy"]
    n_malformed = info["n_malformed"]
    poisoned_ids = poisoned_ids if poisoned_ids is not None else info.get("poisoned_ids", [])
    reward_att = reward_att or {}
    reward_def = reward_def or {}

    # Diversity of the committed (possibly multi-client) attack; 0 for one client.
    diversity = perturbation_diversity(
        info.get("poisoned_by_client", {}),
        {cid: env.pool_benign[cid] for cid in poisoned_ids if cid in env.pool_benign},
    )
    a_rew = attacker_reward(ctx.global_accuracy, post_acc, env.goal,
                            poisoned_ids, verdicts, n_malformed,
                            alpha=reward_att.get("alpha", 1.0),
                            beta=reward_att.get("beta", 0.5),
                            gamma=reward_att.get("gamma", 1.0),
                            delta=reward_att.get("delta", 0.0),
                            zeta=reward_att.get("zeta", 0.0),
                            pool_size=env.n_compromisable,
                            diversity=diversity)
    d_rew = defender_reward(verdicts, poisoned_ids,
                            mode=reward_def.get("mode", "soft_f1"),
                            fpr_penalty=reward_def.get("fpr_penalty", 1.0))

    metrics_tracker.update(ctx.round_num, verdicts, post_acc, set(poisoned_ids))
    save_round_log(RoundLog(
        round_num=ctx.round_num,
        attack_goal=env.goal,
        poisoned_client_ids=poisoned_ids,
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
            "budget": ctx.budget,
            "n_used": len(poisoned_ids),
            "controllable_pool": ctx.pool_ids,
            "attack_diversity": round(float(diversity), 4),
            "post_class_accuracy": info.get("post_class_accuracy"),
            "phase_index": phase_index,
            "phase_round": phase_round,
            "learner_success": success,
            "train": {
                "loss": stats["loss"],
                "mean_reward": stats["mean_reward"],
                "max_reward": stats["max_reward"],
                "zero_advantage_fraction": stats["zero_advantage_fraction"],
                "stepped": stats.get("stepped", True),
                "resampled": stats.get("resampled", False),
            },
        },
    ))
    dbg.commit_summary(
        learner, best_index, info, ctx.global_accuracy, post_acc,
        ctx.global_accuracy - post_acc, success, a_rew, d_rew, poisoned_ids,
    )
    dbg.flush()
    logger.info(
        f"Round {ctx.round_num} [learn={learner} ph={phase_index}.{phase_round} "
        f"{'WIN' if success else '...'}]: acc {ctx.global_accuracy:.4f}->{post_acc:.4f} "
        f"| att_reward={a_rew:.3f} def_reward={d_rew:.3f} "
        f"| grpo_loss={stats['loss']:.4f} mean_r={stats['mean_reward']:.3f} "
        f"zero_adv={stats['zero_advantage_fraction']:.2f} "
        f"{'step' if stats.get('stepped', True) else 'SKIP'}"
    )
