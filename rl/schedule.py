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

**Training curriculum.** Orthogonally to the phase schedule, ``rl/curriculum.py``
fixes WHICH defense algorithm and HOW MANY poisoners each round faces: one
(algorithm, #poisoners) pair per block of consecutive rounds, sweeping every pair
in turn. Phases, wins, caps and the FL interlude are unaffected — the interlude in
particular does not consume a curriculum slot, so a block always gets its full
complement of training rounds. The sweep position is checkpointed with the rest of
the resume state (see :func:`_resume_curriculum`).

**Defender LLM disabled.** When ``env.defense`` is set (``defense.mode:
algorithmic``), the defender is a published algorithm fixed per round rather than
a policy, so there is nothing on that side to train: the rotation collapses to
attacker-only phases, no opponent adapter is loaded/borrowed/snapshotted, and the
defender checkpoint on disk is left untouched. The phase machinery still runs —
a phase ends on a sustained win or the cap, which is what schedules the honest FL
interlude between phases and keeps the attacker ratcheting.
"""

import logging

from core.types import RoundLog
from core.debug import dbg
from rl.grpo import grpo_step
from rl.policy import PolicyGenerator
from rl.rewards import (
    DEFAULT_ADVANTAGE_STD_FLOOR, DEFAULT_MIN_REWARD_SPREAD,
    attack_potency, attacker_reward_terms, check_reward_balance, defender_reward,
    goal_target, perturbation_diversity,
)
from rl.switch import PhaseController, SwitchConfig, committed_success
from rl.turns import AttackerTurn, DefenderTurn

logger = logging.getLogger(__name__)


class League:
    """Bounded in-memory pool of past adapter snapshots (CPU LoRA state dicts).

    Each snapshot is a full copy of one adapter's LoRA tensors — for the default
    Qwen2.5-3B + ``lora_r: 16`` over 7 projections × 36 layers that is ~30M
    parameters, i.e. **~115 MB per adapter per snapshot** in fp32. The pool was
    previously unbounded and snapshotted BOTH adapters every
    ``league_snapshot_every`` rounds, so a long run grew it without limit (~22 GB
    of host RAM by round 10k, ~223 GB by round 100k with the shipped settings) and
    eventually OOM'd the box.

    It is now a ring buffer of at most ``max_snapshots`` per adapter: once full,
    the OLDEST snapshot is dropped. That keeps the anti-overfit benefit (the
    learner still faces a spread of past opponents) at a fixed memory ceiling of
    ``max_snapshots × n_adapters × ~115 MB``.
    """

    def __init__(self, rng, max_snapshots: int = 10):
        self.rng = rng
        self.max_snapshots = max(1, int(max_snapshots))
        self.snapshots: dict[str, list[dict]] = {}

    def snapshot(self, policy, names, states=None):
        """Append a snapshot of each named adapter, evicting the oldest past the
        cap. ``states`` optionally overrides the weights stored for a given name —
        used to snapshot a BORROWED opponent's LIVE weights instead of the older
        snapshot temporarily swapped into it, so the league doesn't accumulate
        stale duplicates."""
        evicted = 0
        for name in names:
            st = states[name] if (states and name in states) else policy.get_adapter_state(name)
            pool = self.snapshots.setdefault(name, [])
            pool.append(st)
            while len(pool) > self.max_snapshots:
                pool.pop(0)          # ring buffer: drop the oldest
                evicted += 1
        logger.info(f"League: snapshotted {list(names)} "
                    f"(sizes={ {k: len(v) for k, v in self.snapshots.items()} }, "
                    f"cap={self.max_snapshots}, evicted={evicted})")

    def has(self, name) -> bool:
        return bool(self.snapshots.get(name))

    def sample(self, name) -> dict:
        return self.rng.choice(self.snapshots[name])


def resolve_round_budget(configured: int, total_rounds=None, max_new_rounds=None,
                         start_round: int = 0) -> int:
    """Absolute round budget for this run (the loop runs while ``done < budget``).

    * ``total_rounds`` (``main.py --rounds N``) replaces ``fl.simulation_rounds``
      as the ABSOLUTE budget — matching what ``--rounds`` documents, so a resumed
      run counts the rounds it already did toward N.
    * ``max_new_rounds`` (``main.py --debug``) additionally caps how many rounds
      THIS invocation may add on top of ``start_round``, so a short debug run still
      executes rounds when resuming a long training run.
    """
    budget = int(configured if total_rounds is None else total_rounds)
    if max_new_rounds is not None:
        budget = min(budget, int(start_round) + int(max_new_rounds))
    return budget


def _curriculum_slot(env) -> dict | None:
    """This round's curriculum block as a log-friendly dict, or ``None`` when the
    run has no curriculum (the defense/quota are drawn at random)."""
    slot = getattr(env, "round_curriculum", None)
    return slot.as_log_dict() if slot is not None else None


def _resume_curriculum(env, resume: dict | None, start_round: int) -> None:
    """Put the (defense, #poisoners) sweep back where it stopped.

    Prefer the saved ``curriculum`` snapshot. Older progress files predate it, so
    fall back to ``start_round``: the curriculum advances exactly once per
    committed GRPO round (``begin_round``) and ``rounds_done`` counts exactly
    those rounds — the FL interlude between phases advances neither — so the two
    counters are the same number. Without this a restart would rewind the sweep
    to block 0 and re-train the first algorithm at one poisoner every time.
    """
    curriculum = getattr(env, "curriculum", None)
    if curriculum is None:
        return
    saved = (resume or {}).get("curriculum")
    if saved:
        curriculum.load_state_dict(saved)
    elif start_round:
        curriculum.load_state_dict({"step": int(start_round)})
        logger.info(
            f"No saved curriculum position (older checkpoint) — deriving it from "
            f"rounds_done={start_round}, which counts the same rounds."
        )
    slot = curriculum.peek()
    logger.info(
        f"Curriculum resumes at step {curriculum.step}: block {slot.block} "
        f"(cycle {slot.cycle}, {slot.position + 1}/{curriculum.blocks_per_cycle}) "
        f"round {slot.block_round + 1}/{curriculum.rounds_per_block} — "
        f"defense={slot.algorithm or 'llm'}, {slot.n_poisoners} poisoner(s)"
    )


def _argmax(xs: list[float]) -> int:
    best, bi = xs[0], 0
    for i, x in enumerate(xs):
        if x > best:
            best, bi = x, i
    return bi


def _committed_index(rewards: list[float], mode: str, rng) -> int:
    """Which of the G scored rollouts actually advances the environment.

    ``sample`` (default) — a uniformly random member of the group, i.e. a genuine
    draw from the policy. ``argmax`` — the best-scoring one (the legacy behaviour).

    Committing the argmax is a best-of-G evaluation dressed up as a policy rollout,
    and it distorts three things at once:

    * **The success gate lies.** ``rl/switch.py`` counts a "win" on the committed
      round, so every phase switch, every ``learner_success`` in the logs, and the
      whole arms-race ratchet were measuring max-of-G rather than the policy. The
      benchmark samples ONCE, so those numbers do not reproduce at eval.
    * **Selection amplifies reward noise.** Taking the max over G noisy scores is
      biased upward by roughly the noise scale (and the argmax is chosen on scores
      whose defense draw is re-run at commit time anyway).
    * **The state trajectory is off-policy-optimistic.** The learner only ever sees
      the world its best rollout produced, never the consequences of its median
      behaviour — which is what it will be judged on.

    The argmax is still recorded (``best_index``) so the gap between the two is
    visible in the round logs.
    """
    if not rewards:
        return 0
    if str(mode).lower() == "argmax":
        return _argmax(rewards)
    return rng.randrange(len(rewards))


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
    fl_state_cb=None,
    start_round: int = 0,
    resume: dict | None = None,
    total_rounds: int | None = None,
    max_new_rounds: int | None = None,
):
    """Run the alternating GRPO training loop over ``simulation_rounds`` rounds.

    ``resume`` is the dict from ``storage.load_progress`` — ``rounds_done`` (also
    passed as ``start_round``), plus, for a full resume, the saved FL ``round_index``
    (so round numbering/logs continue) and ``controller`` snapshot (so the arms-race
    schedule continues instead of restarting at the first attacker phase).

    ``total_rounds`` overrides ``fl.simulation_rounds`` as the ABSOLUTE round
    budget (``main.py --rounds N``); ``max_new_rounds`` caps how many rounds THIS
    invocation may add on top of ``start_round`` (``main.py --debug``). Both were
    previously computed in ``main.py`` and then dropped on the floor, so
    ``--rounds``/``--debug`` silently ran the full ``simulation_rounds`` budget.
    """
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
    league_max = int(rl.get("league_max_snapshots", 10))
    skip_zero_adv = bool(rl.get("skip_zero_advantage", True))
    resample_zero_adv = bool(rl.get("resample_on_zero_advantage", True))
    resample_temp = float(rl.get("resample_temperature", 1.3))
    min_reward_spread = float(rl.get("min_reward_spread", DEFAULT_MIN_REWARD_SPREAD))
    advantage_std_floor = float(rl.get("advantage_std_floor", DEFAULT_ADVANTAGE_STD_FLOOR))
    commit_selection = str(rl.get("commit_selection", "sample")).lower()
    if commit_selection not in ("sample", "argmax"):
        raise ValueError(f"rl.commit_selection must be sample|argmax, got {commit_selection!r}")
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
    # Verify up front that damage still outbids the shaping terms. The invariant
    # couples four values across two config blocks and has been broken twice by
    # changing one of them alone; both times the symptom was a policy that trained
    # for hundreds of rounds and learned to look harmless.
    check_reward_balance(reward_att, env.goal, context="train")

    total_rounds = resolve_round_budget(
        int(cfg["fl"]["simulation_rounds"]), total_rounds, max_new_rounds, start_round)
    if start_round >= total_rounds:
        logger.warning(
            f"Nothing to do: {start_round} round(s) already done and the budget is "
            f"{total_rounds}. Raise fl.simulation_rounds or pass a larger --rounds."
        )
    else:
        logger.info(f"Training budget: rounds {start_round} -> {total_rounds}")

    # With an algorithmic defense there is no defender POLICY: only the attacker
    # trains, and the defender adapter (on disk and in the league) is left alone.
    algorithmic_defense = getattr(env, "defense", None) is not None
    trainable = ("attacker",) if algorithmic_defense else ("attacker", "defender")
    if algorithmic_defense:
        if first_learner != "attacker":
            logger.warning(
                f"rl.first_learner={first_learner!r} ignored — the defender LLM is "
                f"disabled (algorithmic defense), so only the attacker trains."
            )
            first_learner = "attacker"
        logger.info(
            f"Defender LLM disabled: attacker-only training against "
            f"{env.defense.describe()}"
        )

    import torch
    optimizers = {
        name: torch.optim.AdamW(policy.adapter_parameters(name), lr=lr)
        for name in trainable
    }
    league = League(rng, max_snapshots=league_max)

    # Per-round knobs bundled so both schedules share one round body.
    knobs = dict(
        G=G, kl_beta=kl_beta, learner_temp=learner_temp, max_new_tokens=max_new_tokens,
        grad_clip=grad_clip, opp_temp=opp_temp, scoring_opp_temp=scoring_opp_temp,
        skip_zero_adv=skip_zero_adv, resample_zero_adv=resample_zero_adv,
        resample_temp=resample_temp, reward_att=reward_att, reward_def=reward_def,
        min_reward_spread=min_reward_spread, advantage_std_floor=advantage_std_floor,
        commit_selection=commit_selection,
    )

    state = dict(
        env=env, policy=policy, attacker_agent=attacker_agent, defender_agent=defender_agent,
        optimizers=optimizers, league=league, knobs=knobs, switch_cfg=switch_cfg,
        metrics_tracker=metrics_tracker, save_round_log=save_round_log,
        adapter_paths=adapter_paths, progress_cb=progress_cb, total_rounds=total_rounds,
        save_every=save_every, snap_every=snap_every, league_prob=league_prob,
        max_new_tokens=max_new_tokens, rng=rng, curriculum_on_cap=curriculum_on_cap,
        fl_interlude=fl_interlude, controller=None,
        fl_state_cb=fl_state_cb, borrowed_opponent=None,
        trainable=trainable, algorithmic_defense=algorithmic_defense,
    )

    # Resume the FL round-number counter so round labels + logs/round_data advance
    # across restarts instead of overwriting from the first Phase-2 round. Prefer the
    # saved round_index; fall back to start_round for old progress files that only
    # stored rounds_done. (env.reset() zeroed round_index just before this.)
    if resume and resume.get("round_index") is not None:
        env.round_index = int(resume["round_index"])
    elif start_round:
        env.round_index = int(start_round)

    _resume_curriculum(env, resume, start_round)

    if switch_mode == "best_response":
        logger.info(
            f"Schedule=best_response: success-gated iterated best response "
            f"(first_learner={first_learner}, streak={switch_cfg.success_streak}, "
            f"min/max phase={switch_cfg.min_phase_rounds}/{switch_cfg.max_phase_rounds})"
        )
        done = _train_best_response(state, first_learner, start_round, resume=resume)
    else:
        logger.info(f"Schedule=fixed: K_a={K_a}, K_d={K_d}")
        done = _train_fixed(state, K_a, K_d, start_round)

    _checkpoint(state, done)   # final save
    logger.info(f"Training complete — {done} rounds. Adapters saved to {adapter_paths}")


def _step_round(state, learner, opp, opp_gen, phase_index, phase_round):
    """Run ONE committed GRPO round for ``learner`` vs the frozen ``opp``.

    Returns ``(stats, committed_drop, success)`` where ``committed_drop`` is the
    accuracy lost on the committed round (prev - post) and ``success`` is whether
    the learner won this round (per the success-gate thresholds)."""
    env = state["env"]
    k = state["knobs"]
    ctx = env.begin_round()

    # With an algorithmic defense the "opponent" is this round's drawn algorithm,
    # not a frozen adapter — label it as such in the debug view.
    opp_label = (f"algo:{env.round_defense}" if state.get("algorithmic_defense") else opp)
    dbg.round_header(ctx.round_num, learner, opp_label, phase_index, phase_round,
                     ctx.pool_ids, ctx.budget, ctx.global_accuracy, k["G"],
                     k["scoring_opp_temp"], k["opp_temp"],
                     curriculum=_curriculum_slot(env))
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

    # The attacker's reward is dominated by the damage term, which is measured
    # against the clean counterfactual. Two round states make that measurement
    # meaningless, and on both we score the group (the environment still needs a
    # rollout to advance) but apply NO gradient. The defender's reward is
    # detection-only, so it is unaffected by either.
    #
    # 1. NO COUNTERFACTUAL — the defense refused to aggregate even the unpoisoned
    #    updates, so the damage term is structurally 0 for every rollout while
    #    LOOKING like a measured round.
    # 2. BROKEN DEFENSE — the defense DID aggregate, but only after rejecting the
    #    honest majority of the unpoisoned cohort. The accuracies are then a reading
    #    of the defense's false-positive rate, and the drop between two such
    #    readings is noise wearing the costume of an attack result. 45 of 262 rounds
    #    in a recorded run were in this state (FLTrust, mean FPR 0.43) and every one
    #    of them contributed a gradient.
    unmeasured = learner == "attacker" and not getattr(ctx, "clean_measured", True)
    malfunctioned = learner == "attacker" and not getattr(ctx, "defense_sane", True)
    void = unmeasured or malfunctioned
    skip_reason = ""
    if unmeasured:
        skip_reason = (f"{env.round_defense or 'the aggregator'} produced no clean "
                       "aggregate, so the damage term is unmeasurable")
    elif malfunctioned:
        skip_reason = (f"{env.round_defense or 'the aggregator'} rejected the honest "
                       "majority of the UNPOISONED cohort, so this round measures the "
                       "defense, not the attack")

    stats = grpo_step(
        state["policy"], learner, state["optimizers"][learner], turn,
        G=k["G"], kl_beta=k["kl_beta"], temperature=k["learner_temp"],
        max_new_tokens=k["max_new_tokens"], grad_clip=k["grad_clip"],
        skip_zero_advantage=k["skip_zero_adv"],
        resample_on_zero_advantage=k["resample_zero_adv"],
        resample_temperature=k["resample_temp"],
        min_reward_spread=k["min_reward_spread"],
        advantage_std_floor=k["advantage_std_floor"],
        skip_update=void,
        skip_reason=skip_reason,
    )

    # Advance the env with an ON-POLICY draw from the group (see _committed_index):
    # committing the argmax made every logged win a best-of-G result.
    best = _argmax(stats["rewards"]) if stats["rewards"] else 0
    committed = _committed_index(stats["rewards"], k["commit_selection"], state["rng"])
    info = turn.commit(stats["completions"][committed])
    poisoned_ids = info["poisoned_ids"]        # the attacker's committed choice

    # Damage vs THIS round's clean counterfactual (what the aggregate would have
    # scored unpoisoned), so the win-gate measures the attack — not the change
    # since the previous round. See FLArmsRaceEnv.clean_reference_accuracy.
    drop = ctx.clean_accuracy - info["post_accuracy"]
    # A void round cannot certify a win either: the phase gate feeds
    # ``success_streak``, and crediting a win on a round whose damage was never
    # measured — or was measured through a defense that had already dropped the
    # honest majority — would switch phases on noise.
    success = (not void) and committed_success(
        learner, drop, info["verdicts"], poisoned_ids, state["switch_cfg"], ctx.goal)

    # The relative damage bar this round had to clear, shared with the metrics
    # tracker so the recorded attack_success means the same thing as the win gate.
    success_drop = state["switch_cfg"].win_fraction * goal_target(ctx.goal)

    _log_round(env, ctx, info, learner, stats, state["metrics_tracker"],
               state["save_round_log"], reward_att=k["reward_att"], reward_def=k["reward_def"],
               phase_index=phase_index, phase_round=phase_round, success=success,
               best_index=best, committed_index=committed, poisoned_ids=poisoned_ids,
               defense_algorithm=env.round_defense, success_drop=success_drop)
    return stats, drop, success


def _post_round_bookkeeping(state, done):
    """League snapshots + checkpoints on the configured cadence."""
    if state["snap_every"] and done % state["snap_every"] == 0:
        borrowed = state.get("borrowed_opponent")
        overrides = {borrowed[0]: borrowed[1]} if borrowed else None
        # Only trainable adapters are snapshotted: with the defender LLM disabled
        # its adapter never changes, so pooling it would just burn host RAM.
        state["league"].snapshot(state["policy"], state["trainable"], states=overrides)
    # Checkpoint adapters + progress TOGETHER so a resume is always consistent
    # (the saved round count never points past the saved adapter weights).
    if state["save_every"] and done % state["save_every"] == 0:
        _checkpoint(state, done)


def _opponent_generator(state, opp, face_snapshot):
    """Return (opp_gen, restore_fn). If ``face_snapshot`` and a snapshot exists,
    temporarily swap the opponent adapter for a past (weaker) snapshot — used for
    the league mix and for the curriculum after a capped phase.

    While swapped, ``state["borrowed_opponent"] = (opp, live_weights)`` so any
    mid-phase checkpoint / league snapshot persists the opponent's LIVE weights,
    not the borrowed snapshot — otherwise the real opponent adapter would be
    clobbered on disk and silently regress on resume. ``restore`` puts the live
    weights back and clears the marker.

    With an algorithmic defense there is no opponent adapter at all, so this is a
    no-op returning ``None`` as the generator — ``AttackerTurn`` does not use one."""
    if state.get("algorithmic_defense"):
        return None, (lambda: None)
    policy = state["policy"]
    live_opp = policy.get_adapter_state(opp)
    used = False
    if face_snapshot and state["league"].has(opp):
        policy.set_adapter_state(opp, state["league"].sample(opp))
        used = True
        state["borrowed_opponent"] = (opp, live_opp)
        logger.info(f"Phase: facing a LEAGUE snapshot of {opp}")
    opp_gen = PolicyGenerator(policy, opp, state["max_new_tokens"])

    def restore():
        if used:
            policy.set_adapter_state(opp, live_opp)
        state["borrowed_opponent"] = None
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


def _train_best_response(state, first_learner, start_round, resume=None):
    """Success-gated iterated best response. Returns the final round count."""
    ctrl = PhaseController(state["switch_cfg"], first_learner=first_learner,
                           learners=state["trainable"])
    if resume and resume.get("controller"):
        ctrl.load_state_dict(resume["controller"])
        logger.info(
            f"Resumed schedule state: learner={ctrl.learner} phase={ctrl.phase_index} "
            f"phase_round={ctrl.phase_round} streak={ctrl.streak} capped={ctrl.capped}"
        )
    state["controller"] = ctrl   # exposed so _checkpoint can persist it
    rng = state["rng"]
    done = start_round

    while done < state["total_rounds"]:
        learner, opp = ctrl.learner, ctrl.opponent
        # Before every phase AFTER the first, run one honest FL round so the
        # incoming learner + frozen opponent + aggregator train against a freshly
        # advanced client state (mirrors a Phase-1 round). Gated on phase_round==0
        # so it fires only at a TRUE phase start — NOT when resuming into the middle
        # of a phase (which would otherwise inject a spurious benign round on every
        # restart, bumping accuracy). The first phase (index 0) uses the Phase-1
        # checkpoint as-is.
        if (ctrl.phase_index > 0 and ctrl.phase_round == 0
                and state.get("fl_interlude", True)):
            _run_fl_interlude(state, next_learner=learner, phase_index=ctrl.phase_index)
        # Curriculum: a phase that capped without a win means the opponent is too
        # strong — let this learner face an earlier snapshot of it. Otherwise mix
        # in a league snapshot with probability league_prob (anti-overfit).
        # Neither applies to an algorithmic defense (no opponent adapter exists).
        face_snapshot = (not state["algorithmic_defense"]) and (
            (state["curriculum_on_cap"] and ctrl.capped)
            or (state["league_prob"] > 0 and rng.random() < state["league_prob"])
        )
        opp_gen, restore = _opponent_generator(state, opp, face_snapshot)
        dbg.phase_event("PHASE START", phase=ctrl.phase_index, learner=learner,
                        opponent=("algorithms" if state["algorithmic_defense"] else opp),
                        facing_snapshot=face_snapshot)

        reason = None
        while done < state["total_rounds"]:
            _stats, _drop, success = _step_round(
                state, learner, opp, opp_gen, ctrl.phase_index, ctrl.phase_round
            )
            done += 1
            # Record the round on the controller FIRST, then checkpoint — so the
            # persisted controller state (phase_round/streak) and rounds_done refer
            # to the SAME completed round. (Previously the checkpoint ran before
            # record(), leaving the two off by one across a resume.)
            switch, reason = ctrl.record(success)
            _post_round_bookkeeping(state, done)
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
        # Persist the just-completed switch (new learner/phase in the controller,
        # the frozen adapters, and the shared FL state) so a crash right after a
        # handoff resumes in the NEW phase instead of re-running the finished one.
        _checkpoint(state, done)

    return done


def _train_fixed(state, K_a, K_d, start_round):
    """Legacy fixed-clock alternation. Returns the final round count."""
    rng = state["rng"]
    done = start_round
    schedule = [(name, K_a if name == "attacker" else K_d)
                for name in state["trainable"]]
    si = 0
    while done < state["total_rounds"]:
        learner, K = schedule[si % len(schedule)]
        si += 1
        opp = "defender" if learner == "attacker" else "attacker"
        face_snapshot = (not state["algorithmic_defense"]
                         and state["league_prob"] > 0
                         and rng.random() < state["league_prob"])
        opp_gen, restore = _opponent_generator(state, opp, face_snapshot)

        for _ in range(K):
            if done >= state["total_rounds"]:
                break
            _step_round(state, learner, opp, opp_gen, si, _)
            done += 1
            _post_round_bookkeeping(state, done)

        restore()
    return done


def _checkpoint(state, done):
    """Atomically-ish save: adapters + the shared FL state, then the resume state
    (round count, FL round index, the PhaseController snapshot and the training
    curriculum's position) so a resume is always consistent — the saved round count
    never points past the saved weights, the shared model continues instead of
    rewinding to the Phase-1 baseline, and the (defense, #poisoners) sweep picks up
    mid-block instead of restarting."""
    _save_adapters(state)
    if state.get("fl_state_cb"):
        state["fl_state_cb"](state["env"].snapshot_fl_state())
    if state["progress_cb"]:
        ctrl = state.get("controller")
        cur = getattr(state["env"], "curriculum", None)
        state["progress_cb"](done, state["env"].round_index,
                             ctrl.state_dict() if ctrl is not None else None,
                             cur.state_dict() if cur is not None else None)


def _save_adapters(state):
    """Persist every adapter. If the opponent adapter is currently BORROWED (its
    live weights temporarily swapped for an older league snapshot during a
    curriculum/league phase), write its LIVE weights instead of the borrowed
    snapshot, so a mid-phase checkpoint can't overwrite the real opponent adapter
    on disk (which would silently regress the frozen opponent on resume).

    Only TRAINABLE adapters are written: with the defender LLM disabled its
    checkpoint must survive untouched so the run can be switched back later."""
    policy = state["policy"]
    borrowed = state.get("borrowed_opponent")
    for name in state["trainable"]:
        path = state["adapter_paths"][name]
        if borrowed and borrowed[0] == name:
            policy.save_adapter_state_dict(name, borrowed[1], path)
        else:
            policy.save_adapter(name, path)


def _warn_defense_malfunction(round_num, defense_algorithm, verdicts, poisoned_ids):
    """Shout when the round's defense behaved in a way that invalidates the reward.

    A robust aggregator that rejects the honest majority is not "a hard round for the
    attacker" — it is a broken defense, and every quantity derived from that round is
    meaningless: the clean counterfactual, the post-attack accuracy, the drop, the
    reward. This went undiagnosed for a whole recorded run because the symptoms all
    look like ordinary numbers (an accuracy, a small drop, a plausible reward).

    The loudest case is a strict subset accept: FLTrust dropping all 19 honest clients
    and keeping ONLY the poisoned one, so the "defended" aggregate is built purely
    from the attack. See ``server.algo_defender.resolve_root_epochs`` for the cause
    this detector was written to catch.
    """
    if not verdicts:
        return
    accepted = [v.client_id for v in verdicts if not v.is_suspicious]
    poisoned = set(poisoned_ids or ())
    tag = f"Round {round_num} [{defense_algorithm or 'llm'}]"
    if not accepted:
        logger.warning("%s: defense REJECTED ALL %d clients — no aggregate is possible "
                       "and this round's reward carries no information",
                       tag, len(verdicts))
        return
    if poisoned and set(accepted) <= poisoned:
        logger.warning(
            "%s: defense accepted ONLY POISONED clients (%s of %d) — the aggregate is "
            "built entirely from the attack, so the measured drop reflects a defense "
            "malfunction, not the attacker's skill", tag, sorted(accepted), len(verdicts))
        return
    honest_total = len(verdicts) - len(poisoned)
    honest_rejected = honest_total - len([c for c in accepted if c not in poisoned])
    if honest_total > 0 and honest_rejected > honest_total / 2:
        logger.warning(
            "%s: defense rejected %d of %d HONEST clients (FPR=%.2f) — a robust "
            "aggregator dropping the honest majority indicates a misconfigured "
            "defense, not a strong attack", tag, honest_rejected, honest_total,
            honest_rejected / honest_total)


def _log_round(env, ctx, info, learner, stats, metrics_tracker, save_round_log,
               reward_att=None, reward_def=None, phase_index=0, phase_round=0,
               success=False, best_index=0, committed_index=0, poisoned_ids=None,
               defense_algorithm=None, success_drop=None):
    verdicts = info["verdicts"]
    post_acc = info["post_accuracy"]
    n_malformed = info["n_malformed"]
    poisoned_ids = poisoned_ids if poisoned_ids is not None else info.get("poisoned_ids", [])
    reward_att = reward_att or {}
    reward_def = reward_def or {}
    # This round's actual goal (per-round target sampling); falls back to the env default.
    goal = ctx.goal if getattr(ctx, "goal", None) is not None else env.goal

    # Diversity of the committed (possibly multi-client) attack; 0 for one client.
    poisoned_by_client = info.get("poisoned_by_client", {})
    refs = {cid: env.pool_benign[cid] for cid in poisoned_ids if cid in env.pool_benign}
    diversity = perturbation_diversity(poisoned_by_client, refs)
    # Size of the committed poison relative to the honest update it hides in; gates
    # the stealth term exactly as it does when the rollouts were scored, so the
    # logged reward is the one the policy was actually trained on.
    potency = attack_potency(poisoned_by_client, refs, env.global_weights)
    terms = attacker_reward_terms(ctx.clean_accuracy, post_acc, goal,
                                  poisoned_ids, verdicts, n_malformed,
                                  alpha=reward_att.get("alpha", 1.0),
                                  beta=reward_att.get("beta", 0.5),
                                  gamma=reward_att.get("gamma", 1.0),
                                  zeta=reward_att.get("zeta", 0.0),
                                  diversity=diversity,
                                  potency=potency)
    a_rew = terms["total"]
    d_rew = defender_reward(verdicts, poisoned_ids,
                            mode=reward_def.get("mode", "soft_f1"),
                            fpr_penalty=reward_def.get("fpr_penalty", 1.0))

    clean_measured = bool(getattr(ctx, "clean_measured", True))
    defense_sane = bool(getattr(ctx, "defense_sane", True))
    _warn_defense_malfunction(ctx.round_num, defense_algorithm, verdicts, poisoned_ids)

    # ``clean_accuracy`` is passed only when it was really measured AND measured
    # through a defense that was working: the tracker derives the damage-based
    # attack_success from it, and must treat neither the current-global fallback
    # (clean_measured=False) nor a counterfactual built after the honest majority
    # was rejected (defense_sane=False) as a counterfactual. Both would credit the
    # attacker with the defense's own noise.
    countable = clean_measured and defense_sane
    metrics_tracker.update(ctx.round_num, verdicts, post_acc, set(poisoned_ids),
                           clean_accuracy=(ctx.clean_accuracy if countable else None),
                           success_drop=success_drop)
    save_round_log(RoundLog(
        round_num=ctx.round_num,
        attack_goal=goal,
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
            # Which defense faced this round: an algorithm name, or "llm" when the
            # defender LLM is in charge. Rounds are only comparable within a defense.
            "defense": defense_algorithm or "llm",
            # The (defense, #poisoners) block this round belongs to, so a run can be
            # sliced by block without re-deriving it from the round number. None when
            # the run draws the defense/quota at random instead of sweeping them.
            "curriculum": _curriculum_slot(env),
            "attack_diversity": round(float(diversity), 4),
            # ‖poison‖/‖honest update‖ squashed into [0,1) — how hard the committed
            # attack actually pushed. Near 0 with a high reward means the policy is
            # farming the stealth term instead of attacking (see attack_potency).
            "attack_potency": round(float(potency), 4),
            # The attacker reward broken into its WEIGHTED terms. `damage` must be
            # the dominant one on any round the attack actually worked; a run where
            # `stealth` carries the reward is a run learning to hide, not to attack
            # — which is exactly what a recorded 262-round run did while looking
            # healthy in every other field. Summing to `attacker_reward`.
            "reward_terms": {k: round(float(v), 6) for k, v in terms.items()},
            # The clean counterfactual and the damage measured against it — the
            # quantities the reward and the win-gate actually use, so monitor.py
            # and the visualizer report the same number the policy is trained on.
            "clean_accuracy": round(float(ctx.clean_accuracy), 6),
            "induced_drop": round(float(ctx.clean_accuracy - post_acc), 6),
            # False = the defense produced no clean aggregate, so `clean_accuracy` is
            # the current global's accuracy standing in for a counterfactual that does
            # not exist and `induced_drop` is not a measurement. Slice these rounds OUT
            # before reporting damage; the policy was not updated on them either.
            "clean_measured": clean_measured,
            # False = the round's defense had already rejected the honest majority of
            # the UNPOISONED cohort, so `clean_accuracy` / `induced_drop` measure its
            # false-positive rate rather than the attack. Slice these out too; no
            # policy gradient was applied on them either.
            "defense_sane": defense_sane,
            "phase_index": phase_index,
            "phase_round": phase_round,
            "learner_success": success,
            "train": {
                "loss": stats["loss"],
                "mean_reward": stats["mean_reward"],
                "max_reward": stats["max_reward"],
                # Raw within-group reward span: the quantity the degeneracy gate
                # tests, so a run of skipped steps is diagnosable from the logs.
                "reward_spread": stats.get("reward_spread"),
                # Which rollout advanced the env vs which scored best. They now
                # differ by design (an on-policy draw, not best-of-G), so the gap is
                # the honest measure of how optimistic the old argmax commit was.
                "committed_index": committed_index,
                "best_index": best_index,
                "committed_reward": (stats["rewards"][committed_index]
                                     if stats["rewards"] else None),
                "total_tokens": stats.get("total_tokens"),
                "completion_tokens": stats.get("completion_tokens"),
                "zero_advantage_fraction": stats["zero_advantage_fraction"],
                # Degeneracy of the FIRST draw, before any re-roll. The field above is
                # post-resample, so on its own it reports a healthy group for a round
                # that only became healthy on the second attempt.
                "zero_advantage_fraction_first_draw":
                    stats.get("zero_advantage_fraction_first_draw"),
                "stepped": stats.get("stepped", True),
                "skipped_by_caller": stats.get("skipped_by_caller", False),
                "resampled": stats.get("resampled", False),
            },
        },
    ))
    dbg.commit_summary(
        learner, committed_index, info, ctx.clean_accuracy, post_acc,
        ctx.clean_accuracy - post_acc, success, a_rew, d_rew, poisoned_ids,
        global_acc=ctx.global_accuracy, best_index=best_index,
    )
    dbg.flush()
    slot = getattr(env, "round_curriculum", None)
    # ph / blk are "index/round" pairs, NOT decimals: `ph=0/14` is phase 0 round 14.
    # They used to be joined with a '.', which read as a float and made `blk=1.9` ->
    # `blk=2.0` look like a 0.1 step when it is block 1 round 9 -> block 2 round 0.
    logger.info(
        f"Round {ctx.round_num} [learn={learner} ph={phase_index}/{phase_round} "
        f"def={defense_algorithm or 'llm'} "
        + (f"blk={slot.block}/{slot.block_round} n_pois={slot.n_poisoners} " if slot else "")
        + f"{'WIN' if success else '...'}]: acc {ctx.global_accuracy:.4f}->{post_acc:.4f} "
        f"(clean_ref={ctx.clean_accuracy:.4f}{'' if clean_measured else '~UNMEASURED'} "
        f"drop={ctx.clean_accuracy - post_acc:+.4f}) "
        # Break the attacker reward out by term. Reading `att_reward` alone hid the
        # central failure of a whole recorded run: a healthy-looking 0.47 that was
        # 93% stealth and ~0% damage.
        f"| att_reward={a_rew:.3f}(dmg={terms['damage']:+.3f} stl={terms['stealth']:+.3f} "
        f"mal={terms['malformed']:+.3f} col={terms['collab']:+.3f} pot={potency:.2f}) "
        f"def_reward={d_rew:.3f} "
        f"| grpo_loss={stats['loss']:.4f} mean_r={stats['mean_reward']:.3f} "
        f"spread={stats.get('reward_spread', 0.0):.3f} "
        f"zero_adv={stats['zero_advantage_fraction']:.2f}"
        # The first draw's degeneracy is the honest read on whether the reward still
        # separates the policy's own samples; the field above hides a re-roll.
        f"(first={stats.get('zero_advantage_fraction_first_draw', 0.0):.2f}) "
        f"n_malformed={n_malformed}/{ctx.budget} "
        + ("resampled " if stats.get("resampled") else "")
        + ("step" if stats.get("stepped", True)
           else ("SKIP-degenerate" if not stats.get("skipped_by_caller")
                 else ("SKIP-unmeasured" if not clean_measured else "SKIP-defense-broken")))
    )
