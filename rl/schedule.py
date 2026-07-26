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

A third mode short-circuits the alternation entirely: when ``train`` is given an
``algorithmic_defender`` (``python main.py --env linux --freeze defender``) the
defender LLM is out of the loop, the defense is a FIXED ensemble of the classical
robust-aggregation algorithms, and ONLY the attacker is trained — see
:func:`_train_attacker_only`.

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
from rl.defenders import LLMDefenderPolicy
from rl.grpo import grpo_step
from rl.policy import PolicyGenerator
from rl.rewards import (
    attacker_reward, defender_reward, goal_drop, perturbation_diversity, targeted_terms,
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
    fl_state_cb=None,
    start_round: int = 0,
    resume: dict | None = None,
    total_rounds: int | None = None,
    max_new_rounds: int | None = None,
    algorithmic_defender=None,
    adapter_paths: dict | None = None,
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

    ``algorithmic_defender`` (``--freeze defender``) replaces the defender LLM with
    a fixed non-LLM ensemble: no defender adapter, no defender optimizer, no
    learner switching, no opponent league — only the attacker is trained.
    ``adapter_paths`` overrides ``rl.adapter_paths`` so the caller can decide where
    this run's adapters live (the frozen-defender run keeps its own, so it can
    never overwrite the arms-race attacker).
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
    switch_mode = str(rl.get("switch_mode", "best_response"))
    first_learner = str(rl.get("first_learner", "attacker"))
    curriculum_on_cap = bool(rl.get("curriculum_on_cap", True))
    fl_interlude = bool(rl.get("fl_interlude_between_phases", True))
    if adapter_paths is None:
        adapter_paths = rl.get("adapter_paths", {
            "attacker": "checkpoints/attacker_adapter",
            "defender": "checkpoints/defender_adapter",
        })
    reward_cfg = rl.get("reward", {})
    reward_att = reward_cfg.get("attacker", {})
    reward_def = reward_cfg.get("defender", {})
    switch_cfg = SwitchConfig.from_cfg(rl)

    total_rounds = resolve_round_budget(
        int(cfg["fl"]["simulation_rounds"]), total_rounds, max_new_rounds, start_round)
    if start_round >= total_rounds:
        logger.warning(
            f"Nothing to do: {start_round} round(s) already done and the budget is "
            f"{total_rounds}. Raise fl.simulation_rounds or pass a larger --rounds."
        )
    else:
        logger.info(f"Training budget: rounds {start_round} -> {total_rounds}")

    import torch
    # With a frozen (non-LLM) defense there is no defender adapter to optimize —
    # asking for its parameters would build an AdamW over an empty list.
    learners = ("attacker",) if algorithmic_defender is not None else ("attacker", "defender")
    optimizers = {name: torch.optim.AdamW(policy.adapter_parameters(name), lr=lr)
                  for name in learners}
    league = League(rng, max_snapshots=league_max)

    if algorithmic_defender is not None:
        # The league only ever supplies OPPONENT snapshots, and the opponent here
        # is a fixed algorithm — so snapshotting costs host RAM for nothing.
        snap_every = 0
        league_prob = 0.0
        adapter_paths = {"attacker": adapter_paths["attacker"]}

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
        fl_interlude=fl_interlude, controller=None,
        fl_state_cb=fl_state_cb, borrowed_opponent=None,
        algorithmic_defender=algorithmic_defender,
    )

    # Resume the FL round-number counter so round labels + logs/round_data advance
    # across restarts instead of overwriting from the first Phase-2 round. Prefer the
    # saved round_index; fall back to start_round for old progress files that only
    # stored rounds_done. (env.reset() zeroed round_index just before this.)
    if resume and resume.get("round_index") is not None:
        env.round_index = int(resume["round_index"])
    elif start_round:
        env.round_index = int(start_round)

    if algorithmic_defender is not None:
        logger.info(
            f"Schedule=frozen_defender: the defender LLM is OFF — only the attacker "
            f"trains, against {algorithmic_defender.describe()} "
            f"(streak={switch_cfg.success_streak}, "
            f"min/max phase={switch_cfg.min_phase_rounds}/{switch_cfg.max_phase_rounds})"
        )
        done = _train_attacker_only(state, start_round, resume=resume)
    elif switch_mode == "best_response":
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


def _defender_for(state, opp_gen):
    """The verdict source an ATTACKER round faces.

    The fixed algorithmic ensemble when the defender is frozen
    (``--freeze defender``), otherwise the frozen defender LLM behind ``opp_gen``.
    """
    algorithmic = state.get("algorithmic_defender")
    if algorithmic is not None:
        return algorithmic
    return LLMDefenderPolicy(state["defender_agent"], opp_gen)


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
            env, state["attacker_agent"], _defender_for(state, opp_gen),
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

    # Damage vs THIS round's clean counterfactual (what the aggregate would have
    # scored unpoisoned), so the win-gate measures the attack — not the change
    # since the previous round. See FLArmsRaceEnv.clean_reference_accuracy.
    # For a targeted_label goal this is the TARGET CLASS's recall drop instead of
    # the overall accuracy drop, and ``terms`` carries the collateral the win-gate
    # additionally requires to stay small (rl.rewards.targeted_terms).
    terms = targeted_terms(ctx.goal, ctx.clean_eval, info.get("post_eval"))
    drop = goal_drop(ctx.goal, ctx.clean_accuracy, info["post_accuracy"],
                     ctx.clean_eval, info.get("post_eval"))
    success = committed_success(learner, drop, info["verdicts"], poisoned_ids,
                                state["switch_cfg"], ctx.goal, terms)

    _log_round(env, ctx, info, learner, stats, state["metrics_tracker"],
               state["save_round_log"], reward_att=k["reward_att"], reward_def=k["reward_def"],
               phase_index=phase_index, phase_round=phase_round, success=success,
               best_index=best, poisoned_ids=poisoned_ids, terms=terms)
    return stats, drop, success


def _post_round_bookkeeping(state, done):
    """League snapshots + checkpoints on the configured cadence."""
    if state["snap_every"] and done % state["snap_every"] == 0:
        borrowed = state.get("borrowed_opponent")
        overrides = {borrowed[0]: borrowed[1]} if borrowed else None
        state["league"].snapshot(state["policy"], state["policy"].adapters, states=overrides)
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
    weights back and clears the marker."""
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
    ctrl = PhaseController(state["switch_cfg"], first_learner=first_learner)
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


def _train_attacker_only(state, start_round, resume=None):
    """Frozen-defender schedule (``--freeze defender``). Returns the final round count.

    The defender LLM is not in the loop at all: every round's verdicts come from
    the fixed algorithmic ensemble (``state["algorithmic_defender"]``), so there is
    no second policy to alternate with and the learner never switches. Only the
    attacker's LoRA adapter is updated by GRPO.

    The PHASE structure is deliberately kept, with the learner pinned
    (``PhaseController(alternate=False)``). A phase still ends on a sustained win
    or on ``max_phase_rounds``, and that boundary is what (a) checkpoints the
    attacker at a meaningful point and (b) triggers the honest FL interlude that
    advances the shared global model + per-client benign weights. Without it a run
    with ``benign_retrain_each_round: false`` would attack one frozen snapshot of
    the federation forever.

    What is dropped versus the arms-race schedule: the opponent league and the
    curriculum (both only supply past OPPONENT snapshots, and the opponent here is
    a deterministic algorithm), and the defender adapter/optimizer.
    """
    defense = state["algorithmic_defender"]
    ctrl = PhaseController(state["switch_cfg"], first_learner="attacker", alternate=False)
    if resume and resume.get("controller"):
        snapshot = dict(resume["controller"])
        if snapshot.get("learner") not in (None, "attacker"):
            logger.warning(
                f"Resume state says learner={snapshot['learner']!r}, but --freeze defender "
                f"only ever trains the attacker — pinning it back to 'attacker'. (Is this "
                f"progress file from an arms-race run?)"
            )
            snapshot["learner"] = "attacker"
        ctrl.load_state_dict(snapshot)
        logger.info(
            f"Resumed schedule state: phase={ctrl.phase_index} "
            f"phase_round={ctrl.phase_round} streak={ctrl.streak} capped={ctrl.capped}"
        )
    state["controller"] = ctrl   # exposed so _checkpoint can persist it
    done = start_round

    while done < state["total_rounds"]:
        # One honest FL round before every phase AFTER the first, exactly as in the
        # arms race: same gate on phase_round==0 so a resume into mid-phase does not
        # inject a spurious benign round on every restart.
        if (ctrl.phase_index > 0 and ctrl.phase_round == 0
                and state.get("fl_interlude", True)):
            _run_fl_interlude(state, next_learner="attacker", phase_index=ctrl.phase_index)
        dbg.phase_event("PHASE START", phase=ctrl.phase_index, learner="attacker",
                        opponent=defense.describe(), facing_snapshot=False)

        reason = None
        while done < state["total_rounds"]:
            _stats, _drop, success = _step_round(
                state, "attacker", "frozen-defense", None,
                ctrl.phase_index, ctrl.phase_round,
            )
            done += 1
            switch, reason = ctrl.record(success)
            _post_round_bookkeeping(state, done)
            if switch:
                break

        state["policy"].save_adapter("attacker", state["adapter_paths"]["attacker"])
        logger.info(
            f"Phase {ctrl.phase_index} [attacker vs frozen defense] ended "
            f"({reason or 'budget'}) after {ctrl.phase_round} rounds "
            f"(streak={ctrl.streak}) — saved the attacker adapter"
        )
        dbg.phase_event("PHASE END", phase=ctrl.phase_index, learner="attacker",
                        reason=reason or "budget", rounds=ctrl.phase_round, streak=ctrl.streak)
        if reason is None:   # ran out of total budget mid-phase
            break
        ctrl.next_phase(reason)
        _checkpoint(state, done)

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


def _checkpoint(state, done):
    """Atomically-ish save: adapters + the shared FL state, then the resume state
    (round count, FL round index, and the PhaseController snapshot) so a resume is
    always consistent — the saved round count never points past the saved weights,
    and the shared model continues instead of rewinding to the Phase-1 baseline."""
    _save_adapters(state)
    if state.get("fl_state_cb"):
        state["fl_state_cb"](state["env"].snapshot_fl_state())
    if state["progress_cb"]:
        ctrl = state.get("controller")
        state["progress_cb"](done, state["env"].round_index,
                             ctrl.state_dict() if ctrl is not None else None)


def _save_adapters(state):
    """Persist every adapter. If the opponent adapter is currently BORROWED (its
    live weights temporarily swapped for an older league snapshot during a
    curriculum/league phase), write its LIVE weights instead of the borrowed
    snapshot, so a mid-phase checkpoint can't overwrite the real opponent adapter
    on disk (which would silently regress the frozen opponent on resume)."""
    policy = state["policy"]
    borrowed = state.get("borrowed_opponent")
    for name, path in state["adapter_paths"].items():
        if borrowed and borrowed[0] == name:
            policy.save_adapter_state_dict(name, borrowed[1], path)
        else:
            policy.save_adapter(name, path)


def _log_round(env, ctx, info, learner, stats, metrics_tracker, save_round_log,
               reward_att=None, reward_def=None, phase_index=0, phase_round=0,
               success=False, best_index=0, poisoned_ids=None, terms=None):
    verdicts = info["verdicts"]
    post_acc = info["post_accuracy"]
    n_malformed = info["n_malformed"]
    poisoned_ids = poisoned_ids if poisoned_ids is not None else info.get("poisoned_ids", [])
    reward_att = reward_att or {}
    reward_def = reward_def or {}
    # This round's actual goal (per-round target sampling); falls back to the env default.
    goal = ctx.goal if getattr(ctx, "goal", None) is not None else env.goal

    # Diversity of the committed (possibly multi-client) attack; 0 for one client.
    diversity = perturbation_diversity(
        info.get("poisoned_by_client", {}),
        {cid: env.pool_benign[cid] for cid in poisoned_ids if cid in env.pool_benign},
    )
    a_rew = attacker_reward(ctx.clean_accuracy, post_acc, goal,
                            poisoned_ids, verdicts, n_malformed,
                            alpha=reward_att.get("alpha", 1.0),
                            beta=reward_att.get("beta", 0.5),
                            gamma=reward_att.get("gamma", 1.0),
                            delta=reward_att.get("delta", 0.0),
                            zeta=reward_att.get("zeta", 0.0),
                            eta=reward_att.get("eta", 1.0),
                            pool_size=env.n_compromisable,
                            diversity=diversity,
                            clean_eval=ctx.clean_eval,
                            post_eval=info.get("post_eval"))
    d_rew = defender_reward(verdicts, poisoned_ids,
                            mode=reward_def.get("mode", "soft_f1"),
                            fpr_penalty=reward_def.get("fpr_penalty", 1.0))

    # Per-class record for a targeted round: everything a reader (or monitor.py /
    # visualize_rounds.py) needs to answer "did ONLY the target class break?".
    targeted_meta = None
    if terms is not None:
        post_eval = info.get("post_eval")
        targeted_meta = {
            "label": terms["label"],
            "clean_recall": round(terms["clean_recall"], 6),
            "post_recall": round(terms["post_recall"], 6),
            "target_class_drop": round(terms["target_drop"], 6),
            "effective_target": round(terms["effective_target"], 6),
            "collateral": round(terms["collateral"], 6),
            "max_collateral": round(terms["max_collateral"], 6),
            "others_clean": round(terms["others_clean"], 6),
            "others_post": round(terms["others_post"], 6),
            "per_class_clean": [round(v, 6) for v in ctx.clean_eval.per_class],
            "per_class_post": ([round(v, 6) for v in post_eval.per_class]
                               if post_eval is not None else None),
        }

    metrics_tracker.update(ctx.round_num, verdicts, post_acc, set(poisoned_ids))
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
            "attack_diversity": round(float(diversity), 4),
            # The clean counterfactual and the damage measured against it — the
            # quantities the reward and the win-gate actually use, so monitor.py
            # and the visualizer report the same number the policy is trained on.
            "clean_accuracy": round(float(ctx.clean_accuracy), 6),
            "induced_drop": round(float(ctx.clean_accuracy - post_acc), 6),
            "targeted": targeted_meta,
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
        learner, best_index, info, ctx.clean_accuracy, post_acc,
        ctx.clean_accuracy - post_acc, success, a_rew, d_rew, poisoned_ids,
        global_acc=ctx.global_accuracy,
    )
    dbg.flush()
    # On a targeted round the headline number is the target class's recall, not
    # overall accuracy — print it so a run can be read at a glance.
    tgt_s = ""
    if terms is not None:
        tgt_s = (f" | TGT[{terms['label']}] {terms['clean_recall']:.3f}->"
                 f"{terms['post_recall']:.3f} (drop={terms['target_drop']:+.3f}/"
                 f"{terms['effective_target']:.3f}) collat={terms['collateral']:.3f}"
                 f"/{terms['max_collateral']:.3f}")
    logger.info(
        f"Round {ctx.round_num} [learn={learner} ph={phase_index}.{phase_round} "
        f"{'WIN' if success else '...'}]: acc {ctx.global_accuracy:.4f}->{post_acc:.4f} "
        f"(clean_ref={ctx.clean_accuracy:.4f} drop={ctx.clean_accuracy - post_acc:+.4f})"
        f"{tgt_s} "
        f"| att_reward={a_rew:.3f} def_reward={d_rew:.3f} "
        f"| grpo_loss={stats['loss']:.4f} mean_r={stats['mean_reward']:.3f} "
        f"zero_adv={stats['zero_advantage_fraction']:.2f} "
        f"{'step' if stats.get('stepped', True) else 'SKIP'}"
    )
