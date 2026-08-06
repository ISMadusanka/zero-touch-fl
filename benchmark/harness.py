"""Benchmark harness: run the trained attacker against multiple defenses.

Each round:
  1. the env (reused UNMODIFIED, purely as a generator) builds the controllable
     pool and benign updates for an exact poison-client quota;
  2. the trained attacker LLM produces ONE attack plan from the reference
     (no-defense) accuracy;
  3. that single set of client updates is fed to EVERY defense, each evolving its
     OWN global model;
  4. we record per-defense detection (vs ground truth) and post-aggregation test
     accuracy.

Holding the attack fixed across defenses and varying only the defense is what
makes the comparison fair.
"""
import logging

from server.fed_server import FedServer

from benchmark.metrics import DefenseMetrics

logger = logging.getLogger("benchmark")

# Retries of an unusable attacker action are sampled at (at least) this
# temperature: a greedy redraw would reproduce the same unusable text verbatim.
_RETRY_TEMPERATURE_FLOOR = 0.7


def _snippet(text, limit: int = 240) -> str:
    """One-line, length-capped view of a model output, for a log line."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else f"{s[:limit]}... [{len(s)} chars total]"


def _hit_token_cap(policy) -> bool:
    """Did the last generation stop at ``max_new_tokens`` instead of emitting EOS?

    ``LLMPolicy.generate`` records this per rollout in
    ``last_generation_completed``, which the next ``generate`` call overwrites —
    including the LLM defender's — so this must be read immediately after the
    attacker's own generation. Generators that don't expose it report ``False``.
    """
    flags = getattr(policy, "last_generation_completed", None)
    if isinstance(flags, list) and flags:
        return not bool(flags[0])
    return False


def _sample_attack(policy, attacker_agent, ctx, system, user, *, adapter,
                   temperature, max_new_tokens, retries):
    """Draw this round's attacker action, redrawing an unusable one.

    Returns ``(poisoned, chosen_ids, n_malformed, attempts)`` from the first
    attempt that filled the exact poison quota, or from the last attempt if none
    did (``len(chosen_ids) != ctx.budget`` is how the caller detects that).

    A single generation can come back unusable in ways that are pure sampling
    noise: truncated mid-JSON at the token cap, or a syntactically fine plan
    whose every operation is an arithmetic no-op (``scale factor=1.0``, an
    unknown operator, a target no layer matches). ``select_and_apply`` rightly
    refuses to call byte-identical benign weights poison, so such a round yields
    fewer effective plans than the quota. Aborting the benchmark there threw away
    every round already measured (tens of rounds x every defense in the panel, on
    the GPU) over a condition the same prompt almost always survives on a redraw,
    so the action is resampled a bounded number of times first.

    This is resampling for *validity*, not best-of-N on attack strength: the
    first action that fills the quota is used, whatever it does to accuracy, so
    the measured attack is not quietly strengthened on the rounds that needed a
    retry. Retries do force sampling (``_RETRY_TEMPERATURE_FLOOR``) even when the
    caller asked for greedy decoding, since re-running greedy returns the same
    unusable text.
    """
    result = ({}, [], ctx.budget)
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        temp = (temperature if attempt == 1
                else max(float(temperature), _RETRY_TEMPERATURE_FLOOR))
        text = policy.generate(adapter, system, user, n=1, temperature=temp,
                               max_new_tokens=max_new_tokens)[0]
        truncated = _hit_token_cap(policy)
        result = attacker_agent.select_and_apply(text, ctx.pool_benign, ctx.budget)
        poisoned, chosen_ids, n_malformed = result
        if len(chosen_ids) == ctx.budget:
            if attempt > 1:
                logger.info(f"round {ctx.round_num}: usable attacker action on "
                            f"attempt {attempt}/{attempts}")
            return poisoned, chosen_ids, n_malformed, attempt
        logger.warning(
            f"round {ctx.round_num}: unusable attacker action on attempt "
            f"{attempt}/{attempts} — {len(chosen_ids)}/{ctx.budget} effective plan(s), "
            f"{n_malformed} malformed quota slot(s)"
            + (f"; the generation hit the {max_new_tokens}-token cap mid-output "
               f"(raise rl.max_new_tokens)" if truncated else "")
            + f". attacker output: {_snippet(text)}"
        )
    return (*result, attempts)


def _check_prompt_fits(policy, system, user, max_new_tokens, pool_size):
    """Log the attacker prompt's real token cost once, and warn if it crowds the context.

    The prompt carries ``delta_details`` for EVERY client in the controllable pool, so
    its size scales with the pool — and the benchmark can now widen that pool up to
    ``fl.n_clients`` via ``--max-poison-clients``. At the top of that range the prompt
    plus ``max_new_tokens`` can approach ``rl.max_seq_len``, at which point generations
    get truncated and the JSON the attacker emits stops parsing (which would show up as
    a mysteriously ineffective attack rather than as a context error). Measure it rather
    than let the user guess.
    """
    try:
        ids = policy._prompt_ids(system, user)
        n_prompt = int(ids.shape[1])
        limit = int(getattr(policy, "max_seq_len", 0)) or None
    except Exception:                       # a stub/inference generator: nothing to check
        return
    budget_msg = (f"attacker prompt = {n_prompt} tokens for a pool of {pool_size} "
                  f"client(s), + up to {max_new_tokens} generated")
    if limit is None:
        logger.info(budget_msg)
        return
    needed = n_prompt + int(max_new_tokens)
    logger.info(f"{budget_msg} = {needed}/{limit} of rl.max_seq_len")
    if needed > limit:
        logger.warning(
            f"Attacker prompt + max_new_tokens ({needed}) EXCEEDS rl.max_seq_len "
            f"({limit}). Generations will be cut short and the attack JSON will fail to "
            f"parse. Raise rl.max_seq_len, lower rl.max_new_tokens, or reduce "
            f"--max-poison-clients (the pool of {pool_size} is what makes the prompt big)."
        )
    elif needed > 0.9 * limit:
        logger.warning(
            f"Attacker prompt + max_new_tokens ({needed}) is within 10% of "
            f"rl.max_seq_len ({limit}) — a longer plan may get truncated."
        )


def run_benchmark(env, policy, attacker_agent, defenses, test_loader,
                  init_global, baseline_accuracy, n_rounds, *,
                  attack_temperature: float = 0.7, max_new_tokens: int = 512,
                  device: str = "cpu", attacker_adapter: str = "attacker",
                  log_every: int = 10, target_drop: float | None = None,
                  attack_retries: int = 3):
    """Run ``n_rounds`` of attacker-vs-defenses. Returns (summaries, metrics) where
    summaries = {name: summary-dict} and metrics = {name: DefenseMetrics}.

    ``target_drop`` (the goal's requested accuracy drop) enables the per-defense
    goal-success rate: the fraction of rounds that defense's accuracy fell to/below
    ``baseline - target_drop`` (i.e. the attack met its degradation goal).

    ``attack_retries`` bounds how many extra samples a round may draw when the
    attacker's action does not fill the exact poison quota (see
    :func:`_sample_attack`). A round that still has no usable action is SKIPPED —
    logged, counted, and excluded from every defense's metrics — rather than
    aborting the run or, worse, feeding the panel an all-honest round labelled as
    an attack. Each summary's ``rounds`` is therefore the number of rounds actually
    measured, which the caller should prefer over ``n_rounds`` when reporting."""
    if "fedavg" not in defenses:
        logger.warning("no 'fedavg' defense in the panel — the attacker's reference accuracy "
                       "will stay frozen at the clean baseline for the whole run.")
    for d in defenses.values():
        d.reset(init_global)
    eval_server = FedServer(device=device)
    metrics = {name: DefenseMetrics(name, baseline_accuracy, target_drop) for name in defenses}
    reference_acc = float(baseline_accuracy)   # what the attacker observes (no-defense world)
    n_unusable = 0                             # rounds with no usable attacker action

    for r in range(1, n_rounds + 1):
        ctx = env.begin_round()

        # The trained attacker SELECTS exactly the eval-budget count from its
        # controllable pool and plans ONE attack against the reference state;
        # the SAME poisoned updates go to every defense (vary defense, hold attack).
        system = attacker_agent.system_prompt()
        user = attacker_agent.build_user_prompt(ctx.round_num, reference_acc,
                                                ctx.pool_benign, env.global_weights, ctx.budget)
        if r == 1:
            _check_prompt_fits(policy, system, user, max_new_tokens, len(ctx.pool_benign))
        poisoned, chosen_ids, n_malformed, attempts = _sample_attack(
            policy, attacker_agent, ctx, system, user, adapter=attacker_adapter,
            temperature=attack_temperature, max_new_tokens=max_new_tokens,
            retries=attack_retries)
        if len(chosen_ids) != ctx.budget:
            # No usable action after every retry. Measuring this round anyway would
            # score the panel on an attack that never happened (all-honest updates,
            # ground truth claiming `budget` poisoners), so it is dropped from the
            # metrics instead — and the run continues, since the rounds already
            # measured are still valid and expensive to reproduce.
            n_unusable += 1
            logger.error(
                f"round {ctx.round_num}: SKIPPED — no usable attacker action after "
                f"{attempts} attempt(s) (last: {len(chosen_ids)}/{ctx.budget} effective "
                f"plan(s), {n_malformed} malformed quota slot(s)). Excluded from every "
                f"defense's metrics; {n_unusable} round(s) skipped so far. If this "
                f"recurs, raise rl.max_new_tokens or inspect the attacker adapter."
            )
            continue
        poisoned_ids = set(chosen_ids)
        env.set_committed_poison(chosen_ids)
        updates = env.build_updates(poisoned)

        for name, d in defenses.items():
            res = d.step(updates, poisoned_ids)
            gw = d.global_weights()
            if gw is not None:
                # On a skip the defense kept its previous global, so this still
                # reflects that defense's actual current model accuracy.
                eval_server.set_global_weights(gw)
                acc = eval_server.evaluate(test_loader)
            else:
                acc = metrics[name].last_acc      # no model yet (shouldn't happen post-reset)
            metrics[name].record(ctx.round_num, res.verdicts, poisoned_ids, acc,
                                 skipped=(res.new_global is None))

        # The attacker observes the undefended (no-defense) accuracy next round.
        if "fedavg" in metrics:
            reference_acc = metrics["fedavg"].last_acc

        if r == 1 or r % log_every == 0 or r == n_rounds:
            status = " | ".join(
                f"{n}: det={metrics[n].summary()['detection_rate']:.0%} acc={metrics[n].last_acc:.3f}"
                for n in defenses
            )
            logger.info(f"[round {r}/{n_rounds}] poisoned={sorted(poisoned_ids)} | {status}")

    if n_unusable:
        logger.warning(
            f"{n_unusable} of {n_rounds} round(s) produced no usable attacker action and "
            f"were skipped: every defense's metrics cover {n_rounds - n_unusable} measured "
            f"round(s). Rerun with a larger rl.max_new_tokens (or --attack-retries) if that "
            f"fraction is material to the result."
        )
    return {name: m.summary() for name, m in metrics.items()}, metrics
