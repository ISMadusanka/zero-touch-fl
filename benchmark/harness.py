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

from data.datasets import DEFAULT_DATASET
from server.fed_server import FedServer

from benchmark.metrics import DefenseMetrics

logger = logging.getLogger("benchmark")


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
                  device: str = "cpu", dataset: str | None = None,
                  attacker_adapter: str = "attacker",
                  log_every: int = 10, target_drop: float | None = None):
    """Run ``n_rounds`` of attacker-vs-defenses. Returns (summaries, metrics) where
    summaries = {name: summary-dict} and metrics = {name: DefenseMetrics}.

    ``target_drop`` (the goal's requested accuracy drop) enables the per-defense
    goal-success rate: the fraction of rounds that defense's accuracy fell to/below
    ``baseline - target_drop`` (i.e. the attack met its degradation goal)."""
    if "fedavg" not in defenses:
        logger.warning("no 'fedavg' defense in the panel — the attacker's reference accuracy "
                       "will stay frozen at the clean baseline for the whole run.")
    for d in defenses.values():
        d.reset(init_global)
    # The scoring model must have the same architecture as the weights every
    # defense produces; default to the env's dataset so callers can't desync them.
    eval_server = FedServer(
        device=device,
        dataset=dataset or getattr(env, "dataset", DEFAULT_DATASET),
    )
    metrics = {name: DefenseMetrics(name, baseline_accuracy, target_drop) for name in defenses}
    reference_acc = float(baseline_accuracy)   # what the attacker observes (no-defense world)

    for r in range(1, n_rounds + 1):
        ctx = env.begin_round()

        # The trained attacker SELECTS exactly the eval-budget count from its
        # controllable pool and plans ONE attack against the reference state;
        # the SAME poisoned updates go to every defense (vary defense, hold attack).
        system = attacker_agent.system_prompt()
        user = attacker_agent.build_user_prompt(
            ctx.round_num, reference_acc, ctx.pool_benign, env.global_weights,
            ctx.budget, dataset=getattr(env, "dataset", None))
        if r == 1:
            _check_prompt_fits(policy, system, user, max_new_tokens, len(ctx.pool_benign))
        text = policy.generate(attacker_adapter, system, user, n=1,
                               temperature=attack_temperature, max_new_tokens=max_new_tokens)[0]
        poisoned, chosen_ids, _n_malformed = attacker_agent.select_and_apply(
            text, ctx.pool_benign, ctx.budget)
        if len(chosen_ids) != ctx.budget:
            raise RuntimeError(
                f"round {ctx.round_num}: attacker could not satisfy the exact "
                f"poison quota {ctx.budget}; only {len(chosen_ids)} effective "
                f"plan(s) were produced ({_n_malformed} malformed quota slots). "
                f"Increase rl.max_new_tokens or inspect the attacker output."
            )
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

    return {name: m.summary() for name, m in metrics.items()}, metrics
