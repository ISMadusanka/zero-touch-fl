"""Benchmark harness: run the trained attacker against multiple defenses.

Each round:
  1. the env (reused UNMODIFIED, purely as a generator) builds the controllable
     pool and benign updates for an exact poison-client quota;
  2. the trained attacker LLM produces ONE attack plan from the reference
     (no-defense) accuracy;
  3. every defense is first probed on the UNPOISONED updates to get its own clean
     counterfactual for the round (see below);
  4. that single set of poisoned client updates is fed to EVERY defense, each
     evolving its OWN global model;
  5. we record per-defense detection (vs ground truth), post-aggregation test
     accuracy, and how large the attacker's perturbation actually was.

Holding the attack fixed across defenses and varying only the defense is what
makes the comparison fair.

**The clean counterfactual.** Scoring every defense against ONE global baseline
conflates two different things: what the ATTACK cost, and what the DEFENSE costs
by itself. They are not close to equal — a defense that rejects most of an honest
non-IID federation loses several points of accuracy on a round with no poison in
it at all, and against a single baseline the attacker is credited for all of it.
(FLTrust is the standard example: it can exclude every poisoner in 80% of rounds
and still be reported as suffering a successful attack.) So each round every
defense is also run on the unpoisoned updates via ``Defense.probe`` — a real
``step`` whose effects on the global model and cross-round memory are rolled back
— giving the accuracy that defense would have reached with no attacker present.
Damage then splits cleanly:

    defense_cost = baseline    - clean_accuracy      (the defense's own price)
    attack_drop  = clean_accuracy - post_accuracy    (what the attack actually did)

which is the same definition the training reward already uses
(``rl.env.FLArmsRaceEnv.clean_reference_accuracy``); the benchmark was the odd one
out. It costs one extra aggregation + test-set evaluation per defense per round —
and, for the ``llm_defender`` column only, one extra defender generation, since its
``step`` is an LLM call. ``clean_counterfactual=False`` falls back to the
single-baseline behaviour.
"""
import logging

from agents.attack_ops import perturbation_sizes
from data.datasets import DEFAULT_DATASET
from server.fed_server import FedServer

from benchmark.metrics import INERT_POISON_RATIO, DefenseMetrics

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
                  log_every: int = 10, target_drop: float | None = None,
                  clean_counterfactual: bool = True):
    """Run ``n_rounds`` of attacker-vs-defenses. Returns (summaries, metrics) where
    summaries = {name: summary-dict} and metrics = {name: DefenseMetrics}.

    ``target_drop`` (the goal's requested accuracy drop) enables the per-defense
    goal-success rate: how much of the requested degradation the attack achieved.

    ``clean_counterfactual`` probes every defense on the unpoisoned updates each
    round so the attack's damage is separated from the defense's own cost (see the
    module docstring). Turning it off halves the evaluations and reverts every
    drop to being measured against the single clean baseline."""
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
        # The same honest updates with the poison left out — the input to every
        # defense's clean counterfactual this round.
        clean_updates = env.build_updates({}) if clean_counterfactual else None

        # How large the attack actually is, measured against the honest update it
        # replaces. Without this a policy that emits near-no-ops is indistinguishable
        # in the report from one that evades a real defense: both read as 100%
        # throughput. See agents.attack_ops.perturbation_size.
        sizes = perturbation_sizes(
            poisoned, {cid: ctx.pool_benign[cid] for cid in chosen_ids},
            env.global_weights)
        poison_ratios = [s["rel_update"] for s in sizes.values()
                         if s["rel_update"] is not None]
        mean_poison = (sum(poison_ratios) / len(poison_ratios)) if poison_ratios else 0.0

        def _eval(state, fallback):
            if state is None:                 # no model yet (shouldn't happen post-reset)
                return fallback
            eval_server.set_global_weights(state)
            return eval_server.evaluate(test_loader)

        for name, d in defenses.items():
            clean_acc = None
            if clean_updates is not None:
                # PROBE FIRST: the counterfactual has to face the same state the real
                # step will. Defense.probe rolls back the global model and any
                # cross-round memory it touches, so the defense is left untouched and
                # only the returned aggregate is scored. A probe that declined to
                # aggregate would have kept the current global — score that instead.
                probe = d.probe(clean_updates, set())
                clean_acc = _eval(probe.new_global if probe.new_global is not None
                                  else d.global_weights(), metrics[name].last_clean_acc)

            res = d.step(updates, poisoned_ids)
            # ``step`` installs its own aggregate; on a skip the defense kept its
            # previous global, so this reflects its actual current model either way.
            acc = _eval(d.global_weights(), metrics[name].last_acc)
            metrics[name].record(ctx.round_num, res.verdicts, poisoned_ids, acc,
                                 skipped=(res.new_global is None),
                                 clean_accuracy=clean_acc,
                                 poison_ratios=poison_ratios)

        # The attacker observes the undefended (no-defense) accuracy next round.
        if "fedavg" in metrics:
            reference_acc = metrics["fedavg"].last_acc

        if r == 1 or r % log_every == 0 or r == n_rounds:
            status = " | ".join(
                f"{n}: det={metrics[n].summary()['detection_rate']:.0%} acc={metrics[n].last_acc:.3f}"
                for n in defenses
            )
            logger.info(f"[round {r}/{n_rounds}] poisoned={sorted(poisoned_ids)} "
                        f"poison_size={mean_poison:.3g}x honest | {status}")

    _warn_if_attack_is_inert(metrics, logger)
    return {name: m.summary() for name, m in metrics.items()}, metrics


def _warn_if_attack_is_inert(metrics, log) -> None:
    """Say it out loud when the attacker did not actually attack.

    A trained policy can collapse onto perturbations far smaller than the honest
    client-to-client spread. Every number in the table then looks defensible in
    isolation — 100% throughput, 0% detection — while meaning the opposite of what
    it reads as: nothing was detected because nothing happened. The one number that
    exposes it is the perturbation size, so check it explicitly rather than leaving
    it to be noticed.
    """
    any_metric = next(iter(metrics.values()), None)
    if any_metric is None or not any_metric.rounds:
        return
    mean_poison = any_metric.mean_poison_ratio()
    if mean_poison >= INERT_POISON_RATIO:
        return
    log.warning(
        f"THE ATTACK IS EFFECTIVELY INERT: the attacker's mean perturbation is "
        f"{mean_poison:.3g}x the honest update it replaces (< {INERT_POISON_RATIO}). "
        f"That is well inside the spread between honest non-IID clients, so a robust "
        f"aggregator ranks a poisoned update as MORE central than a real one — it "
        f"survives every filter and moves the global by nothing. Read the detection "
        f"columns accordingly: 0% detection and 100% throughput here mean 'there was "
        f"nothing to detect', NOT 'the defense failed'. Check the attacker's plans "
        f"(rl.reward.attacker.stealth_floor gates the stealth term on attack size) "
        f"before reading this table as a defense result."
    )
