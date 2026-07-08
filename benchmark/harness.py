"""Benchmark harness: run the trained attacker against multiple defenses.

Each round:
  1. the env (reused UNMODIFIED, purely as a generator) samples the poisoned
     client subset + builds the benign updates;
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


def run_benchmark(env, policy, attacker_agent, defenses, test_loader,
                  init_global, baseline_accuracy, n_rounds, *,
                  attack_temperature: float = 0.7, max_new_tokens: int = 512,
                  device: str = "cpu", attacker_adapter: str = "attacker",
                  log_every: int = 10):
    """Run ``n_rounds`` of attacker-vs-defenses. Returns (summaries, metrics) where
    summaries = {name: summary-dict} and metrics = {name: DefenseMetrics}."""
    if "fedavg" not in defenses:
        logger.warning("no 'fedavg' defense in the panel — the attacker's reference accuracy "
                       "will stay frozen at the clean baseline for the whole run.")
    for d in defenses.values():
        d.reset(init_global)
    eval_server = FedServer(device=device)
    metrics = {name: DefenseMetrics(name, baseline_accuracy) for name in defenses}
    reference_acc = float(baseline_accuracy)   # what the attacker observes (no-defense world)

    for r in range(1, n_rounds + 1):
        ctx = env.begin_round()

        # The trained attacker SELECTS which of its controllable pool to poison
        # (<= the eval budget) and plans ONE attack against the reference state;
        # the SAME poisoned updates go to every defense (vary defense, hold attack).
        system = attacker_agent.system_prompt()
        user = attacker_agent.build_user_prompt(ctx.round_num, reference_acc,
                                                ctx.pool_benign, env.global_weights,
                                                ctx.budget, ctx.target_neuron_indices)
        text = policy.generate(attacker_adapter, system, user, n=1,
                               temperature=attack_temperature, max_new_tokens=max_new_tokens)[0]
        poisoned, chosen_ids, _n_malformed = attacker_agent.select_and_apply(
            text, ctx.pool_benign, ctx.budget)
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
