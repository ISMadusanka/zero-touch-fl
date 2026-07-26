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
                  log_every: int = 10, target_drop: float | None = None,
                  goal: dict | None = None, win_fraction: float = 0.6,
                  n_classes: int = 10):
    """Run ``n_rounds`` of attacker-vs-defenses. Returns (summaries, metrics) where
    summaries = {name: summary-dict} and metrics = {name: DefenseMetrics}.

    ``target_drop`` (the goal's requested accuracy drop) enables the per-defense
    goal-success rate: the fraction of rounds that defense's accuracy fell to/below
    ``baseline - target_drop`` (i.e. the attack met its degradation goal).

    ``goal`` is the run's fixed attack goal. Every round is evaluated PER CLASS
    (one pass, same cost as plain accuracy — see ``FedServer.evaluate_per_class``),
    so a ``targeted_label`` run can report the target class's recall against the
    other classes' for each defense. The clean per-class reference is taken from
    round 1's clean counterfactual (``ctx.clean_eval``)."""
    if "fedavg" not in defenses:
        logger.warning("no 'fedavg' defense in the panel — the attacker's reference accuracy "
                       "will stay frozen at the clean baseline for the whole run.")
    for d in defenses.values():
        d.reset(init_global)
    eval_server = FedServer(device=device)
    metrics = {name: DefenseMetrics(name, baseline_accuracy, target_drop,
                                    goal=goal, win_fraction=win_fraction)
               for name in defenses}
    reference_acc = float(baseline_accuracy)   # what the attacker observes (no-defense world)
    clean_eval = None                          # clean per-class reference (set on round 1)

    for r in range(1, n_rounds + 1):
        ctx = env.begin_round()
        if clean_eval is None:
            # The unpoisoned counterfactual for this frozen setup. The benchmark
            # never commits to ``env``, so it is the same every round — take it once.
            clean_eval = ctx.clean_eval
            for m in metrics.values():
                m.set_clean_eval(clean_eval)
            logger.info("clean per-class recall: "
                        + " ".join(f"{i}={v:.3f}" for i, v in enumerate(clean_eval.per_class)))

        # The trained attacker SELECTS which of its controllable pool to poison
        # (<= the eval budget) and plans ONE attack against the reference state;
        # the SAME poisoned updates go to every defense (vary defense, hold attack).
        system = attacker_agent.system_prompt()
        user = attacker_agent.build_user_prompt(ctx.round_num, reference_acc,
                                                ctx.pool_benign, env.global_weights, ctx.budget,
                                                goal=ctx.goal)
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
            class_eval = None
            if gw is not None:
                # On a skip the defense kept its previous global, so this still
                # reflects that defense's actual current model accuracy.
                eval_server.set_global_weights(gw)
                class_eval = eval_server.evaluate_per_class(test_loader, n_classes)
                acc = class_eval.overall
            else:
                acc = metrics[name].last_acc      # no model yet (shouldn't happen post-reset)
            metrics[name].record(ctx.round_num, res.verdicts, poisoned_ids, acc,
                                 skipped=(res.new_global is None), class_eval=class_eval)

        # The attacker observes the undefended (no-defense) accuracy next round.
        if "fedavg" in metrics:
            reference_acc = metrics["fedavg"].last_acc

        if r == 1 or r % log_every == 0 or r == n_rounds:
            def _one(n):
                m = metrics[n]
                s = f"{n}: det={m.summary()['detection_rate']:.0%} acc={m.last_acc:.3f}"
                # On a targeted run the number that matters is the target class's
                # recall, so show it inline next to overall accuracy.
                if m.target_label is not None and m.last_per_class:
                    s += f" tgt[{m.target_label}]={m.last_per_class[m.target_label]:.3f}"
                return s
            status = " | ".join(_one(n) for n in defenses)
            logger.info(f"[round {r}/{n_rounds}] poisoned={sorted(poisoned_ids)} | {status}")

    return {name: m.summary() for name, m in metrics.items()}, metrics
