"""Benchmark harness: run the label-flip attack against multiple defenses.

Each round:
  1. the env (reused UNMODIFIED, purely as a generator) trains every client, flips
     the ladder's share of the insiders' labels, and exposes the round's ground
     truth;
  2. that single set of client updates is fed to EVERY defense, each evolving its
     OWN global model;
  3. we record per-defense detection (vs ground truth) and post-aggregation test
     accuracy.

Holding the attack fixed across defenses and varying only the defense is what
makes the comparison fair — and here the attack is fixed by construction, since
it is a deterministic schedule rather than a sampled policy. That removes the old
harness's whole retry/skip apparatus: there is no unusable action to redraw, so
every requested round is a measured round.

THE LADDER AND THE PANEL. The attack adapts to detection, but the panel has N
defenses with N different verdicts, so ``ladder_feedback`` names which one closes
the loop (default: the first non-``fedavg`` defense). Every defense still faces the
identical attack each round; one of them just also decides how strong the next
round is. With ``ladder_feedback: None`` the ladder is fed the union of nothing —
it never advances and the attack stays at its starting level for the whole run,
which is the right choice for a straight "how do these defenses handle a 100%
label flip" sweep.
"""
import logging

from server.fed_server import FedServer

from benchmark.metrics import DefenseMetrics

logger = logging.getLogger("benchmark")


def run_benchmark(env, defenses, test_loader, init_global, baseline_accuracy,
                  n_rounds, *, device: str = "cpu", log_every: int = 10,
                  target_drop: float | None = None,
                  ladder_feedback: str | None = None):
    """Run ``n_rounds`` of label-flip-vs-defenses. Returns (summaries, metrics) where
    summaries = {name: summary-dict} and metrics = {name: DefenseMetrics}.

    ``target_drop`` (the goal's requested accuracy drop) enables the per-defense
    goal-success rate: the fraction of rounds that defense's accuracy fell to/below
    ``baseline - target_drop`` (i.e. the attack met its degradation goal).

    ``ladder_feedback`` is the defense whose verdicts drive the attack's ladder, or
    ``None`` to hold the attack at its starting level for the whole run.
    """
    if "fedavg" not in defenses:
        logger.warning("no 'fedavg' defense in the panel — there is no no-defense "
                       "reference column to compare the others against.")
    if ladder_feedback is not None and ladder_feedback not in defenses:
        raise ValueError(f"ladder_feedback={ladder_feedback!r} is not in the panel "
                         f"{list(defenses)}")
    for d in defenses.values():
        d.reset(init_global)
    eval_server = FedServer(device=device)
    metrics = {name: DefenseMetrics(name, baseline_accuracy, target_drop) for name in defenses}

    logger.info(
        f"Attack: {env.attacker.describe()}; ladder feedback from "
        + (f"'{ladder_feedback}'" if ladder_feedback
           else "NOTHING (held at the starting level all run)")
    )

    for r in range(1, n_rounds + 1):
        ctx = env.begin_round()
        poisoned_ids = set(ctx.poisoned_ids)
        # ONE cohort of updates, handed unchanged to every defense.
        updates = env.build_updates()

        feedback_verdicts = None
        for name, d in defenses.items():
            res = d.step(updates, poisoned_ids)
            if name == ladder_feedback:
                feedback_verdicts = res.verdicts
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

        # Close the loop, once, after every defense has seen the round.
        ladder = env.record_detection(feedback_verdicts or [])

        if r == 1 or r % log_every == 0 or r == n_rounds:
            status = " | ".join(
                f"{n}: det={metrics[n].summary()['detection_rate']:.0%} acc={metrics[n].last_acc:.3f}"
                for n in defenses
            )
            logger.info(
                f"[round {r}/{n_rounds}] flip={ctx.flip_fraction:.0%} "
                f"({sum(ctx.flip_plan.values())} labels) poisoned={sorted(poisoned_ids)} "
                f"ladder->{ladder.get('next_flip_fraction', ctx.flip_fraction):.0%} | {status}"
            )

    return {name: m.summary() for name, m in metrics.items()}, metrics
