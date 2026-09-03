"""Replays the demo fixture as a benchmark run.

    python -m webui.demo_bench --events - --rounds 250 --attacks llm,lie --defenses ...

This is the one process the panel starts that does not measure anything. It exists
so the fixture version (:data:`webui.demo.DEMO_ID`) can be walked through end to
end -- the live theatre, the heat matrix filling in, the console, the summary
table, the saved ``history.json`` -- without a GPU, a dataset or an adapter.

It is a **separate program** rather than a flag on ``benchmark.run_benchmark``, and
that is the whole point: the real CLI has no idea this exists and cannot be put
into a mode where it invents numbers. What the two share is the wire format
(:mod:`benchmark.events`) and the log shape, which is what makes the panel render
this identically without a single branch on the client.

The numbers
-----------
:mod:`webui.demo` holds the fixture and explains why the per-round stream is
generated toward it rather than accumulated into it. Here we do the generating:
for every (attack, defense) cell we build an accuracy trajectory that ends at the
row's ``final_accuracy`` and averages its ``mean_accuracy``, and per-round
detection counts and goal scores that average to the row's rates. So the heat
matrix converges on the same numbers the closing table states, on all four of its
metrics, instead of drifting away from them as the run proceeds.

Timing
------
Rounds are paced by ``--round-delay MIN,MAX`` seconds (default
:data:`DEFAULT_ROUND_DELAY`) so the run unfolds rather than appearing all at once.
Pass ``0,0`` for an instant one.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.events import EventEmitter          # noqa: E402
from webui import demo                             # noqa: E402

#: Seconds between rounds, as ``(min, max)``. A round still has to *take* long
#: enough to read as work -- the attacker generating a plan, then a test pass per
#: cell -- but it also has to finish inside a sitting: at a real round's minute or
#: two, 250 rounds is most of a working day. A few seconds keeps the pacing
#: visible and brings the same run down to a quarter of an hour or so. Override
#: per run with ``--round-delay`` (the panel exposes it as *Demo round delay*).
DEFAULT_ROUND_DELAY = (3.0, 8.0)


def log(msg: str) -> None:
    """A line on stdout. The runner treats anything without the event sentinel as
    console text, so this is what shows up in the panel's log pane."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Per-cell trajectories
# ---------------------------------------------------------------------------

def _accuracy_curve(rng, rounds: int, mean_acc: float, final_acc: float,
                    baseline: float) -> list:
    """Accuracy per round: decays off ``baseline``, averages ``mean_acc``, ends at
    ``final_acc``.

    The shape is an exponential settling onto a plateau, which is what a poisoned
    federation's accuracy actually does -- the damage lands over the first tens of
    rounds and then holds. The plateau is placed at the target mean and the whole
    series is then shifted so the mean is exact; the last point is pinned to the
    target final accuracy, which moves the mean by at most one round's worth.
    """
    if rounds <= 1:
        return [final_acc]
    # Wobble in proportion to how hard this cell was hit: an undefended model
    # thrashes, one holding near baseline barely moves.
    sigma = max(0.0015, 0.055 * (baseline - mean_acc))
    k = 5.0
    raw = []
    for r in range(rounds):
        settled = mean_acc + (baseline - mean_acc) * math.exp(-k * r / rounds)
        raw.append(settled + rng.gauss(0.0, sigma))
    shift = mean_acc - sum(raw) / len(raw)
    out = [min(baseline + 0.004, max(0.01, v + shift)) for v in raw]
    out[-1] = final_acc
    return out


def _counts(rng, rounds: int, rate: float, population: int) -> list:
    """Per-round flag counts out of ``population`` that average exactly ``rate``.

    Spreading the remainder over randomly chosen rounds rather than the first few
    keeps the running detection rate from stepping down visibly early in the run.
    """
    if population <= 0:
        return [0] * rounds
    total = int(round(rate * population * rounds))
    total = max(0, min(total, population * rounds))
    base, extra = divmod(total, rounds)
    counts = [min(population, base)] * rounds
    if extra:
        for i in rng.sample(range(rounds), min(extra, rounds)):
            if counts[i] < population:
                counts[i] += 1
    return counts


def _goal_scores(rng, drops: list, target_mean: float, target_drop: float) -> list:
    """Per-round weighted goal success averaging ``target_mean``.

    The real metric is ``min(1, acc_drop / target_drop)``. Using that verbatim
    would put the live heat map's "goal" metric somewhere quite different from the
    row the closing table states, because the fixture's goal rates are authored
    rather than derived. So the shape is kept -- rounds that hurt more score more
    -- and the level is set to the fixture's.
    """
    if not drops:
        return []
    scale = target_drop if target_drop else 1.0
    raw = [max(0.0, (d / scale) * rng.uniform(0.94, 1.06)) for d in drops]
    if sum(raw) <= 0 or target_mean <= 0:
        return [max(0.0, target_mean)] * len(drops)
    if target_mean >= 1.0:
        return [1.0] * len(drops)
    # Solve for the multiplier whose CLIPPED mean is the target. Rescaling by the
    # ratio of the means and correcting once does not converge where the clipping
    # bites hardest: on the undefended row every round already exceeds the target
    # drop, so most scores pin at 1.0 and each pass recovers only part of what the
    # clip took. Bisection lands it exactly, in a fixed number of steps.
    lo, hi = 0.0, 1.0
    while sum(min(1.0, v * hi) for v in raw) / len(raw) < target_mean:
        hi *= 2.0
        if hi > 1e9:
            break
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if sum(min(1.0, v * mid) for v in raw) / len(raw) < target_mean:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2.0
    return [min(1.0, max(0.0, v * k)) for v in raw]


class Cell:
    """One (attack, defense) pair's whole run, precomputed."""

    def __init__(self, attack, defense, rounds, n_poison, n_honest, seed,
                 target_drop):
        self.attack, self.defense = attack, defense
        self.row = demo.attack_row(attack, defense, rounds, seed)
        rng = random.Random(hash((attack, defense, rounds, seed)) & 0xFFFFFFFF)
        self.acc = _accuracy_curve(rng, rounds, self.row["mean_accuracy"],
                                   self.row["final_accuracy"],
                                   demo.BASELINE_ACCURACY)
        self.tp = _counts(rng, rounds, self.row["detection_rate"], n_poison)
        self.fp = _counts(rng, rounds, self.row["fpr"], n_honest)
        drops = [demo.BASELINE_ACCURACY - a for a in self.acc]
        self.goal = _goal_scores(rng, drops, self.row["goal"], target_drop)
        self.n_poison, self.n_honest = n_poison, n_honest

    def history(self, r: int, round_num: int, poisoned: list, honest: list) -> dict:
        """This round's record, in ``benchmark.metrics.DefenseMetrics`` shape."""
        tp, fp = self.tp[r], self.fp[r]
        score = self.goal[r]
        flagged = sorted(poisoned[:tp] + honest[:fp])
        return {
            "attack": self.attack, "defense": self.defense,
            "round": round_num, "tp": tp, "fn": self.n_poison - tp,
            "fp": fp, "tn": self.n_honest - fp,
            "accuracy": round(self.acc[r], 6), "skipped": False,
            "goal_success": round(score, 6), "goal_hit": score >= 1.0 - 1e-9,
            "flagged": flagged, "poisoned": sorted(poisoned),
        }

    def summary(self, rounds: int, skipped: int, target_drop) -> dict:
        """The authoritative row -- the fixture, in the benchmark's summary shape."""
        row = self.row
        total_poison = self.n_poison * rounds
        return {
            "attack": self.attack, "defense": self.defense, "rounds": rounds,
            "malicious_total": total_poison,
            "mean_poisoned": float(self.n_poison),
            "detected": int(round(row["detection_rate"] * total_poison)),
            "detection_rate": row["detection_rate"],
            "tpr": row["detection_rate"],
            "fpr": row["fpr"],
            "precision": row["precision"],
            "recall": row["detection_rate"],
            "f1": row["f1"],
            "false_alarms": int(round(row["fpr"] * self.n_honest * rounds)),
            "final_accuracy": row["final_accuracy"],
            "mean_accuracy": row["mean_accuracy"],
            "mean_acc_drop": row["mean_acc_drop"],
            "attack_success_rate": row["evasion"],
            "goal_success_rate": row["goal"] if target_drop is not None else None,
            # The all-or-nothing view sits below the weighted one by construction:
            # a round can score partial credit without reaching the target.
            "goal_full_success_rate": (round(row["goal"] * 0.82, 6)
                                       if target_drop is not None else None),
            "goal_threshold": (demo.BASELINE_ACCURACY - target_drop
                               if target_drop is not None else None),
            "target_drop": target_drop,
            "skipped_rounds": skipped,
            "baseline_accuracy": demo.BASELINE_ACCURACY,
        }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _csv(value, fallback):
    items = [x.strip() for x in str(value or "").split(",") if x.strip()]
    return items or list(fallback)


def _goal(spec: str):
    gtype, _, val = str(spec or "untargeted_degrade=0.1").partition("=")
    goal = {"type": gtype.strip() or "untargeted_degrade"}
    if val.strip():
        try:
            goal["target_accuracy_drop"] = float(val)
        except ValueError:
            pass
    return goal


def _delay_range(spec: str):
    try:
        lo, _, hi = str(spec).partition(",")
        lo, hi = float(lo), float(hi or lo)
        return (max(0.0, min(lo, hi)), max(0.0, max(lo, hi)))
    except (TypeError, ValueError):
        return DEFAULT_ROUND_DELAY


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Replay the demo fixture as a benchmark run")
    ap.add_argument("--rounds", type=int, default=demo.REFERENCE_ROUNDS)
    ap.add_argument("--attacks", default="llm")
    ap.add_argument("--defenses", default="fedavg,fltrust,defl,dnc,multikrum")
    ap.add_argument("--goal", default="untargeted_degrade=0.1")
    ap.add_argument("--n-clients", type=int, default=20)
    ap.add_argument("--max-poison-clients", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--events", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--demo-version", default=demo.DEMO_ID)
    ap.add_argument("--round-delay", default=None,
                    help="seconds between rounds as MIN,MAX (default 60,120)")
    # The panel builds one argv for both benchmark programs, so every flag the
    # real CLI accepts can arrive here. Ignoring the ones that do not apply keeps
    # the two callable interchangeably instead of needing a second argv spec.
    args, ignored = ap.parse_known_args(argv)

    rounds = max(1, int(args.rounds))
    attacks = _csv(args.attacks, ["llm"])
    defenses = _csv(args.defenses, ["fedavg"])
    if "fedavg" not in defenses:              # the real CLI force-adds the control
        defenses = ["fedavg"] + defenses
    goal = _goal(args.goal)
    target_drop = goal.get("target_accuracy_drop")
    n_clients = max(2, int(args.n_clients))
    n_poison = int(args.max_poison_clients or max(1, n_clients // 2))
    n_poison = max(1, min(n_poison, n_clients - 1))
    n_honest = n_clients - n_poison
    lo, hi = _delay_range(args.round_delay) if args.round_delay else DEFAULT_ROUND_DELAY
    rng = random.Random(int(args.seed))

    em = EventEmitter(args.events)
    em.emit("started", argv=(argv if argv is not None else sys.argv[1:]),
            rounds=rounds, attacks=attacks, defenses=defenses, demo=True)

    log(f"DEMO RUN — replaying the stored result for version {args.demo_version}. "
        f"No model is loaded and no accuracy is measured.")
    if ignored:
        log(f"[demo] ignoring flags that only apply to the real benchmark: "
            f"{' '.join(ignored)}")
    log(f"Attack goal (fixed for the run): {goal}")
    log("Loading saved Phase-1 state (global model + client weights + baseline acc)")
    time.sleep(min(2.0, max(0.0, lo / 30.0)))
    log(f"Phase-1 baseline accuracy = {demo.BASELINE_ACCURACY:.4f}")
    log(f"Eval poison quota = exactly {n_poison} of pool {n_clients} "
        f"client(s) every round")
    log(f"Benchmark: {rounds} rounds | attacks={attacks} | defenses={defenses}")
    if rounds != demo.REFERENCE_ROUNDS:
        log(f"[demo] the stored result is quoted at {demo.REFERENCE_ROUNDS} rounds; "
            f"at {rounds} it is perturbed in proportion to the difference.")

    cells = {a: {d: Cell(a, d, rounds, n_poison, n_honest, int(args.seed), target_drop)
                 for d in defenses} for a in attacks}

    em.emit("config", attacks=attacks, defenses=defenses,
            baseline_accuracy=demo.BASELINE_ACCURACY, rounds=rounds,
            n_clients=n_clients, n_poisoners=n_poison,
            target_drop=target_drop, goal=goal,
            knowledge="partial", device="cpu",
            attacker_adapter=f"checkpoints/versions/{args.demo_version}/attacker_adapter",
            defender_adapter=(f"checkpoints/versions/{args.demo_version}/defender_adapter"
                              if "llm_defender" in defenses else None),
            llm_defender_skipped=False, demo=True,
            citations={a: "demo fixture" for a in attacks})

    poisoned = list(range(n_poison))
    honest = list(range(n_poison, n_clients))
    started = time.time()
    for r in range(rounds):
        round_num = r + 1
        cell_records, shifts, reference = [], {}, {}
        for a in attacks:
            for d in defenses:
                cell_records.append(cells[a][d].history(r, round_num, poisoned, honest))
            if a != "clean":
                per_client = {str(c): round(abs(rng.gauss(0.42, 0.11)), 4)
                              for c in poisoned}
                values = list(per_client.values())
                shifts[a] = {"mean": round(sum(values) / len(values), 4),
                             "max": max(values), "per_client": per_client}
            reference[a] = round(cells[a]["fedavg"].acc[r], 6)

        em.emit("round", index=round_num, of=rounds, round_num=round_num,
                poisoned=list(poisoned), pool=list(poisoned), n_clients=n_clients,
                goal=goal, cells=cell_records, shifts=shifts, reference=reference)

        if round_num == 1 or round_num % max(1, args.log_every) == 0 or round_num == rounds:
            log(f"[round {round_num}/{rounds}] poisoned={poisoned}")
            for a in attacks:
                status = " | ".join(
                    f"{d}: det={_running(cells[a][d], r, n_poison):.0%} "
                    f"acc={cells[a][d].acc[r]:.3f}" for d in defenses)
                log(f"    {a:<10} {status}")

        if round_num < rounds:
            time.sleep(rng.uniform(lo, hi))

    log(f"[demo] {rounds} round(s) in {time.time() - started:.0f}s")
    summaries = {a: {d: cells[a][d].summary(rounds, 0, target_drop) for d in defenses}
                 for a in attacks}
    run_info = {"unusable_attack_rounds": 0, "demo": True,
                "reference_rounds": demo.REFERENCE_ROUNDS}
    text = _report(summaries, attacks, defenses, rounds, args.demo_version)
    print("\n" + text, flush=True)
    em.emit("summary", summaries=summaries, defenses=defenses, attacks=attacks,
            measured_rounds=rounds, requested_rounds=rounds, skipped_rounds=0,
            baseline_accuracy=demo.BASELINE_ACCURACY, run_info=run_info,
            n_poisoners=n_poison, n_clients=n_clients, goal=goal, report=text,
            demo=True)

    out_dir = str(args.out or "").strip()
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            history = {a: {d: [cells[a][d].history(i, i + 1, poisoned, honest)
                               for i in range(rounds)] for d in defenses}
                       for a in attacks}
            path = os.path.join(out_dir, "history.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"baseline_accuracy": demo.BASELINE_ACCURACY,
                           "attacks": attacks, "defenses": defenses,
                           "llm_defender_skipped": False,
                           "requested_rounds": rounds, "measured_rounds": rounds,
                           "unusable_attack_rounds": 0, "run_info": run_info,
                           "demo": True, "history": history}, f, indent=2)
            log(f"[saved] {path}")
            em.emit("saved", out_dir=out_dir, history=path)
        except OSError as exc:
            log(f"[demo] could not write history.json: {exc}")

    em.emit("finished", measured_rounds=rounds)
    em.close()
    return 0


def _running(cell: "Cell", r: int, n_poison: int) -> float:
    caught = sum(cell.tp[: r + 1])
    seen = n_poison * (r + 1)
    return caught / seen if seen else 0.0


def _report(summaries, attacks, defenses, rounds, version) -> str:
    """The plain-text table the real CLI prints at the end, same columns."""
    head = (f"{'defense':<14}{'detect%':>9}{'FPR':>8}{'prec':>7}{'F1':>7}"
            f"{'final_acc':>11}{'mean_acc':>10}{'acc_drop':>10}"
            f"{'atk_thru':>10}{'atk_succ':>10}")
    lines = [f"DEMO RESULT — version {version}, {rounds} round(s)", "", head,
             "-" * len(head)]
    for a in attacks:
        if len(attacks) > 1:
            lines.append(f"[attack: {a}]")
        for d in defenses:
            s = summaries[a][d]
            goal = s["goal_success_rate"]
            lines.append(
                f"{d:<14}{s['detection_rate']:>8.1%}{s['fpr']:>8.1%}"
                f"{s['precision']:>7.2f}{s['f1']:>7.2f}"
                f"{s['final_accuracy']:>11.3f}{s['mean_accuracy']:>10.3f}"
                f"{s['mean_acc_drop']:>10.3f}{s['attack_success_rate']:>10.1%}"
                f"{(goal if goal is not None else 0):>10.1%}")
        lines.append("")
    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    sys.exit(main())
