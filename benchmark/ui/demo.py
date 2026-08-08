"""Synthetic replay of a targeted run, for checking the dashboard without a GPU.

``python -m benchmark.ui --demo`` uses this instead of spawning the real
benchmark. It emits the *same* events (``benchmark.events``) in the same order,
so the UI cannot tell the difference — which is the point: it exercises the
dashboard's whole path, including the final summary, on a laptop with no torch,
no adapter and no CUDA.

The per-round numbers are invented. Everything downstream of them is real: the
accumulation runs through ``benchmark.metrics.DefenseMetrics`` and the summary
table through ``benchmark.targeted_report.render``, exactly as a live run does,
so the demo cannot drift away from what the CLI would print.
"""
import random

from core.types import ClassEval, DetectionVerdict

# Clean per-class recall of the shipped MNIST MLP — uneven on purpose (classes 5
# and 8 are genuinely harder), so the display is exercised with realistic spread.
CLEAN = [0.918, 0.961, 0.903, 0.788, 0.812, 0.601, 0.874, 0.845, 0.669, 0.703]
SUPPORT = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]

# name -> (per-round detection probability, false-alarm rate, how much of the
# attack's damage the defense removes when it catches it)
PROFILES = {
    "fedavg":       (0.00, 0.000, 0.00),      # no defense: the attack's true effect
    "oracle":       (1.00, 0.000, 1.00),      # perfect detection: must match `clean`
    "llm_defender": (0.72, 0.031, 0.92),
    "fltrust":      (0.41, 0.080, 0.85),
    "defl":         (0.55, 0.055, 0.80),
    "dnc":          (0.63, 0.045, 0.88),
    "multikrum":    (0.48, 0.070, 0.83),
    "ensemble":     (0.80, 0.040, 0.90),
}
DEFAULT_PROFILE = (0.50, 0.05, 0.80)


def _argv_value(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def demo_events(argv):
    """Yield ``(delay_seconds, event)`` replaying a run of the given argv."""
    rounds = int(_argv_value(argv, "--rounds", 20))
    label = int(_argv_value(argv, "--label", 0))
    pool = [int(c) for c in str(_argv_value(argv, "--poison-client-ids", "0")).split(",") if c]
    names = [n for n in str(_argv_value(argv, "--defenses", "fedavg")).split(",") if n]
    if "fedavg" not in names:
        names = ["fedavg"] + names
    budget = int(_argv_value(argv, "--poison-clients", len(pool)))
    n_clients, n_classes = 20, len(CLEAN)
    rng = random.Random(1234)

    goal = {"type": "targeted_label", "label": label,
            "target_class_drop": 0.50, "max_collateral": 0.05}
    clean_eval = ClassEval(overall=_overall(CLEAN), per_class=list(CLEAN), support=list(SUPPORT))

    yield 0.2, {"event": "log", "line": "[demo] no model is loaded — these numbers are invented"}
    for phase, msg in (("setup", f"label={label} defenses={names} rounds={rounds}"),
                       ("data", f"partitioning mnist across {n_clients} clients"),
                       ("phase1", "restoring / running Phase-1 honest baseline"),
                       ("policy", "loading Qwen2.5-3B + attacker adapter")):
        yield 0.35, {"event": "phase", "phase": phase, "message": msg}

    yield 0.4, {"event": "start", "n_rounds": rounds, "target_label": label, "pool": pool,
                "poison_budget": budget, "defenses": names,
                "baseline_accuracy": clean_eval.overall, "n_classes": n_classes,
                "n_clients": n_clients, "goal": goal, "device": "demo",
                "attack_temperature": 0.7, "out_dir": None}
    yield 0.2, {"event": "clean", "per_class": list(CLEAN), "support": list(SUPPORT),
                "overall": clean_eval.overall}

    # Real accumulators, so the summary at the end is produced by the same code
    # path the CLI uses.
    from benchmark.metrics import DefenseMetrics
    metrics = {n: DefenseMetrics(n, clean_eval.overall, None, goal=goal, win_fraction=0.6)
               for n in names}
    for m in metrics.values():
        m.set_clean_eval(clean_eval)

    for r in range(1, rounds + 1):
        # The attacker warms up over the first few rounds, then mostly lands.
        strength = min(1.0, 0.35 + 0.9 * (r / max(8, rounds * 0.15))) * rng.uniform(0.82, 1.0)
        wasted = rng.random() < 0.05                    # a malformed / no-op plan
        poisoned = [] if wasted else sorted(rng.sample(pool, min(budget, len(pool))))
        factor = round(1 - n_clients / max(1, len(poisoned) or 1), 1)

        payload = {
            "round": 46 + r, "index": r, "n_rounds": rounds,
            "poisoned": poisoned, "pool": pool, "budget": budget,
            "n_malformed": 1 if wasted else 0,
            "plan": [{"client": c, "ops": [
                {"op": "scale", "target": "net.4", "rows": [label],
                 "factor": round(factor * rng.uniform(1.0, 1.6), 2)}]}
                for c in (poisoned or pool[:1])],
            "defenses": {},
        }
        for name in names:
            p_det, p_fp, suppress = PROFILES.get(name, DEFAULT_PROFILE)
            caught = bool(poisoned) and rng.random() < p_det
            flagged = [c for c in poisoned if caught]
            flagged += [c for c in range(n_clients)
                        if c not in poisoned and rng.random() < p_fp]
            flagged = sorted(set(flagged))
            landed = 0.0 if not poisoned else strength * (1.0 - (suppress if caught else 0.0))

            per_class = []
            for c in range(n_classes):
                if c == label:
                    per_class.append(max(0.005, CLEAN[c] * (1 - landed)))
                else:                                    # a little collateral + noise
                    per_class.append(max(0.0, CLEAN[c] - landed * rng.uniform(0.0, 0.012)))
            ce = ClassEval(overall=_overall(per_class), per_class=per_class,
                           support=list(SUPPORT))
            verdicts = [DetectionVerdict(client_id=c, is_suspicious=(c in flagged),
                                         confidence=0.8, reason="demo")
                        for c in range(n_clients)]
            m = metrics[name]
            m.record(payload["round"], verdicts, set(poisoned), ce.overall, class_eval=ce)
            s, last = m.summary(), m.history[-1]
            payload["defenses"][name] = {
                "accuracy": last["accuracy"], "per_class": last["per_class"],
                "flagged": last["flagged"], "tp": last["tp"], "fn": last["fn"],
                "fp": last["fp"], "tn": last["tn"], "skipped": last["skipped"],
                "targeted": last["targeted"], "detection_rate": s["detection_rate"],
                "fpr": s["fpr"], "f1": s["f1"],
                "attack_success_rate": s["attack_success_rate"],
                "targeted_success_rate": s.get("targeted_success_rate"),
                "mean_collateral": s.get("mean_collateral"),
            }
        yield 0.28, {"event": "round", **payload}
        if r == 1 or r % 10 == 0 or r == rounds:
            tgt = payload["defenses"]["fedavg"]["per_class"][label]
            yield 0.02, {"event": "log",
                         "line": f"[round {r}/{rounds}] poisoned={poisoned} | "
                                 f"fedavg tgt[{label}]={tgt:.3f}"}

    from benchmark import targeted_report
    summaries = [metrics[n].summary() for n in names]
    report = targeted_report.render(summaries, rounds, clean_eval.overall, label,
                                    out_dir=None, goal=goal, n_poisoners=budget)
    yield 0.3, {"event": "summary", "target_label": label, "n_rounds": rounds,
                "baseline_accuracy": clean_eval.overall, "n_poisoners": budget,
                "results": summaries, "report": report, "out_dir": None}
    yield 0.1, {"event": "end", "status": "ok"}


def _overall(per_class) -> float:
    """Support-weighted accuracy from per-class recall — what a real evaluation
    would report, so the demo's `overall` column moves the way a live one does."""
    total = sum(SUPPORT)
    return sum(r * s for r, s in zip(per_class, SUPPORT)) / total
