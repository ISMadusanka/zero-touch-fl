"""GOAL-06 pre-flight: measure the clean-counterfactual noise floor per defense.

Measures the round-to-round swing of ``FLArmsRaceEnv.clean_reference_accuracy()``
under each defense judging alone, in ``single`` mode. Every round submits only
honest client updates — there is no poisoning client and no LLM anywhere in
this measurement — because the sole purpose of this tool is the GOAL-06 gate:
confirming the attack-goal ladder's bottom rung is larger than the noise a
budget-1 round's ``drop`` is measured against (see ``.planning/ROADMAP.md``
Phase 1 Success Criterion 3). A rung that does not clear the noise by a
comfortable margin would make a budget-1 round's measured drop mostly noise,
corrupting every per-(defense x budget) cell built on it in later phases.

Committed (not thrown away) so the gate can be re-run whenever the ladder,
``fl.n_compromisable``, ``attack.max_poison_clients``, or a ``defense:``
hyperparameter changes.

    python -m benchmark.noise_probe --rounds 20 --device cpu
"""

import argparse
import copy
import json
import logging
import math
import os
import random
import statistics

import yaml

logger = logging.getLogger("noise_probe")


# ---------------------------------------------------------------------------
# Pure statistics + verdict logic — torch-free, importable in isolation.
# ---------------------------------------------------------------------------

def summarize_noise(samples: list[float], rung: float, sigma_margin: float = 3.0) -> dict:
    """Summarize a defense's per-round clean-counterfactual samples into a verdict.

    Raises ``ValueError`` naming the sample count when fewer than two samples are
    given: a single sample would make ``statistics.pstdev`` return ``0.0``,
    which would certify ANY rung as clearing — an aborted or empty probe run
    must fail loudly instead of silently passing the gate.

    ``sd`` is ``statistics.pstdev(samples)`` over the raw floats with no
    pre-rounding. ``threshold = sigma_margin * sd``; ``clears = rung >=
    threshold`` — an exactly-equal case is recorded as clearing. Because ``sd``
    is produced by a square root, a mathematically-exact margin (e.g. samples
    constructed so ``sd`` is exactly ``rung / sigma_margin``) can land a few
    ULPs above ``rung`` purely from floating-point rounding; ``math.isclose``
    with a tolerance many orders of magnitude below the ladder's 0.01
    granularity absorbs that noise without masking a genuine failure.

    ``required_rung`` is reported in every result, not only on failure, so the
    recorded artifact shows the headroom either way. It is
    ``math.ceil(threshold / 0.01) * 0.01`` rounded to 2 decimal places to avoid
    a binary-float tail; the ratio is rounded to 9 decimal places before the
    ceiling so the same float noise that ``clears`` tolerates cannot inflate
    the required rung by a spurious extra cent.
    """
    n = len(samples)
    if n < 2:
        raise ValueError(
            f"summarize_noise needs at least 2 samples to compute a standard "
            f"deviation and certify a rung; got {n}. A single sample would "
            f"make the standard deviation 0.0 and silently certify any rung "
            f"from an aborted or empty probe run."
        )

    sd = statistics.pstdev(samples)
    threshold = sigma_margin * sd
    clears = threshold <= rung or math.isclose(
        threshold, rung, rel_tol=1e-9, abs_tol=1e-12)
    ratio = round(threshold / 0.01, 9)
    required_rung = round(math.ceil(ratio) * 0.01, 2)

    return {
        "n": n,
        "mean": sum(samples) / n,
        "sd": sd,
        "min": min(samples),
        "max": max(samples),
        "sigma_margin": sigma_margin,
        "threshold": threshold,
        "rung": rung,
        "clears": clears,
        "required_rung": required_rung,
    }


# ---------------------------------------------------------------------------
# Measurement rig — needs the real FL environment (heavy imports, deferred).
# ---------------------------------------------------------------------------

def measure_defense(config: dict, defense_name: str, n_rounds: int, *,
                    device: str, seed: int, client_loaders, test_loader,
                    start_state) -> list[float]:
    """Per-round clean-counterfactual accuracies for ONE defense, judging alone.

    Deep-copies ``config`` and forces ``defense.algorithms`` to
    ``[defense_name]``, ``defense.mode`` to ``"single"`` and
    ``defense.selection`` to ``"fixed"``, so exactly one algorithm judges every
    round and rotation cannot swap it mid-measurement — the counterfactual is
    only comparable across rounds when the same filter produced each value.

    ``start_state`` is the ``(global_weights, client_weights,
    baseline_accuracy)`` Phase-1 tuple; it is deep-copied here (on top of
    ``env.reset``'s own copy) so every defense starts from the identical global
    model and client weights and the four measured standard deviations stay
    comparable.

    Every round: ``env.begin_round()`` (which measures and caches this round's
    clean counterfactual), record ``ctx.clean_accuracy``, then commit an
    all-honest round — ``env.build_updates({})`` is exactly the all-honest
    update set, with no poisoned client anywhere. Committing is what advances
    the global model and the defense's own cross-round state (e.g. DeFL's Beta
    trust counts), which is the source of the round-to-round variation being
    measured.
    """
    from rl.env import FLArmsRaceEnv
    from server.defense_ensemble import build_ensemble

    cfg_copy = copy.deepcopy(config)
    dcfg = cfg_copy.setdefault("defense", {})
    dcfg["algorithms"] = [defense_name]
    dcfg["mode"] = "single"
    dcfg["selection"] = "fixed"

    ensemble = build_ensemble(cfg_copy, device=device, seed=seed, rng=random.Random(seed))
    env = FLArmsRaceEnv(cfg_copy, client_loaders, test_loader, random.Random(seed),
                        defense=ensemble)

    global_weights, client_weights, baseline_accuracy = start_state
    env.reset(copy.deepcopy(global_weights), copy.deepcopy(client_weights),
              baseline_accuracy)

    samples: list[float] = []
    for round_num in range(1, n_rounds + 1):
        ctx = env.begin_round()
        samples.append(ctx.clean_accuracy)

        updates = env.build_updates({})
        verdicts, _info = env.defense.verdicts(updates, env.global_weights, commit=True)
        committed_acc = env.commit(updates, verdicts)

        logger.info(
            f"[{defense_name}] round {round_num}/{n_rounds}: "
            f"clean_reference_accuracy={ctx.clean_accuracy:.6f} "
            f"committed_accuracy={committed_acc:.6f}"
        )
    return samples


def run_probe(config: dict, defense_names: list[str], n_rounds: int, rung: float,
             sigma_margin: float, *, device: str, seed: int, client_loaders,
             test_loader, start_state) -> dict:
    """Measure and summarize every defense in ``defense_names``, in that order.

    Preserves input order in the ``defenses`` list — two runs of the probe must
    produce comparable tables.
    """
    defenses = []
    for name in defense_names:
        samples = measure_defense(
            config, name, n_rounds, device=device, seed=seed,
            client_loaders=client_loaders, test_loader=test_loader,
            start_state=start_state,
        )
        summary = summarize_noise(samples, rung, sigma_margin)
        defenses.append({"name": name, **summary, "samples": samples})

    all_clear = all(d["clears"] for d in defenses)
    required_rung = (
        None if all_clear
        else max(d["required_rung"] for d in defenses if not d["clears"])
    )

    return {
        "sigma_margin": sigma_margin,
        "rung": rung,
        "n_rounds": n_rounds,
        "device": device,
        "seed": seed,
        "defenses": defenses,
        "all_clear": all_clear,
        "required_rung": required_rung,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _render_table(result: dict) -> str:
    """Markdown table for stdout — values rounded to 6 decimals for DISPLAY
    only; the pass/fail comparison already happened on the unrounded floats."""
    margin = result["sigma_margin"]
    lines = [
        f"| defense | n | mean | sd | {margin:g}·sd | bottom rung | clears | required rung |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in result["defenses"]:
        lines.append(
            "| {name} | {n} | {mean:.6f} | {sd:.6f} | {threshold:.6f} | "
            "{rung:.6f} | {clears} | {required_rung:.6f} |".format(
                name=d["name"], n=d["n"], mean=d["mean"], sd=d["sd"],
                threshold=d["threshold"], rung=d["rung"],
                clears="yes" if d["clears"] else "no",
                required_rung=d["required_rung"],
            )
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="GOAL-06 pre-flight: measure the round-to-round standard "
                    "deviation of the clean counterfactual per defense, in "
                    "single mode, with no poisoning client and no LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rounds", type=int, default=20,
                    help="clean rounds measured per defense (default: 20)")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--defenses", default="fltrust,multikrum,dnc,defl",
                    help="comma-separated defense names, each judged alone in single mode")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sigma-margin", type=float, default=3.0,
                    help="the rung clears when rung >= sigma_margin * sd (default: 3.0)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override fl.poison_seed")
    ap.add_argument("--out", default="logs/noise_probe.json")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any saved Phase-1 checkpoint and train fresh")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # Heavy / FL imports deferred so --help works without torch.
    import torch
    from data.mnist_loader import get_data_loaders
    from storage.checkpoint import state_exists, load_state
    from benchmark.phase1 import run_phase1
    from rl.rewards import DEFAULT_TARGET_LADDER

    config = yaml.safe_load(open(args.config))
    # configs/base.yaml defaults fl.device to "cuda" (RL training needs a GPU);
    # --device is this probe's own override and must reach run_phase1 too, since
    # it reads config["fl"]["device"] directly rather than taking a device arg.
    config.setdefault("fl", {})["device"] = args.device
    fl = config["fl"]
    attack = config.get("attack", {})
    data_cfg = config.get("data", {})

    seed = args.seed if args.seed is not None else int(fl.get("poison_seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    client_loaders, test_loader = get_data_loaders(
        n_clients=fl["n_clients"], batch_size=fl["batch_size"],
        data_dir=data_cfg.get("data_dir", "./data/mnist_raw"),
        iid=data_cfg.get("iid", True), bias_q=float(data_cfg.get("noniid_bias", 0.5)),
        seed=seed,
    )

    if state_exists() and not args.fresh:
        logger.info("Loading saved Phase-1 checkpoint as the probe's start state")
        start_state = load_state()
    else:
        start_state = None
    if start_state is None:
        logger.info("No (usable) checkpoint or --fresh — running Phase-1 honest "
                    "training to build the probe's start state")
        start_state = run_phase1(config, client_loaders, test_loader)

    raw_ladder = attack.get("target_ladder", DEFAULT_TARGET_LADDER)
    ladder = {int(k): float(v) for k, v in raw_ladder.items()}
    rung = ladder[min(ladder)]

    defense_names = [n.strip() for n in args.defenses.split(",") if n.strip()]

    result = run_probe(
        config, defense_names, args.rounds, rung, args.sigma_margin,
        device=args.device, seed=seed, client_loaders=client_loaders,
        test_loader=test_loader, start_state=start_state,
    )

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"[saved] {out_path}")

    print("\n" + _render_table(result))

    if result["all_clear"]:
        print(f"\nVerdict: the {rung:.6f} bottom rung clears the measured noise "
              f"by at least {args.sigma_margin} sigma for all "
              f"{len(defense_names)} defenses.")
    else:
        failing = [d["name"] for d in result["defenses"] if not d["clears"]]
        print(f"\nVerdict: the {rung:.6f} bottom rung does NOT clear "
              f"{args.sigma_margin} sigma for {', '.join(failing)}; required "
              f"bottom rung >= {result['required_rung']:.6f}.")


if __name__ == "__main__":
    main()
