#!/usr/bin/env python3
"""Run the defense benchmark: the label-flip attack vs a panel of defenses.

Pits the detection-adaptive label-flip attack against {fedavg, oracle,
llm_defender, fltrust, defl, dnc, multikrum} for N rounds and prints how much of
the attack each defense detected + how well it preserved accuracy.

Needs no attacker model — the attack is a deterministic schedule. It needs a GPU
box only for the ``llm_defender`` column (torch/unsloth/peft + the trained
defender adapter in checkpoints/); every other column runs on CPU.

Examples
--------
python -m benchmark.run_benchmark --rounds 200
python -m benchmark.run_benchmark --rounds 200 \
    --defenses fedavg,oracle,fltrust,llm_defender \
    --poison-clients 0,1 --root-size 100 --out logs/benchmark
"""
import argparse
import copy
import logging
import os
import random
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml


def _parse_args():
    ap = argparse.ArgumentParser(description="Defense benchmark for zero-touch-fl",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=200, help="number of attack rounds")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--goal", default=None,
                    help="the DAMAGE BAR the attack is reported against, e.g. "
                         "'untargeted_degrade=0.1'. Does not change what the attack does. "
                         "Default: attack.goal from --config.")
    ap.add_argument("--defenses", default="fedavg,oracle,llm_defender,fltrust,defl,dnc,multikrum",
                    help="comma-separated; 'fedavg' is always included (no-defense reference)")
    ap.add_argument("--defender-adapter", default=None, help="override defender checkpoint dir")
    ap.add_argument("--defender-temperature", type=float, default=0.0,
                    help="LLM-defender sampling temperature")
    ap.add_argument("--poison-clients", default=None, metavar="IDS",
                    help="comma-separated client ids that flip labels, e.g. '0' or '0,1,2'. "
                         "Default: attack.poison_client_ids. Anything from a single client "
                         "up to every client is allowed here, so this is the knob for the "
                         "standard 'attack success vs fraction malicious' sweep.")
    ap.add_argument("--flip-fraction", type=float, default=None, metavar="F",
                    help="HOLD the attack at this flip fraction for the whole run instead of "
                         "letting the ladder adapt (e.g. 1.0 = always flip every label). "
                         "Sets start=floor=F, so no step is possible. Use this for a clean "
                         "per-defense comparison at one attack strength.")
    ap.add_argument("--ladder-feedback", default=None, metavar="DEFENSE",
                    help="which defense's verdicts drive the attack ladder (default: the "
                         "first non-fedavg defense in the panel). 'none' holds the attack "
                         "at its starting level for the whole run.")
    ap.add_argument("--root-size", type=int, default=100, help="FLTrust clean root-set size")
    ap.add_argument("--root-epochs", type=int, default=None,
                    help="FLTrust server local epochs R_l (default: defense.fltrust.root_epochs, "
                         "or sized to match an honest client's SGD iteration count)")
    ap.add_argument("--root-lr", type=float, default=None, help="FLTrust server lr (default: fl.lr)")
    ap.add_argument("--eta", type=float, default=1.0, help="FLTrust global learning rate")
    ap.add_argument("--defl-delta", type=float, default=0.05,
                    help="DeFL CLP relative-rise threshold (paper delta)")
    ap.add_argument("--defl-tau", type=float, default=2.5,
                    help="DeFL MOUD per-layer outlier z-threshold")
    ap.add_argument("--dnc-num-byzantine", type=int, default=None,
                    help="DnC assumed #malicious m (default: the poisoned-client count)")
    ap.add_argument("--dnc-c", type=float, default=1.0, help="DnC filtering fraction c")
    ap.add_argument("--dnc-niters", type=int, default=1, help="DnC subsampling iterations")
    ap.add_argument("--dnc-sub-dim", type=int, default=10000,
                    help="DnC subsample dimension b (clamped to the model's dim)")
    ap.add_argument("--multikrum-f", type=int, default=None,
                    help="Multi-Krum assumed #Byzantine f (default: the poisoned-client count)")
    ap.add_argument("--multikrum-m", type=int, default=None,
                    help="Multi-Krum #selected/averaged (default: n - f)")
    ap.add_argument("--device", default=None, help="override fl.device")
    ap.add_argument("--seed", type=int, default=None, help="override fl.poison_seed")
    ap.add_argument("--out", default="logs/benchmark", help="output dir for json/csv/png (or '' to skip)")
    ap.add_argument("--no-plot", action="store_true", help="skip drawing per-round graphs")
    ap.add_argument("--fresh", action="store_true", help="force fresh Phase-1 instead of loading checkpoint")
    ap.add_argument("--log-every", type=int, default=10)
    return ap.parse_args()


def _parse_goal(spec: str) -> dict:
    """Parse a --goal string into an attack-goal dict.

    Forms (value optional — falls back to the type's default):
        untargeted_degrade=0.1  -> {"type": "untargeted_degrade", "target_accuracy_drop": 0.1}
        slow_degrade=0.02       -> {"type": "slow_degrade", "per_round_drop": 0.02}
        targeted_label=7        -> {"type": "targeted_label", "label": 7}
    """
    gtype, _sep, val = spec.strip().partition("=")
    gtype, val = gtype.strip(), val.strip()
    if gtype == "slow_degrade":
        return {"type": gtype, "per_round_drop": float(val) if val else 0.02}
    if gtype == "targeted_label":
        return {"type": gtype, "label": int(val) if val else 7}
    if gtype in ("untargeted_degrade", ""):
        return {"type": "untargeted_degrade",
                "target_accuracy_drop": float(val) if val else 0.20}
    raise SystemExit(f"ERROR: unknown --goal type {gtype!r}; use "
                     f"untargeted_degrade=<drop> | slow_degrade=<drop> | targeted_label=<label>")


def _build_root_loader(data_cfg, root_size, batch_size, seed):
    from data.mnist_loader import build_root_loader
    return build_root_loader(root_size=root_size, batch_size=batch_size,
                             data_dir=data_cfg.get("data_dir", "./data/mnist_raw"),
                             seed=seed)


def resolve_poison_clients(attack_cfg: dict, requested: str | None, n_clients: int,
                           log=None) -> list[int]:
    """The client ids that flip labels for this evaluation.

    ``--poison-clients`` overrides ``attack.poison_client_ids``. The ceiling here is
    ``fl.n_clients``, so the benchmark can sweep the adversary's share all the way up
    — the standard "attack success vs fraction malicious" experiment — which is the
    point of evaluation as opposed to training. Ids outside the federation are
    dropped with a warning rather than clamped.
    """
    log = log or logging.getLogger("benchmark")
    if requested is None:
        raw = attack_cfg.get("poison_client_ids", [0])
        if isinstance(raw, int):
            raw = [raw]
    else:
        raw = [part for part in str(requested).replace(" ", "").split(",") if part]
    ids, seen = [], set()
    for item in raw:
        cid = int(item)
        if not 0 <= cid < n_clients:
            log.warning(f"--poison-clients: dropping {cid} — outside 0..{n_clients - 1}")
            continue
        if cid in seen:
            log.warning(f"--poison-clients: dropping duplicate {cid}")
            continue
        ids.append(cid)
        seen.add(cid)
    if not ids:
        raise SystemExit("ERROR: no valid poisoned client ids — pass --poison-clients "
                         f"with ids in 0..{n_clients - 1}")
    return sorted(ids)


def _warn_about_adversary_share(n_poisoners, n_clients, log):
    """Flag the thresholds where the benchmark's numbers change meaning.

    Every robust aggregator in the panel assumes the adversary is a MINORITY:
    Multi-Krum needs n >= 2f+3 for its resilience bound, DnC filters c*m spectral
    outliers around an honest bulk, and DeFL's MOUD-Vote calls outliers relative to
    the population. Once the poisoners are half or more of the federation, "the
    honest majority" no longer exists and those algorithms have no guarantee left —
    a low detection rate there is the expected result, not a defense bug. FLTrust is
    the exception by design: its trust comes from a server-held clean root set rather
    than from the client population, so it degrades gracefully instead of inverting.

    Raising the count that far is a legitimate experiment (it is the standard "attack
    success vs fraction malicious" sweep) — it just has to be read with this caveat.
    """
    if n_clients <= 0:
        return
    share = n_poisoners / n_clients
    if n_poisoners >= n_clients:
        log.warning(
            f"EVERY client flips labels ({n_poisoners}/{n_clients}). There are no honest "
            f"updates and no true negatives, so FPR/precision are degenerate and "
            f"detection numbers are not interpretable. Accuracy drop is the only "
            f"meaningful column in this configuration."
        )
    elif share >= 0.5:
        log.warning(
            f"Poisoners are {n_poisoners}/{n_clients} ({share:.0%}) — NO honest majority. "
            f"Multi-Krum/DnC/DeFL assume the adversary is a minority (Multi-Krum's "
            f"bound needs n >= 2f+3), so their guarantees do not hold here and weak "
            f"detection is the expected outcome rather than a defect. FLTrust is the "
            f"exception: its trust is bootstrapped from the server's clean root set."
        )
    elif share > 1.0 / 3.0:
        log.info(
            f"Poisoners are {n_poisoners}/{n_clients} ({share:.0%}) — past the ~1/3 point "
            f"where Byzantine-robust aggregators typically start to degrade. Expected, "
            f"but worth noting when reading the table."
        )


def _resolve_llm_defender(names, adapter_paths, base_cfg, exists=None):
    """Drop ``llm_defender`` from the panel when its adapter is not on disk.

    Returns ``(names, skipped)``.

    ``llm_defender`` is the only column that needs a trained DEFENDER adapter, so on a
    box that has never run training it is simply unavailable. Since it is also in the
    default ``--defenses`` list, a hard exit would abort the run before measuring ANY
    defense — throwing away FLTrust, DeFL, DnC, Multi-Krum, Oracle and FedAvg because
    one optional column was missing. Warn, drop the column, keep going. Only a panel
    with nothing comparable left is fatal.

    ``exists`` is injectable so this is testable without ``storage.checkpoint``.
    """
    if "llm_defender" not in names:
        return list(names), False
    if exists is None:
        from storage.checkpoint import adapter_exists as exists
    path = adapter_paths["defender"]
    if exists(path):
        return list(names), False

    remaining = [n for n in names if n != "llm_defender"]
    log = logging.getLogger("benchmark")
    # `fedavg` is force-added as the no-defense reference, so it does not count as a
    # defense the user asked to compare against.
    if not [n for n in remaining if n != "fedavg"]:
        raise SystemExit(
            f"ERROR: nothing left to benchmark — 'llm_defender' was the only requested "
            f"defense and no defender adapter has been trained yet (looked in {path}).\n"
            f"       Request algorithmic defenses instead, e.g.\n"
            f"       --defenses fltrust,defl,dnc,multikrum\n"
            f"       or pass --defender-adapter <dir> to point at a checkpoint."
        )
    log.warning(
        f"Skipping the 'llm_defender' column: no defender adapter at {path}. "
        f"Continuing with {remaining}. To include it, train one (python main.py) or "
        f"pass --defender-adapter <dir>."
    )
    return remaining, True


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("benchmark")

    # Heavy / FL imports are deferred so --help works without torch.
    from data.mnist_loader import get_data_loaders
    from storage.checkpoint import state_exists, load_state
    from agents.defender_agent import DefenderAgent
    from rl.env import FLArmsRaceEnv
    from benchmark.defenses import AVAILABLE, build_defenses
    from benchmark.harness import run_benchmark
    from benchmark import report
    from benchmark.phase1 import run_phase1
    from rl.rewards import goal_target

    base_cfg = yaml.safe_load(open(args.config))
    defender_cfg = yaml.safe_load(open("configs/defender_agent.yaml"))
    # The DAMAGE BAR the run is reported against. --goal overrides the config.
    goal = _parse_goal(args.goal) if args.goal else base_cfg.get("attack", {}).get("goal")
    if goal:
        base_cfg.setdefault("attack", {})["goal"] = goal
    log.info(f"Damage bar (fixed for the run): {goal}")
    target_drop = goal_target(goal) if goal else None

    fl = base_cfg["fl"]
    rl_cfg = base_cfg.get("rl", {})
    data_cfg = base_cfg["data"]
    device = args.device or fl["device"]
    seed = args.seed if args.seed is not None else int(fl.get("poison_seed", 0))

    # Always include the no-defense baseline.
    names = [n.strip() for n in args.defenses.split(",") if n.strip()]
    if "fedavg" not in names:
        names = ["fedavg"] + names
    bad = [n for n in names if n not in AVAILABLE]
    if bad:
        sys.exit(f"ERROR: unknown defense(s) {bad}; available: {AVAILABLE}")

    # --- The attack under evaluation --------------------------------------
    attack_cfg = dict(base_cfg.get("attack", {}) or {})
    n_clients = int(fl["n_clients"])
    poison_ids = resolve_poison_clients(attack_cfg, args.poison_clients, n_clients, log)
    attack_cfg["poison_client_ids"] = poison_ids
    if args.flip_fraction is not None:
        f = max(1e-6, min(1.0, float(args.flip_fraction)))
        # start == floor makes the ladder a single level, so `advance` can only ever
        # hold: the attack is pinned at f whatever any defense detects.
        attack_cfg["schedule"] = dict(attack_cfg.get("schedule") or {},
                                      start_fraction=f, floor_fraction=f)
        log.info(f"Attack PINNED at {f:.0%} flipped labels for every round "
                 f"(--flip-fraction) — the ladder cannot adapt")
    base_cfg["attack"] = attack_cfg
    _warn_about_adversary_share(len(poison_ids), n_clients, log)

    ladder_feedback = args.ladder_feedback
    if ladder_feedback is None:
        ladder_feedback = next((n for n in names if n != "fedavg"), None)
    elif ladder_feedback.strip().lower() == "none":
        ladder_feedback = None

    random.seed(seed)
    import torch
    torch.manual_seed(seed)

    client_loaders, test_loader = get_data_loaders(
        n_clients=n_clients, batch_size=fl["batch_size"],
        data_dir=data_cfg.get("data_dir", "./data/mnist_raw"), iid=data_cfg.get("iid", True),
        bias_q=float(data_cfg.get("noniid_bias", 0.5)), seed=seed,
    )

    # Phase-1 start state (reuse the saved honest-FedAvg checkpoint, or train fresh).
    # load_state() returns None on a partial/corrupt checkpoint, so guard the unpack.
    state = load_state() if (state_exists() and not args.fresh) else None
    if state is not None and len(state[1]) != n_clients:
        log.warning(f"Checkpoint has {len(state[1])} client(s) but config n_clients={n_clients} "
                    f"— ignoring the stale checkpoint and re-running Phase-1.")
        state = None
    if state is not None:
        log.info("Loading saved Phase-1 state (global model + client weights + baseline acc)")
        global_weights, client_weights, baseline_accuracy = state
    else:
        log.info("No (usable) checkpoint or --fresh — running Phase-1 honest training")
        global_weights, client_weights, baseline_accuracy = run_phase1(
            base_cfg, client_loaders, test_loader)

    # Env: pure round generator (local training + label flipping + build_updates).
    # Every client trains each round, which is not optional — the poison IS the
    # poisoned clients' local training on flipped labels. Every defense in the panel
    # still sees byte-identical updates within a round, because the env builds them
    # once and hands the same list to each.
    rng = random.Random(seed)
    env = FLArmsRaceEnv(base_cfg, client_loaders, test_loader, rng)
    env.reset(copy.deepcopy(global_weights), client_weights, baseline_accuracy)

    # The defender adapter is needed ONLY by the `llm_defender` column, and a missing
    # one is the normal case on a box that has not trained. Resolved BEFORE the model
    # is built so a skipped column does not also materialise an unused LoRA.
    adapter_paths = dict(rl_cfg.get("adapter_paths", {
        "defender": "checkpoints/defender_adapter",
    }))
    if args.defender_adapter:
        adapter_paths["defender"] = args.defender_adapter
    names, skipped_llm_defender = _resolve_llm_defender(names, adapter_paths, base_cfg)

    policy = None
    if "llm_defender" in names:
        from rl.policy import LLMPolicy
        policy = LLMPolicy(
            base_model=rl_cfg.get("model", "unsloth/Qwen2.5-3B-Instruct"),
            max_seq_len=int(rl_cfg.get("max_seq_len", 8192)),
            lora_r=int(rl_cfg.get("lora_r", 16)),
            lora_alpha=int(rl_cfg.get("lora_alpha", 32)),
            load_in_4bit=bool(rl_cfg.get("load_in_4bit", True)),
            seed=seed,
            adapters=("defender",),
            attn_implementation=rl_cfg.get("attn_implementation", "eager"),
            use_fast_generate=bool(rl_cfg.get("use_fast_generate", True)),
        )
        policy.load_adapter("defender", adapter_paths["defender"])
    defender_agent = DefenderAgent(defender_cfg)

    root_loader = None
    root_epochs = 1
    if "fltrust" in names:
        root_loader = _build_root_loader(data_cfg, args.root_size, fl["batch_size"], seed)
        # R_l must match what training used, or the FLTrust column measures a
        # different defense than the defender learned against. --root-epochs wins;
        # otherwise fall back to the config, whose default (null) sizes the root
        # update to an honest client's iteration count. FLTrust rescales every
        # accepted delta to ||g0||, so R_l alone decides how far the global can move
        # per round — see server.algo_defender.resolve_root_epochs.
        from server.algo_defender import DEFAULT_MAX_ROOT_EPOCHS, resolve_root_epochs
        ft_cfg = ((base_cfg.get("defense") or {}).get("fltrust") or {})
        configured = args.root_epochs
        if configured is None:
            configured = ft_cfg.get("root_epochs")
        root_epochs = resolve_root_epochs(
            configured, root_batches=len(root_loader),
            client_iterations=int(fl["local_epochs"]) * len(client_loaders[0]),
            max_epochs=int(ft_cfg.get("max_root_epochs") or DEFAULT_MAX_ROOT_EPOCHS),
        )
        log.info(f"FLTrust root fine-tuning: root_epochs={root_epochs} over "
                 f"{len(root_loader)} batch(es) = ~{root_epochs * len(root_loader)} SGD "
                 f"iterations (honest client: "
                 f"~{int(fl['local_epochs']) * len(client_loaders[0])})")

    # DnC / Multi-Krum assume a known upper bound on #malicious; default it to the
    # number of clients actually flipping labels, clamped to a benign majority.
    #
    # The clamp stays even when more clients are poisoned than that: both algorithms
    # are only defined for a minority adversary (Multi-Krum keeps m = n - f updates,
    # so f >= n/2 leaves it averaging half the federation or fewer), so feeding them
    # f = 12 of 20 does not make them stronger, it makes them ill-posed.
    n_poisoners = len(poison_ids)
    assumed_byz = max(1, min(n_poisoners, (n_clients - 1) // 2))
    if n_poisoners > assumed_byz:
        log.info(
            f"DnC/Multi-Krum assumed_byzantine capped at {assumed_byz} (of {n_clients}) while "
            f"{n_poisoners} clients are actually poisoned — those algorithms are only "
            f"defined for a minority adversary. Override with --dnc-num-byzantine / "
            f"--multikrum-f."
        )
    dnc_m = args.dnc_num_byzantine if args.dnc_num_byzantine is not None else assumed_byz
    mk_f = args.multikrum_f if args.multikrum_f is not None else assumed_byz

    defenses = build_defenses(
        names, device=device, policy=policy, defender_agent=defender_agent,
        root_loader=root_loader, root_lr=args.root_lr or float(fl["lr"]),
        root_epochs=root_epochs, eta=args.eta,
        defender_temperature=args.defender_temperature,
        max_new_tokens=int(rl_cfg.get("max_new_tokens", 512)),
        defl_delta=args.defl_delta, defl_tau=args.defl_tau,
        dnc_num_byzantine=dnc_m, dnc_c=args.dnc_c, dnc_niters=args.dnc_niters,
        dnc_sub_dim=args.dnc_sub_dim, dnc_seed=seed,
        multikrum_num_byzantine=mk_f, multikrum_m=args.multikrum_m,
    )
    if ladder_feedback is not None and ladder_feedback not in defenses:
        ladder_feedback = next((n for n in defenses if n != "fedavg"), None)

    log.info(f"Benchmark: {args.rounds} rounds | defenses={list(defenses)} | "
             f"poisoned={poison_ids} | baseline_acc={baseline_accuracy:.4f}"
             + (" | llm_defender SKIPPED (no defender adapter)"
                if skipped_llm_defender else ""))
    summaries, _metrics = run_benchmark(
        env, defenses, test_loader,
        init_global=copy.deepcopy(global_weights), baseline_accuracy=baseline_accuracy,
        n_rounds=args.rounds, device=device, log_every=args.log_every,
        target_drop=target_drop, ladder_feedback=ladder_feedback,
    )

    out_dir = args.out or None
    print("\n" + report.render([summaries[n] for n in defenses], args.rounds,
                                baseline_accuracy, out_dir=out_dir, goal=goal,
                                n_poisoners=n_poisoners))

    # Persist per-round history + draw the per-round graphs.
    if out_dir:
        import json as _json
        history = {name: m.history for name, m in _metrics.items()}
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            _json.dump({"baseline_accuracy": baseline_accuracy,
                        "defenses": list(defenses),
                        # Recorded so a saved result is self-describing.
                        "llm_defender_skipped": skipped_llm_defender,
                        "poison_client_ids": poison_ids,
                        "flip_fraction_pinned": args.flip_fraction,
                        "ladder_feedback": ladder_feedback,
                        "rounds": args.rounds,
                        "history": history}, f, indent=2)
        log.info(f"[saved] {os.path.join(out_dir, 'history.json')}")
        if not args.no_plot:
            from benchmark.plot import plot_history
            png = plot_history(history, baseline_accuracy, os.path.join(out_dir, "benchmark.png"))
            if png:
                log.info(f"[saved] {png}")


if __name__ == "__main__":
    main()
