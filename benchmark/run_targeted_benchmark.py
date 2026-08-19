#!/usr/bin/env python3
"""Evaluate the TARGETED-poisoning attacker against a panel of defenses.

Separate command from ``benchmark.run_benchmark`` (the untargeted one) so the two
experiments never share a config, an adapter or an output directory.

The two knobs the experiment is parameterised on:

    --label            WHICH class to make the model misclassify (0-9)
    --poison-clients   HOW MANY clients the attacker may poison per round

Everything else defaults to ``configs/targeted.yaml``. The report shows, per
defense, the target class's recall next to every other class's, against a clean
reference row — so whether *only* the target broke is visible directly.

Examples
--------
python -m benchmark.run_targeted_benchmark --label 2 --poison-clients 3 --rounds 100
python -m benchmark.run_targeted_benchmark --label 4 --poison-clients 1 \
    --defenses fedavg,oracle,llm_defender,fltrust,dnc --out logs/targeted/benchmark_l4
"""
import argparse
import copy
import logging
import os
import random
import sys
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Targeted label-poisoning benchmark for zero-touch-fl",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", type=int, default=None,
                    help="the class the attack must make the model misclassify (0-9). "
                         "Default: the label derived at runtime from client "
                         "attack.target_label_from_client's own non-IID shard (the same "
                         "class training attacked), or attack.goal.label when that is null.")
    ap.add_argument("--poison-clients", type=int, default=None,
                    help="how many clients the attacker may poison each round (it chooses "
                         "WHICH of its controllable pool). Default: attack.eval_poison_clients")
    ap.add_argument("--poison-client-ids", default=None,
                    help="comma-separated client ids the attacker controls, e.g. '0' or '0,3,7'. "
                         "Default: the config's insider prefix [0 .. fl.n_compromisable). Naming "
                         "ids also sets the per-round budget to how many were named, unless "
                         "--poison-clients says otherwise. NOTE the policy was trained on the "
                         "class the config's insider holds most of, so compromising a DIFFERENT "
                         "client measures generalization, not the trained attack.")
    ap.add_argument("--rounds", type=int, default=100, help="number of attack rounds")
    ap.add_argument("--config", default="configs/targeted.yaml")
    ap.add_argument("--target-class-drop", type=float, default=None,
                    help="override attack.goal.target_class_drop (how much of the target "
                         "class's recall counts as a full success)")
    ap.add_argument("--max-collateral", type=float, default=None,
                    help="override attack.goal.max_collateral (how much mean recall the OTHER "
                         "classes may lose before the attack stops counting as targeted)")
    ap.add_argument("--defenses", default="fedavg,oracle,llm_defender,fltrust,defl,dnc,multikrum",
                    help="comma-separated; 'fedavg' is always included (no-defense reference)")
    ap.add_argument("--attacker-adapter", default=None, help="override attacker checkpoint dir")
    ap.add_argument("--defender-adapter", default=None, help="override defender checkpoint dir")
    ap.add_argument("--attack-temperature", type=float, default=0.7,
                    help="attacker sampling temperature (0 = greedy/deterministic)")
    ap.add_argument("--defender-temperature", type=float, default=0.0,
                    help="LLM-defender sampling temperature")
    ap.add_argument("--root-size", type=int, default=100, help="FLTrust clean root-set size")
    ap.add_argument("--root-epochs", type=int, default=1, help="FLTrust server local epochs (R_l)")
    ap.add_argument("--root-lr", type=float, default=None, help="FLTrust server lr (default: fl.lr)")
    ap.add_argument("--eta", type=float, default=1.0, help="FLTrust global learning rate")
    ap.add_argument("--defl-delta", type=float, default=0.05, help="DeFL CLP relative-rise threshold")
    ap.add_argument("--defl-tau", type=float, default=2.5, help="DeFL MOUD per-layer outlier z-threshold")
    ap.add_argument("--dnc-num-byzantine", type=int, default=None,
                    help="DnC assumed #malicious m (default: --poison-clients)")
    ap.add_argument("--dnc-c", type=float, default=1.0, help="DnC filtering fraction c")
    ap.add_argument("--dnc-niters", type=int, default=1, help="DnC subsampling iterations")
    ap.add_argument("--dnc-sub-dim", type=int, default=10000, help="DnC subsample dimension b")
    ap.add_argument("--multikrum-f", type=int, default=None,
                    help="Multi-Krum assumed #Byzantine f (default: --poison-clients)")
    ap.add_argument("--multikrum-m", type=int, default=None,
                    help="Multi-Krum #selected/averaged (default: n - f)")
    ap.add_argument("--device", default=None, help="override fl.device")
    ap.add_argument("--seed", type=int, default=None, help="override fl.poison_seed")
    ap.add_argument("--out", default="logs/targeted/benchmark",
                    help="output dir for json/csv/png (or '' to skip)")
    ap.add_argument("--no-plot", action="store_true", help="skip drawing per-round graphs")
    ap.add_argument("--events", default="",
                    help="stream structured per-round JSON events: a file path, or '-' for "
                         "stdout (each line prefixed with the benchmark.events sentinel). "
                         "This is how `python -m benchmark.ui` watches a run live. Off by "
                         "default, so normal CLI output is unchanged.")
    ap.add_argument("--fresh", action="store_true", help="force fresh Phase-1 instead of checkpoint")
    ap.add_argument("--log-every", type=int, default=10)
    return ap.parse_args()


def _build_root_loader(data_cfg, root_size, batch_size, seed):
    from data.mnist_loader import get_root_loader
    return get_root_loader(root_size, batch_size,
                           data_dir=data_cfg.get("data_dir", "./data/mnist_raw"), seed=seed)


def _parse_client_ids(raw, n_clients: int):
    """``"0,3,7"`` -> ``[0, 3, 7]``. Empty/None -> None (use the config's pool)."""
    if not raw or not str(raw).strip():
        return None
    try:
        ids = [int(x) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        sys.exit(f"ERROR: --poison-client-ids must be comma-separated integers, got {raw!r}")
    ids = list(dict.fromkeys(ids))                       # dedup, preserve order
    if not ids:
        sys.exit(f"ERROR: --poison-client-ids named no clients (got {raw!r})")
    bad = [c for c in ids if not 0 <= c < n_clients]
    if bad:
        sys.exit(f"ERROR: --poison-client-ids {bad} outside [0, {n_clients - 1}]")
    return ids


def _clean_op(op):
    """One attack operation, bounded.

    The plan is raw LLM output, so a degenerate round (a 10k-element list, a
    nested blob) must not blow up the event stream a watcher is reading.
    """
    if not isinstance(op, dict):
        return {"op": str(op)[:120]}
    out = {}
    for k, v in list(op.items())[:8]:
        key = str(k)[:32]
        if v is None or isinstance(v, (int, float, bool)):
            out[key] = v
        elif isinstance(v, list):
            out[key] = [x for x in v[:16] if isinstance(x, (int, float, str))]
        else:
            out[key] = str(v)[:120]
    return out


def _plan_digest(text: str, max_ops: int = 8) -> list:
    """The attacker's plan as ``[{client, ops}]`` — what it actually tried this round.

    Best-effort: an unparseable plan is exactly the ``n_malformed`` case the round
    event already reports, so return nothing rather than raising.
    """
    try:
        from agents.attack_ops import extract_selection
        sel = extract_selection(text)
    except Exception:                    # noqa: BLE001 - a display nicety, never fatal
        return []
    if not sel:
        return []
    if sel["per_client"]:
        entries = [(e["id"], e["operations"]) for e in sel["per_client"]]
    elif sel["shared_ids"]:
        entries = [(cid, sel["shared_ops"]) for cid in sel["shared_ids"]]
    elif sel["shared_ops"] is not None:
        entries = [(None, sel["shared_ops"])]            # ids auto-selected downstream
    else:
        return []
    return [{"client": cid, "ops": [_clean_op(o) for o in (ops or [])[:max_ops]]}
            for cid, ops in entries[:16]]


def _round_event(state: dict) -> dict:
    """Compress the harness's per-round state into a UI-sized event.

    The raw state carries the attacker's whole LLM output and every defense's full
    summary dict; streaming that verbatim for 100 rounds is megabytes. Keep the
    fields a live view reads, and turn the attacker's text into its parsed plan.
    """
    out = {k: state[k] for k in ("round", "index", "n_rounds", "poisoned",
                                 "pool", "budget", "n_malformed")}
    out["plan"] = _plan_digest(state.get("attack_text", ""))
    out["defenses"] = {
        name: {
            "accuracy": d["last"]["accuracy"],
            "per_class": d["last"]["per_class"],
            "flagged": d["last"]["flagged"],
            "tp": d["last"]["tp"], "fn": d["last"]["fn"],
            "fp": d["last"]["fp"], "tn": d["last"]["tn"],
            "skipped": d["last"]["skipped"],
            "targeted": d["last"]["targeted"],
            "detection_rate": d["summary"]["detection_rate"],
            "fpr": d["summary"]["fpr"],
            "f1": d["summary"]["f1"],
            "attack_success_rate": d["summary"]["attack_success_rate"],
            "targeted_success_rate": d["summary"].get("targeted_success_rate"),
            "mean_collateral": d["summary"].get("mean_collateral"),
        }
        for name, d in state["defenses"].items()
    }
    return out


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("benchmark")

    # The whole run is wrapped so a watcher (benchmark.ui) is told how it ended
    # rather than being left guessing at a dead pipe. Emitting nothing is the
    # default, so without --events this wrapper is invisible.
    from benchmark.events import EventEmitter
    events = EventEmitter(args.events)
    status = "ok"
    try:
        _run(args, log, events)
    except KeyboardInterrupt:
        status = "cancelled"
        raise
    except SystemExit as e:
        # The failure paths below use sys.exit("ERROR: ..."), so the message IS
        # the exit code on this path.
        if e.code not in (0, None):
            status = "error"
            events.emit("error", message=str(e.code))
        raise
    except BaseException as e:                # noqa: BLE001 - report it, then re-raise
        status = "error"
        events.emit("error", message=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc())
        raise
    finally:
        events.emit("end", status=status)
        events.close()


def _run(args, log, events):
    # Heavy / FL imports are deferred so --help works without torch.
    from data.mnist_loader import get_data_loaders
    from storage.checkpoint import state_exists, load_state, adapter_exists
    from agents.attacker_agent import AttackerAgent
    from agents.defender_agent import DefenderAgent
    from rl.env import FLArmsRaceEnv
    from rl.policy import LLMPolicy
    from benchmark.defenses import AVAILABLE, build_defenses
    from benchmark.harness import run_benchmark
    from benchmark import targeted_report
    from benchmark.phase1 import run_phase1

    base_cfg = yaml.safe_load(open(args.config))
    attacker_cfg = yaml.safe_load(open("configs/attacker_agent.yaml"))
    defender_cfg = yaml.safe_load(open("configs/defender_agent.yaml"))

    # ---- Build the FIXED evaluation goal: one label, no per-round sampling. ----
    goal = dict(base_cfg.get("attack", {}).get("goal") or {})
    goal["type"] = "targeted_label"
    if args.label is not None:
        goal["label"] = int(args.label)
    goal.setdefault("label", 2)
    if args.target_class_drop is not None:
        goal["target_class_drop"] = float(args.target_class_drop)
    if args.max_collateral is not None:
        goal["max_collateral"] = float(args.max_collateral)
    label = int(goal["label"])

    n_classes = int(base_cfg.get("data", {}).get("n_classes", 10))
    if not 0 <= label < n_classes:
        sys.exit(f"ERROR: --label must be in [0, {n_classes - 1}], got {label}")

    base_cfg.setdefault("attack", {})["goal"] = goal
    attacker_cfg["attack_goal"] = goal
    attacker_cfg["n_clients"] = int(base_cfg["fl"]["n_clients"])
    attacker_cfg["n_classes"] = n_classes
    # NOTE: `label` may still be replaced below, once the data is partitioned, when the
    # config derives it from a client's own shard (attack.target_label_from_client).
    log.info(f"Targeted goal from config/flags: {goal}")

    fl = base_cfg["fl"]
    rl_cfg = base_cfg.get("rl", {})
    data_cfg = base_cfg["data"]
    device = args.device or fl["device"]
    seed = args.seed if args.seed is not None else int(fl.get("poison_seed", 0))

    names = [n.strip() for n in args.defenses.split(",") if n.strip()]
    if "fedavg" not in names:
        names = ["fedavg"] + names
    bad = [n for n in names if n not in AVAILABLE]
    if bad:
        sys.exit(f"ERROR: unknown defense(s) {bad}; available: {AVAILABLE}")

    # Parsed here rather than at the point of use so a typo fails in a second,
    # not after Phase-1 and a multi-GB model load.
    pool_ids = _parse_client_ids(args.poison_client_ids, int(fl["n_clients"]))

    events.emit("phase", phase="setup", message=(
        f"label={label} defenses={names} rounds={args.rounds} device={device}"))
    events.emit("phase", phase="data",
                message=f"partitioning {data_cfg.get('dataset', 'mnist')} across "
                        f"{fl['n_clients']} clients (iid={data_cfg.get('iid', True)})")

    random.seed(seed)
    import torch
    torch.manual_seed(seed)

    client_loaders, test_loader = get_data_loaders(
        n_clients=fl["n_clients"], batch_size=fl["batch_size"],
        data_dir=data_cfg.get("data_dir", "./data/mnist_raw"), iid=data_cfg.get("iid", True),
        bias_q=float(data_cfg.get("noniid_bias", 0.5)), seed=seed,
        n_classes=n_classes,
    )

    # Match training's label choice: when the config derives the target label from a
    # client's own non-IID shard, evaluation must attack that SAME class or it is
    # measuring a policy on a goal it was never trained for. An explicit --label wins
    # (that is what it is for: probing generalization to another class).
    from data.target_label import CLIENT_KEY, resolve_client_target_label
    if args.label is None:
        info = resolve_client_target_label(base_cfg, client_loaders)
        if info is not None:
            label = int(goal["label"])          # goal was mutated in place
            log.info(f"Targeted goal (fixed for the run, label read off client "
                     f"{info['client_id']}'s data): {goal}")
    elif (base_cfg.get("attack") or {}).get(CLIENT_KEY) is not None:
        log.info(f"--label {label} given explicitly — ignoring attack.{CLIENT_KEY}="
                 f"{base_cfg['attack'][CLIENT_KEY]} (the label training derived from "
                 f"that client's data may differ)")

    if fl.get("benign_retrain_each_round", False):
        log.warning("benign_retrain_each_round=true: the benchmark assumes frozen benign "
                    "replay (false). With retrain on, the cross-defense comparison may skew.")

    # Phase-1 start state — SHARED with the untargeted experiment on purpose, so
    # both are measured from the identical honest baseline.
    events.emit("phase", phase="phase1", message="restoring / running Phase-1 honest baseline")
    state = load_state() if (state_exists() and not args.fresh) else None
    if state is not None and len(state[1]) != fl["n_clients"]:
        log.warning(f"Checkpoint has {len(state[1])} client(s) but config n_clients={fl['n_clients']} "
                    f"— ignoring the stale checkpoint and re-running Phase-1.")
        state = None
    if state is not None:
        log.info("Loading saved Phase-1 state (global model + client weights + baseline acc)")
        global_weights, client_weights, baseline_accuracy = state
    else:
        log.info("No (usable) checkpoint or --fresh — running Phase-1 honest training")
        global_weights, client_weights, baseline_accuracy = run_phase1(
            base_cfg, client_loaders, test_loader)

    rng = random.Random(seed)
    env = FLArmsRaceEnv(base_cfg, client_loaders, test_loader, rng)
    env.reset(copy.deepcopy(global_weights), client_weights, baseline_accuracy)

    # WHICH clients the attacker controls: the named set, else the config's insider
    # prefix [0 .. n_compromisable).
    if pool_ids:
        env.pool_override = pool_ids
    pool = pool_ids or list(range(env.n_compromisable))

    # Fixed poison budget + fixed label: evaluation must not randomize either.
    # Naming ids implies "poison all of them" unless --poison-clients narrows it.
    default_budget = (len(pool) if pool_ids
                      else int(base_cfg.get("attack", {}).get("eval_poison_clients", 1)))
    eval_budget = (args.poison_clients if args.poison_clients is not None
                   else default_budget)
    eval_budget = max(1, min(eval_budget, len(pool)))
    env.sample_budget = False
    env.budget_cap = eval_budget
    env.sample_target = False           # one label for the whole run
    env.goal = goal
    log.info(f"Eval: label={label}, poison budget={eval_budget} of pool {len(pool)} "
             f"(clients {pool}); attacker selects which to poison")
    if pool_ids and pool_ids != list(range(env.n_compromisable)):
        log.warning(f"--poison-client-ids {pool_ids} overrides the config's insider pool "
                    f"{list(range(env.n_compromisable))}. The policy was trained on the class "
                    f"the config's insider owns, so this measures generalization.")

    adapter_paths = dict(rl_cfg.get("adapter_paths", {
        "attacker": "checkpoints/targeted/attacker_adapter",
        "defender": "checkpoints/targeted/defender_adapter",
    }))
    if args.attacker_adapter:
        adapter_paths["attacker"] = args.attacker_adapter
    if args.defender_adapter:
        adapter_paths["defender"] = args.defender_adapter

    events.emit("phase", phase="policy",
                message=f"loading {rl_cfg.get('model', 'unsloth/Qwen2.5-3B-Instruct')} "
                        f"+ attacker adapter (this is the slow step)")
    policy = LLMPolicy(
        base_model=rl_cfg.get("model", "unsloth/Qwen2.5-3B-Instruct"),
        max_seq_len=int(rl_cfg.get("max_seq_len", 8192)),
        lora_r=int(rl_cfg.get("lora_r", 16)),
        lora_alpha=int(rl_cfg.get("lora_alpha", 32)),
        load_in_4bit=bool(rl_cfg.get("load_in_4bit", True)),
        seed=seed,
        attn_implementation=rl_cfg.get("attn_implementation", "eager"),
        use_fast_generate=bool(rl_cfg.get("use_fast_generate", True)),
    )
    if not adapter_exists(adapter_paths["attacker"]):
        sys.exit(f"ERROR: no trained TARGETED attacker adapter at {adapter_paths['attacker']}. "
                 f"Train one first:  python train_targeted.py")
    policy.load_adapter("attacker", adapter_paths["attacker"])
    if "llm_defender" in names:
        if not adapter_exists(adapter_paths["defender"]):
            sys.exit(f"ERROR: llm_defender requested but no defender adapter at "
                     f"{adapter_paths['defender']}")
        policy.load_adapter("defender", adapter_paths["defender"])

    attacker_agent = AttackerAgent(attacker_cfg)
    defender_agent = DefenderAgent(defender_cfg)
    # The `ensemble` entry contains fltrust by default, so it needs a root set too.
    ensemble_members = (base_cfg.get("defense", {}) or {}).get("members")
    ensemble_vote = (base_cfg.get("defense", {}) or {}).get("vote", "majority")

    root_loader = None
    if "fltrust" in names or ("ensemble" in names
                              and "fltrust" in (ensemble_members or ["fltrust"])):
        root_loader = _build_root_loader(data_cfg, args.root_size, fl["batch_size"], seed)

    n_cl = int(fl["n_clients"])
    assumed_byz = max(1, min(eval_budget, (n_cl - 1) // 2))
    dnc_m = args.dnc_num_byzantine if args.dnc_num_byzantine is not None else assumed_byz
    mk_f = args.multikrum_f if args.multikrum_f is not None else assumed_byz

    defenses = build_defenses(
        names, device=device, policy=policy, defender_agent=defender_agent,
        root_loader=root_loader, root_lr=args.root_lr or float(fl["lr"]),
        root_epochs=args.root_epochs, eta=args.eta,
        defender_temperature=args.defender_temperature,
        max_new_tokens=int(rl_cfg.get("max_new_tokens", 512)),
        defl_delta=args.defl_delta, defl_tau=args.defl_tau,
        dnc_num_byzantine=dnc_m, dnc_c=args.dnc_c, dnc_niters=args.dnc_niters,
        dnc_sub_dim=args.dnc_sub_dim, dnc_seed=seed,
        multikrum_num_byzantine=mk_f, multikrum_m=args.multikrum_m,
        ensemble_members=ensemble_members, ensemble_vote=ensemble_vote,
    )

    log.info(f"Targeted benchmark: {args.rounds} rounds | label={label} | "
             f"poison_clients={eval_budget} | defenses={list(defenses)} | "
             f"baseline_acc={baseline_accuracy:.4f} | attack_temp={args.attack_temperature}")
    events.emit("start", n_rounds=args.rounds, target_label=label, pool=pool,
                poison_budget=eval_budget, defenses=list(defenses),
                baseline_accuracy=baseline_accuracy, n_classes=n_classes,
                n_clients=int(fl["n_clients"]), goal=goal, device=device,
                attack_temperature=args.attack_temperature, out_dir=args.out or None)
    summaries, _metrics = run_benchmark(
        env, policy, attacker_agent, defenses, test_loader,
        init_global=copy.deepcopy(global_weights), baseline_accuracy=baseline_accuracy,
        n_rounds=args.rounds, attack_temperature=args.attack_temperature,
        max_new_tokens=int(rl_cfg.get("max_new_tokens", 512)), device=device,
        log_every=args.log_every, target_drop=None,
        goal=goal, win_fraction=float(rl_cfg.get("win_fraction", 0.6)),
        n_classes=n_classes,
        on_start=(lambda ce: events.emit(
            "clean", per_class=[round(float(v), 6) for v in ce.per_class],
            support=list(ce.support), overall=float(ce.overall))) if events.enabled else None,
        on_round=(lambda st: events.emit("round", **_round_event(st))) if events.enabled else None,
    )

    out_dir = args.out or None
    report = targeted_report.render(
        [summaries[n] for n in defenses], args.rounds, baseline_accuracy, label,
        out_dir=out_dir, goal=goal, n_poisoners=eval_budget)
    print("\n" + report)
    events.emit("summary", target_label=label, n_rounds=args.rounds,
                baseline_accuracy=baseline_accuracy, n_poisoners=eval_budget,
                results=[summaries[n] for n in defenses], report=report, out_dir=out_dir)

    if out_dir:
        import json as _json
        history = {name: m.history for name, m in _metrics.items()}
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            _json.dump({"baseline_accuracy": baseline_accuracy, "target_label": label,
                        "history": history}, f, indent=2)
        log.info(f"[saved] {os.path.join(out_dir, 'history.json')}")
        if not args.no_plot:
            from benchmark.targeted_plot import plot_targeted_history
            png = plot_targeted_history(history, label, os.path.join(out_dir, "targeted.png"))
            if png:
                log.info(f"[saved] {png}")


if __name__ == "__main__":
    main()
