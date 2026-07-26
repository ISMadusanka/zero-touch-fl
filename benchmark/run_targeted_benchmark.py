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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Targeted label-poisoning benchmark for zero-touch-fl",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", type=int, default=None,
                    help="the class the attack must make the model misclassify (0-9). "
                         "Default: attack.goal.label from --config.")
    ap.add_argument("--poison-clients", type=int, default=None,
                    help="how many clients the attacker may poison each round (it chooses "
                         "WHICH of its controllable pool). Default: attack.eval_poison_clients")
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
    ap.add_argument("--fresh", action="store_true", help="force fresh Phase-1 instead of checkpoint")
    ap.add_argument("--log-every", type=int, default=10)
    return ap.parse_args()


def _build_root_loader(data_cfg, root_size, batch_size, seed):
    from data.mnist_loader import get_root_loader
    return get_root_loader(root_size, batch_size,
                           data_dir=data_cfg.get("data_dir", "./data/mnist_raw"), seed=seed)


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    log = logging.getLogger("benchmark")

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
    log.info(f"Targeted goal (fixed for the run): {goal}")

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

    random.seed(seed)
    import torch
    torch.manual_seed(seed)

    client_loaders, test_loader = get_data_loaders(
        n_clients=fl["n_clients"], batch_size=fl["batch_size"],
        data_dir=data_cfg.get("data_dir", "./data/mnist_raw"), iid=data_cfg.get("iid", True),
        bias_q=float(data_cfg.get("noniid_bias", 0.5)), seed=seed,
    )

    if fl.get("benign_retrain_each_round", False):
        log.warning("benign_retrain_each_round=true: the benchmark assumes frozen benign "
                    "replay (false). With retrain on, the cross-defense comparison may skew.")

    # Phase-1 start state — SHARED with the untargeted experiment on purpose, so
    # both are measured from the identical honest baseline.
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

    # Fixed poison budget + fixed label: evaluation must not randomize either.
    eval_budget = (args.poison_clients if args.poison_clients is not None
                   else int(base_cfg.get("attack", {}).get("eval_poison_clients", 1)))
    eval_budget = max(1, min(eval_budget, env.n_compromisable))
    env.sample_budget = False
    env.budget_cap = eval_budget
    env.sample_target = False           # one label for the whole run
    env.goal = goal
    log.info(f"Eval: label={label}, poison budget={eval_budget} of pool {env.n_compromisable} "
             f"(clients {list(range(env.n_compromisable))}); attacker selects which to poison")

    adapter_paths = dict(rl_cfg.get("adapter_paths", {
        "attacker": "checkpoints/targeted/attacker_adapter",
        "defender": "checkpoints/targeted/defender_adapter",
    }))
    if args.attacker_adapter:
        adapter_paths["attacker"] = args.attacker_adapter
    if args.defender_adapter:
        adapter_paths["defender"] = args.defender_adapter

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
    if not attacker_agent.targeted:
        sys.exit("ERROR: attacker agent did not enter targeted mode — check attack.goal.type")

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
    summaries, _metrics = run_benchmark(
        env, policy, attacker_agent, defenses, test_loader,
        init_global=copy.deepcopy(global_weights), baseline_accuracy=baseline_accuracy,
        n_rounds=args.rounds, attack_temperature=args.attack_temperature,
        max_new_tokens=int(rl_cfg.get("max_new_tokens", 512)), device=device,
        log_every=args.log_every, target_drop=None,
        goal=goal, win_fraction=float(rl_cfg.get("win_fraction", 0.6)),
        n_classes=n_classes,
    )

    out_dir = args.out or None
    print("\n" + targeted_report.render(
        [summaries[n] for n in defenses], args.rounds, baseline_accuracy, label,
        out_dir=out_dir, goal=goal, n_poisoners=eval_budget))

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
