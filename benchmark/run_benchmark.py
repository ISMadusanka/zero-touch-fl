#!/usr/bin/env python3
"""Run the defense benchmark: trained attacker vs a panel of defenses.

Pits the trained ATTACKER adapter against {fedavg, oracle, llm_defender, fltrust}
for N rounds and prints how much of the attack each defense detected + how well it
preserved accuracy. Must run on the GPU box (needs torch/unsloth/peft + the
trained adapters in checkpoints/).

Examples
--------
python -m benchmark.run_benchmark --rounds 200
python -m benchmark.run_benchmark --rounds 200 \
    --defenses fedavg,oracle,fltrust,llm_defender \
    --attack-temperature 0.7 --root-size 100 --out logs/benchmark
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
    ap.add_argument("--defenses", default="fedavg,oracle,llm_defender,fltrust",
                    help="comma-separated; 'fedavg' is always included (attacker reference)")
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
    ap.add_argument("--device", default=None, help="override fl.device")
    ap.add_argument("--seed", type=int, default=None, help="override fl.poison_seed")
    ap.add_argument("--out", default="logs/benchmark", help="output dir for json/csv/png (or '' to skip)")
    ap.add_argument("--no-plot", action="store_true", help="skip drawing per-round graphs")
    ap.add_argument("--fresh", action="store_true", help="force fresh Phase-1 instead of loading checkpoint")
    ap.add_argument("--log-every", type=int, default=10)
    return ap.parse_args()


def _build_root_loader(data_cfg, root_size, batch_size, seed):
    import torch
    from torch.utils.data import DataLoader, Subset
    from data.mnist_loader import load_mnist
    train_ds, _ = load_mnist(data_cfg.get("data_dir", "./data/mnist_raw"))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_ds), generator=g)[:root_size].tolist()
    return DataLoader(Subset(train_ds, idx), batch_size=min(batch_size, root_size), shuffle=True)


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
    from benchmark import report
    from benchmark.phase1 import run_phase1

    base_cfg = yaml.safe_load(open(args.config))
    attacker_cfg = yaml.safe_load(open("configs/attacker_agent.yaml"))
    defender_cfg = yaml.safe_load(open("configs/defender_agent.yaml"))
    goal = base_cfg.get("attack", {}).get("goal")
    if goal:
        attacker_cfg["attack_goal"] = goal

    fl = base_cfg["fl"]
    rl_cfg = base_cfg.get("rl", {})
    data_cfg = base_cfg["data"]
    device = args.device or fl["device"]
    seed = args.seed if args.seed is not None else int(fl.get("poison_seed", 0))

    # Always include the no-defense baseline (it is also the attacker's reference).
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
    )

    if fl.get("benign_retrain_each_round", False):
        log.warning("benign_retrain_each_round=true: the benchmark assumes frozen benign "
                    "replay (false). With retrain on, benign updates are retrained against a "
                    "stale env global and the cross-defense comparison may be skewed.")

    # Phase-1 start state (reuse the saved honest-FedAvg checkpoint, or train fresh).
    # load_state() returns None on a partial/corrupt checkpoint, so guard the unpack.
    state = load_state() if (state_exists() and not args.fresh) else None
    if state is not None:
        log.info("Loading saved Phase-1 state (global model + client weights + baseline acc)")
        global_weights, client_weights, baseline_accuracy = state
    else:
        log.info("No (usable) checkpoint or --fresh — running Phase-1 honest training")
        global_weights, client_weights, baseline_accuracy = run_phase1(
            base_cfg, client_loaders, test_loader)

    # Env: pure round generator (poison sampling + benign updates + build_updates).
    rng = random.Random(seed)
    env = FLArmsRaceEnv(base_cfg, client_loaders, test_loader, rng)
    env.reset(copy.deepcopy(global_weights), client_weights, baseline_accuracy)

    # Load the trained policy: the attacker adapter is always needed; the defender
    # adapter only if the LLM defender is in the panel.
    adapter_paths = dict(rl_cfg.get("adapter_paths", {
        "attacker": "checkpoints/attacker_adapter",
        "defender": "checkpoints/defender_adapter",
    }))
    if args.attacker_adapter:
        adapter_paths["attacker"] = args.attacker_adapter
    if args.defender_adapter:
        adapter_paths["defender"] = args.defender_adapter

    policy = LLMPolicy(
        base_model=rl_cfg.get("model", "unsloth/Llama-3.2-3B-Instruct"),
        max_seq_len=int(rl_cfg.get("max_seq_len", 8192)),
        lora_r=int(rl_cfg.get("lora_r", 16)),
        lora_alpha=int(rl_cfg.get("lora_alpha", 32)),
        load_in_4bit=bool(rl_cfg.get("load_in_4bit", True)),
        seed=seed,
        attn_implementation=rl_cfg.get("attn_implementation", "eager"),
        use_fast_generate=bool(rl_cfg.get("use_fast_generate", True)),
    )
    if not adapter_exists(adapter_paths["attacker"]):
        sys.exit(f"ERROR: no trained attacker adapter at {adapter_paths['attacker']}")
    policy.load_adapter("attacker", adapter_paths["attacker"])
    if "llm_defender" in names:
        if not adapter_exists(adapter_paths["defender"]):
            sys.exit(f"ERROR: llm_defender requested but no defender adapter at {adapter_paths['defender']}")
        policy.load_adapter("defender", adapter_paths["defender"])

    attacker_agent = AttackerAgent(attacker_cfg)
    defender_agent = DefenderAgent(defender_cfg)

    root_loader = None
    if "fltrust" in names:
        root_loader = _build_root_loader(data_cfg, args.root_size, fl["batch_size"], seed)

    defenses = build_defenses(
        names, device=device, policy=policy, defender_agent=defender_agent,
        root_loader=root_loader, root_lr=args.root_lr or float(fl["lr"]),
        root_epochs=args.root_epochs, eta=args.eta,
        defender_temperature=args.defender_temperature,
        max_new_tokens=int(rl_cfg.get("max_new_tokens", 512)),
    )

    log.info(f"Benchmark: {args.rounds} rounds | defenses={list(defenses)} | "
             f"baseline_acc={baseline_accuracy:.4f} | attack_temp={args.attack_temperature}")
    summaries, _metrics = run_benchmark(
        env, policy, attacker_agent, defenses, test_loader,
        init_global=copy.deepcopy(global_weights), baseline_accuracy=baseline_accuracy,
        n_rounds=args.rounds, attack_temperature=args.attack_temperature,
        max_new_tokens=int(rl_cfg.get("max_new_tokens", 512)), device=device,
        log_every=args.log_every,
    )

    out_dir = args.out or None
    print("\n" + report.render([summaries[n] for n in defenses], args.rounds,
                                baseline_accuracy, out_dir=out_dir))

    # Persist per-round history + draw the per-round graphs.
    if out_dir:
        import json as _json
        history = {name: m.history for name, m in _metrics.items()}
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            _json.dump({"baseline_accuracy": baseline_accuracy, "history": history}, f, indent=2)
        log.info(f"[saved] {os.path.join(out_dir, 'history.json')}")
        if not args.no_plot:
            from benchmark.plot import plot_history
            png = plot_history(history, baseline_accuracy, os.path.join(out_dir, "benchmark.png"))
            if png:
                log.info(f"[saved] {png}")


if __name__ == "__main__":
    main()
