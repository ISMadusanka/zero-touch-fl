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
    ap.add_argument("--goal", default=None,
                    help="attack goal the attacker aims for, e.g. 'untargeted_degrade=0.1', "
                         "'slow_degrade=0.02', or 'targeted_label=7'. Fixed for the whole run "
                         "(no per-round sampling). Default: attack.goal from --config.")
    ap.add_argument("--defenses", default="fedavg,oracle,llm_defender,fltrust,defl,dnc,multikrum",
                    help="comma-separated; 'fedavg' is always included (attacker reference)")
    ap.add_argument("--attacker-adapter", default=None, help="override attacker checkpoint dir")
    ap.add_argument("--defender-adapter", default=None, help="override defender checkpoint dir")
    ap.add_argument("--attack-temperature", type=float, default=0.7,
                    help="attacker sampling temperature (0 = greedy/deterministic)")
    ap.add_argument("--defender-temperature", type=float, default=0.0,
                    help="LLM-defender sampling temperature")
    ap.add_argument("--max-poison-clients", type=int, default=None,
                    help="eval poison budget: max clients the attacker may poison per round "
                         "(the attacker chooses WHICH of its pool). Default: attack.eval_poison_clients (=1)")
    ap.add_argument("--root-size", type=int, default=100, help="FLTrust clean root-set size")
    ap.add_argument("--root-epochs", type=int, default=1, help="FLTrust server local epochs (R_l)")
    ap.add_argument("--root-lr", type=float, default=None, help="FLTrust server lr (default: fl.lr)")
    ap.add_argument("--eta", type=float, default=1.0, help="FLTrust global learning rate")
    ap.add_argument("--defl-delta", type=float, default=0.05,
                    help="DeFL CLP relative-rise threshold (paper delta)")
    ap.add_argument("--defl-tau", type=float, default=2.5,
                    help="DeFL MOUD per-layer outlier z-threshold")
    ap.add_argument("--dnc-num-byzantine", type=int, default=None,
                    help="DnC assumed #malicious m (default: configured poison count)")
    ap.add_argument("--dnc-c", type=float, default=1.0, help="DnC filtering fraction c")
    ap.add_argument("--dnc-niters", type=int, default=1, help="DnC subsampling iterations")
    ap.add_argument("--dnc-sub-dim", type=int, default=10000,
                    help="DnC subsample dimension b (clamped to the model's dim)")
    ap.add_argument("--multikrum-f", type=int, default=None,
                    help="Multi-Krum assumed #Byzantine f (default: configured poison count)")
    ap.add_argument("--multikrum-m", type=int, default=None,
                    help="Multi-Krum #selected/averaged (default: n - f)")
    ap.add_argument("--ensemble-members", default=None,
                    help="comma-separated members of the 'ensemble' defense "
                         "(default: defense.members from --config, else fltrust,multikrum,dnc,defl)")
    ap.add_argument("--ensemble-vote", default=None,
                    help="ensemble vote rule: majority | any | all | <int> "
                         "(default: defense.vote from --config, else majority)")
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
    from benchmark import report
    from benchmark.phase1 import run_phase1
    from rl.rewards import goal_target

    base_cfg = yaml.safe_load(open(args.config))
    attacker_cfg = yaml.safe_load(open("configs/attacker_agent.yaml"))
    defender_cfg = yaml.safe_load(open("configs/defender_agent.yaml"))
    # Attack goal: --goal overrides the config. Set it on BOTH the base config (so the
    # env's goal/logging match) and the attacker agent (whose self.goal drives the
    # benchmark prompt). Evaluation always uses a FIXED goal — never per-round sampling.
    goal = _parse_goal(args.goal) if args.goal else base_cfg.get("attack", {}).get("goal")
    if goal:
        base_cfg.setdefault("attack", {})["goal"] = goal
        attacker_cfg["attack_goal"] = goal
    log.info(f"Attack goal (fixed for the run): {goal}")
    # Requested accuracy drop for the goal-success metric (attack "succeeds" a round
    # when a defense's accuracy falls to/below baseline - target_drop).
    target_drop = goal_target(goal) if goal else None

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
        bias_q=float(data_cfg.get("noniid_bias", 0.5)), seed=seed,
    )

    if fl.get("benign_retrain_each_round", False):
        log.warning("benign_retrain_each_round=true: the benchmark assumes frozen benign "
                    "replay (false). With retrain on, benign updates are retrained against a "
                    "stale env global and the cross-defense comparison may be skewed.")

    # Phase-1 start state (reuse the saved honest-FedAvg checkpoint, or train fresh).
    # load_state() returns None on a partial/corrupt checkpoint, so guard the unpack.
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

    # Env: pure round generator (controllable pool + benign updates + build_updates).
    rng = random.Random(seed)
    env = FLArmsRaceEnv(base_cfg, client_loaders, test_loader, rng)
    env.reset(copy.deepcopy(global_weights), client_weights, baseline_accuracy)

    # Evaluation uses a FIXED poison budget: the attacker chooses which <= budget of
    # its controllable pool to poison each round. Default from config (=1); override
    # with --max-poison-clients.
    eval_budget = (args.max_poison_clients if args.max_poison_clients is not None
                   else int(base_cfg.get("attack", {}).get("eval_poison_clients", 1)))
    eval_budget = max(1, min(eval_budget, env.n_compromisable))
    env.sample_budget = False
    env.budget_cap = eval_budget
    # Evaluation uses the FIXED attack goal above (no per-round target sampling), so the
    # attacker is measured against one requested target for the whole run.
    env.sample_target = False
    log.info(f"Eval poison budget = {eval_budget} of pool {env.n_compromisable} "
             f"(clients {list(range(env.n_compromisable))}); attacker selects which to poison")

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
        sys.exit(f"ERROR: no trained attacker adapter at {adapter_paths['attacker']}")
    policy.load_adapter("attacker", adapter_paths["attacker"])
    if "llm_defender" in names:
        if not adapter_exists(adapter_paths["defender"]):
            sys.exit(f"ERROR: llm_defender requested but no defender adapter at {adapter_paths['defender']}")
        policy.load_adapter("defender", adapter_paths["defender"])

    attacker_agent = AttackerAgent(attacker_cfg)
    defender_agent = DefenderAgent(defender_cfg)

    # The `ensemble` entry contains fltrust by default, so it needs a root set too.
    defense_cfg = base_cfg.get("defense", {}) or {}
    ensemble_members = [n.strip() for n in args.ensemble_members.split(",") if n.strip()] \
        if args.ensemble_members else defense_cfg.get("members")
    ensemble_vote = args.ensemble_vote or defense_cfg.get("vote", "majority")

    root_loader = None
    if "fltrust" in names or ("ensemble" in names
                              and "fltrust" in (ensemble_members or ["fltrust"])):
        root_loader = _build_root_loader(data_cfg, args.root_size, fl["batch_size"], seed)

    # DnC / Multi-Krum assume a known upper bound on #malicious; default it to the
    # eval poison budget (the max clients the attacker may actually poison), clamped
    # to a benign majority. This is an assumed adversary budget, NOT per-round truth.
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

    log.info(f"Benchmark: {args.rounds} rounds | defenses={list(defenses)} | "
             f"baseline_acc={baseline_accuracy:.4f} | attack_temp={args.attack_temperature}")
    summaries, _metrics = run_benchmark(
        env, policy, attacker_agent, defenses, test_loader,
        init_global=copy.deepcopy(global_weights), baseline_accuracy=baseline_accuracy,
        n_rounds=args.rounds, attack_temperature=args.attack_temperature,
        max_new_tokens=int(rl_cfg.get("max_new_tokens", 512)), device=device,
        log_every=args.log_every, target_drop=target_drop,
    )

    out_dir = args.out or None
    print("\n" + report.render([summaries[n] for n in defenses], args.rounds,
                                baseline_accuracy, out_dir=out_dir, goal=goal,
                                n_poisoners=eval_budget))

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
