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
    ap.add_argument("--attack-retries", type=int, default=3, metavar="N",
                    help="extra attacker samples to draw when a round's action does not "
                         "fill the exact poison quota (truncated or no-op plan). A round "
                         "with no usable action after these is skipped and excluded from "
                         "the metrics, not fatal. 0 = one attempt per round")
    ap.add_argument("--defender-temperature", type=float, default=0.0,
                    help="LLM-defender sampling temperature")
    ap.add_argument("--max-poison-clients", type=int, default=None, metavar="N",
                    help="exact eval poison quota per round (the attacker chooses WHICH "
                         "clients fill it). Anything from 1 up to "
                         "fl.n_clients (20) is allowed here — the controllable pool is widened "
                         "to match, so this is NOT limited to fl.n_compromisable the way "
                         "training is. Default: attack.eval_poison_clients")
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
                    help="DnC assumed #malicious m (default: configured poison count)")
    ap.add_argument("--dnc-c", type=float, default=1.0, help="DnC filtering fraction c")
    ap.add_argument("--dnc-niters", type=int, default=1, help="DnC subsampling iterations")
    ap.add_argument("--dnc-sub-dim", type=int, default=10000,
                    help="DnC subsample dimension b (clamped to the model's dim)")
    ap.add_argument("--multikrum-f", type=int, default=None,
                    help="Multi-Krum assumed #Byzantine f (default: configured poison count)")
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


def _resolve_eval_budget(env, requested, log=None):
    """Pin ``env`` to a fixed evaluation poison budget and return it.

    The benchmark's ceiling is ``fl.n_clients``, **not** ``fl.n_compromisable``.
    Training pins the attacker to a fixed insider pool (5 of 20 by default) because
    that is the threat model it learns under. Evaluation is where you ask "how do these
    defenses hold up as the adversary controls more of the federation?", which has to
    go past that pool and up to every client. Clamping the eval budget to
    ``n_compromisable`` made ``--max-poison-clients 10`` silently behave exactly like 5.

    Raising the cap alone is not enough. The attacker may only pick from
    ``env.pool_ids == range(n_compromisable)``, and ``AttackerAgent.select_and_apply``
    clamps the budget to the pool size — so the pool is widened to match. That widening
    mutates only THIS env instance, after ``reset()``: training reads
    ``fl.n_compromisable`` from the config and is untouched.

    Also switches off the per-round budget/target randomisation that training can use,
    so every evaluated round must poison exactly the requested count and faces the
    same goal. The attacker chooses WHICH clients fill the quota, never how many.
    """
    log = log or logging.getLogger("benchmark")
    n_all = int(env.n_clients)
    budget = max(1, min(int(requested), n_all))
    if budget != int(requested):
        log.warning(f"--max-poison-clients {requested} clamped to {budget} "
                    f"(1..fl.n_clients={n_all} is the valid range)")

    trained_pool = int(env.n_compromisable)
    if budget > trained_pool:
        env.n_compromisable = budget
        log.warning(
            f"Widening the attacker's controllable pool from {trained_pool} to {budget} "
            f"client(s) for this evaluation (clients 0..{budget - 1}). The policy was "
            f"TRAINED against a pool of {trained_pool}, so this measures "
            f"out-of-distribution generalization to a larger insider foothold — "
            f"informative, but not the setting it was fitted to. Raise "
            f"fl.n_compromisable and retrain to make it in-distribution."
        )

    env.sample_budget = False       # eval never randomises the budget
    env.budget_cap = budget
    env.sample_target = False       # ...nor the goal
    # ...nor sweeps it. The training curriculum walks the poison quota through
    # 1..5 in blocks; evaluation must hold it at exactly `budget` for every round
    # so the panel's columns are comparable. (The benchmark builds its env without
    # a curriculum, so this is belt-and-braces against that changing.)
    env.curriculum = None
    log.info(f"Eval poison quota = exactly {budget} of pool {env.n_compromisable} "
             f"(clients {list(range(env.n_compromisable))}); "
             f"attacker selects which to poison")
    _warn_about_adversary_share(budget, n_all, log)
    return budget


def _warn_about_adversary_share(budget, n_clients, log):
    """Flag the thresholds where the benchmark's numbers change meaning.

    Every robust aggregator in the panel assumes the adversary is a MINORITY:
    Multi-Krum needs n >= 2f+3 for its resilience bound, DnC filters c*m spectral
    outliers around an honest bulk, and DeFL's MOUD-Vote calls outliers relative to
    the population. Once the poisoners are half or more of the federation, "the
    honest majority" no longer exists and those algorithms have no guarantee left —
    a low detection rate there is the expected result, not a defense bug. FLTrust is
    the exception by design: its trust comes from a server-held clean root set rather
    than from the client population, so it degrades gracefully instead of inverting.

    Raising the budget that far is a legitimate experiment (it is the standard
    "attack success vs fraction malicious" sweep) — it just has to be read with this
    caveat, so we say it plainly rather than letting the table look like a defense
    collapse at ordinary settings.
    """
    if n_clients <= 0:
        return
    share = budget / n_clients
    if budget >= n_clients:
        log.warning(
            f"EVERY client is a poisoner ({budget}/{n_clients}). There are no honest "
            f"updates and no true negatives, so FPR/precision are degenerate and "
            f"detection numbers are not interpretable. Accuracy drop is the only "
            f"meaningful column in this configuration."
        )
    elif share >= 0.5:
        log.warning(
            f"Poisoners are {budget}/{n_clients} ({share:.0%}) — NO honest majority. "
            f"Multi-Krum/DnC/DeFL assume the adversary is a minority (Multi-Krum's "
            f"bound needs n >= 2f+3), so their guarantees do not hold here and weak "
            f"detection is the expected outcome rather than a defect. FLTrust is the "
            f"exception: its trust is bootstrapped from the server's clean root set."
        )
    elif share > 1.0 / 3.0:
        log.info(
            f"Poisoners are {budget}/{n_clients} ({share:.0%}) — past the ~1/3 point "
            f"where Byzantine-robust aggregators typically start to degrade. Expected, "
            f"but worth noting when reading the table."
        )


def _resolve_llm_defender(names, adapter_paths, base_cfg, exists=None):
    """Drop ``llm_defender`` from the panel when its adapter is not on disk.

    Returns ``(names, skipped)``.

    ``llm_defender`` is the only column that needs a trained DEFENDER adapter, and
    with ``defense.mode: algorithmic`` — the shipped default — that adapter is never
    written: the defender LLM is disabled and ``rl/schedule.py`` trains the attacker
    only, deliberately leaving the defender checkpoint untouched. Since
    ``llm_defender`` is also in the default ``--defenses`` list, the old hard
    ``sys.exit`` meant a plain ``run_benchmark`` on a normally-trained run aborted
    before measuring ANY defense — throwing away FLTrust, DeFL, DnC, Multi-Krum,
    Oracle and FedAvg because one optional column was unavailable.

    A missing defender adapter is therefore expected, not exceptional: warn, drop the
    column, keep going. Only a panel with nothing comparable left is fatal — and that
    is reported with the flag needed to fix it rather than as a stack trace.

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
    mode = str(((base_cfg.get("defense") or {}).get("mode") or "algorithmic")).lower()
    why = ("the defender LLM is disabled in this config (defense.mode: algorithmic), "
           "so no defender adapter is ever trained"
           if mode == "algorithmic" else "no defender adapter has been trained yet")
    log = logging.getLogger("benchmark")
    # `fedavg` is force-added as the no-defense reference, so it does not count as a
    # defense the user asked to compare against.
    if not [n for n in remaining if n != "fedavg"]:
        raise SystemExit(
            f"ERROR: nothing left to benchmark — 'llm_defender' was the only requested "
            f"defense and {why} (looked in {path}).\n"
            f"       Request algorithmic defenses instead, e.g.\n"
            f"       --defenses fltrust,defl,dnc,multikrum\n"
            f"       or pass --defender-adapter <dir> to point at a checkpoint."
        )
    log.warning(
        f"Skipping the 'llm_defender' column: {why} (looked in {path}). "
        f"Continuing with {remaining}. To include it, pass --defender-adapter <dir>, "
        f"or set defense.mode: llm and train a defender."
    )
    return remaining, True


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
    # Frozen benign replay is a BENCHMARK requirement, not a training preference, so it
    # is pinned here rather than read from fl.benign_retrain_each_round (which training
    # now sets to true — see configs/base.yaml). Every defense in the panel must see
    # byte-identical benign updates in a round for "same attack to everyone" to hold,
    # and each Defense owns its own global while the env's stays at the Phase-1 state,
    # so retraining here would re-draw the honest updates against a stale reference for
    # no benefit. This used to be a warning that the comparison "may be skewed".
    if env.benign_retrain:
        log.info("benchmark: pinning frozen benign replay (fl.benign_retrain_each_round "
                 "is true for training, but the panel needs identical benign updates)")
        env.benign_retrain = False
    env.reset(copy.deepcopy(global_weights), client_weights, baseline_accuracy)

    attack_cfg = base_cfg.get("attack", {})
    requested_budget = (
        args.max_poison_clients if args.max_poison_clients is not None
        else int(attack_cfg.get(
            "eval_poison_clients", attack_cfg.get("max_poison_clients", 1)))
    )
    eval_budget = _resolve_eval_budget(env, requested_budget)

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

    # The ATTACKER adapter is what is under evaluation — without it there is nothing
    # to benchmark, so this one really is fatal.
    if not adapter_exists(adapter_paths["attacker"]):
        sys.exit(f"ERROR: no trained attacker adapter at {adapter_paths['attacker']}")

    # The DEFENDER adapter is needed ONLY by the `llm_defender` column, and a missing
    # one is the NORMAL case rather than an error: with `defense.mode: algorithmic`
    # (the shipped default) the defender LLM is disabled and rl/schedule.py
    # deliberately never writes that checkpoint. Aborting the run therefore threw away
    # every algorithmic-defense result the benchmark exists to produce. Drop the column
    # and carry on. Resolved BEFORE the model is built so a skipped column does not
    # also materialise an unused defender LoRA (~115 MB of VRAM at lora_r=16).
    names, skipped_llm_defender = _resolve_llm_defender(names, adapter_paths, base_cfg)

    adapter_names = (("attacker", "defender") if "llm_defender" in names
                     else ("attacker",))
    policy = LLMPolicy(
        base_model=rl_cfg.get("model", "unsloth/Qwen2.5-3B-Instruct"),
        max_seq_len=int(rl_cfg.get("max_seq_len", 8192)),
        lora_r=int(rl_cfg.get("lora_r", 16)),
        lora_alpha=int(rl_cfg.get("lora_alpha", 32)),
        load_in_4bit=bool(rl_cfg.get("load_in_4bit", True)),
        seed=seed,
        adapters=adapter_names,
        attn_implementation=rl_cfg.get("attn_implementation", "eager"),
        use_fast_generate=bool(rl_cfg.get("use_fast_generate", True)),
    )
    policy.load_adapter("attacker", adapter_paths["attacker"])
    if "llm_defender" in names:
        policy.load_adapter("defender", adapter_paths["defender"])

    attacker_agent = AttackerAgent(attacker_cfg)
    # Only meaningful for the llm_defender column; harmless to build either way.
    defender_agent = DefenderAgent(defender_cfg)

    root_loader = None
    root_epochs = 1
    if "fltrust" in names:
        root_loader = _build_root_loader(data_cfg, args.root_size, fl["batch_size"], seed)
        # R_l must match what the attacker TRAINED against, or the FLTrust column
        # measures a different defense than the one the policy learned to beat.
        # --root-epochs wins; otherwise fall back to the config, whose default (null)
        # sizes the root update to an honest client's iteration count. FLTrust rescales
        # every accepted delta to ||g0||, so R_l alone decides how far the global can
        # move per round — see server.algo_defender.resolve_root_epochs.
        from server.algo_defender import DEFAULT_MAX_ROOT_EPOCHS, resolve_root_epochs
        ft_cfg = ((base_cfg.get("defense") or {}).get("fltrust") or {})
        configured = args.root_epochs
        if configured is None:
            configured = ft_cfg.get("root_epochs")
        root_epochs = resolve_root_epochs(
            configured, root_batches=len(root_loader),
            client_iterations=int(fl["local_epochs"]) * len(client_loaders[0]),
            # Same cap as training, or the FLTrust column faces a different (and
            # in the un-capped case malfunctioning) defense than the policy trained on.
            max_epochs=int(ft_cfg.get("max_root_epochs") or DEFAULT_MAX_ROOT_EPOCHS),
        )
        log.info(f"FLTrust root fine-tuning: root_epochs={root_epochs} over "
                 f"{len(root_loader)} batch(es) = ~{root_epochs * len(root_loader)} SGD "
                 f"iterations (honest client: "
                 f"~{int(fl['local_epochs']) * len(client_loaders[0])})")

    # DnC / Multi-Krum assume a known upper bound on #malicious; default it to the
    # eval poison quota (the exact number of clients the attacker must poison), clamped
    # to a benign majority. This is an assumed adversary budget, NOT per-round truth.
    #
    # The clamp stays even when --max-poison-clients exceeds it: both algorithms are
    # only defined for a minority adversary (Multi-Krum keeps m = n - f updates, so
    # f >= n/2 would leave it averaging half the federation or fewer), so feeding them
    # f = 12 of 20 does not make them stronger, it makes them ill-posed. They are
    # deliberately given the strongest well-formed assumption instead, and the
    # honest-majority warning above explains why detection falls off past that point.
    n_cl = int(fl["n_clients"])
    assumed_byz = max(1, min(eval_budget, (n_cl - 1) // 2))
    if eval_budget > assumed_byz:
        log.info(
            f"DnC/Multi-Krum assumed_byzantine capped at {assumed_byz} (of {n_cl}) while "
            f"{eval_budget} clients are actually poisoned — those algorithms are only "
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

    log.info(f"Benchmark: {args.rounds} rounds | defenses={list(defenses)} | "
             f"baseline_acc={baseline_accuracy:.4f} | attack_temp={args.attack_temperature}"
             + (" | llm_defender SKIPPED (no defender adapter)"
                if skipped_llm_defender else ""))
    summaries, _metrics = run_benchmark(
        env, policy, attacker_agent, defenses, test_loader,
        init_global=copy.deepcopy(global_weights), baseline_accuracy=baseline_accuracy,
        n_rounds=args.rounds, attack_temperature=args.attack_temperature,
        max_new_tokens=int(rl_cfg.get("max_new_tokens", 512)), device=device,
        log_every=args.log_every, target_drop=target_drop,
        attack_retries=max(0, args.attack_retries),
    )

    # Rounds the harness could not get a usable attacker action for are skipped, so
    # report the MEASURED count rather than the requested one — the table's header
    # and its per-defense columns then describe the same set of rounds.
    measured_rounds = max((s.get("rounds", 0) for s in summaries.values()),
                          default=args.rounds)
    skipped_rounds = args.rounds - measured_rounds
    if skipped_rounds > 0:
        log.warning(f"{skipped_rounds} of {args.rounds} round(s) had no usable attacker "
                    f"action and are excluded; the report covers {measured_rounds} round(s)")

    out_dir = args.out or None
    print("\n" + report.render([summaries[n] for n in defenses], measured_rounds,
                                baseline_accuracy, out_dir=out_dir, goal=goal,
                                n_poisoners=eval_budget))

    # Persist per-round history + draw the per-round graphs.
    if out_dir:
        import json as _json
        history = {name: m.history for name, m in _metrics.items()}
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            _json.dump({"baseline_accuracy": baseline_accuracy,
                        "defenses": list(defenses),
                        # Recorded so a saved result is self-describing: a missing
                        # llm_defender column is a deliberate skip, not a lost run,
                        # and a measured count below --rounds is skipped attack
                        # rounds rather than a truncated run.
                        "llm_defender_skipped": skipped_llm_defender,
                        "requested_rounds": args.rounds,
                        "measured_rounds": measured_rounds,
                        "unusable_attack_rounds": skipped_rounds,
                        "history": history}, f, indent=2)
        log.info(f"[saved] {os.path.join(out_dir, 'history.json')}")
        if not args.no_plot:
            from benchmark.plot import plot_history
            png = plot_history(history, baseline_accuracy, os.path.join(out_dir, "benchmark.png"))
            if png:
                log.info(f"[saved] {png}")


if __name__ == "__main__":
    main()
