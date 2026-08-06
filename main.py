"""Zero-Touch Federated Learning — Main Entry Point.

Two phases:
  Phase 1 (training_rounds): honest FedAvg training, then checkpoint.
  Phase 2 (simulation_rounds): LLM-direct adversarial arms race with RL.

In Phase 2 an exact-budget subset of clients is poisoned each round. An attacker LLM
emits an attack plan (primitive weight operators) applied to the benign weights,
and the server defends.

The DEFENDER LLM IS CURRENTLY DISABLED (``defense.mode: algorithmic`` in
configs/base.yaml). The server instead defends with the published algorithms —
FLTrust, DeFL, DnC and Multi-Krum — using ONE of them, drawn at random, per
round (see ``server/algo_defender.py``). Only the attacker is trained with GRPO.
Setting ``defense.mode: llm`` restores the original two-sided arms race, where a
defender LLM classifies each client benign/malicious from per-client per-layer
statistics and both sides get a verifiable per-round reward and train with GRPO
(separate LoRA adapters over one frozen Qwen2.5-3B-Instruct base).

Datasets:
  ``--dataset {mnist,cifar10}`` selects what the federation trains on. The FL
  model architecture, the data directory, the Phase-1 checkpoint and the logs are
  all per-dataset (``checkpoints/<dataset>/``, ``logs/<dataset>/``), so the two
  runs never overwrite each other. The **LLM is not**: the attacker LoRA adapter
  lives at ``checkpoints/attacker_adapter`` and is shared, so every run — MNIST or
  CIFAR-10 — continues fine-tuning the same policy from its last checkpoint.

Modes:
  python main.py --env linux                 # full GRPO training (needs a GPU)
  python main.py --env linux --dataset cifar10   # ...on CIFAR-10, same LLM
  python main.py --env linux --dry-run       # frozen-LLM round loop, no training
  python main.py --baseline                  # best-of-N reward-harness sanity (no LLM)
  python main.py --fresh                      # force fresh Phase-1 training
  python main.py --rounds 8                   # override simulation_rounds (quick runs)
"""

import os
# Reduce CUDA fragmentation OOMs — must be set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import logging
import random
import sys
from dataclasses import asdict

import yaml

from data.datasets import DATASET_NAMES, describe
from data.loaders import get_data_loaders, build_root_loader
from clients.benign_client import BenignClient
from server.fed_server import FedServer
from server.aggregation import FedAvgAggregator
from server.algo_defender import build_algorithmic_defender
from agents.attacker_agent import AttackerAgent
from agents.defender_agent import DefenderAgent
from agents.llm_client import create_llm_client
from storage.checkpoint import (
    save_state, load_state, state_exists, save_progress, load_progress, adapter_exists,
    save_fl_state, load_fl_state,
)
from core.run_config import apply_dataset, describe_run, run_paths
from core.types import RoundLog, DetectionVerdict
from core.debug import dbg
from metrics import MetricsTracker
from rl.env import FLArmsRaceEnv

# ---------------------------------------------------------------------------
# Logging / config
# ---------------------------------------------------------------------------

def setup_logging(paths: dict, debug: bool = False):
    """Configure logging into this run's per-dataset log directory.

    ``paths`` comes from ``core.run_config.run_paths(dataset)``, so MNIST and
    CIFAR-10 runs write to ``logs/mnist/`` and ``logs/cifar10/`` and never
    interleave rounds (round numbering restarts per dataset).
    """
    os.makedirs(paths["round_data_dir"], exist_ok=True)
    file_handler = logging.FileHandler(paths["system_log"], mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[file_handler, stream_handler],
    )
    if debug:
        # In debug mode the rich per-round picture comes from the structured
        # ``core.debug`` logger; here we just make sure third-party libraries stay
        # quiet so the console shows only OUR federated-learning system logs.
        for noisy in (
            "transformers", "unsloth", "peft", "bitsandbytes", "accelerate",
            "torch", "datasets", "huggingface_hub", "filelock", "urllib3",
            "httpx", "httpcore", "asyncio", "PIL",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("main")

# In --debug, if the user doesn't pass --rounds we cap Phase 2 to a short,
# fully-logged run (each round is identical in logic — this only stops early).
DEBUG_DEFAULT_ROUNDS = 3


def quiet_noisy_warnings():
    """Silence the high-frequency, harmless library chatter during training:
    the per-generation 'max_new_tokens vs max_length' notice and Transformers'
    AttentionMaskConverter deprecation FutureWarnings.

    IMPORTANT: this must NOT import ``transformers``. It runs at program startup,
    and importing ``transformers`` here would pull it in BEFORE Unsloth (which is
    imported lazily in ``rl/policy.py`` on the training path) — tripping Unsloth's
    "import unsloth before transformers" warning and potentially skipping some of
    its optimizations. Transformers' own log verbosity is lowered in
    ``LLMPolicy.__init__`` instead, right after Unsloth is imported. The
    ``warnings.filterwarnings`` calls below are pure ``warnings``-module filters
    and do not import ``transformers``.
    """
    import warnings
    warnings.filterwarnings("ignore", message=r".*max_new_tokens.*max_length.*")
    warnings.filterwarnings("ignore", message=r".*AttentionMaskConverter.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers.*")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Per-round logs are APPENDED to one JSONL file rather than written as
# ``round_NNN.json`` per round. At the shipped ``fl.simulation_rounds`` a
# file-per-round sink produces millions of tiny files (and the same again under
# logs/<dataset>/metrics/), which exhausts inodes and makes the directory
# unusable. One append-only stream is O(1) per round and stays greppable.
# ``monitor.py`` and ``visualize_rounds.py`` read this file, and still read legacy
# ``round_NNN.json`` files so older runs keep working.


def make_round_log_saver(path: str):
    """Return a ``RoundLog -> None`` sink appending to ``path``.

    A factory rather than a module constant because the path is per dataset
    (``logs/<dataset>/round_data/rounds.jsonl``) and only known once the run's
    dataset is resolved.
    """
    import json

    def save_round_log(log: RoundLog):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log), default=str) + "\n")
        logger.info(f"Round {log.round_num} appended to {path}")

    return save_round_log


# ---------------------------------------------------------------------------
# Phase 1: honest FedAvg training
# ---------------------------------------------------------------------------

def run_training_phase(config: dict, client_loaders, test_loader, dataset: str):
    """Train all clients honestly for ``training_rounds`` rounds; checkpoint.

    The resulting state is saved under ``checkpoints/<dataset>/`` — it is one
    dataset's global model plus the per-client weights that match it, and a
    CIFAR-10 conv ``state_dict`` cannot be loaded into the MNIST MLP.
    """
    fl = config["fl"]
    logger.info("=" * 60)
    logger.info(f"PHASE 1: Honest Federated Learning Training [{dataset}]")
    logger.info("=" * 60)

    server = FedServer(device=fl["device"], dataset=dataset)
    clients = [
        BenignClient(
            client_id=i,
            data_loader=client_loaders[i],
            lr=fl["lr"],
            local_epochs=fl["local_epochs"],
            device=fl["device"],
        )
        for i in range(fl["n_clients"])
    ]
    aggregator = FedAvgAggregator()

    updates = []
    for round_num in range(1, fl["training_rounds"] + 1):
        logger.info(f"--- Training Round {round_num}/{fl['training_rounds']} ---")
        updates = [client.train(server.model) for client in clients]
        # No detection in Phase 1 — everyone is honest.
        clean = [DetectionVerdict(u.client_id, False, 0.0, "phase1") for u in updates]
        new_weights = aggregator.aggregate(updates, clean)
        server.set_global_weights(new_weights)
        accuracy = server.evaluate(test_loader)
        logger.info(f"  Round {round_num} accuracy: {accuracy:.4f}")

    baseline_accuracy = server.evaluate(test_loader)
    logger.info(f"Baseline accuracy after Phase 1: {baseline_accuracy:.4f}")

    client_weights = [u.weights for u in updates]
    save_state(server.get_global_weights(), client_weights, baseline_accuracy,
               dataset=dataset)
    logger.info(f"Phase 1 state saved to checkpoints/{dataset}/")
    return server.get_global_weights(), client_weights, baseline_accuracy


# ---------------------------------------------------------------------------
# Phase 2: LLM-direct adversarial arms race
# ---------------------------------------------------------------------------

def run_phase2(
    global_weights, client_weights, baseline_accuracy,
    client_loaders, test_loader, config, attacker_config, defender_config,
    mode: str, n_rounds: int, llm_backend: str,
    dataset: str, paths: dict, save_round_log,
    max_new_rounds: int | None = None,
):
    fl = config["fl"]
    logger.info("=" * 60)
    attack_cfg = config.get("attack", {})
    logger.info(f"PHASE 2: LLM-direct arms race  (mode={mode}, dataset={dataset})")
    logger.info(f"  simulation_rounds={n_rounds}, n_compromisable={fl.get('n_compromisable')}, "
                f"max_poison_clients={attack_cfg.get('max_poison_clients')}, "
                f"sample_budget={attack_cfg.get('sample_budget_in_training')}")
    logger.info(f"  baseline_accuracy={baseline_accuracy:.4f}")
    logger.info("=" * 60)

    seed = int(fl.get("poison_seed", 0))
    rng = random.Random(seed)
    # Server-side defense. With ``defense.mode: algorithmic`` (the default) the
    # defender LLM is disabled and one published algorithm — FLTrust / DeFL / DnC /
    # Multi-Krum — defends each round, drawn at random; only the attacker trains.
    # ``defense.mode: llm`` restores the trainable defender LLM.
    # An honest client's per-round SGD iteration count. FLTrust's root fine-tuning is
    # sized to match it (defense.fltrust.root_epochs: null), because FLTrust rescales
    # every accepted update to ||g0|| — so the server's reference update, not the
    # clients', sets how far the global model can move per round. See
    # server.algo_defender.resolve_root_epochs.
    client_iterations = (
        int(fl["local_epochs"]) * len(client_loaders[0]) if client_loaders else None
    )
    defense = build_algorithmic_defender(
        config, seed=seed, client_iterations=client_iterations,
        root_loader_factory=lambda: build_root_loader(
            dataset,
            root_size=int(((config.get("defense") or {}).get("fltrust") or {})
                          .get("root_size", 100)),
            batch_size=int(fl["batch_size"]),
            data_dir=config.get("data", {}).get("data_dir"),
            seed=seed,
        ),
    )
    env = FLArmsRaceEnv(config, client_loaders, test_loader, rng, defense=defense)
    env.reset(global_weights, client_weights, baseline_accuracy)

    metrics_tracker = MetricsTracker(baseline_accuracy=baseline_accuracy,
                                     output_dir=paths["metrics_dir"])
    attacker_agent = AttackerAgent(attacker_config)
    defender_agent = DefenderAgent(defender_config)

    if mode == "baseline":
        from rl.baseline import run_baseline
        run_baseline(env, n_rounds, metrics_tracker, save_round_log)

    elif mode == "dry-run":
        from rl.inference import InferenceGenerator, run_inference
        llm_cfg = attacker_config.get("llm", {})
        model = llm_cfg.get("ollama_model" if llm_backend == "ollama" else "model")
        backend = create_llm_client(
            backend=llm_backend, model=model,
            temperature=float(llm_cfg.get("temperature", 0.7)),
            ollama_base_url=llm_cfg.get("ollama_base_url", "http://localhost:11434"),
        )
        gen = InferenceGenerator(backend, max_new_tokens=int(config.get("rl", {}).get("max_new_tokens", 2048)))
        run_inference(env, attacker_agent, defender_agent, gen, n_rounds,
                      metrics_tracker, save_round_log,
                      temperature=float(llm_cfg.get("temperature", 0.7)))

    else:  # full GRPO training
        from rl.policy import LLMPolicy
        from rl.schedule import train
        rl_cfg = config.get("rl", {})
        # NOT dataset-scoped, deliberately: the same attacker adapter keeps
        # fine-tuning across datasets, so an mnist run and a cifar10 run resume
        # the identical policy checkpoint. Only the FL state below is per dataset.
        adapter_paths = rl_cfg.get("adapter_paths", {
            "attacker": "checkpoints/attacker_adapter",
            "defender": "checkpoints/defender_adapter",
        })
        # With the defender LLM disabled there is no defender policy to train, so
        # we don't even materialise its LoRA adapter (one fewer ~115 MB copy of
        # LoRA tensors on the GPU). Its checkpoint on disk is left untouched, so
        # flipping ``defense.mode`` back to ``llm`` resumes it unchanged.
        adapter_names = ("attacker",) if defense is not None else ("attacker", "defender")
        policy = LLMPolicy(
            base_model=rl_cfg.get("model", "unsloth/Qwen2.5-3B-Instruct"),
            max_seq_len=int(rl_cfg.get("max_seq_len", 8192)),
            lora_r=int(rl_cfg.get("lora_r", 16)),
            lora_alpha=int(rl_cfg.get("lora_alpha", 32)),
            load_in_4bit=bool(rl_cfg.get("load_in_4bit", True)),
            seed=int(fl.get("poison_seed", 0)),
            adapters=adapter_names,
            attn_implementation=rl_cfg.get("attn_implementation", "eager"),
            use_fast_generate=bool(rl_cfg.get("use_fast_generate", True)),
        )
        # Resume adapters if present. This is the continual-fine-tuning hinge:
        # the adapter path does not depend on `dataset`, so switching datasets
        # keeps training the SAME policy rather than restarting from the base model.
        for name, path in adapter_paths.items():
            if name in adapter_names and adapter_exists(path):
                logger.info(f"Continuing {name} LoRA from {path} "
                            f"(shared across datasets; this run: {dataset})")
                policy.load_adapter(name, path)
        progress = load_progress(dataset=dataset)
        start_round = progress["rounds_done"]
        if start_round:
            logger.info(f"Resuming Phase-2 training from round {start_round} [{dataset}]")
            # Restore the LIVE shared FL state (the evolving global model + per-client
            # weights) so the arms race continues from where it stopped instead of
            # rewinding to the Phase-1 baseline. env.reset() above already loaded the
            # Phase-1 baseline; this overrides it with the saved Phase-2 state.
            saved_fl = load_fl_state(dataset=dataset)
            if saved_fl is not None:
                env.restore_fl_state(saved_fl)
            else:
                logger.warning(
                    "No saved Phase-2 FL state found — the shared model resumes from the "
                    "Phase-1 baseline (older checkpoint predating fl_state.pt)."
                )

        def progress_cb(done, round_index=None, controller=None):
            save_progress(done, round_index=round_index, controller=controller,
                          dataset=dataset)

        def fl_state_cb(fl_state):
            save_fl_state(fl_state, dataset=dataset)

        train(env, policy, attacker_agent, defender_agent, config,
              metrics_tracker, save_round_log, rng,
              progress_cb=progress_cb, fl_state_cb=fl_state_cb,
              start_round=start_round, resume=progress,
              total_rounds=n_rounds, max_new_rounds=max_new_rounds)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 COMPLETE")
    logger.info(f"Final accuracy: {env.current_accuracy:.4f} (baseline: {baseline_accuracy:.4f})")
    logger.info("=" * 60)
    metrics_tracker.save_summary()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-Touch Federated Learning")
    parser.add_argument("--fresh", action="store_true", help="Force fresh Phase 1 training")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Path to the base config (default: configs/base.yaml)")
    parser.add_argument("--env", choices=["linux", "windows"], default="linux",
                        help="'linux' uses Ollama (qwen2.5), 'windows' uses OpenAI (default: linux)")
    parser.add_argument("--dataset", default=None, metavar="NAME",
                        help=f"which dataset the federation trains on: "
                             f"{', '.join(DATASET_NAMES)} (default: data.dataset in "
                             f"the config). FL state and logs are per dataset "
                             f"(checkpoints/<dataset>/, logs/<dataset>/); the LLM "
                             f"adapter is SHARED, so the same policy keeps "
                             f"fine-tuning from its last checkpoint either way.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the Phase-2 loop with a frozen LLM (no training, no GPU needed)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run the best-of-N reward-harness sanity baseline (no LLM)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override simulation_rounds (handy for quick smoke runs)")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose Phase-2 debug run: print every attacker/defender LLM "
                             "prompt+output, poisoning step, FL update and reward to the console "
                             "AND to logs/<dataset>/debug.json. Library noise is silenced. "
                             "If --rounds is not given, Phase 2 is capped at "
                             f"{DEBUG_DEFAULT_ROUNDS} rounds.")
    args = parser.parse_args()

    # Resolve the dataset (and its config overrides) BEFORE logging is configured:
    # the log directory itself is per-dataset. An unknown --dataset raises here,
    # with the valid names, rather than after Phase 1 has started.
    base_config = load_config(args.config)
    base_config, dataset = apply_dataset(base_config, args.dataset)
    paths = run_paths(dataset)

    setup_logging(paths, debug=args.debug)
    quiet_noisy_warnings()
    logger.info("Starting Zero-Touch Federated Learning System")
    logger.info(f"Dataset: {describe(dataset)}")
    logger.info(f"Resolved run config: {describe_run(base_config, dataset)}")
    logger.info(f"Artifacts: checkpoints={paths['checkpoint_dir']}/ logs={paths['log_dir']}/ "
                f"| LLM adapters are SHARED across datasets (checkpoints/*_adapter)")
    save_round_log = make_round_log_saver(paths["round_log"])

    attacker_config = load_config("configs/attacker_agent.yaml")
    defender_config = load_config("configs/defender_agent.yaml")

    # LLM backend + shared Ollama defaults (used by --dry-run / OpenAI paths).
    llm_backend = "ollama" if args.env == "linux" else "openai"
    llm_defaults = base_config.get("llm", {})
    for agent_cfg in (attacker_config, defender_config):
        agent_cfg.setdefault("llm", {})
        agent_cfg["llm"]["backend"] = llm_backend
        agent_cfg["llm"].setdefault("ollama_base_url", llm_defaults.get("ollama_base_url", "http://localhost:11434"))
        agent_cfg["llm"].setdefault("ollama_model", llm_defaults.get("ollama_model", "qwen2.5:3b"))

    # Single source of truth for the attack goal: base config -> attacker agent.
    goal = base_config.get("attack", {}).get("goal")
    if goal:
        attacker_config["attack_goal"] = goal

    # Reproducibility.
    seed = int(base_config["fl"].get("poison_seed", 0))
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    fl = base_config["fl"]
    data_cfg = base_config["data"]
    client_loaders, test_loader = get_data_loaders(
        dataset, n_clients=fl["n_clients"], batch_size=fl["batch_size"],
        data_dir=data_cfg.get("data_dir"), iid=data_cfg.get("iid", True),
        bias_q=float(data_cfg.get("noniid_bias", 0.5)), seed=seed,
        n_classes=data_cfg.get("n_classes"),
    )

    # Phase-1 state is looked up in THIS dataset's checkpoint dir, so switching
    # datasets never loads a state_dict the model cannot accept.
    state = (load_state(dataset=dataset)
             if (state_exists(dataset=dataset) and not args.fresh) else None)
    if state is not None and len(state[1]) != fl["n_clients"]:
        logger.warning(
            f"Checkpoint has {len(state[1])} client(s) but config n_clients={fl['n_clients']} "
            f"— ignoring the stale checkpoint and re-running Phase 1."
        )
        state = None
    if state is not None:
        logger.info(f"Checkpoint found for '{dataset}' — skipping Phase 1, loading saved state")
        global_weights, client_weights, baseline_accuracy = state
    else:
        logger.info(f"No (usable) '{dataset}' checkpoint or --fresh — running Phase 1")
        global_weights, client_weights, baseline_accuracy = run_training_phase(
            base_config, client_loaders, test_loader, dataset
        )

    mode = "baseline" if args.baseline else ("dry-run" if args.dry_run else "train")
    n_rounds = args.rounds if args.rounds is not None else int(fl["simulation_rounds"])

    # ``--debug`` without ``--rounds`` caps how many rounds THIS RUN adds rather
    # than the absolute budget: the training loop resumes from `rounds_done`, so an
    # absolute cap of 3 would make a resumed debug run exit without executing
    # anything. baseline/dry-run always start from zero, so there the two coincide.
    max_new_rounds = None
    if args.debug and args.rounds is None:
        max_new_rounds = DEBUG_DEFAULT_ROUNDS
        if mode != "train":
            n_rounds = min(n_rounds, DEBUG_DEFAULT_ROUNDS)
        logger.info(f"[debug] no --rounds given -> capping this run at "
                    f"{DEBUG_DEFAULT_ROUNDS} Phase-2 round(s)")
    if args.debug:
        dbg.enable(
            output_dir=paths["debug_dir"], filename="debug.json", mode=mode,
            config_summary={
                "dataset": dataset,
                "model": base_config.get("rl", {}).get("model"),
                "defense_mode": (base_config.get("defense", {}) or {}).get("mode", "algorithmic"),
                "defense_algorithms": (base_config.get("defense", {}) or {}).get("algorithms"),
                "defense_selection": (base_config.get("defense", {}) or {}).get("selection", "random"),
                "n_clients": fl.get("n_clients"),
                "n_compromisable": fl.get("n_compromisable"),
                "max_poison_clients": base_config.get("attack", {}).get("max_poison_clients"),
                "sample_budget": base_config.get("attack", {}).get("sample_budget_in_training"),
                "noniid_bias": base_config.get("data", {}).get("noniid_bias"),
                "G": base_config.get("rl", {}).get("G"),
                "switch_mode": base_config.get("rl", {}).get("switch_mode"),
                "first_learner": base_config.get("rl", {}).get("first_learner"),
                "success_streak": base_config.get("rl", {}).get("success_streak"),
                "fl_interlude_between_phases": base_config.get("rl", {}).get("fl_interlude_between_phases"),
                "baseline_accuracy": round(float(baseline_accuracy), 4),
                "n_rounds": n_rounds,
                "max_new_rounds": max_new_rounds,
            },
        )

    try:
        run_phase2(
            global_weights=copy.deepcopy(global_weights),
            client_weights=client_weights,
            baseline_accuracy=baseline_accuracy,
            client_loaders=client_loaders,
            test_loader=test_loader,
            config=base_config,
            attacker_config=attacker_config,
            defender_config=defender_config,
            mode=mode,
            n_rounds=n_rounds,
            llm_backend=llm_backend,
            dataset=dataset,
            paths=paths,
            save_round_log=save_round_log,
            max_new_rounds=max_new_rounds,
        )
    finally:
        dbg.close()


if __name__ == "__main__":
    main()
