"""Zero-Touch Federated Learning — Main Entry Point.

Two phases:
  Phase 1 (training_rounds): honest FedAvg training, then checkpoint.
  Phase 2 (simulation_rounds): LLM-direct adversarial arms race with RL.

In Phase 2 a random subset of clients is poisoned each round. An attacker LLM
emits an attack plan (primitive weight operators) applied to the benign weights;
a defender LLM classifies each client benign/
malicious from per-client per-layer statistics. Both get a verifiable per-round
reward and are trained with GRPO (separate LoRA adapters over one frozen
Llama-3.2-3B-Instruct base).

Modes:
  python main.py --env linux                 # full GRPO training (needs a GPU)
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

from data.mnist_loader import get_data_loaders
from clients.benign_client import BenignClient
from server.fed_server import FedServer
from server.aggregation import FedAvgAggregator
from agents.attacker_agent import AttackerAgent
from agents.defender_agent import DefenderAgent
from agents.llm_client import create_llm_client
from storage.checkpoint import (
    save_state, load_state, state_exists, save_progress, load_progress, adapter_exists,
)
from core.types import RoundLog, DetectionVerdict
from core.debug import dbg
from metrics import MetricsTracker
from rl.env import FLArmsRaceEnv

# ---------------------------------------------------------------------------
# Logging / config
# ---------------------------------------------------------------------------

def setup_logging(debug: bool = False):
    os.makedirs("logs/round_data", exist_ok=True)
    file_handler = logging.FileHandler("logs/system.log", mode="a", encoding="utf-8")
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
    AttentionMaskConverter deprecation FutureWarnings."""
    import warnings
    warnings.filterwarnings("ignore", message=r".*max_new_tokens.*max_length.*")
    warnings.filterwarnings("ignore", message=r".*AttentionMaskConverter.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers.*")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_round_log(log: RoundLog):
    path = f"logs/round_data/round_{log.round_num:03d}.json"
    with open(path, "w") as f:
        import json
        json.dump(asdict(log), f, indent=2, default=str)
    logger.info(f"Round data saved to {path}")


# ---------------------------------------------------------------------------
# Phase 1: honest FedAvg training
# ---------------------------------------------------------------------------

def run_training_phase(config: dict, client_loaders, test_loader):
    """Train all clients honestly for ``training_rounds`` rounds; checkpoint."""
    fl = config["fl"]
    logger.info("=" * 60)
    logger.info("PHASE 1: Honest Federated Learning Training")
    logger.info("=" * 60)

    server = FedServer(device=fl["device"])
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
    save_state(server.get_global_weights(), client_weights, baseline_accuracy)
    logger.info("Phase 1 state saved to checkpoints/")
    return server.get_global_weights(), client_weights, baseline_accuracy


# ---------------------------------------------------------------------------
# Phase 2: LLM-direct adversarial arms race
# ---------------------------------------------------------------------------

def run_phase2(
    global_weights, client_weights, baseline_accuracy,
    client_loaders, test_loader, config, attacker_config, defender_config,
    mode: str, n_rounds: int, llm_backend: str,
):
    fl = config["fl"]
    logger.info("=" * 60)
    attack_cfg = config.get("attack", {})
    logger.info(f"PHASE 2: LLM-direct arms race  (mode={mode})")
    logger.info(f"  simulation_rounds={n_rounds}, n_compromisable={fl.get('n_compromisable')}, "
                f"max_poison_clients={attack_cfg.get('max_poison_clients')}, "
                f"sample_budget={attack_cfg.get('sample_budget_in_training')}")
    logger.info(f"  baseline_accuracy={baseline_accuracy:.4f}")
    logger.info("=" * 60)

    rng = random.Random(int(fl.get("poison_seed", 0)))
    env = FLArmsRaceEnv(config, client_loaders, test_loader, rng)
    env.reset(global_weights, client_weights, baseline_accuracy)

    metrics_tracker = MetricsTracker(baseline_accuracy=baseline_accuracy, output_dir="logs/metrics")
    attacker_agent = AttackerAgent(attacker_config)
    defender_agent = DefenderAgent(defender_config)

    if mode == "baseline":
        from rl.baseline import run_baseline
        run_baseline(env, n_rounds, metrics_tracker, _save_round_log)

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
                      metrics_tracker, _save_round_log,
                      temperature=float(llm_cfg.get("temperature", 0.7)))

    else:  # full GRPO training
        from rl.policy import LLMPolicy
        from rl.schedule import train
        rl_cfg = config.get("rl", {})
        adapter_paths = rl_cfg.get("adapter_paths", {
            "attacker": "checkpoints/attacker_adapter",
            "defender": "checkpoints/defender_adapter",
        })
        policy = LLMPolicy(
            base_model=rl_cfg.get("model", "unsloth/Llama-3.2-3B-Instruct"),
            max_seq_len=int(rl_cfg.get("max_seq_len", 8192)),
            lora_r=int(rl_cfg.get("lora_r", 16)),
            lora_alpha=int(rl_cfg.get("lora_alpha", 32)),
            load_in_4bit=bool(rl_cfg.get("load_in_4bit", True)),
            seed=int(fl.get("poison_seed", 0)),
            attn_implementation=rl_cfg.get("attn_implementation", "eager"),
            use_fast_generate=bool(rl_cfg.get("use_fast_generate", True)),
        )
        # Resume adapters if present.
        for name, path in adapter_paths.items():
            if adapter_exists(path):
                policy.load_adapter(name, path)
        start_round = load_progress()
        if start_round:
            logger.info(f"Resuming Phase-2 training from round {start_round}")

        def progress_cb(done):
            save_progress(done)

        train(env, policy, attacker_agent, defender_agent, config,
              metrics_tracker, _save_round_log, rng,
              progress_cb=progress_cb, start_round=start_round)

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
    parser.add_argument("--env", choices=["linux", "windows"], default="linux",
                        help="'linux' uses Ollama (llama3.2), 'windows' uses OpenAI (default: linux)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the Phase-2 loop with a frozen LLM (no training, no GPU needed)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run the best-of-N reward-harness sanity baseline (no LLM)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override simulation_rounds (handy for quick smoke runs)")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose Phase-2 debug run: print every attacker/defender LLM "
                             "prompt+output, poisoning step, FL update and reward to the console "
                             "AND to logs/debug.json. Library noise is silenced. If --rounds is "
                             f"not given, Phase 2 is capped at {DEBUG_DEFAULT_ROUNDS} rounds.")
    args = parser.parse_args()

    setup_logging(debug=args.debug)
    quiet_noisy_warnings()
    logger.info("Starting Zero-Touch Federated Learning System")

    base_config = load_config("configs/base.yaml")
    attacker_config = load_config("configs/attacker_agent.yaml")
    defender_config = load_config("configs/defender_agent.yaml")

    # LLM backend + shared Ollama defaults (used by --dry-run / OpenAI paths).
    llm_backend = "ollama" if args.env == "linux" else "openai"
    llm_defaults = base_config.get("llm", {})
    for agent_cfg in (attacker_config, defender_config):
        agent_cfg.setdefault("llm", {})
        agent_cfg["llm"]["backend"] = llm_backend
        agent_cfg["llm"].setdefault("ollama_base_url", llm_defaults.get("ollama_base_url", "http://localhost:11434"))
        agent_cfg["llm"].setdefault("ollama_model", llm_defaults.get("ollama_model", "llama3.2:3b"))

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
        n_clients=fl["n_clients"], batch_size=fl["batch_size"],
        data_dir=data_cfg.get("data_dir", "./data/mnist_raw"), iid=data_cfg.get("iid", True),
        bias_q=float(data_cfg.get("noniid_bias", 0.5)), seed=seed,
    )

    state = load_state() if (state_exists() and not args.fresh) else None
    if state is not None and len(state[1]) != fl["n_clients"]:
        logger.warning(
            f"Checkpoint has {len(state[1])} client(s) but config n_clients={fl['n_clients']} "
            f"— ignoring the stale checkpoint and re-running Phase 1."
        )
        state = None
    if state is not None:
        logger.info("Checkpoint found — skipping Phase 1, loading saved state")
        global_weights, client_weights, baseline_accuracy = state
    else:
        logger.info("No (usable) checkpoint or --fresh — running Phase 1")
        global_weights, client_weights, baseline_accuracy = run_training_phase(
            base_config, client_loaders, test_loader
        )

    mode = "baseline" if args.baseline else ("dry-run" if args.dry_run else "train")
    n_rounds = args.rounds if args.rounds is not None else int(fl["simulation_rounds"])

    if args.debug:
        if args.rounds is None:
            n_rounds = DEBUG_DEFAULT_ROUNDS
            logger.info(f"[debug] no --rounds given -> capping Phase 2 at {n_rounds} rounds")
        dbg.enable(
            output_dir="logs", filename="debug.json", mode=mode,
            config_summary={
                "model": base_config.get("rl", {}).get("model"),
                "n_clients": fl.get("n_clients"),
                "n_compromisable": fl.get("n_compromisable"),
                "max_poison_clients": base_config.get("attack", {}).get("max_poison_clients"),
                "sample_budget": base_config.get("attack", {}).get("sample_budget_in_training"),
                "noniid_bias": base_config.get("data", {}).get("noniid_bias"),
                "G": base_config.get("rl", {}).get("G"),
                "switch_mode": base_config.get("rl", {}).get("switch_mode"),
                "first_learner": base_config.get("rl", {}).get("first_learner"),
                "baseline_accuracy": round(float(baseline_accuracy), 4),
                "n_rounds": n_rounds,
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
        )
    finally:
        dbg.close()


if __name__ == "__main__":
    main()
