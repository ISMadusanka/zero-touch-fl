"""Zero-Touch Federated Learning — Main Entry Point.

Two phases:
  Phase 1 (training_rounds): honest FedAvg training, then checkpoint.
  Phase 2 (simulation_rounds): defender-learning rounds SIMULATED on the frozen
    Phase-1 global.

THE ATTACK IS LABEL FLIPPING, and it is not learned. A fixed, configurable set of
insider clients (``attack.poison_client_ids``, default ``[0]``) trains each round
on its own data with some fraction of the labels flipped symmetrically
(``y -> 9-y``). That fraction follows a detection-adaptive ladder:

  * start at 100% of the client's round data,
  * every round the defense CATCHES it, back off one step (10% by default),
  * every round the defense MISSES it, hold the level and send it again,
  * once it is caught at the floor (50%), RESET to 100% and descend again.

So the attack strength is a saw-tooth driven by the defender's own competence,
and it keeps re-testing the whole range instead of parking at one setting. See
``agents/label_flip_attacker.py``.

Phase 2 does not run a continuing federation. Each round is an independent episode
branching off the SAME Phase-1 final model:

  1. that frozen global is sent to every client;
  2. every client trains on NEW local data (a fresh slice of its own shard);
  3. the poisoned clients re-train that same data with the ladder's share of the
     labels flipped;
  4. poisoned + honest updates go to the server, which defends and aggregates;
  5. the aggregate is evaluated on the test set, the defender is rewarded on how
     well its verdicts matched ground truth, and GRPO updates it;
  6. those verdicts feed the ladder, and the next round starts from the anchor.

Set ``fl.freeze_global_in_phase2: false`` to restore the continuing federation,
where each committed aggregate becomes the next round's global.

The DEFENDER LLM is the only learner (``defense.mode: llm``). It classifies each
client benign/malicious from per-client per-layer weight statistics and trains with
GRPO over a LoRA adapter on a frozen Qwen2.5-3B-Instruct base. Setting
``defense.mode: algorithmic`` swaps it for the published algorithms — FLTrust,
DeFL, DnC, Multi-Krum — which is useful for ``--dry-run`` / ``--baseline`` and the
benchmark, but leaves nothing to train.

Modes:
  python main.py --env linux                 # full GRPO training (needs a GPU)
  python main.py --env linux --dry-run       # frozen-LLM round loop, no training
  python main.py --baseline                  # no-LLM round loop + fixed heuristic defense
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

from data.mnist_loader import get_data_loaders, build_root_loader
from data.round_sampler import build_round_data_sampler
from clients.benign_client import BenignClient
from server.fed_server import FedServer
from server.aggregation import FedAvgAggregator
from server.algo_defender import build_algorithmic_defender
from agents.defender_agent import DefenderAgent
from agents.label_flip_attacker import build_attacker
from agents.llm_client import create_llm_client
from storage.checkpoint import (
    save_state, load_state, state_exists, save_progress, load_progress, adapter_exists,
    save_fl_state, load_fl_state,
)
from core.types import RoundLog, DetectionVerdict
from core.debug import dbg
from metrics import MetricsTracker
from rl.curriculum import build_training_curriculum
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
    AttentionMaskConverter deprecation FutureWarnings.

    IMPORTANT: this must NOT import ``transformers``. It runs at program startup,
    and importing ``transformers`` here would pull it in BEFORE Unsloth (which is
    imported lazily in ``rl/policy.py`` on the training path) — tripping Unsloth's
    "import unsloth before transformers" warning and potentially skipping some of
    its optimizations. Transformers' own log verbosity is lowered in
    ``LLMPolicy.__init__`` instead, right after Unsloth is imported.
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
# logs/metrics/), which exhausts inodes and makes the directory unusable. One
# append-only stream is O(1) per round and stays greppable. ``monitor.py`` and
# ``visualize_rounds.py`` read this file, and still read legacy
# ``round_NNN.json`` files so older runs keep working.
ROUND_LOG_PATH = "logs/round_data/rounds.jsonl"


def _save_round_log(log: RoundLog):
    import json
    os.makedirs(os.path.dirname(ROUND_LOG_PATH), exist_ok=True)
    with open(ROUND_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(log), default=str) + "\n")
    logger.info(f"Round {log.round_num} appended to {ROUND_LOG_PATH}")


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
        # No detection in Phase 1 — everyone is honest, no labels are flipped.
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
# Phase 2: label-flip attack vs the learning defender
# ---------------------------------------------------------------------------

def run_phase2(
    global_weights, client_weights, baseline_accuracy,
    client_loaders, test_loader, config, defender_config,
    mode: str, n_rounds: int, llm_backend: str,
    max_new_rounds: int | None = None,
):
    fl = config["fl"]
    attack_cfg = config.get("attack", {}) or {}
    frozen = bool(fl.get("freeze_global_in_phase2", True))
    seed = int(fl.get("poison_seed", 0))

    logger.info("=" * 60)
    logger.info(f"PHASE 2: defender-learning rounds  (mode={mode}, "
                f"{'SIMULATED on the frozen Phase-1 global' if frozen else 'continuing federation'})")
    logger.info(f"  client_data_refresh={fl.get('client_data_refresh', 'rotate')} "
                f"fraction={fl.get('client_round_fraction', 0.25)}")

    # The attack is built here (not inside the env) so its ladder state can be
    # restored from the progress file before the first round runs.
    attacker = build_attacker(config, n_clients=int(fl["n_clients"]), seed=seed)
    logger.info(f"  simulation_rounds={n_rounds}, baseline_accuracy={baseline_accuracy:.4f}")
    logger.info(f"  attack_goal={(attack_cfg.get('goal') or {}).get('type')} "
                f"target_accuracy_drop="
                f"{(attack_cfg.get('goal') or {}).get('target_accuracy_drop')} "
                f"(the DAMAGE BAR the round is reported against — it does not change "
                f"what the attack does)")
    logger.info("=" * 60)

    rng = random.Random(seed)
    # Step 2 of a simulated round: every client gets NEW local data each round (a
    # fresh slice of its own shard). Without this the frozen global would hand the
    # clients an identical starting point AND identical data, so every round would
    # reproduce the same honest updates and the defender would face one static
    # problem restated N times. ``fl.client_data_refresh: none`` turns it off.
    round_data = build_round_data_sampler(config, client_loaders, seed=seed)
    # Server-side defense. ``defense.mode: llm`` (the default) trains the defender
    # LLM; ``algorithmic`` swaps in one published algorithm per round, which leaves
    # nothing to train and is only useful for --dry-run / --baseline.
    #
    # An honest client's per-round SGD iteration count. FLTrust's root fine-tuning is
    # sized to match it (defense.fltrust.root_epochs: null), because FLTrust rescales
    # every accepted update to ||g0|| — so the server's reference update, not the
    # clients', sets how far the global model can move per round. It must be counted
    # over the data a client actually trains on THIS round, which with the per-round
    # refresh on is a slice of the shard, not the whole thing.
    batches_per_client = (
        round_data.batches_per_round if round_data is not None
        else (len(client_loaders[0]) if client_loaders else 0)
    )
    client_iterations = (
        int(fl["local_epochs"]) * batches_per_client if client_loaders else None
    )
    defense = build_algorithmic_defender(
        config, seed=seed, client_iterations=client_iterations,
        root_loader_factory=lambda: build_root_loader(
            root_size=int(((config.get("defense") or {}).get("fltrust") or {})
                          .get("root_size", 100)),
            batch_size=int(fl["batch_size"]),
            data_dir=config.get("data", {}).get("data_dir", "./data/mnist_raw"),
            seed=seed,
        ),
    )
    # With an algorithmic defense the curriculum sweeps WHICH algorithm defends each
    # block; under the defender LLM there is no algorithm axis and the attack
    # strength is the ladder's, so there is nothing to sweep and it returns None.
    curriculum = build_training_curriculum(
        config, algorithms=(defense.names if defense is not None else None))
    env = FLArmsRaceEnv(config, client_loaders, test_loader, rng,
                        defense=defense, curriculum=curriculum, round_data=round_data,
                        attacker=attacker)
    env.reset(global_weights, client_weights, baseline_accuracy)

    metrics_tracker = MetricsTracker(baseline_accuracy=baseline_accuracy, output_dir="logs/metrics")
    defender_agent = DefenderAgent(defender_config)

    if mode == "baseline":
        from rl.baseline import run_baseline
        run_baseline(env, n_rounds, metrics_tracker, _save_round_log)

    elif mode == "dry-run":
        from rl.inference import InferenceGenerator, run_inference
        llm_cfg = defender_config.get("llm", {})
        model = llm_cfg.get("ollama_model" if llm_backend == "ollama" else "model")
        backend = create_llm_client(
            backend=llm_backend, model=model,
            temperature=float(llm_cfg.get("temperature", 0.7)),
            ollama_base_url=llm_cfg.get("ollama_base_url", "http://localhost:11434"),
        )
        gen = InferenceGenerator(backend, max_new_tokens=int(config.get("rl", {}).get("max_new_tokens", 2048)))
        run_inference(env, defender_agent, gen, n_rounds, metrics_tracker,
                      _save_round_log,
                      temperature=float(llm_cfg.get("temperature", 0.7)))

    else:  # full GRPO training
        from rl.policy import LLMPolicy
        from rl.schedule import train
        rl_cfg = config.get("rl", {})
        adapter_paths = rl_cfg.get("adapter_paths", {
            "defender": "checkpoints/defender_adapter",
        })
        # Only the defender has a policy now, so only its LoRA adapter is
        # materialised (one fewer ~115 MB copy of LoRA tensors on the GPU).
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
        if adapter_exists(adapter_paths["defender"]):
            policy.load_adapter("defender", adapter_paths["defender"])
        progress = load_progress()
        start_round = progress["rounds_done"]
        if start_round:
            logger.info(f"Resuming Phase-2 training from round {start_round}")
            # Restore the LIVE shared FL state (the evolving global model + per-client
            # weights) so the run continues from where it stopped instead of rewinding
            # to the Phase-1 baseline. env.reset() above already loaded the Phase-1
            # baseline; this overrides it with the saved Phase-2 state.
            saved_fl = load_fl_state()
            if saved_fl is not None:
                env.restore_fl_state(saved_fl)
            else:
                logger.warning(
                    "No saved Phase-2 FL state found — the shared model resumes from the "
                    "Phase-1 baseline (older checkpoint predating fl_state.pt)."
                )

        def progress_cb(done, round_index=None, controller=None, curriculum=None,
                        attacker_state=None):
            save_progress(done, round_index=round_index, controller=controller,
                          curriculum=curriculum, attacker=attacker_state)

        def fl_state_cb(fl_state):
            save_fl_state(fl_state)

        train(env, policy, defender_agent, config,
              metrics_tracker, _save_round_log, rng,
              progress_cb=progress_cb, fl_state_cb=fl_state_cb,
              start_round=start_round, resume=progress,
              total_rounds=n_rounds, max_new_rounds=max_new_rounds)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 COMPLETE")
    if frozen:
        # The global never moved (that is the point), so the informative number is
        # the LAST round's post-attack accuracy — what the anchor degrades to under
        # the committed round — not the anchor's own unchanged accuracy.
        logger.info(f"Frozen anchor accuracy: {env.current_accuracy:.4f} "
                    f"(baseline: {baseline_accuracy:.4f}) — last simulated round scored "
                    f"{env.last_round_accuracy:.4f}")
    else:
        logger.info(f"Final accuracy: {env.current_accuracy:.4f} (baseline: {baseline_accuracy:.4f})")
    logger.info(f"Label-flip ladder ended at {env.attacker.fraction:.0%} "
                f"(cycle {env.attacker.ladder.cycle})")
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the Phase-2 loop with a frozen LLM (no training, no GPU needed)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run the Phase-2 loop with no LLM at all (fixed heuristic defense)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override simulation_rounds (handy for quick smoke runs)")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose Phase-2 debug run: print the label-flip plan, every "
                             "defender LLM prompt+output, every FL update and reward to the "
                             "console AND to logs/debug.json. Library noise is silenced. If "
                             "--rounds is not given, Phase 2 is capped at "
                             f"{DEBUG_DEFAULT_ROUNDS} rounds.")
    args = parser.parse_args()

    setup_logging(debug=args.debug)
    quiet_noisy_warnings()
    logger.info("Starting Zero-Touch Federated Learning System")

    base_config = load_config(args.config)
    defender_config = load_config("configs/defender_agent.yaml")

    # LLM backend + shared Ollama defaults (used by --dry-run / OpenAI paths).
    llm_backend = "ollama" if args.env == "linux" else "openai"
    llm_defaults = base_config.get("llm", {})
    defender_config.setdefault("llm", {})
    defender_config["llm"]["backend"] = llm_backend
    defender_config["llm"].setdefault(
        "ollama_base_url", llm_defaults.get("ollama_base_url", "http://localhost:11434"))
    defender_config["llm"].setdefault(
        "ollama_model", llm_defaults.get("ollama_model", "qwen2.5:3b"))

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
        attack_cfg = base_config.get("attack", {}) or {}
        schedule = attack_cfg.get("schedule") or {}
        dbg.enable(
            output_dir="logs", filename="debug.json", mode=mode,
            config_summary={
                "model": base_config.get("rl", {}).get("model"),
                "defense_mode": (base_config.get("defense", {}) or {}).get("mode", "llm"),
                "defense_algorithms": (base_config.get("defense", {}) or {}).get("algorithms"),
                "curriculum": (base_config.get("curriculum") or None),
                "freeze_global_in_phase2": fl.get("freeze_global_in_phase2", True),
                "client_data_refresh": fl.get("client_data_refresh", "rotate"),
                "client_round_fraction": fl.get("client_round_fraction", 0.25),
                "n_clients": fl.get("n_clients"),
                "attack_type": attack_cfg.get("type", "label_flip"),
                "poison_client_ids": attack_cfg.get("poison_client_ids"),
                "flip_ladder": {
                    "start_fraction": schedule.get("start_fraction"),
                    "step_fraction": schedule.get("step_fraction"),
                    "floor_fraction": schedule.get("floor_fraction"),
                    "caught_rule": schedule.get("caught_rule"),
                },
                "noniid_bias": base_config.get("data", {}).get("noniid_bias"),
                "G": base_config.get("rl", {}).get("G"),
                "success_streak": base_config.get("rl", {}).get("success_streak"),
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
            defender_config=defender_config,
            mode=mode,
            n_rounds=n_rounds,
            llm_backend=llm_backend,
            max_new_rounds=max_new_rounds,
        )
    finally:
        dbg.close()


if __name__ == "__main__":
    main()
