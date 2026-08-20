"""Zero-Touch Federated Learning — Main Entry Point.

Two phases:
  Phase 1 (training_rounds): honest FedAvg training, then checkpoint.
  Phase 2 (simulation_rounds): attacker-learning rounds SIMULATED on the frozen
    Phase-1 global.

Phase 2 does not run a continuing federation. Each round is an independent episode
branching off the SAME Phase-1 final model:

  1. that frozen global is sent to every client;
  2. every client trains on NEW local data (a fresh slice of its own shard);
  3. the attacker LLM picks an exact-budget subset of its controllable pool and
     poisons those clients (primitive weight operators over the benign weights);
  4. poisoned + honest updates go to the server, which defends and aggregates;
  5. the aggregate is evaluated on the test set — that accuracy is the attacker's
     reward — and then DISCARDED. The attacker improves on it with GRPO;
  6. the next round starts again from the Phase-1 global.

Set ``fl.freeze_global_in_phase2: false`` to restore the original continuing
federation, where each committed aggregate becomes the next round's global.

The DEFENDER LLM IS CURRENTLY DISABLED (``defense.mode: algorithmic`` in
configs/base.yaml). The server instead defends with the published algorithms —
FLTrust, DeFL, DnC and Multi-Krum — using ONE of them, drawn at random, per
round (see ``server/algo_defender.py``). Only the attacker is trained with GRPO.
Setting ``defense.mode: llm`` restores the original two-sided arms race, where a
defender LLM classifies each client benign/malicious from per-client per-layer
statistics and both sides get a verifiable per-round reward and train with GRPO
(separate LoRA adapters over one frozen Qwen2.5-3B-Instruct base).

Modes:
  python main.py --env linux                 # full GRPO training (needs a GPU)
  python main.py --env linux --dry-run       # frozen-LLM round loop, no training
  python main.py --baseline                  # best-of-N reward-harness sanity (no LLM)
  python main.py --fresh                      # force fresh Phase-1 training
  python main.py --rounds 8                   # override simulation_rounds (quick runs)
  python main.py --poisoners 8                # poison 8 clients per round, not the config's
  python main.py --learn attacker             # train ONE side; the other plays frozen
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

from data.nidd_loader import get_data_loaders, build_root_loader
from data.round_sampler import build_round_data_sampler
from clients.benign_client import BenignClient
from model import set_default_hidden
from server.fed_server import FedServer
from server.aggregation import FedAvgAggregator
from server.algo_defender import build_algorithmic_defender
from agents.attacker_agent import AttackerAgent
from agents.defender_agent import DefenderAgent
from agents.llm_client import create_llm_client
from storage.checkpoint import (
    save_state, load_state, state_exists, save_progress, load_progress, adapter_exists,
    save_fl_state, load_fl_state, shape_mismatch as checkpoint_shape_mismatch,
)
from core.types import RoundLog, DetectionVerdict
from core.debug import dbg
from core.config_overrides import LEARN_CHOICES, apply_learner_choice, apply_poisoner_count
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
    max_new_rounds: int | None = None,
):
    fl = config["fl"]
    logger.info("=" * 60)
    attack_cfg = config.get("attack", {})
    frozen = bool(fl.get("freeze_global_in_phase2", True))
    logger.info(f"PHASE 2: attacker-learning rounds  (mode={mode}, "
                f"{'SIMULATED on the frozen Phase-1 global' if frozen else 'continuing federation'})")
    logger.info(f"  client_data_refresh={fl.get('client_data_refresh', 'rotate')} "
                f"fraction={fl.get('client_round_fraction', 0.25)}")
    _fixed_poison = attack_cfg.get("fixed_poison_clients")
    if _fixed_poison not in (None, False, 0, ""):
        logger.info(f"  simulation_rounds={n_rounds}, FIXED poison set = clients "
                    f"0..{int(_fixed_poison) - 1} ({_fixed_poison} of "
                    f"{fl.get('n_clients')}), poisoned every round; the attacker LLM "
                    f"chooses HOW to poison them, not which")
    else:
        logger.info(f"  simulation_rounds={n_rounds}, n_compromisable={fl.get('n_compromisable')}, "
                    f"max_poison_clients={attack_cfg.get('max_poison_clients')}, "
                    f"sample_budget={attack_cfg.get('sample_budget_in_training')}")
    goal_cfg = attack_cfg.get("goal", {}) or {}
    logger.info(f"  attack_goal={goal_cfg.get('type')} "
                f"target_accuracy_drop={goal_cfg.get('target_accuracy_drop')} "
                f"sample_target={attack_cfg.get('sample_target_in_training')}")
    logger.info(f"  baseline_accuracy={baseline_accuracy:.4f}")
    logger.info("=" * 60)

    seed = int(fl.get("poison_seed", 0))
    rng = random.Random(seed)
    # Step 2 of a simulated round: every client gets NEW local data each round (a
    # fresh slice of its own shard). Without this the frozen global would hand the
    # clients an identical starting point AND identical data, so every round would
    # reproduce the same honest updates and there would be nothing for the attacker
    # to generalize over. ``fl.client_data_refresh: none`` turns it off.
    round_data = build_round_data_sampler(config, client_loaders, seed=seed)
    # Server-side defense. With ``defense.mode: algorithmic`` (the default) the
    # defender LLM is disabled and one published algorithm — FLTrust / DeFL / DnC /
    # Multi-Krum — defends each round, drawn at random; only the attacker trains.
    # ``defense.mode: llm`` restores the trainable defender LLM.
    # An honest client's per-round SGD iteration count. FLTrust's root fine-tuning is
    # sized to match it (defense.fltrust.root_epochs: null), because FLTrust rescales
    # every accepted update to ||g0|| — so the server's reference update, not the
    # clients', sets how far the global model can move per round. See
    # server.algo_defender.resolve_root_epochs. It must be counted over the data a
    # client actually trains on THIS round, which with the per-round refresh on is a
    # slice of the shard, not the whole thing — sizing g0 off the full shard would
    # make it several times too large and FLTrust would rescale every honest update
    # to match.
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
            data_cfg=config.get("data", {}),
            seed=seed,
        ),
    )
    # Training curriculum: instead of drawing the defense algorithm and the poison
    # quota at random every round, sweep them — one algorithm held for
    # `rounds_per_block` rounds at 1 poisoner, then at 2, ... then the next
    # algorithm, then the cycle repeats. Every (defense, #poisoners) pair therefore
    # gets an equal, contiguous share of the attacker's training. None = the old
    # random draws (no `curriculum:` block, or `enabled: false`).
    curriculum = build_training_curriculum(
        config, algorithms=(defense.names if defense is not None else None))
    env = FLArmsRaceEnv(config, client_loaders, test_loader, rng,
                        defense=defense, curriculum=curriculum, round_data=round_data)
    env.reset(global_weights, client_weights, baseline_accuracy)

    metrics_tracker = MetricsTracker(baseline_accuracy=baseline_accuracy, output_dir="logs/metrics")
    attacker_agent = AttackerAgent(attacker_config)
    defender_agent = DefenderAgent(defender_config)

    if mode == "baseline":
        from rl.baseline import run_baseline
        # Same attacker reward weights training uses — the baseline COMMITS the
        # highest-scoring action, so different weights would report a different
        # attack (see rl.rewards.check_reward_balance).
        run_baseline(env, n_rounds, metrics_tracker, _save_round_log,
                     reward_cfg=config.get("rl", {}).get("reward", {}).get("attacker", {}))

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
                      temperature=float(llm_cfg.get("temperature", 0.7)),
                      reward_cfg=config.get("rl", {}).get("reward", {}).get("attacker", {}))

    else:  # full GRPO training
        from rl.policy import LLMPolicy
        from rl.schedule import train
        rl_cfg = config.get("rl", {})
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
        # Measure the attacker's context fill with the REAL tokenizer from here on
        # (until now the agent used the character heuristic). This is what makes
        # rl.max_context_fill an exact cap rather than an approximate one.
        attacker_agent.bind_tokenizer(policy.count_prompt_tokens)
        # Resume adapters if present.
        for name, path in adapter_paths.items():
            if name in adapter_names and adapter_exists(path):
                policy.load_adapter(name, path)
        progress = load_progress()
        start_round = progress["rounds_done"]
        if start_round:
            logger.info(f"Resuming Phase-2 training from round {start_round}")
            # Restore the LIVE shared FL state (the evolving global model + per-client
            # weights) so the arms race continues from where it stopped instead of
            # rewinding to the Phase-1 baseline. env.reset() above already loaded the
            # Phase-1 baseline; this overrides it with the saved Phase-2 state.
            saved_fl = load_fl_state()
            if saved_fl is not None:
                env.restore_fl_state(saved_fl)
            else:
                logger.warning(
                    "No saved Phase-2 FL state found — the shared model resumes from the "
                    "Phase-1 baseline (older checkpoint predating fl_state.pt)."
                )

        def progress_cb(done, round_index=None, controller=None, curriculum=None):
            save_progress(done, round_index=round_index, controller=controller,
                          curriculum=curriculum)

        def fl_state_cb(fl_state):
            save_fl_state(fl_state)

        train(env, policy, attacker_agent, defender_agent, config,
              metrics_tracker, _save_round_log, rng,
              progress_cb=progress_cb, fl_state_cb=fl_state_cb,
              start_round=start_round, resume=progress,
              total_rounds=n_rounds, max_new_rounds=max_new_rounds)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 COMPLETE")
    if frozen:
        # The global never moved (that is the point), so the informative number is
        # the LAST round's post-attack accuracy — what the anchor degrades to under
        # the attacker's committed plan — not the anchor's own unchanged accuracy.
        logger.info(f"Frozen anchor accuracy: {env.current_accuracy:.4f} "
                    f"(baseline: {baseline_accuracy:.4f}) — last simulated round scored "
                    f"{env.last_round_accuracy:.4f}")
    else:
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the Phase-2 loop with a frozen LLM (no training, no GPU needed)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run the best-of-N reward-harness sanity baseline (no LLM)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override simulation_rounds (handy for quick smoke runs)")
    parser.add_argument("--poisoners", type=int, default=None, metavar="N",
                        help="Poison exactly N clients every Phase-2 round, overriding the "
                             "config's count (attack.fixed_poison_clients, or the per-round "
                             "quota when the attacker picks its own set). Also retargets the "
                             "curriculum's poisoner sweep, the evaluation quota, and the "
                             "#malicious DnC/Multi-Krum assume. Default: the config's value.")
    parser.add_argument("--learn", choices=list(LEARN_CHOICES), default=None,
                        help="Which side TRAINS. 'attacker' / 'defender' train that adapter "
                             "only, with the other side playing frozen (its checkpoint is "
                             "left untouched); 'both' is the two-sided arms race. "
                             "'defender'/'both' need defense.mode: llm. Default: the config "
                             "(both under defense.mode: llm, attacker-only otherwise).")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose Phase-2 debug run: print every attacker/defender LLM "
                             "prompt+output, poisoning step, FL update and reward to the console "
                             "AND to logs/debug.json. Library noise is silenced. If --rounds is "
                             f"not given, Phase 2 is capped at {DEBUG_DEFAULT_ROUNDS} rounds.")
    args = parser.parse_args()

    setup_logging(debug=args.debug)
    quiet_noisy_warnings()
    logger.info("Starting Zero-Touch Federated Learning System")

    base_config = load_config(args.config)
    attacker_config = load_config("configs/attacker_agent.yaml")
    defender_config = load_config("configs/defender_agent.yaml")

    # CLI overrides FIRST: everything below — the attacker's system prompt, the
    # data partition, the curriculum, the algorithmic defender's assumed #malicious
    # and the env itself — reads these keys, so they have to be settled before the
    # first reader. Each flag rewrites the whole SET of keys that implements it
    # (see core/config_overrides.py) and logs what it changed.
    try:
        if args.poisoners is not None:
            apply_poisoner_count(base_config, args.poisoners)
        if args.learn is not None:
            apply_learner_choice(base_config, args.learn)
    except ValueError as exc:
        parser.error(str(exc))

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

    # ...and for the two things the attacker's PROMPT depends on but that live in
    # the base config: whether the poisoned set is fixed (which changes the system
    # prompt from "select k of n" to "plan for all of these"), and the context-fill
    # budget the observation is compacted to fit.
    _attack_cfg = base_config.get("attack", {}) or {}
    attacker_config["fixed_poison_set"] = (
        _attack_cfg.get("fixed_poison_clients") not in (None, False, 0, ""))
    attacker_config["rl"] = base_config.get("rl", {})

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
    # The architecture knob, installed before any FedServer is built (the four
    # construction sites have no config access — see model.set_default_hidden).
    set_default_hidden((base_config.get("model") or {}).get("hidden"))
    client_loaders, test_loader = get_data_loaders(
        n_clients=fl["n_clients"], batch_size=fl["batch_size"],
        data_cfg=data_cfg, iid=data_cfg.get("iid", True),
        bias_q=float(data_cfg.get("noniid_bias", 0.5)), seed=seed,
    )

    state = load_state() if (state_exists() and not args.fresh) else None
    if state is not None and len(state[1]) != fl["n_clients"]:
        logger.warning(
            f"Checkpoint has {len(state[1])} client(s) but config n_clients={fl['n_clients']} "
            f"— ignoring the stale checkpoint and re-running Phase 1."
        )
        state = None
    # A checkpoint from a different feature/class count cannot be loaded into this
    # run's model at all, and `load_state_dict` would otherwise raise deep inside
    # Phase-2 resume. Catch it here, where the fix (re-run Phase 1) is automatic.
    # This is the shape guard that makes changing `data.n_features`,
    # `data.label_mode` or `model.hidden` safe rather than a confusing crash.
    if state is not None:
        stale = checkpoint_shape_mismatch(state[0])
        if stale:
            logger.warning(f"Checkpoint {stale} — ignoring it and re-running Phase 1.")
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
    # --poisoners applies in every mode (the env reads it), but --learn only means
    # something where there is a GRPO schedule to point at.
    if args.learn is not None and mode != "train":
        logger.warning(f"--learn {args.learn} has no effect in --{mode} (nothing trains).")

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
            output_dir="logs", filename="debug.json", mode=mode,
            config_summary={
                "model": base_config.get("rl", {}).get("model"),
                "defense_mode": (base_config.get("defense", {}) or {}).get("mode", "algorithmic"),
                "defense_algorithms": (base_config.get("defense", {}) or {}).get("algorithms"),
                "defense_selection": (base_config.get("defense", {}) or {}).get("selection", "random"),
                "curriculum": (base_config.get("curriculum") or None),
                "freeze_global_in_phase2": fl.get("freeze_global_in_phase2", True),
                "client_data_refresh": fl.get("client_data_refresh", "rotate"),
                "client_round_fraction": fl.get("client_round_fraction", 0.25),
                "n_clients": fl.get("n_clients"),
                "n_compromisable": fl.get("n_compromisable"),
                "fixed_poison_clients": base_config.get("attack", {}).get("fixed_poison_clients"),
                "max_poison_clients": base_config.get("attack", {}).get("max_poison_clients"),
                "sample_budget": base_config.get("attack", {}).get("sample_budget_in_training"),
                "noniid_bias": base_config.get("data", {}).get("noniid_bias"),
                "G": base_config.get("rl", {}).get("G"),
                "switch_mode": base_config.get("rl", {}).get("switch_mode"),
                "first_learner": base_config.get("rl", {}).get("first_learner"),
                "learners": base_config.get("rl", {}).get("learners"),
                "cli_poisoners": args.poisoners,
                "cli_learn": args.learn,
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
            max_new_rounds=max_new_rounds,
        )
    finally:
        dbg.close()


if __name__ == "__main__":
    main()
