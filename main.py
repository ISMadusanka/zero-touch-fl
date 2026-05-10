"""Zero-Touch Federated Learning — Main Entry Point.

Two-phase system:
  Phase 1 (Rounds 1-3): Honest FL training, then save state.
  Phase 2 (Round 4+):   Attack/defend simulation with LLM agents.

Usage:
  python main.py                    # Windows/OpenAI (default)
  python main.py --env linux        # Linux/Ollama backend
  python main.py --fresh            # Force fresh training (Phase 1)
  python main.py --env linux --fresh
"""

import argparse
import copy
import json
import logging
import os
import sys

import yaml
# import torch

from data.mnist_loader import get_data_loaders
from model.mnist_net import MnistNet, count_parameters
from clients.benign_client import BenignClient
from clients.malicious_client import MaliciousClient
from server.fed_server import FedServer
from server.aggregation import FedAvgAggregator
from detector.layered_detector import LayeredDetector
from detector.explainability import ExplainabilityEngine
from agents.attacker_agent import AttackerAgent
# from agents.defender_agent import DefenderAgent # LLM defender bypassed for Phase 1 output
from storage.checkpoint import save_state, load_state, state_exists
from core.types import RoundLog

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging():
    os.makedirs("logs/round_data", exist_ok=True)
    # Force UTF-8 to avoid Windows cp1252 encoding errors
    file_handler = logging.FileHandler("logs/system.log", mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[file_handler, stream_handler],
    )

logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Phase 1: Honest FL training for 3 rounds
# ---------------------------------------------------------------------------

def run_training_phase(config: dict):
    """Train all clients honestly for `training_rounds` rounds. Returns saved state."""
    fl = config["fl"]
    data_cfg = config["data"]

    logger.info("=" * 60)
    logger.info("PHASE 1: Honest Federated Learning Training")
    logger.info("=" * 60)

    # Data
    client_loaders, test_loader, root_loader = get_data_loaders(
        n_clients=fl["n_clients"],
        batch_size=fl["batch_size"],
        data_dir=data_cfg.get("data_dir", "./data/mnist_raw"),
    )

    # Log data sizes
    for i, loader in enumerate(client_loaders):
        logger.info(f"  Client {i} training samples: {len(loader.dataset)}")
    logger.info(f"  Test samples: {len(test_loader.dataset)}")
    logger.info(f"  Total training samples: {sum(len(l.dataset) for l in client_loaders)}")

    # Server
    server = FedServer(device=fl["device"])

    # Clients
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

    # Aggregator
    aggregator = FedAvgAggregator()

    # Training loop
    for round_num in range(1, fl["training_rounds"] + 1):
        logger.info(f"--- Training Round {round_num}/{fl['training_rounds']} ---")

        global_weights = server.get_global_weights()

        # All clients train honestly
        updates = []
        for client in clients:
            update = client.train(server.model)
            updates.append(update)
            meta = update.metadata
            logger.info(
                f"  Client {client.client_id} trained — "
                f"acc: {meta.get('train_accuracy', 0):.4f}, "
                f"loss: {meta.get('train_loss', 0):.4f}, "
                f"samples: {meta.get('train_samples', 0)}"
            )

        # Aggregate (no detection in Phase 1)
        from core.types import DetectionVerdict
        clean_verdicts = [
            DetectionVerdict(u.client_id, False, 0.0, "phase1") for u in updates
        ]
        new_weights = aggregator.aggregate(updates, clean_verdicts)
        server.set_global_weights(new_weights)

        # Evaluate
        accuracy = server.evaluate(test_loader)
        logger.info(f"  Round {round_num} accuracy: {accuracy:.4f}")

    # Baseline accuracy (before any attacks)
    baseline_accuracy = server.evaluate(test_loader)
    logger.info(f"Baseline accuracy after Phase 1: {baseline_accuracy:.4f}")

    # Save state: global model + each client's weights from final round
    client_weights = [u.weights for u in updates]
    save_state(server.get_global_weights(), client_weights, baseline_accuracy)
    logger.info("Phase 1 state saved to checkpoints/")

    return server.get_global_weights(), client_weights, baseline_accuracy, test_loader, root_loader

# ---------------------------------------------------------------------------
# Phase 2: Attack/Defend Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    global_weights: dict,
    client_weights: list[dict],
    baseline_accuracy: float,
    test_loader,
    root_loader,
    config: dict,
    attacker_config: dict,
):
    """Run the attack/defend simulation loop."""
    fl = config["fl"]
    malicious_id = fl["malicious_client_id"]

    logger.info("=" * 60)
    logger.info("PHASE 2: Attack / Defend Simulation")
    logger.info(f"  Malicious client: {malicious_id}")
    logger.info(f"  Simulation rounds: {fl['simulation_rounds']}")
    logger.info(f"  Baseline accuracy: {baseline_accuracy:.4f}")
    logger.info("=" * 60)

    # Purely Mathematical Components (No LLMs in Phase 1)
    server = FedServer(device=fl["device"])
    server.set_global_weights(copy.deepcopy(global_weights))
    aggregator = FedAvgAggregator()
    detector = LayeredDetector(root_loader=root_loader, device=fl["device"])
    explain_engine = ExplainabilityEngine()
    malicious_client = MaliciousClient(client_id=malicious_id)

    # State tracking
    last_attack_detected = None    # None on first round
    last_attack_passed = None      # None on first round
    current_accuracy = baseline_accuracy

    for sim_round in range(1, fl["simulation_rounds"] + 1):
        round_num = fl["training_rounds"] + sim_round
        logger.info(f"\n{'='*60}")
        logger.info(f"SIMULATION ROUND {sim_round} (Global Round {round_num})")
        logger.info(f"{'='*60}")

        current_global = server.get_global_weights()

        # ------------------------------------------------------------------
        # Step 1: Attack is hardcoded for Phase 1 research
        # ------------------------------------------------------------------
        attack_name = "sign_flip"
        attack_params = {}
        logger.info(f"Phase 1 Attack (Static): {attack_name}")

        # ------------------------------------------------------------------
        # Step 2: Build all client updates
        # ------------------------------------------------------------------
        from core.types import ModelUpdate

        updates = []
        for cid in range(fl["n_clients"]):
            if cid == malicious_id:
                # Poisoned update
                update = malicious_client.poison(
                    saved_weights=client_weights[cid],
                    global_weights=current_global,
                    attack_name=attack_name,
                    attack_params=attack_params,
                )
                logger.info(f"  Client {cid}: POISONED ({attack_name})")
            else:
                # Honest update (from saved Phase 1 weights)
                update = ModelUpdate(client_id=cid, weights=copy.deepcopy(client_weights[cid]))
                logger.info(f"  Client {cid}: honest (saved weights)")
            updates.append(update)

        # ------------------------------------------------------------------
        # Step 3 & 4: 4-Layer Defense Pipeline + SHAP Explainability
        # ------------------------------------------------------------------
        # In Phase 1, we don't consult the LLM for strategy yet. 
        # We just run the pipeline and output the evidence.
        evidence = detector.analyze(updates, server.model)
        
        # Add SHAP explanations for each client
        for cid_str, features in evidence.items():
            logger.info(f"Feature Vector for {cid_str}: {json.dumps(features)}")
            explanation = explain_engine.explain(features)
            evidence[cid_str]["explainability"] = explanation

        # TODO [Step 2.2]: Temporal Analysis - Track drift of client features over last N rounds
        # TODO [Step 2.2]: Correlation Analysis - Compare feature similarity between clients to detect collusion
        
        logger.info("\n" + "="*40)
        logger.info("PHASE 1 DEFENSE OUTPUT (Security Evidence)")
        logger.info("="*40)
        logger.info(json.dumps(evidence, indent=2))
        logger.info("="*40 + "\n")

        # Save evidence to file for research
        with open(f"logs/round_data/evidence_round_{round_num}.json", "w") as f:
            json.dump(evidence, f, indent=2)
        logger.info(f"Evidence saved to logs/round_data/evidence_round_{round_num}.json")

        # For aggregation in this modular mode, we can use a simple rule:
        # reject any client that is flagged by Layer 4 (Trimmed) or has low trust.
        from core.types import DetectionVerdict
        verdicts = []
        for cid_str, e in evidence.items():
            cid = int(cid_str.split("_")[1])
            is_suspicious = e["layer_4_is_trimmed"] or e["layer_1_fl_trust"] < 0.5
            verdicts.append(DetectionVerdict(cid, is_suspicious, e["layer_1_fl_trust"], "statistical_pipeline"))

        # Check if the malicious client was detected
        malicious_verdict = next(v for v in verdicts if v.client_id == malicious_id)
        attack_detected = malicious_verdict.is_suspicious
        logger.info(f"Detection result: malicious client {'DETECTED' if attack_detected else 'PASSED THROUGH'}")

        # ------------------------------------------------------------------
        # Step 5: Aggregation (exclude detected clients)
        # ------------------------------------------------------------------
        new_weights = aggregator.aggregate(updates, verdicts)
        server.set_global_weights(new_weights)

        # ------------------------------------------------------------------
        # Step 6: Evaluate
        # ------------------------------------------------------------------
        current_accuracy = server.evaluate(test_loader)
        logger.info(f"Test accuracy after aggregation: {current_accuracy:.4f} (baseline: {baseline_accuracy:.4f})")

        # ------------------------------------------------------------------
        # Step 8: Save round data
        # ------------------------------------------------------------------
        round_log = RoundLog(
            round_num=round_num,
            attack_strategy={"type": attack_name, "params": attack_params, "reasoning": "Static Phase 1 Attack"},
            defend_strategy={"method": "4_layer_pipeline", "params": {}},
            verdicts=[
                {"client_id": v.client_id, "suspicious": v.is_suspicious, "confidence": v.confidence, "reason": v.reason}
                for v in verdicts
            ],
            test_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            attack_detected=attack_detected,
            attacker_adapted=False,
            defender_adapted=False,
        )
        _save_round_log(round_log)

        # STOP AFTER ONE ROUND as per user request ("no need to automate again and again")
        logger.info("Modular Phase 1 Defense complete. Stopping execution.")
        break

    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info(f"Final accuracy: {current_accuracy:.4f} (baseline: {baseline_accuracy:.4f})")
    logger.info("=" * 60)


def _save_round_log(log: RoundLog):
    """Save a round's complete data to JSON."""
    path = f"logs/round_data/round_{log.round_num:03d}.json"
    data = {
        "round_num": log.round_num,
        "attack_strategy": log.attack_strategy,
        "defend_strategy": log.defend_strategy,
        "verdicts": log.verdicts,
        "test_accuracy": log.test_accuracy,
        "baseline_accuracy": log.baseline_accuracy,
        "attack_detected": log.attack_detected,
        "attacker_adapted": log.attacker_adapted,
        "defender_adapted": log.defender_adapted,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Round data saved to {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-Touch Federated Learning")
    parser.add_argument("--fresh", action="store_true", help="Force fresh Phase 1 training")
    parser.add_argument(
        "--env",
        choices=["linux", "windows"],
        default="windows",
        help="Running environment: 'linux' uses Ollama, 'windows' uses OpenAI (default: windows)",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting Zero-Touch Federated Learning System")
    logger.info(f"Environment: {args.env}")

    # Load configs
    base_config = load_config("configs/base.yaml")
    attacker_config = load_config("configs/attacker_agent.yaml")
    defender_config = load_config("configs/defender_agent.yaml")

    # Inject LLM backend based on --env flag
    llm_backend = "ollama" if args.env == "linux" else "openai"
    llm_defaults = base_config.get("llm", {})

    for agent_cfg in (attacker_config, defender_config):
        agent_cfg.setdefault("llm", {})
        agent_cfg["llm"]["backend"] = llm_backend
        # Propagate global Ollama settings (agent-level values take priority)
        agent_cfg["llm"].setdefault("ollama_base_url", llm_defaults.get("ollama_base_url", "http://localhost:11434"))
        agent_cfg["llm"].setdefault("ollama_model", llm_defaults.get("ollama_model", "deepseek-r1:70b"))

    logger.info(f"LLM backend: {llm_backend}")

    fl = base_config["fl"]
    data_cfg = base_config["data"]

    # Prepare data loaders
    client_loaders, test_loader, root_loader = get_data_loaders(
        n_clients=fl["n_clients"],
        batch_size=fl["batch_size"],
        data_dir=data_cfg.get("data_dir", "./data/mnist_raw"),
    )

    if state_exists() and not args.fresh:
        logger.info("Checkpoint found — skipping Phase 1, loading saved state")
        loaded = load_state()
        global_weights, client_weights, baseline_accuracy = loaded
    else:
        logger.info("No checkpoint found (or --fresh) — running Phase 1")
        global_weights, client_weights, baseline_accuracy, test_loader, root_loader = run_training_phase(base_config)

    # Phase 2 (Modular Single-Round)
    run_simulation(
        global_weights=global_weights,
        client_weights=client_weights,
        baseline_accuracy=baseline_accuracy,
        test_loader=test_loader,
        root_loader=root_loader,
        config=base_config,
        attacker_config=attacker_config,
    )


if __name__ == "__main__":
    main()
