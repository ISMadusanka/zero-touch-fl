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
from detector.anomaly_detector import AnomalyDetector
from agents.attacker_agent import AttackerAgent
from agents.defender_agent import DefenderAgent
from storage.checkpoint import save_state, load_state, state_exists
from core.types import RoundLog
from metrics import MetricsTracker

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
    client_loaders, test_loader = get_data_loaders(
        n_clients=fl["n_clients"],
        batch_size=fl["batch_size"],
        data_dir=data_cfg.get("data_dir", "./data/mnist_raw"),
        iid=data_cfg.get("iid", True),
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

    return server.get_global_weights(), client_weights, baseline_accuracy, test_loader

# ---------------------------------------------------------------------------
# Phase 2: Attack/Defend Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    global_weights: dict,
    client_weights: list[dict],
    baseline_accuracy: float,
    test_loader,
    config: dict,
    attacker_config: dict,
    defender_config: dict,
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

    # Components
    server = FedServer(device=fl["device"])
    server.set_global_weights(copy.deepcopy(global_weights))
    aggregator = FedAvgAggregator()
    detector = AnomalyDetector()
    attacker_agent = AttackerAgent(attacker_config)
    defender_agent = DefenderAgent(defender_config)
    malicious_client = MaliciousClient(client_id=malicious_id)

    # Metrics tracker — ground truth = the configured malicious client id(s)
    metrics_tracker = MetricsTracker(
        malicious_ids={malicious_id},
        baseline_accuracy=baseline_accuracy,
        output_dir="logs/metrics",
    )

    # State tracking
    last_attack_detected = None    # None on first round
    last_attack_passed = None      # None on first round
    last_all_clients_flagged = None  # None on first round
    current_accuracy = baseline_accuracy

    for sim_round in range(1, fl["simulation_rounds"] + 1):
        round_num = fl["training_rounds"] + sim_round
        logger.info(f"\n{'='*60}")
        logger.info(f"SIMULATION ROUND {sim_round} (Global Round {round_num})")
        logger.info(f"{'='*60}")

        current_global = server.get_global_weights()

        # ------------------------------------------------------------------
        # Step 1: Attacker decides strategy
        # ------------------------------------------------------------------
        # Compute windowed metrics for agent feedback (uses rounds so far)
        windowed = metrics_tracker.get_windowed_metrics(window=5)

        attacker_context = {
            "baseline_accuracy": baseline_accuracy,
            "current_accuracy": current_accuracy,
            "was_detected": last_attack_detected,
            # Windowed KPIs — attacker gets ASR, FPR, APR
            "attack_success_rate_recent": windowed["attack_success_rate"],
            "fpr_recent": windowed["fpr"],
            "accuracy_preservation_rate": windowed["accuracy_preservation_rate"],
        }
        attack_strategy = attacker_agent.decide(attacker_context)
        attack_name = attack_strategy.get("attack_type", "sign_flip")
        attack_params = attack_strategy.get("params", {})
        logger.info(f"Attacker strategy: {attack_name} with params={attack_params}")

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
        # Step 3: Defender decides strategy
        # ------------------------------------------------------------------
        update_features = detector.get_features(updates, current_global)
        defender_context = {
            "update_features": update_features,
            "attack_passed_through": last_attack_passed,
            "all_clients_flagged": last_all_clients_flagged,
            # Windowed KPIs — defender gets TPR, FPR, APR
            "tpr_recent": windowed["tpr"],
            "fpr_recent": windowed["fpr"],
            "accuracy_preservation_rate": windowed["accuracy_preservation_rate"],
        }
        defend_strategy = defender_agent.decide(defender_context)
        logger.info(f"Defender strategy: {defend_strategy.get('method')} with params={defend_strategy.get('params')}")

        # ------------------------------------------------------------------
        # Step 4: Anomaly detection
        # ------------------------------------------------------------------
        verdicts = detector.analyze(updates, current_global, defend_strategy)

        # Check if the malicious client was detected
        malicious_verdict = next(v for v in verdicts if v.client_id == malicious_id)
        attack_detected = malicious_verdict.is_suspicious
        attack_passed = not attack_detected
        n_flagged = sum(1 for v in verdicts if v.is_suspicious)
        all_clients_flagged = n_flagged == len(verdicts)

        logger.info(f"Detection result: malicious client {'DETECTED' if attack_detected else 'PASSED THROUGH'}")
        logger.info(f"Detection summary: {n_flagged}/{len(verdicts)} clients flagged")

        # ------------------------------------------------------------------
        # Step 5: Aggregation (exclude detected clients)
        # ------------------------------------------------------------------
        new_weights = aggregator.aggregate(updates, verdicts, strategy=defend_strategy)

        if new_weights is None:
            # All clients flagged → skip round, keep global model unchanged
            logger.warning(
                f"Round {round_num}: all clients flagged — global model NOT updated"
            )
        else:
            server.set_global_weights(new_weights)

        # ------------------------------------------------------------------
        # Step 6: Evaluate
        # ------------------------------------------------------------------
        current_accuracy = server.evaluate(test_loader)
        logger.info(f"Test accuracy after aggregation: {current_accuracy:.4f} (baseline: {baseline_accuracy:.4f})")

        # ------------------------------------------------------------------
        # Step 6b: Compute & log evaluation metrics for this round
        # ------------------------------------------------------------------
        round_metrics = metrics_tracker.update(
            round_num=round_num,
            verdicts=verdicts,
            current_accuracy=current_accuracy,
        )

        # ------------------------------------------------------------------
        # Step 7: Record outcomes for both agents
        # ------------------------------------------------------------------
        # Extract attack metadata (e.g. flipped indices) from the malicious update
        malicious_update = updates[malicious_id]
        attack_metadata = malicious_update.metadata.get("attack_metadata", {})
        if attack_metadata:
            layer_info = attack_metadata.get("flipped_per_layer", attack_metadata.get("affected_per_layer", {}))
            logger.info(
                f"Attack metadata: k={attack_metadata.get('k', 'N/A')}, "
                f"total_params={attack_metadata.get('total_params', '?')}, "
                f"layers_affected={list(layer_info.keys())}"
            )

        # Recompute windowed metrics AFTER this round's data is recorded
        windowed_after = metrics_tracker.get_windowed_metrics(window=5)

        attacker_agent.record_outcome(
            round_num=round_num,
            strategy=attack_strategy,
            was_detected=attack_detected,
            accuracy=current_accuracy,
            attack_metadata=attack_metadata,
            # Windowed KPIs for attacker history
            attack_success_rate_recent=windowed_after["attack_success_rate"],
            fpr_recent=windowed_after["fpr"],
            accuracy_preservation_rate=windowed_after["accuracy_preservation_rate"],
        )
        defender_agent.record_outcome(
            round_num=round_num,
            strategy=defend_strategy,
            attack_passed=attack_passed,
            all_clients_flagged=all_clients_flagged,
            verdicts=[
                {"client_id": v.client_id, "suspicious": v.is_suspicious, "confidence": v.confidence, "reason": v.reason}
                for v in verdicts
            ],
            # Windowed KPIs for defender history
            tpr_recent=windowed_after["tpr"],
            fpr_recent=windowed_after["fpr"],
            accuracy_preservation_rate=windowed_after["accuracy_preservation_rate"],
        )

        # ------------------------------------------------------------------
        # Step 8: Save round data to file
        # ------------------------------------------------------------------
        round_log = RoundLog(
            round_num=round_num,
            attack_strategy={"type": attack_name, "params": attack_params, "reasoning": attack_strategy.get("reasoning", "")},
            defend_strategy=defend_strategy,
            verdicts=[
                {"client_id": v.client_id, "suspicious": v.is_suspicious, "confidence": v.confidence, "reason": v.reason}
                for v in verdicts
            ],
            test_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            attack_detected=attack_detected,
            attacker_adapted=last_attack_detected is True,    # adapted this round because caught last round
            defender_adapted=last_attack_passed is True,      # adapted this round because failed last round
            all_clients_flagged=all_clients_flagged,
            round_skipped=new_weights is None,
        )
        _save_round_log(round_log, extra={"metrics": round_metrics.to_dict()})

        # Update state for next round
        last_attack_detected = attack_detected
        last_attack_passed = attack_passed
        last_all_clients_flagged = all_clients_flagged

    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info(f"Final accuracy: {current_accuracy:.4f} (baseline: {baseline_accuracy:.4f})")
    logger.info("=" * 60)

    # Aggregate metrics over the whole simulation
    metrics_tracker.save_summary()


def _save_round_log(log: RoundLog, extra: dict | None = None):
    """Save a round's complete data to JSON.

    `extra` is merged into the payload (e.g. evaluation metrics for the round).
    """
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
        "all_clients_flagged": log.all_clients_flagged,
        "round_skipped": log.round_skipped,
    }
    if extra:
        data.update(extra)
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

    # Prepare test loader (needed for both phases)
    _, test_loader = get_data_loaders(
        n_clients=fl["n_clients"],
        batch_size=fl["batch_size"],
        data_dir=data_cfg.get("data_dir", "./data/mnist_raw"),
        iid=data_cfg.get("iid", True),
    )

    if state_exists() and not args.fresh:
        logger.info("Checkpoint found — skipping Phase 1, loading saved state")
        loaded = load_state()
        global_weights, client_weights, baseline_accuracy = loaded
    else:
        logger.info("No checkpoint found (or --fresh) — running Phase 1")
        global_weights, client_weights, baseline_accuracy, test_loader = run_training_phase(base_config)

    # Phase 2
    run_simulation(
        global_weights=global_weights,
        client_weights=client_weights,
        baseline_accuracy=baseline_accuracy,
        test_loader=test_loader,
        config=base_config,
        attacker_config=attacker_config,
        defender_config=defender_config,
    )


if __name__ == "__main__":
    main()
