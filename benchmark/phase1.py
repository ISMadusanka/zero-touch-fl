"""Phase-1 honest FedAvg start state for the benchmark.

Mirrors ``main.run_training_phase`` (honest FedAvg for ``training_rounds`` rounds)
but is self-contained — it reuses the FL components directly and does NOT import
``main`` (whose import chain pulls inference-only LLM clients). The benchmark
normally LOADS the saved Phase-1 checkpoint; this is only the fresh fallback. It
does not write any checkpoint (so it never clobbers training artifacts).
"""
import logging

from clients.benign_client import BenignClient
from core.types import DetectionVerdict
from data.datasets import DEFAULT_DATASET
from server.aggregation import FedAvgAggregator
from server.fed_server import FedServer

logger = logging.getLogger("benchmark")


def run_phase1(config: dict, client_loaders, test_loader):
    """Run honest FedAvg for ``fl.training_rounds`` rounds.

    Returns (global_weights, client_weights, baseline_accuracy) — the same tuple
    shape ``storage.checkpoint.load_state`` returns.
    """
    fl = config["fl"]
    dataset = (config.get("data") or {}).get("dataset", DEFAULT_DATASET)
    server = FedServer(device=fl["device"], dataset=dataset)
    clients = [
        BenignClient(client_id=i, data_loader=client_loaders[i], lr=fl["lr"],
                     local_epochs=fl["local_epochs"], device=fl["device"])
        for i in range(fl["n_clients"])
    ]
    aggregator = FedAvgAggregator()

    updates = []
    for round_num in range(1, int(fl["training_rounds"]) + 1):
        updates = [c.train(server.model) for c in clients]
        clean = [DetectionVerdict(u.client_id, False, 0.0, "phase1") for u in updates]
        new_weights = aggregator.aggregate(updates, clean)
        server.set_global_weights(new_weights)
        if round_num % 5 == 0 or round_num == int(fl["training_rounds"]):
            logger.info(f"  Phase-1 round {round_num}/{fl['training_rounds']} "
                        f"acc={server.evaluate(test_loader):.4f}")

    baseline_accuracy = server.evaluate(test_loader)
    client_weights = [u.weights for u in updates]
    logger.info(f"Phase-1 baseline accuracy: {baseline_accuracy:.4f}")
    return server.get_global_weights(), client_weights, baseline_accuracy
