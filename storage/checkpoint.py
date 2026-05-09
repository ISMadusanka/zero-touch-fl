"""Checkpoint management — save/load system state between phases."""

import json
import os
import torch

CHECKPOINT_DIR = "checkpoints"


def _ensure_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_state(global_model_state: dict, client_updates: list[dict], baseline_accuracy: float):
    """Persist Phase 1 results to disk."""
    _ensure_dir()
    torch.save(global_model_state, os.path.join(CHECKPOINT_DIR, "global_model.pt"))
    torch.save(client_updates, os.path.join(CHECKPOINT_DIR, "client_updates.pt"))
    with open(os.path.join(CHECKPOINT_DIR, "baseline.json"), "w") as f:
        json.dump({"baseline_accuracy": baseline_accuracy}, f)


def load_state():
    """Load saved state. Returns (global_state, client_updates, baseline_acc) or None."""
    try:
        global_state = torch.load(
            os.path.join(CHECKPOINT_DIR, "global_model.pt"), weights_only=False
        )
        client_updates = torch.load(
            os.path.join(CHECKPOINT_DIR, "client_updates.pt"), weights_only=False
        )
        with open(os.path.join(CHECKPOINT_DIR, "baseline.json")) as f:
            baseline = json.load(f)["baseline_accuracy"]
        return global_state, client_updates, baseline
    except FileNotFoundError:
        return None


def state_exists() -> bool:
    return os.path.exists(os.path.join(CHECKPOINT_DIR, "global_model.pt"))
