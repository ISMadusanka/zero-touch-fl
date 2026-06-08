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


def save_midway_state(global_model_state: dict, client_updates: list[dict], round_num: int):
    """Persist midway Phase 1 results to disk to allow resuming."""
    _ensure_dir()
    torch.save(global_model_state, os.path.join(CHECKPOINT_DIR, "midway_global_model.pt"))
    torch.save(client_updates, os.path.join(CHECKPOINT_DIR, "midway_client_updates.pt"))
    with open(os.path.join(CHECKPOINT_DIR, "midway_meta.json"), "w") as f:
        json.dump({"round_num": round_num}, f)


def load_midway_state():
    """Load midway state. Returns (global_state, client_updates, round_num) or None."""
    try:
        global_state = torch.load(
            os.path.join(CHECKPOINT_DIR, "midway_global_model.pt"), weights_only=False
        )
        client_updates = torch.load(
            os.path.join(CHECKPOINT_DIR, "midway_client_updates.pt"), weights_only=False
        )
        with open(os.path.join(CHECKPOINT_DIR, "midway_meta.json")) as f:
            round_num = json.load(f)["round_num"]
        return global_state, client_updates, round_num
    except FileNotFoundError:
        return None


def midway_state_exists() -> bool:
    return os.path.exists(os.path.join(CHECKPOINT_DIR, "midway_meta.json"))


def clear_midway_state():
    """Remove midway checkpoints after Phase 1 finishes."""
    for file_name in ["midway_global_model.pt", "midway_client_updates.pt", "midway_meta.json"]:
        path = os.path.join(CHECKPOINT_DIR, file_name)
        if os.path.exists(path):
            os.remove(path)

