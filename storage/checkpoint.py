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


# ---------------------------------------------------------------------------
# RL training progress (for resuming the Phase-2 GRPO loop)
# ---------------------------------------------------------------------------

_PROGRESS_FILE = "rl_progress.json"


def save_progress(rounds_done: int):
    """Persist how many Phase-2 rounds have been trained (resume support)."""
    _ensure_dir()
    with open(os.path.join(CHECKPOINT_DIR, _PROGRESS_FILE), "w") as f:
        json.dump({"rounds_done": int(rounds_done)}, f)


def load_progress() -> int:
    """Return rounds already trained (0 if none)."""
    try:
        with open(os.path.join(CHECKPOINT_DIR, _PROGRESS_FILE)) as f:
            return int(json.load(f)["rounds_done"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0


def adapter_exists(path: str) -> bool:
    """True if a saved LoRA adapter directory looks present at ``path``."""
    return os.path.exists(os.path.join(path, "adapter_config.json"))
