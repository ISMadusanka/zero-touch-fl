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
# Phase-2 SHARED FL state (the evolving global model + per-client benign
# weights). Phase 1 saves the *initial* baseline via save_state(); this saves
# the LIVE Phase-2 arms-race state at each checkpoint so a resume continues the
# shared model where it left off instead of rewinding to the Phase-1 baseline
# (the global model is a ~970-param MnistNet, so this is a few KB).
# ---------------------------------------------------------------------------

_FL_STATE_FILE = "fl_state.pt"


def save_fl_state(fl_state: dict):
    """Persist the live Phase-2 FL state dict (see ``FLArmsRaceEnv.snapshot_fl_state``)."""
    _ensure_dir()
    torch.save(fl_state, os.path.join(CHECKPOINT_DIR, _FL_STATE_FILE))


def load_fl_state():
    """Return the saved Phase-2 FL state dict, or ``None`` if there is none."""
    try:
        return torch.load(
            os.path.join(CHECKPOINT_DIR, _FL_STATE_FILE), weights_only=False
        )
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# RL training progress (for resuming the Phase-2 GRPO loop)
# ---------------------------------------------------------------------------

_PROGRESS_FILE = "rl_progress.json"


def save_progress(rounds_done: int, round_index: int | None = None,
                  controller: dict | None = None):
    """Persist Phase-2 resume state.

    Backward compatible: ``rounds_done`` is always written. For a FULL resume we also
    persist ``round_index`` (the FL round-number counter, so round labels and
    ``logs/round_data`` continue across restarts instead of overwriting from the first
    Phase-2 round) and ``controller`` (the arms-race ``PhaseController`` snapshot, so
    the learner/phase/streak resume instead of restarting at the first attacker phase).
    """
    _ensure_dir()
    payload = {"rounds_done": int(rounds_done)}
    if round_index is not None:
        payload["round_index"] = int(round_index)
    if controller is not None:
        payload["controller"] = controller
    with open(os.path.join(CHECKPOINT_DIR, _PROGRESS_FILE), "w") as f:
        json.dump(payload, f)


def load_progress() -> dict:
    """Return the Phase-2 resume state as a dict:
    ``{"rounds_done": int, "round_index": int|None, "controller": dict|None}``.

    Old progress files that only hold ``rounds_done`` load fine (the new keys come
    back ``None`` → the caller falls back gracefully); a missing or corrupt file
    yields a fresh-start dict (``rounds_done=0``).
    """
    try:
        with open(os.path.join(CHECKPOINT_DIR, _PROGRESS_FILE)) as f:
            data = json.load(f)
        return {
            "rounds_done": int(data.get("rounds_done", 0)),
            "round_index": data.get("round_index"),
            "controller": data.get("controller"),
        }
    except (FileNotFoundError, ValueError, TypeError):
        return {"rounds_done": 0, "round_index": None, "controller": None}


def adapter_exists(path: str) -> bool:
    """True if a saved LoRA adapter directory looks present at ``path``."""
    return os.path.exists(os.path.join(path, "adapter_config.json"))
