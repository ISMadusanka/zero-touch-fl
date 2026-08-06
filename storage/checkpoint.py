"""Checkpoint management — save/load system state between phases.

**Layout (per dataset).** Everything in here describes ONE dataset's federated
run — a global ``state_dict``, the per-client weights that match it, the accuracy
those produce, and the round/phase counters that index them — and none of it is
portable across datasets (a CIFAR-10 conv state_dict cannot load into the MNIST
MLP). So each dataset gets its own directory::

    checkpoints/
      attacker_adapter/          <- SHARED: the LLM keeps fine-tuning across datasets
      defender_adapter/          <- SHARED
      mnist/
        global_model.pt  client_updates.pt  baseline.json  fl_state.pt  rl_progress.json
      cifar10/
        global_model.pt  client_updates.pt  baseline.json  fl_state.pt  rl_progress.json

The LoRA adapters sit at the ROOT on purpose: the policy observes only
dimensionless per-layer statistics and emits architecture-agnostic operator
plans, so one attacker adapter continues learning whichever dataset a run
selects. See ``core.run_config.adapter_paths``.

``dataset=None`` addresses the root directory directly — the pre-multi-dataset
layout, kept for tests and for reading old checkpoints (see :func:`_read_path`).
"""

import json
import logging
import os

import torch

from data.datasets import canonical

logger = logging.getLogger(__name__)

#: Root of the checkpoint tree. Per-dataset state lives in ``CHECKPOINT_DIR/<dataset>``.
CHECKPOINT_DIR = "checkpoints"

#: The dataset whose artifacts a pre-multi-dataset checkpoint tree contains: back
#: then MNIST was the only option, so a bare ``checkpoints/global_model.pt`` can
#: only be MNIST state.
_LEGACY_DATASET = "mnist"

_GLOBAL_FILE = "global_model.pt"
_CLIENTS_FILE = "client_updates.pt"
_BASELINE_FILE = "baseline.json"
_FL_STATE_FILE = "fl_state.pt"
_PROGRESS_FILE = "rl_progress.json"


def dataset_dir(dataset: str | None = None) -> str:
    """Directory holding ``dataset``'s FL state (the root when ``dataset`` is None)."""
    if dataset is None:
        return CHECKPOINT_DIR
    return os.path.join(CHECKPOINT_DIR, canonical(dataset))


def _ensure_dir(dataset: str | None = None) -> str:
    path = dataset_dir(dataset)
    os.makedirs(path, exist_ok=True)
    return path


def _write_path(filename: str, dataset: str | None) -> str:
    """Where to WRITE ``filename`` — always the dataset's own directory."""
    return os.path.join(_ensure_dir(dataset), filename)


def _read_path(filename: str, dataset: str | None) -> str:
    """Where to READ ``filename`` from, with a one-step legacy fallback.

    A checkpoint tree written before multi-dataset support has its files flat in
    ``checkpoints/`` and is MNIST by construction. Rather than move a user's
    artifacts behind their back, an MNIST run reads them in place when the new
    per-dataset copy does not exist yet; the next save writes to
    ``checkpoints/mnist/``, which then wins on every subsequent read.
    """
    path = os.path.join(dataset_dir(dataset), filename)
    if os.path.exists(path) or dataset is None:
        return path
    if canonical(dataset) != _LEGACY_DATASET:
        return path
    legacy = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(legacy):
        logger.info(
            f"Reading pre-multi-dataset checkpoint {legacy} (no {path} yet); "
            f"the next save writes to {dataset_dir(dataset)}/"
        )
        return legacy
    return path


def save_state(global_model_state: dict, client_updates: list[dict],
               baseline_accuracy: float, dataset: str | None = None):
    """Persist Phase 1 results for ``dataset`` to disk."""
    torch.save(global_model_state, _write_path(_GLOBAL_FILE, dataset))
    torch.save(client_updates, _write_path(_CLIENTS_FILE, dataset))
    with open(_write_path(_BASELINE_FILE, dataset), "w") as f:
        json.dump({"baseline_accuracy": baseline_accuracy,
                   "dataset": canonical(dataset) if dataset else None}, f)


def load_state(dataset: str | None = None):
    """Load saved Phase-1 state for ``dataset``.

    Returns ``(global_state, client_updates, baseline_acc)`` or ``None`` when any
    of the three files is missing (a partial checkpoint is not resumable).
    """
    try:
        global_state = torch.load(
            _read_path(_GLOBAL_FILE, dataset), weights_only=False
        )
        client_updates = torch.load(
            _read_path(_CLIENTS_FILE, dataset), weights_only=False
        )
        with open(_read_path(_BASELINE_FILE, dataset)) as f:
            baseline = json.load(f)["baseline_accuracy"]
        return global_state, client_updates, baseline
    except FileNotFoundError:
        return None


def state_exists(dataset: str | None = None) -> bool:
    return os.path.exists(_read_path(_GLOBAL_FILE, dataset))


# ---------------------------------------------------------------------------
# Phase-2 SHARED FL state (the evolving global model + per-client benign
# weights). Phase 1 saves the *initial* baseline via save_state(); this saves
# the LIVE Phase-2 arms-race state at each checkpoint so a resume continues the
# shared model where it left off instead of rewinding to the Phase-1 baseline
# (a few KB for MNIST's ~970-param net, ~140 KB for the CIFAR-10 CNN).
# ---------------------------------------------------------------------------

def save_fl_state(fl_state: dict, dataset: str | None = None):
    """Persist the live Phase-2 FL state dict (see ``FLArmsRaceEnv.snapshot_fl_state``)."""
    torch.save(fl_state, _write_path(_FL_STATE_FILE, dataset))


def load_fl_state(dataset: str | None = None):
    """Return the saved Phase-2 FL state dict, or ``None`` if there is none."""
    try:
        return torch.load(_read_path(_FL_STATE_FILE, dataset), weights_only=False)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# RL training progress (for resuming the Phase-2 GRPO loop)
# ---------------------------------------------------------------------------

def save_progress(rounds_done: int, round_index: int | None = None,
                  controller: dict | None = None, dataset: str | None = None):
    """Persist Phase-2 resume state for ``dataset``.

    Backward compatible: ``rounds_done`` is always written. For a FULL resume we also
    persist ``round_index`` (the FL round-number counter, so round labels and
    ``logs/<dataset>/round_data`` continue across restarts instead of overwriting from
    the first Phase-2 round) and ``controller`` (the arms-race ``PhaseController``
    snapshot, so the learner/phase/streak resume instead of restarting at the first
    attacker phase).

    Counted PER DATASET, matching ``fl.simulation_rounds`` (a per-run budget) and the
    ``round_index`` that labels this dataset's FL rounds. The LoRA adapter is what
    carries learning across datasets, and it is saved separately and shared.
    """
    payload = {"rounds_done": int(rounds_done)}
    if round_index is not None:
        payload["round_index"] = int(round_index)
    if controller is not None:
        payload["controller"] = controller
    with open(_write_path(_PROGRESS_FILE, dataset), "w") as f:
        json.dump(payload, f)


def load_progress(dataset: str | None = None) -> dict:
    """Return the Phase-2 resume state as a dict:
    ``{"rounds_done": int, "round_index": int|None, "controller": dict|None}``.

    Old progress files that only hold ``rounds_done`` load fine (the new keys come
    back ``None`` → the caller falls back gracefully); a missing or corrupt file
    yields a fresh-start dict (``rounds_done=0``).
    """
    try:
        with open(_read_path(_PROGRESS_FILE, dataset)) as f:
            data = json.load(f)
        return {
            "rounds_done": int(data.get("rounds_done", 0)),
            "round_index": data.get("round_index"),
            "controller": data.get("controller"),
        }
    except (FileNotFoundError, ValueError, TypeError):
        return {"rounds_done": 0, "round_index": None, "controller": None}


def adapter_exists(path: str) -> bool:
    """True if a saved LoRA adapter directory looks present at ``path``.

    Adapters are NOT dataset-scoped — one attacker adapter continues training
    across datasets — so this takes an explicit path rather than a dataset.
    """
    return os.path.exists(os.path.join(path, "adapter_config.json"))
