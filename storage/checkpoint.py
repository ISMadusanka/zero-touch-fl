"""Checkpoint management — save/load system state between phases."""

import json
import os
import torch

CHECKPOINT_DIR = "checkpoints"


def _ensure_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


#: The feature spec the saved Phase-1 model was trained under.
_SPEC_FILE = "feature_spec.json"


def saved_spec():
    """The :class:`~data.feature_spec.FeatureSpec` Phase 1 was trained under, or ``None``.

    ``None`` for a checkpoint written before specs were recorded — treated as
    "unknown provenance", not as a mismatch.
    """
    from data.feature_spec import FeatureSpec

    return FeatureSpec.from_json(os.path.join(CHECKPOINT_DIR, _SPEC_FILE))


def shape_mismatch(global_weights: dict) -> str | None:
    """Describe why the saved Phase 1 must not be reused, or ``None`` if it is fine.

    On MNIST the model's shape was a constant, so a checkpoint either existed or
    did not. 5G-NIDD's follows the PREPROCESSING — ``data.n_features``,
    ``data.label_mode`` (9 classes or 2), and ``model.hidden`` — so it can
    legitimately differ between two runs of the same repo. Two failure modes
    follow, and both are caught here so the fix (re-run Phase 1) is automatic:

    * **Unloadable.** Different layer names or tensor shapes. Checking up front
      turns "RuntimeError: size mismatch for net.0.weight", raised deep inside a
      Phase-2 resume, into a warning and a fresh Phase 1.
    * **Loadable but wrong.** The shapes match and the weights load, yet the model
      was trained on different data — most dangerously a ``data.source:
      synthetic`` smoke run, whose 32-feature/9-class model is shape-identical to a
      real one. Resuming from it would make every accuracy, drop and defense number
      in the run a measurement of generated traffic. The recorded spec is what
      distinguishes them.

    Callers: ``main.py`` and ``benchmark/run_benchmark.py``, right after
    :func:`load_state`.
    """
    from data.feature_spec import active
    from model import build_model            # deferred: keeps this module torch-only

    expected = build_model().state_dict()
    saved = global_weights or {}
    missing = sorted(set(expected) - set(saved))
    extra = sorted(set(saved) - set(expected))
    if missing or extra:
        return (f"has different layers than the current model "
                f"(missing {missing or 'none'}, unexpected {extra or 'none'})")
    bad = [f"{k}: saved {tuple(saved[k].shape)} vs model {tuple(v.shape)}"
           for k, v in expected.items()
           if tuple(saved[k].shape) != tuple(v.shape)]
    if bad:
        return "was trained at a different feature/class count — " + "; ".join(bad)

    # Shapes agree; now the provenance the shapes cannot reveal.
    was, now = saved_spec(), active()
    if was is not None and was.source != now.source:
        return (f"was trained on {was.source!r} data but this run loads {now.source!r} "
                f"— a synthetic-trained baseline would invalidate every number in the run")
    if was is not None and was.feature_names and now.feature_names \
            and was.feature_names != now.feature_names:
        return (f"was trained on a different feature set "
                f"({len(was.feature_names)} columns, first differing entry "
                f"{next((a for a, b in zip(was.feature_names, now.feature_names) if a != b), '?')!r})")
    return None


def save_state(global_model_state: dict, client_updates: list[dict], baseline_accuracy: float):
    """Persist Phase 1 results to disk, including the feature spec it was trained
    under so a later run can tell whether reusing it is valid (see
    :func:`shape_mismatch`)."""
    from data.feature_spec import active

    _ensure_dir()
    torch.save(global_model_state, os.path.join(CHECKPOINT_DIR, "global_model.pt"))
    torch.save(client_updates, os.path.join(CHECKPOINT_DIR, "client_updates.pt"))
    active().to_json(os.path.join(CHECKPOINT_DIR, _SPEC_FILE))
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
# (the global model is a ~681-param NiddNet, so this is a few KB).
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
                  controller: dict | None = None, curriculum: dict | None = None):
    """Persist Phase-2 resume state.

    Backward compatible: ``rounds_done`` is always written. For a FULL resume we also
    persist ``round_index`` (the FL round-number counter, so round labels and
    ``logs/round_data`` continue across restarts instead of overwriting from the first
    Phase-2 round), ``controller`` (the arms-race ``PhaseController`` snapshot, so
    the learner/phase/streak resume instead of restarting at the first attacker phase),
    and ``curriculum`` (the :class:`rl.curriculum.TrainingCurriculum` position, so the
    (defense, #poisoners) sweep continues mid-block instead of restarting at the first
    algorithm with one poisoner on every restart).
    """
    _ensure_dir()
    payload = {"rounds_done": int(rounds_done)}
    if round_index is not None:
        payload["round_index"] = int(round_index)
    if controller is not None:
        payload["controller"] = controller
    if curriculum is not None:
        payload["curriculum"] = curriculum
    with open(os.path.join(CHECKPOINT_DIR, _PROGRESS_FILE), "w") as f:
        json.dump(payload, f)


def load_progress() -> dict:
    """Return the Phase-2 resume state as a dict: ``{"rounds_done": int,
    "round_index": int|None, "controller": dict|None, "curriculum": dict|None}``.

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
            "curriculum": data.get("curriculum"),
        }
    except (FileNotFoundError, ValueError, TypeError):
        return {"rounds_done": 0, "round_index": None, "controller": None,
                "curriculum": None}


def adapter_exists(path: str) -> bool:
    """True if a saved LoRA adapter directory looks present at ``path``."""
    return os.path.exists(os.path.join(path, "adapter_config.json"))
