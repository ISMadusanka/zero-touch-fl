"""Dataset-aware run configuration: one resolution path for every entry point.

``main.py``, ``benchmark/run_benchmark.py``, ``monitor.py`` and
``visualize_rounds.py`` all have to answer the same three questions the same way,
or a training run and the benchmark that evaluates it will silently disagree:

1. **Which dataset is this run?**  ``--dataset`` beats ``data.dataset`` in the
   config, which beats :data:`data.datasets.DEFAULT_DATASET`.
2. **What are this dataset's hyperparameters?**  A ``datasets:`` block in the
   config carries per-dataset overrides (CIFAR-10 needs a different learning rate
   and more Phase-1 rounds than MNIST) that are deep-merged over the shared
   ``fl:`` / ``data:`` / ``defense:`` blocks.
3. **Where do this dataset's artifacts live?**  ``checkpoints/<dataset>/`` and
   ``logs/<dataset>/``.

The one thing deliberately NOT scoped per dataset is the **LoRA adapters**
(``checkpoints/attacker_adapter``): the same LLM keeps fine-tuning from its last
checkpoint no matter which dataset the current run trains on. See
:func:`adapter_paths`.
"""

import copy
import logging
import os

from data.datasets import DATASETS, DEFAULT_DATASET, canonical, resolve

logger = logging.getLogger(__name__)

#: Roots under which per-dataset artifact directories are created.
CHECKPOINT_ROOT = "checkpoints"
LOG_ROOT = "logs"

#: Config sections a ``datasets.<name>`` block may override. Anything else in it
#: is a typo, and silently ignoring it would mean training with settings the user
#: believes are in effect.
OVERRIDABLE_SECTIONS = ("fl", "data", "attack", "defense", "rl", "curriculum")


# ---------------------------------------------------------------------------
# Dataset selection + per-dataset config overrides
# ---------------------------------------------------------------------------

def resolve_dataset(cfg: dict, cli_dataset=None) -> str:
    """Canonical dataset name for this run: CLI > ``data.dataset`` > default."""
    if cli_dataset:
        return canonical(cli_dataset)
    return canonical((cfg.get("data") or {}).get("dataset") or DEFAULT_DATASET)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; ``override`` wins on leaves. Returns a new dict."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _check_data_dir(cfg: dict, dataset: str) -> None:
    """Fail loudly if ``data.data_dir`` points at a DIFFERENT dataset's cache.

    ``data_dir: "./data/mnist_raw"`` left over from a single-dataset config would
    otherwise make ``--dataset cifar10`` download CIFAR-10 into the MNIST folder —
    which works, so nothing would look wrong until two runs' artifacts were
    tangled together. ``null`` (the shipped value) means "use this dataset's own
    default directory" and is always safe.
    """
    data_dir = (cfg.get("data") or {}).get("data_dir")
    if not data_dir:
        return
    norm = os.path.normpath(str(data_dir))
    for other in DATASETS.values():
        if other.name != dataset and norm == os.path.normpath(other.data_dir):
            raise ValueError(
                f"data.data_dir is {data_dir!r}, which is the default cache for the "
                f"'{other.name}' dataset, but this run is '{dataset}'. Set "
                f"data.data_dir to null (use the dataset's own directory) or give "
                f"'{dataset}' its own path under datasets.{dataset}.data.data_dir."
            )


def apply_dataset(cfg: dict, cli_dataset=None) -> tuple[dict, str]:
    """Return ``(resolved_config, dataset_name)``.

    Deep-merges ``cfg["datasets"][<name>]`` over the shared sections (so the
    per-dataset value WINS over the top-level one), pins ``data.dataset`` to the
    canonical name, and defaults ``data.data_dir`` to the dataset's own directory.
    The input config is not mutated.
    """
    dataset = resolve_dataset(cfg, cli_dataset)
    spec = resolve(dataset)
    out = copy.deepcopy(cfg)

    overrides = (out.get("datasets") or {}).get(dataset) or {}
    unknown = [k for k in overrides if k not in OVERRIDABLE_SECTIONS]
    if unknown:
        raise ValueError(
            f"datasets.{dataset} may only override {list(OVERRIDABLE_SECTIONS)}; "
            f"got unknown section(s) {unknown}"
        )
    for section, values in overrides.items():
        if not isinstance(values, dict):
            raise ValueError(f"datasets.{dataset}.{section} must be a mapping")
        if section == "rl" and "adapter_paths" in values:
            # Per-dataset adapter paths would give each dataset its own policy —
            # the exact opposite of the continual fine-tuning this switch exists
            # for. Legal (someone may want isolated ablations), but never silent.
            logger.warning(
                f"datasets.{dataset}.rl.adapter_paths overrides the LoRA checkpoint "
                f"location for this dataset only. The adapter is normally SHARED so "
                f"training continues across datasets; with a per-dataset path, a "
                f"'{dataset}' run trains a SEPARATE policy. Remove it unless that is "
                f"deliberate."
            )
        out[section] = _deep_merge(out.get(section) or {}, values)
        # Logging may not be configured yet (main.py resolves the dataset first, to
        # find the log dir), so this is a best-effort trace; the authoritative
        # report is describe_run(), which the entry points log once it is.
        logger.info(f"[{dataset}] config override {section}: {dict(values)}")

    out.setdefault("data", {})
    out["data"]["dataset"] = dataset
    _check_data_dir(out, dataset)
    if not out["data"].get("data_dir"):
        out["data"]["data_dir"] = spec.data_dir
    out["data"].setdefault("n_classes", spec.n_classes)
    return out, dataset


def dataset_of(cfg: dict) -> str:
    """The canonical dataset name recorded in an already-resolved config."""
    return canonical((cfg.get("data") or {}).get("dataset") or DEFAULT_DATASET)


def dataset_overrides(cfg: dict, dataset: str) -> dict:
    """The raw ``datasets.<dataset>`` block (``{}`` when there is none).

    Survives :func:`apply_dataset` (which copies the whole config), so a caller
    that configures logging *after* resolving the dataset can still report what
    was overridden.
    """
    return (cfg.get("datasets") or {}).get(canonical(dataset)) or {}


# ---------------------------------------------------------------------------
# Per-dataset artifact layout
# ---------------------------------------------------------------------------

def checkpoint_dir(dataset: str) -> str:
    """Where this dataset's FL state lives: global model, per-client weights,
    baseline accuracy, live Phase-2 FL state, RL progress.

    Per-dataset because none of it is portable — a CIFAR-10 conv state_dict
    cannot be loaded into the MNIST MLP, and the round counters/phase controller
    describe one dataset's arms race.
    """
    return os.path.join(CHECKPOINT_ROOT, canonical(dataset))


def log_dir(dataset: str) -> str:
    """Root for this dataset's logs (``logs/<dataset>/``).

    Per-dataset because round numbering restarts per dataset: interleaving two
    datasets in one ``rounds.jsonl`` would give ``monitor.py`` and
    ``visualize_rounds.py`` duplicate round numbers from incomparable runs.
    """
    return os.path.join(LOG_ROOT, canonical(dataset))


def run_paths(dataset: str) -> dict:
    """Every per-dataset path an entry point needs, in one dict."""
    logs = log_dir(dataset)
    return {
        "dataset": canonical(dataset),
        "checkpoint_dir": checkpoint_dir(dataset),
        "log_dir": logs,
        "system_log": os.path.join(logs, "system.log"),
        "debug_dir": logs,
        "round_data_dir": os.path.join(logs, "round_data"),
        "round_log": os.path.join(logs, "round_data", "rounds.jsonl"),
        "metrics_dir": os.path.join(logs, "metrics"),
        "benchmark_dir": os.path.join(logs, "benchmark"),
        "visualizations_dir": os.path.join(logs, "visualizations"),
        "monitor_dir": os.path.join(logs, "monitor"),
    }


def adapter_paths(cfg: dict) -> dict:
    """The LoRA adapter directories — **shared across datasets, on purpose**.

    The FL model is dataset-specific; the LLM policy is not. It observes
    dimensionless per-layer statistics and emits operator plans, both of which are
    architecture-agnostic, so one attacker adapter can (and should) keep learning
    across MNIST and CIFAR-10 runs. Scoping the adapter per dataset would restart
    the policy from the base model on every switch, which is exactly the opposite
    of continual fine-tuning.

    Returns ``rl.adapter_paths`` from the config, falling back to the defaults.
    """
    paths = ((cfg.get("rl") or {}).get("adapter_paths")) or {}
    return {
        "attacker": paths.get("attacker", os.path.join(CHECKPOINT_ROOT, "attacker_adapter")),
        "defender": paths.get("defender", os.path.join(CHECKPOINT_ROOT, "defender_adapter")),
    }


def describe_run(cfg: dict, dataset: str) -> str:
    """One-line summary of the resolved dataset settings, for the run header.

    Includes the ``datasets.<name>`` overrides that were applied, because
    ``main.py`` has to resolve the dataset *before* it can configure logging (the
    log directory is per-dataset) — so this is where those show up.
    """
    spec = resolve(dataset)
    fl = cfg.get("fl", {})
    data = cfg.get("data", {})
    c, h, w = spec.input_shape
    overrides = dataset_overrides(cfg, dataset)
    return (f"dataset={spec.name} ({c}x{h}x{w}, {spec.n_classes} classes) "
            f"data_dir={data.get('data_dir')} iid={data.get('iid')} "
            f"training_rounds={fl.get('training_rounds')} lr={fl.get('lr')} "
            f"local_epochs={fl.get('local_epochs')} batch_size={fl.get('batch_size')}"
            + (f" | datasets.{spec.name} overrides applied: {overrides}"
               if overrides else " | no per-dataset overrides"))
