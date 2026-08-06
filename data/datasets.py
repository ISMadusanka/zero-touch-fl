"""Dataset registry — the single place that knows what a dataset *is*.

Everything dataset-dependent in the system resolves a name through here:

    data/loaders.py      how to download it, normalize it, partition it
    model/__init__.py    which FL model architecture the clients train
    core/run_config.py   which config overrides / checkpoint dir / log dir apply
    main.py, benchmark/  the ``--dataset`` CLI choices

Adding a third dataset is one :class:`DatasetSpec` entry below plus one model
builder in ``model/__init__.py`` — no change to the round loop, the attacker's
operator DSL, the reward, or any defense (all of those are architecture-agnostic:
they only ever see a ``state_dict``).

**This module deliberately imports nothing heavy** (no torch, no torchvision), so
``--help`` and argparse validation work on a box without a deep-learning stack;
``torchvision.datasets`` is resolved lazily by name in ``data/loaders.py``.
"""

from dataclasses import dataclass

#: Used when neither the CLI nor the config names a dataset.
DEFAULT_DATASET = "mnist"


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the system needs to know about one dataset.

    ``torchvision_name`` is the attribute looked up on ``torchvision.datasets``
    (kept as a string so this module stays import-light). ``mean``/``std`` are the
    standard per-channel normalization constants for that dataset.
    """

    name: str
    torchvision_name: str
    n_classes: int
    in_channels: int
    image_size: int
    mean: tuple
    std: tuple
    data_dir: str
    description: str

    @property
    def input_shape(self) -> tuple:
        """(C, H, W) of one example — what the model's first layer must accept."""
        return (self.in_channels, self.image_size, self.image_size)


_SPECS = (
    DatasetSpec(
        name="mnist",
        torchvision_name="MNIST",
        n_classes=10,
        in_channels=1,
        image_size=28,
        mean=(0.1307,),
        std=(0.3081,),
        data_dir="./data/mnist_raw",
        description="28x28 grayscale handwritten digits (60k train / 10k test)",
    ),
    DatasetSpec(
        name="cifar10",
        torchvision_name="CIFAR10",
        n_classes=10,
        in_channels=3,
        image_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        data_dir="./data/cifar10_raw",
        description="32x32 RGB natural images in 10 classes (50k train / 10k test)",
    ),
)

#: Canonical name -> spec.
DATASETS = {spec.name: spec for spec in _SPECS}

#: Valid ``--dataset`` values (canonical spellings only; aliases below are also
#: accepted but are not advertised in ``--help``).
DATASET_NAMES = tuple(DATASETS)

#: Spellings that turn up in commands and branch names, mapped to the canonical
#: id. Accepting them costs nothing and stops a typo from silently running the
#: wrong dataset (or aborting a long job at argparse).
_ALIASES = {
    "cifar": "cifar10",
    "cifar-10": "cifar10",
    "cifar_10": "cifar10",
    "cifar10": "cifar10",
    "ciffar": "cifar10",
    "ciffar10": "cifar10",
    "ciffar-10": "cifar10",
    "ciffar_10": "cifar10",
}


def canonical(name) -> str:
    """Normalize a user-supplied dataset name (case/alias tolerant).

    Raises ``ValueError`` naming the valid choices — never returns a name that
    :func:`resolve` would then reject.
    """
    key = str(name or DEFAULT_DATASET).strip().lower().replace(" ", "")
    key = _ALIASES.get(key, key)
    if key not in DATASETS:
        raise ValueError(
            f"unknown dataset {name!r}; available: {', '.join(DATASET_NAMES)}"
        )
    return key


def resolve(name) -> DatasetSpec:
    """The :class:`DatasetSpec` for ``name`` (alias/case tolerant)."""
    return DATASETS[canonical(name)]


def describe(name) -> str:
    """One-line human description, for logs and ``--help``."""
    spec = resolve(name)
    c, h, w = spec.input_shape
    return (f"{spec.name}: {spec.description} — input {c}x{h}x{w}, "
            f"{spec.n_classes} classes")
