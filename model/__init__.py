"""Model factory — maps a dataset name to the FL model every component shares.

The FL model is the schema both LLMs operate over (the attacker addresses its
layers by ``state_dict`` key; the defender reads per-layer statistics of it), and
it is necessarily dataset-specific: a 49-input MLP cannot consume 3x32x32 images.
Everything else in the system is architecture-agnostic, so this one function is
the whole of the model's dataset dependency.

    from model import build_model
    net = build_model("cifar10")

``server.FedServer`` is the only production caller; tests and the defenses reach
the model through it.
"""

import torch.nn as nn

from data.datasets import DEFAULT_DATASET, canonical, resolve
from model.cifar_net import Cifar10Net
from model.mnist_net import MnistNet, count_parameters

__all__ = ["MnistNet", "Cifar10Net", "build_model", "count_parameters"]


#: dataset name -> zero-argument model constructor.
_BUILDERS = {
    "mnist": MnistNet,
    "cifar10": Cifar10Net,
}


def build_model(dataset: str = DEFAULT_DATASET) -> nn.Module:
    """Instantiate the FL model for ``dataset`` (alias/case tolerant).

    Raises ``ValueError`` for an unknown dataset, and ``NotImplementedError`` for
    a dataset that is registered in ``data.datasets`` but has no model here — the
    two failure modes are distinct on purpose, because the second one tells you
    exactly what to add.
    """
    name = canonical(dataset)
    builder = _BUILDERS.get(name)
    if builder is None:                     # registered dataset, no architecture yet
        spec = resolve(name)
        raise NotImplementedError(
            f"no FL model registered for dataset '{name}' ({spec.description}); "
            f"add one to model/__init__.py:_BUILDERS"
        )
    return builder()
