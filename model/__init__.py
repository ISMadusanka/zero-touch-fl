"""The FL model shared by every component.

The FL model is the schema both LLMs operate over — the attacker addresses its
layers by ``state_dict`` key, the defender reads per-layer statistics of it — and
it is necessarily dataset-specific: an image CNN cannot consume a row of Argus
flow features. ``model/nidd_net.py`` is that model for 5G-NIDD, and
:func:`build_model` is the whole of the system's dependency on its shape.

    from model import build_model
    net = build_model()                      # sized from the active feature spec

``server.FedServer`` is the only production caller; tests and the defenses reach
the model through it.
"""

from model.nidd_net import (
    DEFAULT_HIDDEN, NiddNet, build_model, count_parameters, default_hidden,
    set_default_hidden,
)

__all__ = ["NiddNet", "build_model", "count_parameters", "DEFAULT_HIDDEN",
           "default_hidden", "set_default_hidden"]
