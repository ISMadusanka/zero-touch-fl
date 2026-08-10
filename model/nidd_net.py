"""Tiny fully-connected network for 5G-NIDD — 681 trainable parameters by default.

Architecture (two learnable layers, no convolution and no pooling):

    Linear(32, 16) -> ReLU -> Linear(16, 9)
    params: (32*16 + 16) + (16*9 + 9) = 528 + 153 = 681

Why not the old image model
---------------------------
The MNIST model was ``AvgPool2d(4) -> Flatten -> Linear(49,16) -> ReLU ->
Linear(16,10)``: 970 parameters, and a front end that only makes sense on a 2-D
pixel grid. Average-pooling neighbouring *flow features* would be meaningless —
column 7 and column 8 of an Argus record ("source TTL", "destination TTL") have
no spatial adjacency to exploit — so the pooling stage is gone and the model
consumes the preprocessed feature vector directly. That also removes the last
image-shaped assumption from the system: clients now receive ``(batch, 32)``
tensors rather than ``(batch, 1, 28, 28)``.

Parameter count
---------------
681 < 970, so this is strictly smaller than the model it replaces, as intended.
The count is a config constant rather than a property of the CSV, because
``data.n_features`` selects exactly K columns (see ``data/nidd_loader.py``):

    params = (K + 1) * H + (H + 1) * C

H=16 is the knee of the curve, not an arbitrary pick. Measured over 45 honest
FedAvg rounds, 20 clients, K=32 (synthetic source, so read the DIFFERENCES rather
than the absolute numbers):

    H=8    345 params   0.7655
    H=16   681 params   0.8298      <- shipped
    H=32 1,353 params   0.8302
    H=64 2,697 params   0.8379

Doubling the parameter count buys 0.0004; halving it costs 6.4 points. So 681 is
about the smallest this model gets before accuracy starts paying for it — which
matters here because accuracy IS the reward signal, and a capacity-starved model
has less accuracy left for the attacker to take away.

Two layers rather than three is deliberate — depth buys little on tabular flow
data, and every extra layer is another tensor the attacker's operator DSL can
target and every defense has to flatten.

State_dict keys are ``net.0.{weight,bias}`` and ``net.2.{weight,bias}``, i.e. two
logical layers under ``detector.features.layer_groups``' prefix grouping, exactly
as the two-layer MNIST model had. Nothing addresses layers by hardcoded name —
the attacker is told the live keys and shapes in its prompt, DeFL groups by
prefix, and every other defense flattens the whole model — so the renaming from
``net.2``/``net.4`` is transparent to the round loop.

**No BatchNorm and no Dropout, deliberately.** BN running statistics live in the
state_dict, so they would be FedAvg-averaged, handed to the attacker's operators
and clamped like weights — and ``num_batches_tracked`` is an int64 counter that
has no meaningful "scale by 1.5". Keeping the model BN-free keeps every tensor in
the state_dict a genuine learnable parameter, which is the contract the attack DSL
and the defenses assume. (Feature standardization, which is what BN would be
approximating here, is done once in the loader instead.)
"""

import torch.nn as nn

from data.feature_spec import FeatureSpec, active

#: Hidden width when the config does not say. With the default 32 features and 9
#: classes this is the 681-parameter model described above.
DEFAULT_HIDDEN = 16

_default_hidden = DEFAULT_HIDDEN


def set_default_hidden(hidden: int | None) -> int:
    """Install the process-wide hidden width from ``model.hidden`` in the config.

    ``FedServer`` is constructed in four places that have no access to the config
    (``rl/env.py``, ``benchmark/{phase1,harness}.py``,
    ``benchmark/defenses/fltrust.py``), so the knob is installed once at startup
    by ``main.py`` / ``benchmark/run_benchmark.py`` rather than threaded through
    the round loop — the same reason the feature spec is published rather than
    passed. ``None`` leaves the default alone.
    """
    global _default_hidden
    if hidden:
        _default_hidden = max(1, int(hidden))
    return _default_hidden


def default_hidden() -> int:
    """The hidden width :func:`build_model` uses when not given one."""
    return _default_hidden


class NiddNet(nn.Module):
    """Linear -> ReLU -> Linear over preprocessed 5G-NIDD flow features.

    Args:
        input_dim: Number of selected flow features (``data.n_features``).
        n_classes: Attack classes including benign (9 for the full dataset).
        hidden: Hidden layer width.

    Defaults come from :data:`data.feature_spec.DEFAULT_SPEC` so ``NiddNet()``
    works before any data is loaded (unit tests, ``--help``); a real run builds it
    from the spec the loader published. Accepts ``(batch, input_dim)`` and also
    ``(batch, 1, input_dim)`` — the leading ``Flatten`` costs no parameters and
    makes the model tolerant of an extra singleton dimension.
    """

    def __init__(self, input_dim: int | None = None, n_classes: int | None = None,
                 hidden: int | None = None):
        super().__init__()
        spec = active()
        self.input_dim = int(input_dim if input_dim is not None else spec.input_dim)
        self.n_classes = int(n_classes if n_classes is not None else spec.n_classes)
        self.hidden = int(hidden) if hidden else default_hidden()
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden),    # 0
            nn.ReLU(),                                 # 1
            nn.Linear(self.hidden, self.n_classes),    # 2
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(spec: FeatureSpec | None = None, hidden: int | None = None) -> NiddNet:
    """The FL model every component shares, sized from the active feature spec.

    ``server.FedServer`` is the only production caller; tests and the defenses
    reach the model through it. Passing ``spec`` explicitly is preferred where the
    caller has one; omitting it uses whatever ``data.nidd_loader`` published (or
    the documented default, offline).
    """
    spec = spec if spec is not None else active()
    return NiddNet(input_dim=spec.input_dim, n_classes=spec.n_classes,
                   hidden=int(hidden) if hidden else default_hidden())
