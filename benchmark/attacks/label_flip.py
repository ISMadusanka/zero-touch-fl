"""Label flipping — the standard untargeted DATA-poisoning baseline
(Biggio et al., ICML 2012; Fang et al., USENIX Security 2020 Sec. 3; Tolpegin et
al., ESORICS 2020).

Every other attack in this panel is a *model*-poisoning attack: it edits the
weight vector a client submits. This one does not touch the weights at all. The
compromised clients train honestly — same optimizer, same epochs, same data — on
relabelled examples, and submit whatever that produces. The resulting update is a
legitimate gradient of a legitimate (wrong) objective, which is why it is the
hardest kind for a statistics-based defense to separate: there is nothing
anomalous about its norm, its direction relative to its own data, or its layer
profile.

It is included as a *different threat model*, not as a stronger attack — a real
adversary who only controls the training pipeline of some clients, rather than
their outgoing weights, is limited to exactly this. Expect it to be weaker than
the optimized model-poisoning attacks and much harder to detect.

Flip rules (``--labelflip-mode``):
    ``reverse``  ``y -> C-1-y``      the usual formulation; deterministic, maximal
                                     class displacement, and self-inverse
    ``next``     ``y -> (y+1) % C``  a milder rotation
    ``random``   ``y -> uniform over the other classes``, re-drawn per batch

Cost and consistency. This is the only attack that runs local training, so it adds
one SGD pass per compromised client per round. It also requires the HONEST updates
to be produced by the same process it uses — freshly trained from the round's
global rather than replayed from the Phase-1 checkpoint — or the comparison would
attribute the difference between "trained now" and "replayed" to the attack. The
runner therefore switches on per-round benign retraining whenever this attack is
in the panel, and says so.
"""
import logging

from benchmark.attacks.base import Attack

logger = logging.getLogger("benchmark")

MODES = ("reverse", "next", "random")


class _FlippedLoader:
    """Wraps a client's DataLoader, relabelling each batch on the way out.

    Deliberately a lazy wrapper rather than a rebuilt dataset: the underlying
    loader keeps its own shuffling and batching, so a label-flipped client sees
    exactly the data an honest one would, only mislabelled.
    """

    def __init__(self, loader, flip):
        self._loader = loader
        self._flip = flip

    def __iter__(self):
        for data, target in self._loader:
            yield data, self._flip(target)

    def __len__(self):
        return len(self._loader)


def make_flip(mode: str, n_classes: int, generator=None):
    """Return a ``targets -> flipped targets`` callable for ``mode``."""
    import torch
    c = int(n_classes)
    if mode == "reverse":
        return lambda y: (c - 1 - y).clamp_(0, c - 1)
    if mode == "next":
        return lambda y: (y + 1) % c

    if mode == "random":
        def flip(y):
            # Draw in [1, C-1] and add modulo C: uniform over the OTHER classes,
            # so a "random" flip never silently leaves a label untouched.
            offset = torch.randint(1, max(2, c), y.shape, generator=generator,
                                   dtype=y.dtype)
            return (y + offset) % c
        return flip
    raise ValueError(f"unknown label-flip mode {mode!r}; use one of {MODES}")


class LabelFlip(Attack):
    name = "label_flip"
    citation = "Biggio et al., ICML 2012 / Fang et al., USENIX Sec 2020"

    #: The runner reads this to force per-round benign retraining (see module docstring).
    needs_benign_retrain = True

    def __init__(self, client_loaders, lr: float, local_epochs: int,
                 device: str = "cpu", n_classes=None, mode: str = "reverse",
                 seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"unknown label-flip mode {mode!r}; use one of {MODES}")
        self.client_loaders = client_loaders
        self.lr = float(lr)
        self.local_epochs = int(local_epochs)
        self.device = device
        self.mode = mode
        self.seed = int(seed)
        self._n_classes = n_classes
        self._gen = None
        self._logged = False

    def reset(self) -> None:
        self._gen = None

    def _n_class(self) -> int:
        if self._n_classes is None:
            from data.feature_spec import active
            self._n_classes = int(active().n_classes)
        return int(self._n_classes)

    def _generator(self):
        import torch
        if self._gen is None:
            self._gen = torch.Generator(device="cpu")
            self._gen.manual_seed(self.seed)
        return self._gen

    def craft(self, ctx) -> dict:
        from clients.benign_client import BenignClient
        from model import build_model

        model = build_model()
        model.load_state_dict(ctx.global_weights)
        model.to(self.device)
        flip = make_flip(self.mode, self._n_class(), self._generator())
        if not self._logged:
            logger.info("label_flip: mode=%s over %d class(es), lr=%g, %d local "
                        "epoch(s)", self.mode, self._n_class(), self.lr,
                        self.local_epochs)
            self._logged = True

        out = {}
        for cid in ctx.poisoned_ids:
            cid = int(cid)
            client = BenignClient(client_id=cid,
                                  data_loader=_FlippedLoader(self.client_loaders[cid], flip),
                                  lr=self.lr, local_epochs=self.local_epochs,
                                  device=self.device)
            out[cid] = client.train(model).weights
        return out
