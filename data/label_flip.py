"""Label-flipping poison at the DATA layer.

This is the project's ONLY attack. A poisoned client is not handed doctored
weights — it trains honestly, with honest SGD, on a dataset whose LABELS have
been corrupted. Whatever weight update comes out is the attack, and the defense
sees exactly what it would see from any other client: a weight delta.

That is the whole point of moving here from the old weight-space operator DSL.
A weight-space edit is a post-hoc transform of a vector the client already
computed, so its magnitude is a free parameter and the attacker can dial it to
whatever the defense tolerates. A label flip cannot be dialled that way: the only
knob is *how many* of the client's own examples carry a wrong label, and the
resulting update is always a real gradient of a real (if wrong) objective. It
lands inside the honest update distribution by construction, which is what makes
it a meaningful test of a detector.

FLIP RULE — ``symmetric``: ``y -> (n_classes - 1) - y``. On MNIST that is
``0<->9, 1<->8, ... 4<->5``. This is the standard label-flipping attack in the
FL-poisoning literature (Fang et al., USENIX Security 2020; Tolpegin et al.,
ESORICS 2020): deterministic, maximally wrong under an ordinal reading of the
label space, and it perturbs every class rather than one pair, so no class is
left as a clean anchor.

WHICH examples get flipped is a seeded, deterministic function of the round and
the client (:func:`choose_flip_positions`), so a resumed run reproduces exactly
the poison an interrupted one would have sent.
"""

import logging
import random

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

#: The only flip rule. Named so configs read as a choice of *rule*, not a choice
#: of attack — there is one attack, and it is label flipping.
FLIP_SCHEME = "symmetric"


def flip_label(label, n_classes: int = 10):
    """``y -> (n_classes - 1) - y`` — the symmetric flip.

    Involutive (flipping twice is the identity) and fixed-point-free for an even
    ``n_classes``, so every flipped example really does carry a wrong label.

    The RETURN TYPE mirrors the input. Datasets disagree about what a label is —
    torchvision's MNIST yields a Python ``int``, a ``TensorDataset`` yields a
    0-dim tensor — and PyTorch's ``default_collate`` refuses to stack a batch that
    mixes the two. Returning a bare ``int`` therefore crashed collation on any
    round where some of a batch's labels were flipped and some were not, which is
    every partially-poisoned round.
    """
    flipped = int(n_classes) - 1 - int(label)
    if isinstance(label, torch.Tensor):
        return torch.full_like(label, flipped)
    return flipped


def choose_flip_positions(n_samples: int, n_flip: int, seed: int) -> frozenset[int]:
    """Which POSITIONS of a client's round dataset get a flipped label.

    Positions are indices into the client's per-round ``Subset``, not into MNIST,
    so they stay valid as ``data.round_sampler`` re-cuts the shard each round.

    Deterministic in ``seed`` and drawn from a dedicated ``random.Random`` rather
    than the ambient RNG: the ladder in :mod:`agents.label_flip_attacker` re-plans
    the same round after a resume, and it must select the same examples or the
    "same flip count" round is not the same round.
    """
    n_samples = max(0, int(n_samples))
    n_flip = max(0, min(int(n_flip), n_samples))
    if n_flip == 0:
        return frozenset()
    if n_flip == n_samples:
        return frozenset(range(n_samples))
    return frozenset(random.Random(int(seed)).sample(range(n_samples), n_flip))


class LabelFlippedDataset(Dataset):
    """A view of ``base`` whose labels are flipped at ``flip_positions``.

    Wraps rather than copies: MNIST tensors stay shared with the underlying
    dataset, so a poisoned client costs no extra memory over an honest one. The
    inputs are untouched — only the target changes, which is what makes this a
    *label*-flipping attack rather than a feature-space one.
    """

    def __init__(self, base, flip_positions, n_classes: int = 10):
        self.base = base
        self.flip_positions = frozenset(int(i) for i in flip_positions)
        self.n_classes = int(n_classes)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index):
        x, y = self.base[index]
        if index in self.flip_positions:
            return x, flip_label(y, self.n_classes)
        return x, y

    @property
    def n_flipped(self) -> int:
        return len(self.flip_positions)


def build_flipped_loader(loader: DataLoader, n_flip: int, *, seed: int,
                         n_classes: int = 10) -> DataLoader:
    """``loader`` with ``n_flip`` of its examples relabelled by the symmetric flip.

    Batch size and shuffling match the honest loader, so the poisoned client runs
    the SAME local-training procedure over the SAME examples in the same-sized
    batches as it would have honestly. The only difference on the wire is the
    labels it optimized against — which is the attack.

    ``n_flip`` is clamped to the dataset size, so a ladder level that asks for
    more examples than the client holds this round simply flips all of them.

    The shuffle runs off a generator seeded from ``seed`` rather than the ambient
    RNG, so the poisoned client's whole local training — which examples are
    mislabelled AND the order SGD visits them in — is a pure function of the round
    and the run seed. Without that, a resumed run would flip the same examples but
    walk them in a different order, and the "same ladder level" round would not
    reproduce the update it produced before the interruption.
    """
    base = loader.dataset
    n_samples = len(base)
    positions = choose_flip_positions(n_samples, n_flip, seed)
    generator = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    return DataLoader(
        LabelFlippedDataset(base, positions, n_classes=n_classes),
        batch_size=loader.batch_size or 1,
        shuffle=True,
        generator=generator,
    )
