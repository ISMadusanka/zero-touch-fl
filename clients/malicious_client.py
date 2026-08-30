"""Label-flipping client — honest training on dishonest labels.

The ONLY difference from :class:`clients.benign_client.BenignClient` is the
dataset it optimizes against: the same examples, in the same batch size, for the
same number of local epochs, at the same learning rate — but with ``n_flip`` of
the labels replaced by the symmetric flip (see :mod:`data.label_flip`).

Everything downstream of that is genuinely honest. There is no post-hoc edit of
the weights, no scaling, no injected noise: the update the server receives is a
real SGD trajectory, so it carries the statistical signature of ordinary local
training and the defense has to detect the attack from what the wrong labels did
to the gradients, not from an artefact of how the poison was applied.
"""

from data.label_flip import build_flipped_loader
from clients.benign_client import BenignClient


class LabelFlipClient(BenignClient):
    """A client whose local labels are flipped before it trains.

    ``n_flip`` is set per round by the attacker's ladder (see
    :class:`agents.label_flip_attacker.LabelFlipAttacker`), and ``flip_seed``
    fixes WHICH of the round's examples are hit — both deterministic, so a
    resumed run reproduces the same poison.
    """

    def __init__(self, client_id: int, data_loader, lr: float, local_epochs: int,
                 device: str, n_classes: int = 10):
        super().__init__(client_id, data_loader, lr, local_epochs, device)
        self.n_classes = int(n_classes)
        self.n_flip = 0
        self.flip_seed = 0

    def set_flip(self, n_flip: int, flip_seed: int) -> None:
        """Set this round's poison: how many labels to flip, and which."""
        self.n_flip = max(0, int(n_flip))
        self.flip_seed = int(flip_seed)

    def train(self, global_model):
        """Train on the flipped dataset and report the poison in the metadata.

        The metadata is the round's ground truth about how hard the attack pushed
        (``n_flipped`` / ``flip_fraction``), which the round log carries so a run's
        attack strength can be read back without re-deriving it from the ladder.
        ``train_accuracy`` here is measured against the FLIPPED labels, so it is
        the client's fit to its poisoned objective — expect it to be high even
        though the model is being pushed away from the true task.
        """
        honest_loader = self.data_loader
        n_samples = len(honest_loader.dataset)
        n_flip = min(self.n_flip, n_samples)
        if n_flip > 0:
            self.data_loader = build_flipped_loader(
                honest_loader, n_flip, seed=self.flip_seed, n_classes=self.n_classes)
        try:
            update = super().train(global_model)
        finally:
            # Always restore the honest loader: the env keeps one client object per
            # id across rounds, and leaving a poisoned loader attached would carry
            # this round's flip set into the next round's plan.
            self.data_loader = honest_loader
        update.metadata.update({
            "poisoned": n_flip > 0,
            "attack": "label_flip",
            "n_flipped": n_flip,
            "n_local_samples": n_samples,
            "flip_fraction": (n_flip / n_samples) if n_samples else 0.0,
        })
        return update
