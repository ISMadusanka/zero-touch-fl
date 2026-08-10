"""Per-round local training data for Phase 2's simulated rounds.

Phase 2 no longer advances the federated model: every attacker-learning round
hands the clients the SAME frozen Phase-1 global (see
``FLArmsRaceEnv.freeze_global``). If the clients also trained on the same
examples every round, the honest updates would differ only by SGD shuffle order,
so the attacker would face one static problem restated N times — nothing to
generalize over, and a clean counterfactual that barely moves.

This module gives each client a DIFFERENT slice of its own shard each round, so
"the clients train with new data" is literally true: the local dataset the
poison has to hide among changes from round to round while the starting point
stays fixed. The client's data DISTRIBUTION is untouched — a slice is drawn from
that client's own shard, so the non-IID skew ``partition_noniid_fltrust`` built
is preserved.

Both refresh modes are STATELESS functions of the round index, so a resumed run
reproduces exactly the data the interrupted one would have used.
"""

import logging
import math
import random

from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


def _flatten(loader):
    """``(base_dataset, absolute indices)`` behind a client's loader.

    Client loaders are built as ``DataLoader(Subset(flows, shard))``; unwrapping
    to the base dataset + absolute index list lets us re-cut the shard every
    round without touching ``data.nidd_loader``'s partitioning.
    """
    ds = loader.dataset
    idx = list(range(len(ds)))
    while isinstance(ds, Subset):
        idx = [int(ds.indices[i]) for i in idx]
        ds = ds.dataset
    return ds, idx


class RoundDataSampler:
    """Fresh local training data for every client, every round.

    Modes:
      ``rotate``   — the shard is shuffled and cut into disjoint per-round
                     slices; round r gets slice r. Every example is used once
                     before any is reused, and the shard is re-shuffled (so the
                     slices are re-cut differently) each time it is exhausted.
      ``resample`` — an independent random draw from the shard each round, so
                     consecutive rounds may overlap but every round is an
                     unbiased sample of that client's distribution.
    """

    def __init__(self, client_loaders, *, fraction: float = 0.25,
                 mode: str = "rotate", batch_size: int = 64, seed: int = 0):
        if mode not in ("rotate", "resample"):
            raise ValueError(
                f"unknown fl.client_data_refresh mode {mode!r} — expected "
                f"'rotate', 'resample' or 'none'"
            )
        self.mode = mode
        self.seed = int(seed)
        self.batch_size = max(1, int(batch_size))
        self.fraction = min(1.0, max(1e-6, float(fraction)))

        self._base, self._shards = [], []
        for loader in client_loaders:
            base, idx = _flatten(loader)
            self._base.append(base)
            self._shards.append(idx)
        # Per client, because a non-IID partition gives clients unequal shards — and on
        # 5G-NIDD they are unequal by a wide margin: the FLTrust partition builds one
        # group per class, and the two dominant classes (Benign ~39% of flows, UDPFlood
        # ~38%) swell their groups, giving a measured 3.6x spread across the 20 clients
        # (2510 / 2946 / 9078 examples at min / median / max). A single federation-wide
        # slice size would over-draw the small shards and starve the large ones.
        self._per_round = [max(1, min(len(s), int(round(self.fraction * len(s)))))
                           for s in self._shards]
        # Disjoint slices a shard yields before it is re-shuffled (rotate only).
        self._slices = [max(1, len(s) // k)
                        for s, k in zip(self._shards, self._per_round)]

    # ------------------------------------------------------------------
    @property
    def samples_per_round(self) -> int:
        """Examples client 0 trains on per round (the representative client, matching
        the ``client_loaders[0]`` convention the FLTrust sizing already uses)."""
        return self._per_round[0] if self._per_round else 0

    @property
    def batches_per_round(self) -> int:
        """Batches one client's round loader yields.

        FLTrust's root fine-tuning is iteration-matched against
        ``local_epochs * batches_per_client`` (see
        ``server.algo_defender.resolve_root_epochs``), and shrinking the per-round
        dataset shrinks that count — so ``main.run_phase2`` must size the root
        update from THIS, not from the full shard, or ``||g0||`` ends up several
        times too large and FLTrust rescales every honest update along with it.
        """
        return (math.ceil(self.samples_per_round / self.batch_size)
                if self._per_round else 0)

    # ------------------------------------------------------------------
    def _stream(self, cid: int, epoch: int) -> random.Random:
        """A dedicated RNG per (client, epoch), derived arithmetically from the seed
        so it is stable across processes and independent of the ambient RNG."""
        return random.Random((self.seed * 1_000_003 + int(cid)) * 1_000_003 + int(epoch))

    def indices_for_round(self, round_index: int, cid: int) -> list[int]:
        """This client's example indices for ``round_index`` (a pure function of it)."""
        shard, k = self._shards[cid], self._per_round[cid]
        if self.mode == "resample":
            return self._stream(cid, round_index).sample(shard, k)
        cycle, slot = divmod(int(round_index), self._slices[cid])
        order = list(shard)
        self._stream(cid, cycle).shuffle(order)   # a fresh cut of the shard per cycle
        return order[slot * k:(slot + 1) * k]

    def loaders_for_round(self, round_index: int) -> list[DataLoader]:
        """One loader per client, in client-id order."""
        return [
            DataLoader(Subset(self._base[cid], self.indices_for_round(round_index, cid)),
                       batch_size=self.batch_size, shuffle=True)
            for cid in range(len(self._shards))
        ]


def build_round_data_sampler(config: dict, client_loaders, seed: int = 0):
    """Build the per-round data refresh from config, or ``None`` when it is off
    (``fl.client_data_refresh: none``), in which case every round replays the
    client's whole fixed shard."""
    fl = config.get("fl", {}) or {}
    mode = str(fl.get("client_data_refresh", "rotate") or "none").lower()
    if mode == "none" or not client_loaders:
        logger.info("Per-round client data refresh OFF — every round replays the same shard")
        return None
    sampler = RoundDataSampler(
        client_loaders,
        fraction=float(fl.get("client_round_fraction", 0.25)),
        mode=mode,
        batch_size=int(fl.get("batch_size", 64)),
        seed=int(seed),
    )
    logger.info(
        f"Per-round client data refresh: {mode} — each client trains on "
        f"{sampler.samples_per_round} fresh example(s) "
        f"({sampler.batches_per_round} batch(es)) per round"
    )
    return sampler
