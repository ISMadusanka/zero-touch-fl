"""Dataset loading and per-client partitioning (IID + FLTrust non-IID).

Dataset-agnostic: which dataset is downloaded, how it is normalized and how many
classes it has all come from :mod:`data.datasets`, so MNIST and CIFAR-10 (and any
future entry in the registry) go through exactly the same partition, root-set and
DataLoader code. Only the ``dataset`` argument changes.

    client_loaders, test_loader = get_data_loaders("cifar10", n_clients=20,
                                                   batch_size=64, iid=False)

The partition functions themselves never touch pixels — they work off the label
list — so they are identical for every dataset and are covered by
``tests/test_partition.py`` without any download.
"""

import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from data.datasets import DEFAULT_DATASET, resolve


def build_transform(spec):
    """The evaluation/training transform for ``spec``: tensor + normalization.

    Deliberately augmentation-free for every dataset. Random crops/flips would
    raise CIFAR-10's clean accuracy, but they would also make each client's local
    training (and FLTrust's root fine-tuning, which draws from the same train set)
    non-reproducible run to run — and this testbed compares a poisoned aggregate
    against a *clean counterfactual* computed from the same weights, so injected
    randomness shows up directly as reward noise.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(spec.mean, spec.std),
    ])


def load_dataset(dataset: str = DEFAULT_DATASET, data_dir: str | None = None):
    """Download (if needed) and return ``(train_dataset, test_dataset)``.

    ``data_dir`` defaults to the registry's per-dataset directory
    (``./data/mnist_raw``, ``./data/cifar10_raw``), so two datasets never share a
    download cache.
    """
    spec = resolve(dataset)
    root = data_dir or spec.data_dir
    cls = getattr(datasets, spec.torchvision_name)
    transform = build_transform(spec)
    train_dataset = cls(root, train=True, download=True, transform=transform)
    test_dataset = cls(root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def build_root_loader(dataset: str = DEFAULT_DATASET, root_size: int = 100,
                      batch_size: int = 64, data_dir: str | None = None,
                      seed: int = 0):
    """A small CLEAN root dataset held by the server, for FLTrust.

    FLTrust (Cao et al., NDSS 2021) bootstraps trust from a handful of clean
    examples the server collects itself; we sample ``root_size`` of them from the
    training set with a fixed seed so runs are reproducible. Used by both the
    benchmark panel and the arms race's algorithmic defender.
    """
    train_dataset, _ = load_dataset(dataset, data_dir)
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(len(train_dataset), generator=g)[:int(root_size)].tolist()
    # The same seeded generator drives the shuffle, so FLTrust's root fine-tuning
    # (and therefore its trusted reference direction g0) is reproducible across runs
    # instead of drawing from the ambient global RNG. Within a round g0 is computed
    # once and cached — see benchmark.defenses.fltrust.FLTrust._root_update.
    return DataLoader(Subset(train_dataset, idx),
                      batch_size=max(1, min(int(batch_size), int(root_size))),
                      shuffle=True, generator=g)


def partition_iid(dataset, n_clients: int, seed: int | None = None):
    """Split dataset into n_clients equal IID shards.

    ``seed`` drives a dedicated generator, so the split is reproducible from the
    argument alone. Without it the shuffle came from the ambient global torch RNG,
    which meant ``get_data_loaders(seed=...)`` was honoured under ``iid=False`` and
    silently ignored under ``iid=True`` — two runs with the same seed could then
    partition differently depending on what had consumed the global RNG first.
    ``None`` keeps the old global-RNG behaviour for callers that want it.
    """
    g = None
    if seed is not None:
        g = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=g).tolist()
    shard_size = len(dataset) // n_clients
    return [indices[i * shard_size : (i + 1) * shard_size] for i in range(n_clients)]


def _dataset_targets(dataset) -> list:
    """Integer labels for every sample, robust across torchvision versions and
    datasets (MNIST exposes a tensor, CIFAR-10 a plain Python list)."""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "train_labels", None)
    if targets is None:  # exotic dataset: fall back to (slow) per-item access
        return [int(dataset[i][1]) for i in range(len(dataset))]
    if isinstance(targets, torch.Tensor):
        return targets.tolist()
    return [int(t) for t in targets]


def partition_noniid_fltrust(dataset, n_clients: int, n_classes: int = 10,
                             bias_q: float = 0.5, seed: int = 0):
    """Non-IID partition following FLTrust (Cao et al., NDSS 2021).

    Clients are split into ``M = n_classes`` groups. A training example with
    label ``l`` is assigned to group ``l`` with probability ``bias_q`` and to any
    OTHER group with probability ``(1 - bias_q) / (M - 1)``. Within a group the
    assigned examples are split evenly across that group's clients.

    ``bias_q = 1/M`` reproduces an IID split; a larger ``bias_q`` gives a stronger
    non-IID skew (each group is dominated by its own class). The paper's default
    is ``0.5``. For a 10-class dataset with 20 clients this yields 10 groups of 2
    clients each.

    Returns a list of ``n_clients`` index lists (shards), ordered by client id;
    the shards are disjoint and cover every sample.
    """
    rng = random.Random(seed)
    M = max(1, int(n_classes))

    # --- Split clients into M groups, as evenly as possible (20/10 -> 2 each). ---
    # Round-robin keeps groups balanced; earlier groups get the extra client when
    # n_clients is not divisible by M.
    groups: list[list[int]] = [[] for _ in range(M)]
    for cid in range(n_clients):
        groups[cid % M].append(cid)
    nonempty = [g for g in range(M) if groups[g]]  # guards n_clients < M

    # --- Route each sample to a group by the bias rule ---
    targets = _dataset_targets(dataset)
    group_indices: list[list[int]] = [[] for _ in range(M)]
    for idx, lbl in enumerate(targets):
        l = int(lbl) % M
        if rng.random() < bias_q:
            g = l
        else:  # pick uniformly among the M-1 groups other than l
            g = rng.randrange(M - 1) if M > 1 else 0
            if g >= l:
                g += 1
        if not groups[g]:  # chosen group has no clients (only when n_clients < M)
            g = l if groups[l] else rng.choice(nonempty)
        group_indices[g].append(idx)

    # --- Within each group, split its indices evenly across the group's clients ---
    shards: list[list[int]] = [[] for _ in range(n_clients)]
    for g in range(M):
        members = groups[g]
        if not members:
            continue
        idxs = group_indices[g]
        rng.shuffle(idxs)
        k = len(members)
        base, rem = divmod(len(idxs), k)
        start = 0
        for j, cid in enumerate(members):
            count = base + (1 if j < rem else 0)
            shards[cid] = idxs[start:start + count]
            start += count

    return shards


def get_data_loaders(dataset: str = DEFAULT_DATASET, *, n_clients: int,
                     batch_size: int, data_dir: str | None = None,
                     iid: bool = True, bias_q: float = 0.5, seed: int = 0,
                     n_classes: int | None = None):
    """Return per-client train loaders and a global test loader.

    Args:
        dataset: registry name ("mnist", "cifar10", ...). Decides what is
             downloaded, how it is normalized, and the default ``data_dir``.
        n_clients: Number of federated clients.
        batch_size: Training batch size.
        data_dir: Override the dataset's default data directory.
        iid: If True, use IID partitioning. If False, use the FLTrust non-IID
             partition (``partition_noniid_fltrust``).
        bias_q: FLTrust bias probability (only used when ``iid=False``). ``1/M``
             is IID; larger is more non-IID; paper default ``0.5``.
        seed: RNG seed for the partition (reproducibility) — honoured by BOTH the
             IID and the non-IID path.
        n_classes: Label-class count (also the non-IID group count). Defaults to
             the dataset's own class count from the registry.
    """
    spec = resolve(dataset)
    train_dataset, test_dataset = load_dataset(spec.name, data_dir)

    if iid:
        shards = partition_iid(train_dataset, n_clients, seed=seed)
    else:
        shards = partition_noniid_fltrust(
            train_dataset, n_clients,
            n_classes=spec.n_classes if n_classes is None else int(n_classes),
            bias_q=bias_q, seed=seed,
        )

    client_loaders = [
        DataLoader(Subset(train_dataset, shard), batch_size=batch_size, shuffle=True)
        for shard in shards
    ]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return client_loaders, test_loader
