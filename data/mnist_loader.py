"""MNIST data loading and partitioning across clients (IID + FLTrust non-IID)."""

import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_mnist(data_dir: str = "./data/mnist_raw"):
    """Download MNIST and return train/test datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def partition_iid(dataset, n_clients: int):
    """Split dataset into n_clients equal IID shards."""
    indices = torch.randperm(len(dataset)).tolist()
    shard_size = len(dataset) // n_clients
    return [indices[i * shard_size : (i + 1) * shard_size] for i in range(n_clients)]


def _dataset_targets(dataset) -> list:
    """Integer labels for every sample, robust across torchvision versions."""
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
    is ``0.5``. For MNIST with 20 clients this yields 10 groups of 2 clients each.

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


def get_root_loader(root_size: int, batch_size: int, data_dir: str = "./data/mnist_raw",
                    seed: int = 0):
    """A small CLEAN "root" dataset held by the server.

    FLTrust (Cao et al., NDSS 2021) bootstraps trust by fine-tuning the current
    global model on such a set each round and comparing every client's update
    against that reference direction, so both the benchmark's ``fltrust`` defense
    and the ensemble that ``--freeze defender`` trains against need one. Sampled
    uniformly (and reproducibly, via ``seed``) from the MNIST training split.
    """
    train_dataset, _ = load_mnist(data_dir)
    size = max(1, min(int(root_size), len(train_dataset)))
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(train_dataset), generator=generator)[:size].tolist()
    return DataLoader(Subset(train_dataset, indices),
                      batch_size=max(1, min(int(batch_size), size)), shuffle=True)


def get_data_loaders(n_clients: int, batch_size: int, data_dir: str = "./data/mnist_raw",
                     iid: bool = True, bias_q: float = 0.5, seed: int = 0,
                     n_classes: int = 10):
    """Return per-client train loaders and a global test loader.

    Args:
        n_clients: Number of federated clients.
        batch_size: Training batch size.
        data_dir: Path to MNIST data directory.
        iid: If True, use IID partitioning. If False, use the FLTrust non-IID
             partition (``partition_noniid_fltrust``).
        bias_q: FLTrust bias probability (only used when ``iid=False``). ``1/M``
             is IID; larger is more non-IID; paper default ``0.5``.
        seed: RNG seed for the non-IID partition (reproducibility).
        n_classes: Number of label classes (MNIST = 10); also the group count.
    """
    train_dataset, test_dataset = load_mnist(data_dir)

    if iid:
        shards = partition_iid(train_dataset, n_clients)
    else:
        shards = partition_noniid_fltrust(
            train_dataset, n_clients, n_classes=n_classes, bias_q=bias_q, seed=seed
        )

    client_loaders = [
        DataLoader(Subset(train_dataset, shard), batch_size=batch_size, shuffle=True)
        for shard in shards
    ]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return client_loaders, test_loader
