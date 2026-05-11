"""MNIST data loading and IID partitioning across clients."""

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

def partition_noniid(dataset, n_clients: int = 5):
    """Split dataset into non-IID shards with overlapping label subsets.

    Each client receives samples from only 4 digit classes:
        Client 0: labels {0, 1, 2, 3}
        Client 1: labels {2, 3, 4, 5}
        Client 2: labels {4, 5, 6, 7}
        Client 3: labels {6, 7, 8, 9}
        Client 4: labels {8, 9, 0, 1}
    """
    # Define overlapping label assignments per client
    client_labels = {
        0: [0, 1, 2, 3],
        1: [2, 3, 4, 5],
        2: [4, 5, 6, 7],
        3: [6, 7, 8, 9],
        4: [8, 9, 0, 1],
    }

    # Build a mapping: label -> list of indices
    if isinstance(dataset, Subset):
        full_targets = dataset.dataset.targets if hasattr(dataset.dataset, 'targets') else dataset.dataset.train_labels
        subset_targets = [full_targets[i] for i in dataset.indices]
    else:
        subset_targets = dataset.targets if hasattr(dataset, 'targets') else dataset.train_labels

    label_to_indices = {}
    for idx, label in enumerate(subset_targets):
        lbl = int(label)
        label_to_indices.setdefault(lbl, []).append(idx)

    # Shuffle each label's indices
    for lbl in label_to_indices:
        perm = torch.randperm(len(label_to_indices[lbl])).tolist()
        label_to_indices[lbl] = [label_to_indices[lbl][i] for i in perm]

    # Track how many samples of each label have been consumed
    label_offset = {lbl: 0 for lbl in label_to_indices}

    shards = []
    for cid in range(n_clients):
        labels = client_labels[cid]
        client_indices = []
        for lbl in labels:
            available = label_to_indices[lbl]
            # Each label is shared by exactly 2 clients → each gets half
            half = len(available) // 2
            start = label_offset[lbl]
            end = start + half
            client_indices.extend(available[start:end])
            label_offset[lbl] = end
        shards.append(client_indices)

    return shards


def get_data_loaders(n_clients: int, batch_size: int, data_dir: str = "./data/mnist_raw",
                     iid: bool = True):
    """Return per-client train loaders, a global test loader, and a server root loader."""
    train_dataset, test_dataset = load_mnist(data_dir)
    # Take 1500 samples for the server root dataset (crucial for FLTrust)
    indices = torch.randperm(len(train_dataset)).tolist()
    root_indices = indices[:1500]
    remaining_indices = indices[500:]

    root_loader = DataLoader(Subset(train_dataset, root_indices), batch_size=batch_size, shuffle=False)
    
    # Partition the remaining data
    train_subset = Subset(train_dataset, remaining_indices)

    if iid:
        shards = partition_iid(train_subset, n_clients)
    else:
        shards = partition_noniid(train_subset, n_clients)

    client_loaders = [
        DataLoader(Subset(train_subset, shard), batch_size=batch_size, shuffle=True)
        for shard in shards
    ]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader, root_loader
