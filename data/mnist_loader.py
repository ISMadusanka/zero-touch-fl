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


def get_data_loaders(n_clients: int, batch_size: int, data_dir: str = "./data/mnist_raw"):
    """Return per-client train loaders and a global test loader."""
    train_dataset, test_dataset = load_mnist(data_dir)
    shards = partition_iid(train_dataset, n_clients)

    client_loaders = [
        DataLoader(Subset(train_dataset, shard), batch_size=batch_size, shuffle=True)
        for shard in shards
    ]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return client_loaders, test_loader
