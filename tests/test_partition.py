"""Tests for the FLTrust non-IID partition (data/mnist_loader.partition_noniid_fltrust).

Uses a tiny fake dataset (a `.targets` list), so no MNIST download is needed.
Run on any box with torch/torchvision installed:  python tests/test_partition.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mnist_loader import partition_noniid_fltrust  # noqa: E402


class FakeDS:
    """Minimal dataset stand-in exposing `.targets` (what the partition reads)."""

    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)


def _balanced_targets(per_class: int, n_classes: int = 10):
    """`per_class` samples of each label 0..n_classes-1 (interleaved)."""
    return [c for _ in range(per_class) for c in range(n_classes)]


def _group_label_counts(shards, targets, members, n_classes=10):
    counts = [0] * n_classes
    for cid in members:
        for idx in shards[cid]:
            counts[targets[idx]] += 1
    return counts


def test_shape_disjoint_and_full_coverage():
    targets = _balanced_targets(100)                 # 1000 samples, 10 classes
    ds = FakeDS(targets)
    shards = partition_noniid_fltrust(ds, n_clients=20, n_classes=10, bias_q=0.5, seed=0)
    assert len(shards) == 20
    flat = [i for s in shards for i in s]
    assert len(flat) == len(targets)                 # no samples dropped/duplicated
    assert sorted(flat) == list(range(len(targets)))  # exact, disjoint cover


def test_own_class_dominates_each_group_when_biased():
    targets = _balanced_targets(200)                 # 2000 samples
    ds = FakeDS(targets)
    shards = partition_noniid_fltrust(ds, n_clients=20, n_classes=10, bias_q=0.7, seed=0)
    # Round-robin groups: group g = clients {g, g+10}. Each group should be
    # dominated by its own class g for q=0.7 (>> IID share of 0.1).
    for g in range(10):
        counts = _group_label_counts(shards, targets, members=[g, g + 10])
        assert counts[g] == max(counts), f"group {g} not dominated by class {g}: {counts}"
        assert counts[g] > 0.5 * sum(counts), f"group {g} own-class share too low: {counts}"


def test_iid_bias_is_not_dominated():
    targets = _balanced_targets(200)
    ds = FakeDS(targets)
    # q = 1/M = 0.1 is the IID point: no class should dominate a group.
    shards = partition_noniid_fltrust(ds, n_clients=20, n_classes=10, bias_q=0.1, seed=0)
    counts = _group_label_counts(shards, targets, members=[0, 10])
    assert counts[0] < 0.25 * sum(counts), f"IID split unexpectedly biased: {counts}"


def test_bias_increases_own_class_share():
    targets = _balanced_targets(200)
    ds = FakeDS(targets)
    low = _group_label_counts(
        partition_noniid_fltrust(ds, 20, 10, bias_q=0.1, seed=1), targets, [0, 10])
    high = _group_label_counts(
        partition_noniid_fltrust(ds, 20, 10, bias_q=0.9, seed=1), targets, [0, 10])
    assert high[0] > low[0]                           # more bias -> more own-class mass


def test_reproducible_and_seed_sensitive():
    ds = FakeDS(_balanced_targets(100))
    a = partition_noniid_fltrust(ds, 20, 10, bias_q=0.5, seed=7)
    b = partition_noniid_fltrust(ds, 20, 10, bias_q=0.5, seed=7)
    c = partition_noniid_fltrust(ds, 20, 10, bias_q=0.5, seed=8)
    assert a == b                                    # same seed -> identical
    assert a != c                                    # different seed -> different


def test_fewer_clients_than_classes_still_covers():
    # n_clients < n_classes leaves some groups empty; those samples must be rerouted,
    # never dropped.
    targets = _balanced_targets(100)
    ds = FakeDS(targets)
    shards = partition_noniid_fltrust(ds, n_clients=5, n_classes=10, bias_q=0.6, seed=0)
    assert len(shards) == 5
    flat = [i for s in shards for i in s]
    assert sorted(flat) == list(range(len(targets)))


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} partition tests passed.")


if __name__ == "__main__":
    _run()
