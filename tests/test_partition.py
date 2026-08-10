"""Tests for the FLTrust non-IID partition (data/nidd_loader.partition_noniid_fltrust).

Uses a tiny fake dataset (a `.targets` list), so no 5G-NIDD CSV is needed.
Run on any box with torch installed:  python tests/test_partition.py

The partition is dataset-agnostic — it only reads labels — so most cases here use a
balanced 10-class set to keep the arithmetic obvious. The 5G-NIDD-specific cases
(9 classes, and the severe class imbalance that dataset really has) are at the end.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nidd_loader import partition_noniid_fltrust  # noqa: E402


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


# --- 5G-NIDD specifics: 9 classes, and a severely imbalanced label mix ---------

def test_nine_classes_over_twenty_clients_covers_exactly():
    """The shipped configuration: 9 attack classes (incl. benign), 20 clients.

    9 does not divide 20, so groups get 2 or 3 clients — the round-robin split must
    still produce an exact, disjoint cover.
    """
    targets = _balanced_targets(90, n_classes=9)          # 810 samples, 9 classes
    ds = FakeDS(targets)
    shards = partition_noniid_fltrust(ds, n_clients=20, n_classes=9, bias_q=0.5, seed=0)
    assert len(shards) == 20
    assert all(len(s) > 0 for s in shards)                # nobody starved
    flat = [i for s in shards for i in s]
    assert sorted(flat) == list(range(len(targets)))


def test_imbalanced_labels_still_cover_and_stay_bounded():
    """5G-NIDD's real mix: two dominant classes and one that is ~0.1% of flows.

    Two properties matter downstream. Coverage must hold (no flow silently
    dropped), and at bias_q=0.5 the shard spread must stay MODEST despite a 400x
    class ratio — half of every class is spread uniformly over the other groups,
    which floors the rare-class groups. `data/round_sampler.py` and the
    `client_round_fraction` comment in configs/base.yaml both quote that bound, so
    a change in this behaviour should fail here rather than silently invalidate
    those notes.
    """
    # Benign 39%, UDPFlood 38%, then a long tail down to ICMPFlood at ~0.1%.
    mix = [3900, 3800, 1150, 600, 164, 164, 127, 79, 9]
    targets = [c for c, n in enumerate(mix) for _ in range(n)]
    ds = FakeDS(targets)
    shards = partition_noniid_fltrust(ds, n_clients=20, n_classes=9, bias_q=0.5, seed=0)

    flat = [i for s in shards for i in s]
    assert sorted(flat) == list(range(len(targets)))       # exact, disjoint cover
    sizes = sorted(len(s) for s in shards)
    assert sizes[0] > 0
    # Measured ~3.6x at these proportions; assert an order-of-magnitude bound so the
    # test documents the effect without being brittle to the exact draw.
    assert sizes[-1] / sizes[0] < 10, sizes


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} partition tests passed.")


if __name__ == "__main__":
    _run()
