"""Tests for the FLTrust trust/aggregation math (needs torch).

Run on the GPU box (or any box with torch):  python tests/test_fltrust.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from benchmark.defenses.fltrust import _flatten, _unflatten, fltrust_combine  # noqa: E402


def test_flatten_unflatten_roundtrip():
    ref = {"a": torch.randn(2, 3), "b": torch.randn(5)}
    keys = list(ref.keys())
    flat = _flatten(ref, keys)
    assert flat.numel() == 6 + 5
    back = _unflatten(flat, ref, keys)
    for k in keys:
        assert torch.allclose(back[k], ref[k].float())


def test_trust_excludes_anti_aligned_and_orthogonal():
    g0 = torch.tensor([1.0, 0.0])
    deltas = [
        torch.tensor([2.0, 0.0]),    # aligned  -> cos=1   -> trust 1, kept
        torch.tensor([-1.0, 0.0]),   # opposite -> cos=-1  -> ReLU 0, dropped
        torch.tensor([0.0, 3.0]),    # orthogonal -> cos=0 -> ReLU 0, dropped
    ]
    agg, trust = fltrust_combine(deltas, g0)
    assert abs(trust[0] - 1.0) < 1e-6
    assert trust[1] == 0.0 and trust[2] == 0.0
    # only client 0 survives; normalized to ||g0||=1 -> direction [1,0]
    assert torch.allclose(agg, torch.tensor([1.0, 0.0]), atol=1e-5)


def test_magnitude_attack_is_normalized_away():
    g0 = torch.tensor([1.0, 0.0])
    # huge but aligned update: trust=1 but rescaled to ||g0|| so it can't dominate
    agg, trust = fltrust_combine([torch.tensor([1000.0, 0.0])], g0)
    assert abs(trust[0] - 1.0) < 1e-6
    assert torch.allclose(agg, torch.tensor([1.0, 0.0]), atol=1e-5)


def test_all_zero_trust_returns_none():
    g0 = torch.tensor([1.0, 0.0])
    agg, trust = fltrust_combine([torch.tensor([-1.0, 0.0]), torch.tensor([-2.0, 0.0])], g0)
    assert agg is None and trust == [0.0, 0.0]


def test_trust_weighted_average_direction():
    # two aligned-ish clients, different angles -> weighted by ReLU(cos)
    g0 = torch.tensor([1.0, 0.0])
    a = torch.tensor([1.0, 0.0])           # cos 1
    b = torch.tensor([1.0, 1.0])           # cos ~0.707
    agg, trust = fltrust_combine([a, b], g0)
    assert abs(trust[0] - 1.0) < 1e-6 and abs(trust[1] - (2 ** -0.5)) < 1e-4
    # result points into the positive quadrant, dominated by client a
    assert agg[0] > agg[1] > 0


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} FLTrust tests passed.")


if __name__ == "__main__":
    _run()
