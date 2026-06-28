"""Tests for DeFL's tensor math + end-to-end step (needs torch).

Run on the GPU box (or any box with torch):  python tests/test_defl.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.types import ModelUpdate  # noqa: E402
from benchmark.defenses.defl import DeFL, fgnv_for_update, group_layers  # noqa: E402


def _zeros_global():
    """A tiny 2-layer (net.2, net.4) zero state_dict matching MnistNet's key style."""
    return {
        "net.2.weight": torch.zeros(2, 2),
        "net.2.bias": torch.zeros(2),
        "net.4.weight": torch.zeros(2, 2),
        "net.4.bias": torch.zeros(2),
    }


def _const_weights(value):
    return {k: torch.full_like(v, float(value)) for k, v in _zeros_global().items()}


def _updates(poison_value=10.0, benign_value=0.1):
    """Client 0 = poison (large delta), clients 1..4 = identical benign."""
    ups = [ModelUpdate(client_id=0, weights=_const_weights(poison_value))]
    ups += [ModelUpdate(client_id=c, weights=_const_weights(benign_value)) for c in range(1, 5)]
    return ups


def test_fgnv_is_per_layer_squared_norm():
    groups = group_layers(["w"])
    fg = fgnv_for_update({"w": torch.tensor([3.0, 4.0, 0.0])},
                         {"w": torch.zeros(3)}, groups)
    assert len(fg) == 1 and abs(fg[0] - 25.0) < 1e-6          # 3^2 + 4^2


def test_fgnv_groups_weight_and_bias_into_one_layer():
    g = _zeros_global()
    groups = group_layers(list(g.keys()))
    fg = fgnv_for_update(_const_weights(0.1), g, groups)
    # net.2 = 6 entries * 0.1^2 = 0.06 ; net.4 = same.
    assert len(fg) == 2
    assert abs(fg[0] - 0.06) < 1e-6 and abs(fg[1] - 0.06) < 1e-6


def test_clp_round_removes_detected_malicious():
    d = DeFL(delta=0.05, tau=2.5)
    d.reset(_zeros_global())
    res = d.step(_updates(), poisoned_ids={0})
    by_id = {v.client_id: v for v in res.verdicts}
    assert by_id[0].is_suspicious is True                      # poison detected
    assert all(by_id[c].is_suspicious is False for c in range(1, 5))
    assert res.info["in_clp"] is True
    # First round is a CLP -> poison gets weight 0 -> global is the benign value, the
    # poison's 10.0 left no trace.
    assert res.new_global is not None
    for v in res.new_global.values():
        assert torch.allclose(v, torch.full_like(v, 0.1), atol=1e-5)


def test_post_clp_soft_weights_keep_but_shrink_malicious():
    d = DeFL(delta=0.05, tau=2.5)
    d.reset(_zeros_global())
    d._prev_total_fgnv = 1e12          # force this round to NOT be a CLP (FGNV "fell")
    res = d.step(_updates(), poisoned_ids={0})
    by_id = {v.client_id: v for v in res.verdicts}
    assert by_id[0].is_suspicious is True                      # still detected...
    assert res.info["in_clp"] is False
    # ...but NOT removed: poison kept at Beta weight 1/3 vs benign 2/3.
    # new_global = 4*(2/9)*0.1 + (1/9)*10 = 1.2 everywhere.
    for v in res.new_global.values():
        assert torch.allclose(v, torch.full_like(v, 1.2), atol=1e-4)


def test_new_global_preserves_keys_and_dtype():
    d = DeFL()
    g = _zeros_global()
    d.reset(g)
    res = d.step(_updates(), poisoned_ids={0})
    assert set(res.new_global.keys()) == set(g.keys())
    for k in g:
        assert res.new_global[k].shape == g[k].shape
        assert res.new_global[k].dtype == g[k].dtype


def test_repeated_malicious_votes_accumulate_in_beta():
    d = DeFL(delta=0.05, tau=2.5)
    d.reset(_zeros_global())
    for _ in range(3):
        d.step(_updates(), poisoned_ids={0})
    # client 0 voted malicious 3x (beta=4, alpha=1) -> p≈0.2 ; benign p high.
    assert d._beta.prob(0) < 0.25
    assert d._beta.prob(1) > 0.75


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} DeFL tests passed.")


if __name__ == "__main__":
    _run()
