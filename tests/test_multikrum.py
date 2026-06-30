"""Tests for Multi-Krum's pairwise distances + end-to-end step (needs torch).

Run on the GPU box (or any box with torch):  python tests/test_multikrum.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.types import ModelUpdate  # noqa: E402
from benchmark.defenses.multikrum import MultiKrum, pairwise_sq_dists  # noqa: E402


def _zeros_global():
    return {
        "net.2.weight": torch.zeros(2, 2),
        "net.2.bias": torch.zeros(2),
        "net.4.weight": torch.zeros(2, 2),
        "net.4.bias": torch.zeros(2),
    }


def _const_weights(value):
    return {k: torch.full_like(v, float(value)) for k, v in _zeros_global().items()}


def _updates(poison_value=10.0, benign_value=0.1):
    ups = [ModelUpdate(client_id=0, weights=_const_weights(poison_value))]
    ups += [ModelUpdate(client_id=c, weights=_const_weights(benign_value)) for c in range(1, 5)]
    return ups


def test_pairwise_sq_dists_matches_manual():
    mat = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    d = pairwise_sq_dists(mat)
    assert abs(d[0][1] - 25.0) < 1e-5 and abs(d[1][0] - 25.0) < 1e-5
    assert abs(d[0][0]) < 1e-5 and abs(d[0][2]) < 1e-5          # identical rows -> 0


def test_step_drops_the_outlier():
    d = MultiKrum(num_byzantine=1)               # m = n - f = 4 -> drop 1
    d.reset(_zeros_global())
    res = d.step(_updates(), poisoned_ids={0})
    by_id = {v.client_id: v for v in res.verdicts}
    assert by_id[0].is_suspicious is True                       # poison not selected
    assert all(by_id[c].is_suspicious is False for c in range(1, 5))
    assert res.info["n_dropped"] == 1 and res.info["m"] == 4 and res.info["k_closest"] == 2
    for v in res.new_global.values():
        assert torch.allclose(v, torch.full_like(v, 0.1), atol=1e-5)


def test_krum_m_one_selects_single_most_central():
    d = MultiKrum(num_byzantine=1, m=1)          # Krum
    d.reset(_zeros_global())
    res = d.step(_updates(), poisoned_ids={0})
    selected = [v.client_id for v in res.verdicts if not v.is_suspicious]
    assert len(selected) == 1 and selected[0] != 0             # one benign, not the poison
    assert res.info["n_dropped"] == 4
    for v in res.new_global.values():                          # equals that one benign (0.1)
        assert torch.allclose(v, torch.full_like(v, 0.1), atol=1e-5)


def test_step_select_all_when_f_zero_is_plain_fedavg():
    d = MultiKrum(num_byzantine=0)               # m = n - 0 = n -> select everyone
    d.reset(_zeros_global())
    res = d.step(_updates(), poisoned_ids={0})
    assert res.info["n_dropped"] == 0
    assert all(v.is_suspicious is False for v in res.verdicts)
    for v in res.new_global.values():                          # (4*0.1 + 10)/5 = 2.08
        assert torch.allclose(v, torch.full_like(v, 2.08), atol=1e-5)


def test_step_drops_non_finite_client():
    d = MultiKrum(num_byzantine=1)
    d.reset(_zeros_global())
    ups = _updates()
    ups[0] = ModelUpdate(client_id=0, weights=_const_weights(float("inf")))
    res = d.step(ups, poisoned_ids={0})
    by_id = {v.client_id: v for v in res.verdicts}
    assert by_id[0].is_suspicious is True and res.info["n_dropped"] == 1
    for v in res.new_global.values():
        assert torch.isfinite(v).all()
        assert torch.allclose(v, torch.full_like(v, 0.1), atol=1e-5)


def test_new_global_preserves_keys_and_dtype():
    g = _zeros_global()
    d = MultiKrum(num_byzantine=1)
    d.reset(g)
    res = d.step(_updates(), poisoned_ids={0})
    assert set(res.new_global.keys()) == set(g.keys())
    for k in g:
        assert res.new_global[k].shape == g[k].shape and res.new_global[k].dtype == g[k].dtype


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} Multi-Krum tests passed.")


if __name__ == "__main__":
    _run()
