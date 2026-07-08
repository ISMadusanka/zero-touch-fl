"""Tests for the attacker's delta-based observation (agents.attack_ops.delta_details).

Pure torch on tiny hand-built state dicts — no MNIST, no GPU, no LLM:
    python tests/test_delta_details.py

The contract these lock in: the attacker's per-client stats are computed from ONLY
its own client weights and the global model (never any cross-client reference),
and every value is a dimensionless ratio/fraction — so it is independent of the
model architecture and scale, and of the poison budget.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from agents.attack_ops import delta_details  # noqa: E402

_LAYER_KEYS = {"rel_update", "rms_delta", "energy_frac", "sign_flip_frac",
               "std_ratio", "absmean_ratio"}


def _global():
    return {
        "net.2.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "net.2.bias": torch.tensor([1.0, -1.0]),
    }


def test_structure_and_keys():
    g = _global()
    c = {k: v.clone() for k, v in g.items()}
    d = delta_details(c, g)
    assert set(d) == {"layers", "whole"}
    assert set(d["layers"]) == set(g.keys())
    for layer in d["layers"].values():
        assert _LAYER_KEYS.issubset(layer)
        assert "shape" in layer
    # Whole-model block drops energy_frac (trivially 1) and adds cos_to_global.
    assert "energy_frac" not in d["whole"]
    assert "cos_to_global" in d["whole"]


def test_zero_delta_is_all_zero():
    g = _global()
    c = {k: v.clone() for k, v in g.items()}   # client == global -> Δ = 0
    d = delta_details(c, g)
    for layer in d["layers"].values():
        for k in _LAYER_KEYS:
            assert layer[k] == 0.0, f"{k} should be 0 when Δ=0"
    assert d["whole"]["cos_to_global"] == 0.0


def test_energy_frac_sums_to_one():
    g = _global()
    c = {k: v.clone() for k, v in g.items()}
    c["net.2.weight"][0, 0] += 0.1        # put ALL the change in one layer
    d = delta_details(c, g)
    total = sum(layer["energy_frac"] for layer in d["layers"].values())
    assert abs(total - 1.0) < 1e-6
    assert d["layers"]["net.2.weight"]["energy_frac"] == 1.0
    assert d["layers"]["net.2.bias"]["energy_frac"] == 0.0


def test_sign_flip_detected():
    g = _global()
    c = {k: v.clone() for k, v in g.items()}
    c["net.2.bias"][0] = -c["net.2.bias"][0]   # flip sign of 1 of the 2 bias coords
    d = delta_details(c, g)
    assert d["layers"]["net.2.bias"]["sign_flip_frac"] == 0.5
    assert d["layers"]["net.2.weight"]["sign_flip_frac"] == 0.0


def test_scale_and_architecture_independence():
    """Multiplying the WHOLE model (global + client) by any constant leaves the
    dimensionless features unchanged — they don't depend on absolute magnitudes,
    hence not on the architecture's parameter scale. Only rms_delta (an absolute
    per-coordinate size) scales with the constant."""
    g = _global()
    c = {k: v.clone() for k, v in g.items()}
    c["net.2.weight"][0, 0] += 0.1
    c["net.2.bias"][1] *= -1

    base = delta_details(c, g)
    k = 7.0
    gk = {kk: v * k for kk, v in g.items()}
    ck = {kk: v * k for kk, v in c.items()}
    scaled = delta_details(ck, gk)

    for lk in g:
        b, s = base["layers"][lk], scaled["layers"][lk]
        for feat in ("rel_update", "energy_frac", "sign_flip_frac",
                     "std_ratio", "absmean_ratio"):
            assert abs(b[feat] - s[feat]) < 1e-4, f"{lk}.{feat} not scale-invariant"
        # rms_delta is an absolute magnitude -> scales by k.
        assert abs(s["rms_delta"] - k * b["rms_delta"]) < 1e-3
    assert abs(base["whole"]["cos_to_global"] - scaled["whole"]["cos_to_global"]) < 1e-4


def test_rel_update_matches_hand_computation():
    g = _global()
    c = {k: v.clone() for k, v in g.items()}
    c["net.2.weight"][0, 0] += 0.1        # Δ_w = [0.1,0,0,0] -> ‖Δ_w‖ = 0.1
    d = delta_details(c, g)
    gw = g["net.2.weight"].flatten().float()
    expected = 0.1 / float(gw.norm())     # ‖Δ‖ / ‖G‖ for that layer
    assert abs(d["layers"]["net.2.weight"]["rel_update"] - round(expected, 4)) < 1e-4


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} delta_details tests passed.")


if __name__ == "__main__":
    _run()
