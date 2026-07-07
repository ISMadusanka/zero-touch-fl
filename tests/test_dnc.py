"""Tests for DnC's spectral scoring + end-to-end step (needs torch).

Run on the GPU box (or any box with torch):  python tests/test_dnc.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.types import ModelUpdate  # noqa: E402
from benchmark.defenses.dnc import DnC, outlier_scores  # noqa: E402


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


def test_outlier_scores_flag_the_spectral_outlier():
    # 4 tight rows + 1 far row -> the far row projects furthest on the top PC.
    sub = torch.tensor([
        [0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.0, 0.0], [10.0, 10.0],
    ])
    scores = outlier_scores(sub)
    assert len(scores) == 5
    assert scores[4] == max(scores) and scores[4] > 10 * max(scores[:4] + [1e-9])


def test_outlier_scores_degenerate_sizes():
    assert outlier_scores(torch.zeros(0, 3)) == []
    assert outlier_scores(torch.zeros(1, 3)) == [0.0]


def test_step_removes_the_obvious_outlier():
    d = DnC(num_byzantine=1, c=1.0)          # remove c*m = 1
    d.reset(_zeros_global())
    res = d.step(_updates(), poisoned_ids={0})
    by_id = {v.client_id: v for v in res.verdicts}
    assert by_id[0].is_suspicious is True                       # poison filtered out
    assert all(by_id[c].is_suspicious is False for c in range(1, 5))
    assert res.info["n_removed"] == 1 and res.info["keep_count"] == 4
    # aggregate = FedAvg over the 4 kept benign (all 0.1) -> poison's 10 left no trace.
    for v in res.new_global.values():
        assert torch.allclose(v, torch.full_like(v, 0.1), atol=1e-5)


def test_step_keep_all_when_m_zero_is_plain_fedavg():
    d = DnC(num_byzantine=0)
    d.reset(_zeros_global())
    res = d.step(_updates(), poisoned_ids={0})
    assert res.info["n_removed"] == 0
    assert all(v.is_suspicious is False for v in res.verdicts)
    # FedAvg over all 5: (4*0.1 + 10)/5 = 2.08
    for v in res.new_global.values():
        assert torch.allclose(v, torch.full_like(v, 2.08), atol=1e-5)


def test_step_c_times_m_removes_two():
    d = DnC(num_byzantine=2, c=1.0)          # remove 2 (1 real poison + 1 false positive)
    d.reset(_zeros_global())
    res = d.step(_updates(), poisoned_ids={0})
    flagged = [v.client_id for v in res.verdicts if v.is_suspicious]
    assert res.info["n_removed"] == 2 and len(flagged) == 2
    assert 0 in flagged                                        # the poison is among them


def test_scoring_is_spectral_not_norm_based():
    # The benign cluster spreads along axis-0 (the dominant variance direction, so
    # the top singular vector is ~axis-0). Client 4 sits OFF that axis with the
    # LARGEST raw norm but a ~zero projection onto axis-0. A correct spectral DnC
    # gives it the SMALLEST score; a norm/distance filter would rank it HIGHEST — so
    # this fails for any non-SVD implementation.
    sub = torch.tensor([[-5.0, 0.0], [-3.0, 0.0], [3.0, 0.0], [5.0, 0.0], [0.0, 8.0]])
    scores = outlier_scores(sub)
    assert scores[4] == min(scores)                            # smallest SPECTRAL score
    assert scores[4] < scores[0] and scores[4] < scores[3]     # ...despite the largest norm


def test_step_niters_and_subsampling_paths():
    # niters>1 (intersection path) AND sub_dim < d=12 (column-slice subsampling path).
    # With constant rows every coordinate subset separates the poison identically, so
    # the per-iteration kept sets agree and their intersection is stable.
    d = DnC(num_byzantine=1, niters=3, sub_dim=4, seed=0)
    d.reset(_zeros_global())
    res = d.step(_updates(), poisoned_ids={0})
    by_id = {v.client_id: v for v in res.verdicts}
    assert by_id[0].is_suspicious is True
    assert res.info["n_removed"] == 1 and res.info["dims"] == 12
    for v in res.new_global.values():
        assert torch.allclose(v, torch.full_like(v, 0.1), atol=1e-5)


def test_step_drops_non_finite_client_and_keeps_global_finite():
    # A client with NaN weights must be forced out (not silently kept) and must not
    # turn the aggregate into NaN.
    d = DnC(num_byzantine=1)
    d.reset(_zeros_global())
    ups = _updates()
    ups[0] = ModelUpdate(client_id=0, weights=_const_weights(float("nan")))
    res = d.step(ups, poisoned_ids={0})
    by_id = {v.client_id: v for v in res.verdicts}
    assert by_id[0].is_suspicious is True and res.info["n_removed"] == 1
    for v in res.new_global.values():
        assert torch.isfinite(v).all()
        assert torch.allclose(v, torch.full_like(v, 0.1), atol=1e-5)


def test_new_global_preserves_keys_and_dtype():
    g = _zeros_global()
    d = DnC(num_byzantine=1)
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
    print(f"\nAll {len(tests)} DnC tests passed.")


if __name__ == "__main__":
    _run()
