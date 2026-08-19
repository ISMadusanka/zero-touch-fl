"""Scripted published attacks (benchmark/attacks): crafting math, the registry,
and end-to-end wiring through run_benchmark against a real defense.

CPU-only, no LLM. Verifies each attack reproduces its published formula, respects
the exact poison quota, produces valid deliverable state_dicts, and — for the
optimization attacks — that all malicious clients collude on one crafted update.
"""
import logging
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from core.types import ModelUpdate                                      # noqa: E402
from benchmark.attacks import AVAILABLE_ATTACKS, build_attacker         # noqa: E402
from benchmark.attacks.base import BenignStats                          # noqa: E402
from benchmark.attacks.min_max import perturbation_direction           # noqa: E402
from benchmark.attacks.lie import lie_z_from_counts                     # noqa: E402

logging.getLogger("benchmark").addHandler(logging.NullHandler())
logging.getLogger("benchmark").propagate = False


# --- fixtures ---------------------------------------------------------------

def _global():
    return {"a": torch.zeros(4), "b": torch.zeros(2, 2)}


def _pool(deltas):
    """{cid: benign_state_dict} with benign = G + delta[cid] for the given deltas."""
    g = _global()
    return {cid: {k: g[k] + d[k] for k in g} for cid, d in enumerate(deltas)}


def _d(a, b):
    return {"a": torch.tensor(a, dtype=torch.float32),
            "b": torch.tensor(b, dtype=torch.float32)}


_GEN = torch.Generator().manual_seed(0)


def _delta(poisoned_sd):
    """poisoned weights - global, per key."""
    g = _global()
    return {k: poisoned_sd[k] - g[k] for k in g}


# --- registry ---------------------------------------------------------------

def test_registry_names_and_llm_sentinel():
    assert AVAILABLE_ATTACKS == ["noise", "sign_flip", "scaling",
                                 "lie", "ipm", "min_max", "min_sum", "fang"]
    assert build_attacker("llm") is None                 # LLM handled by the caller
    for name in AVAILABLE_ATTACKS:
        atk = build_attacker(name, n_clients=20, seed=1)
        assert atk is not None and atk.name == name


def test_unknown_attacker_raises():
    with pytest.raises(ValueError):
        build_attacker("nope")


# --- quota / deliverable shape (all attacks) --------------------------------

@pytest.mark.parametrize("name", AVAILABLE_ATTACKS)
def test_fills_exact_quota_with_valid_state_dicts(name):
    pool = _pool([_d([1, 2, 3, 4], [[1, 0], [0, 1]]),
                  _d([2, 1, 0, 1], [[0, 1], [1, 0]]),
                  _d([0, 1, 2, 1], [[1, 1], [0, 0]])])
    atk = build_attacker(name, n_clients=10, seed=3)
    poisoned, chosen, n_malformed = atk.act(_Env(pool), _Ctx(pool, budget=2))
    assert n_malformed == 0
    assert chosen == [0, 1]                              # first `budget` pool ids
    assert set(poisoned) == {0, 1}
    for sd in poisoned.values():
        assert set(sd) == {"a", "b"}
        assert sd["a"].shape == (4,) and sd["b"].shape == (2, 2)
        assert torch.isfinite(sd["a"]).all() and torch.isfinite(sd["b"]).all()


# --- trivial baselines: exact per-client formula ----------------------------

def test_sign_flip_negates_each_clients_own_update():
    d0, d1 = _d([1, -2, 3, -4], [[1, -1], [2, -2]]), _d([2, 2, 2, 2], [[0, 0], [0, 0]])
    pool = _pool([d0, d1])
    atk = build_attacker("sign_flip", seed=0)            # factor 1.0
    poisoned, _, _ = atk.act(_Env(pool), _Ctx(pool, budget=2))
    # W_mal_i = G - Δ_i, so the delta is exactly -Δ_i (per client, not colluding).
    assert torch.allclose(_delta(poisoned[0])["a"], -d0["a"])
    assert torch.allclose(_delta(poisoned[1])["b"], -d1["b"])


def test_scaling_boosts_each_clients_own_update():
    d0 = _d([1, 2, 3, 4], [[1, 1], [1, 1]])
    pool = _pool([d0, _d([0, 0, 0, 0], [[0, 0], [0, 0]])])
    atk = build_attacker("scaling", attack_scale=3.0, seed=0)
    poisoned, _, _ = atk.act(_Env(pool), _Ctx(pool, budget=1))
    assert torch.allclose(_delta(poisoned[0])["a"], 3.0 * d0["a"])


def test_noise_is_diverse_and_seed_reproducible():
    pool = _pool([_d([1, 1, 1, 1], [[1, 1], [1, 1]]),
                  _d([1, 1, 1, 1], [[1, 1], [1, 1]])])   # identical benign clients
    p1 = build_attacker("noise", seed=7).act(_Env(pool), _Ctx(pool, budget=2))[0]
    p2 = build_attacker("noise", seed=7).act(_Env(pool), _Ctx(pool, budget=2))[0]
    p3 = build_attacker("noise", seed=8).act(_Env(pool), _Ctx(pool, budget=2))[0]
    # Same seed → identical; different seed → different.
    assert torch.equal(p1[0]["a"], p2[0]["a"])
    assert not torch.equal(p1[0]["a"], p3[0]["a"])
    # Independent per-client noise → the two malicious clients differ (not clones).
    assert not torch.equal(p1[0]["a"], p1[1]["a"])


# --- optimization attacks: colluding + exact formula ------------------------

def test_lie_matches_mean_minus_z_sigma_and_colludes():
    d0, d1 = _d([2, 4, 6, 8], [[1, 1], [1, 1]]), _d([0, 0, 0, 0], [[0, 0], [0, 0]])
    pool = _pool([d0, d1])
    stats = BenignStats(pool, _global())
    atk = build_attacker("lie", lie_z=1.0, seed=0)
    poisoned, _, _ = atk.act(_Env(pool), _Ctx(pool, budget=2))
    expect = {k: stats.mean[k] - 1.0 * stats.std[k] for k in ("a", "b")}
    assert torch.allclose(_delta(poisoned[0])["a"], expect["a"], atol=1e-5)
    assert torch.equal(poisoned[0]["a"], poisoned[1]["a"])       # colluding: identical


def test_ipm_matches_negative_epsilon_mean():
    d0, d1 = _d([2, 4, 6, 8], [[2, 2], [2, 2]]), _d([0, 2, 0, 2], [[0, 0], [0, 0]])
    pool = _pool([d0, d1])
    stats = BenignStats(pool, _global())
    atk = build_attacker("ipm", ipm_eps=2.0, seed=0)
    poisoned, _, _ = atk.act(_Env(pool), _Ctx(pool, budget=2))
    assert torch.allclose(_delta(poisoned[0])["a"], -2.0 * stats.mean["a"], atol=1e-5)
    assert torch.equal(poisoned[0]["b"], poisoned[1]["b"])


@pytest.mark.parametrize("name,mode", [("min_max", "max"), ("min_sum", "sum")])
def test_minmax_minsum_respect_their_distance_bound(name, mode):
    pool = _pool([_d([1, 2, 3, 4], [[1, 0], [0, 1]]),
                  _d([2, 0, 1, 3], [[0, 1], [1, 0]]),
                  _d([0, 1, 2, 2], [[1, 1], [0, 1]])])
    stats = BenignStats(pool, _global())
    atk = build_attacker(name, attack_dev="sign", seed=0)
    poisoned, _, _ = atk.act(_Env(pool), _Ctx(pool, budget=3))
    # All colluding clients identical.
    assert torch.equal(poisoned[0]["a"], poisoned[2]["a"])
    # The crafted update honors the published constraint (with binary-search slack).
    deltas = stats.flat_deltas()
    mal = torch.cat([_delta(poisoned[0])[k].reshape(-1) for k in ("a", "b")])
    pdist = torch.cdist(deltas, deltas)
    if mode == "max":
        bound = float(pdist.max())
        got = float((mal.unsqueeze(0) - deltas).norm(dim=1).max())
    else:
        bound = float((pdist ** 2).sum(dim=1).max())
        got = float(((mal.unsqueeze(0) - deltas).norm(dim=1) ** 2).sum())
    assert got <= bound * (1 + 1e-3) + 1e-6
    # ...and it actually deviated (gamma > 0 for a pool with real spread).
    assert mal.norm() > stats.flat_mean().norm() * 0.5


def test_minmax_degrades_to_honest_mean_with_a_single_client():
    d0 = _d([1, 2, 3, 4], [[1, 0], [0, 1]])
    pool = _pool([d0])
    atk = build_attacker("min_max", seed=0)
    poisoned, _, _ = atk.act(_Env(pool), _Ctx(pool, budget=1))
    # No honest spread to hide in → gamma 0 → the crafted update is the honest one.
    assert torch.allclose(_delta(poisoned[0])["a"], d0["a"], atol=1e-5)


def test_fang_deviates_against_the_mean_sign():
    d0, d1 = _d([2, -4, 6, -8], [[1, 1], [1, 1]]), _d([4, -2, 8, -6], [[1, 1], [1, 1]])
    pool = _pool([d0, d1])
    stats = BenignStats(pool, _global())
    atk = build_attacker("fang", fang_lambda=2.0, seed=0)
    poisoned, _, _ = atk.act(_Env(pool), _Ctx(pool, budget=2))
    mal = _delta(poisoned[0])["a"]
    # Deviation moves opposite the sign of the honest mean on every coordinate.
    moved = mal - stats.mean["a"]
    assert torch.all(torch.sign(moved) == -torch.sign(stats.mean["a"]))
    assert torch.equal(poisoned[0]["a"], poisoned[1]["a"])       # colluding


# --- helper-function coverage ----------------------------------------------

def test_lie_z_formula_is_finite_or_none():
    # A degenerate (n, m) returns None; a healthy one returns a positive z.
    assert lie_z_from_counts(1, 1) is None
    z = lie_z_from_counts(50, 10)
    assert z is None or z > 0.0


def test_perturbation_direction_variants():
    pool = _pool([_d([1, -2, 3, -4], [[1, -1], [1, -1]]),
                  _d([3, -1, 1, -2], [[2, -2], [0, 0]])])
    stats = BenignStats(pool, _global())
    for kind in ("sign", "std", "unit"):
        d = perturbation_direction(stats, kind)
        assert d.numel() == stats.flat_mean().numel()
    with pytest.raises(ValueError):
        perturbation_direction(stats, "bogus")


# --- end-to-end: run_benchmark with a scripted attacker vs a real defense ----

class _Env:
    """Round generator with a fixed benign pool; the benchmark never commits to
    the env (each defense owns its own global). ``global_sd``/``honest_sd`` default
    to a synthetic 2-key model for the unit tests (no eval); the end-to-end test
    passes a real MnistNet state_dict so the harness's eval server can load it."""

    def __init__(self, pool, n_clients=6, global_sd=None, honest_sd=None):
        self.pool = pool
        self.n_clients = n_clients
        self.round_num = 0
        self.global_weights = global_sd if global_sd is not None else _global()
        self._honest = honest_sd if honest_sd is not None else {
            k: v + 0.01 for k, v in self.global_weights.items()}   # honest cluster

    def begin_round(self):
        self.round_num += 1
        return _Ctx(self.pool, budget=len(self.pool), round_num=self.round_num)

    def set_committed_poison(self, chosen_ids):
        pass

    def build_updates(self, poisoned_by_client):
        return [ModelUpdate(client_id=cid,
                            weights=poisoned_by_client.get(cid,
                                    {k: v.clone() for k, v in self._honest.items()}),
                            metadata={"poisoned": cid in poisoned_by_client})
                for cid in range(self.n_clients)]


class _Ctx:
    def __init__(self, pool, budget, round_num=1):
        self.round_num = round_num
        self.budget = budget
        self.pool_benign = pool


class _Loader:
    def __iter__(self):
        return iter(())


def test_run_benchmark_end_to_end_with_scripted_attacker():
    """A clearly-outlier scripted attack (scaling) is detected by Multi-Krum but
    not by no-defense, proving the scripted attacker flows through the real harness
    + a real defense and produces sane per-defense detection metrics."""
    from benchmark.harness import run_benchmark
    from benchmark.defenses import build_defenses
    from model.mnist_net import MnistNet

    # Real model weights so the harness's eval server can load each defense's global.
    g = {k: v.detach().clone() for k, v in MnistNet().state_dict().items()}
    honest = {k: v + 0.01 for k, v in g.items()}          # honest cluster (G + 0.01)
    # The 2 malicious clients' benign update is G+0.05; scaling ×10 sends G+0.5,
    # norm outliers Multi-Krum should reject relative to the honest cluster.
    pool = {0: {k: v + 0.05 for k, v in g.items()},
            1: {k: v + 0.05 for k, v in g.items()}}
    env = _Env(pool, n_clients=6, global_sd=g, honest_sd=honest)
    defenses = build_defenses(["fedavg", "multikrum"], device="cpu",
                              multikrum_num_byzantine=2)
    attacker = build_attacker("scaling", attack_scale=10.0, seed=0)

    summaries, metrics = run_benchmark(
        env, None, None, defenses, _Loader(),
        init_global=g, baseline_accuracy=0.9, n_rounds=3,
        device="cpu", scripted_attacker=attacker, log_every=10)

    assert summaries["fedavg"]["rounds"] == 3            # nothing skipped
    assert summaries["fedavg"]["detection_rate"] == 0.0  # no-defense flags nobody
    assert summaries["multikrum"]["detection_rate"] > 0.0  # outliers get caught
