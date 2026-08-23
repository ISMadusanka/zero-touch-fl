"""Tests for the benchmark's ATTACK panel — the published untargeted attacks the
trained policy is compared against, and the harness invariants that make the
comparison fair.

The properties worth asserting are the ones a silent bug would break without
crashing: that each attack computes the formula its paper states, that every attack
in a round poisons exactly the same clients, and that one attack's defenses cannot
contaminate another's.

Needs torch (the attacks are tensor code). Run:  python tests/test_benchmark_attacks.py
"""
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                             # noqa: E402

from benchmark.attacks import AVAILABLE, BASELINES, DEFAULT, build_attacks  # noqa: E402
from benchmark.attacks.agr_agnostic import (              # noqa: E402
    MinMax, MinSum, pairwise_sq_dists, perturbation, solve_gamma,
)
from benchmark.attacks.base import (                      # noqa: E402
    AttackContext, flatten, float_keys, unflatten_like,
)
from benchmark.attacks.classic import GaussianNoise, Scaling, SignFlip  # noqa: E402
from benchmark.attacks.fang import FangKrum, FangTrimmedMean, outside_range  # noqa: E402
from benchmark.attacks.ipm import IPM                     # noqa: E402
from benchmark.attacks.lie import LIE, _norm_ppf, lie_z    # noqa: E402
from benchmark.attacks.mimic import Mimic                  # noqa: E402

# Degenerate-cohort warnings are the expected path in a couple of tests.
logging.getLogger("benchmark").addHandler(logging.NullHandler())
logging.getLogger("benchmark").propagate = False

_N_CLIENTS = 8
_N_POISON = 4


def _sd(seed: int, scale: float = 1.0) -> dict:
    """A real model state_dict, deterministically perturbed."""
    from model.nidd_net import NiddNet
    g = torch.Generator().manual_seed(seed)
    return {k: (v.detach() + scale * torch.randn(v.shape, generator=g))
            for k, v in NiddNet().state_dict().items()}


def _ctx(n_clients: int = _N_CLIENTS, n_poison: int = _N_POISON,
         knowledge: str = "partial", pool: int | None = None) -> AttackContext:
    """A round where the honest clients differ from each other and from the global."""
    g = _sd(0, 0.0)
    honest = {cid: _sd(100 + cid, 0.05) for cid in range(n_clients)}
    pool_ids = list(range(pool if pool is not None else n_poison))
    known = pool_ids if knowledge == "partial" else list(range(n_clients))
    return AttackContext(
        round_num=1, global_weights=g, honest=honest,
        poisoned_ids=list(range(n_poison)), known_ids=known, pool_ids=pool_ids,
        n_clients=n_clients, keys=float_keys(g))


def _delta(ctx: AttackContext, state: dict) -> torch.Tensor:
    return flatten(state, ctx.keys) - flatten(ctx.global_weights, ctx.keys)


# --- flat-vector plumbing ---------------------------------------------------

def test_flatten_unflatten_round_trips_and_does_not_alias():
    sd = _sd(1, 0.1)
    keys = float_keys(sd)
    assert keys == list(sd), "the BN-free model has no non-float entries to skip"
    back = unflatten_like(flatten(sd, keys), sd, keys)
    assert all(torch.equal(back[k], sd[k]) for k in sd)
    # A NEW tensor per key: the honest updates are shared by every attack and every
    # defense in a round, so writing through would corrupt the whole panel.
    back["net.0.weight"] += 1.0
    assert not torch.equal(back["net.0.weight"], sd["net.0.weight"])


def test_context_deltas_are_updates_against_the_round_global():
    ctx = _ctx()
    d = ctx.deltas_for([0, 1])
    assert d.shape == (2, sum(v.numel() for v in ctx.global_weights.values()))
    expect = flatten(ctx.honest[0], ctx.keys) - flatten(ctx.global_weights, ctx.keys)
    assert torch.allclose(d[0], expect)


def test_partial_knowledge_shows_only_the_compromised_clients():
    assert _ctx(knowledge="partial").known_deltas().shape[0] == _N_POISON
    assert _ctx(knowledge="full").known_deltas().shape[0] == _N_CLIENTS


# --- LIE --------------------------------------------------------------------

def test_norm_ppf_matches_known_quantiles():
    for p, want in ((0.5, 0.0), (0.9, 1.2815515655), (0.975, 1.9599639845),
                    (0.01, -2.3263478740)):
        assert abs(_norm_ppf(p) - want) < 1e-6, p


def test_lie_z_follows_the_paper_formula():
    # n=20, m=10 -> s = floor(11) - 10 = 1, q = (20-10-1)/(20-10) = 0.9
    assert abs(lie_z(20, 10) - _norm_ppf(0.9)) < 1e-9
    # n=8, m=2 -> s = 5 - 2 = 3, q = (8-2-3)/(8-2) = 0.5 -> z = 0
    assert abs(lie_z(8, 2) - 0.0) < 1e-9
    # More colluders than the median needs: unbounded, clamped rather than inf.
    assert math.isfinite(lie_z(8, 6)) and lie_z(8, 6) >= 4.0


def test_lie_emits_mu_plus_signed_z_sigma_identically_for_every_client():
    ctx = _ctx()
    known = ctx.known_deltas()
    out = LIE(z=1.5, sign=-1.0).craft(ctx)
    assert sorted(out) == ctx.poisoned_ids
    want = known.mean(0) - 1.5 * known.std(0, unbiased=False)
    for cid in out:
        assert torch.allclose(_delta(ctx, out[cid]), want, atol=1e-6)


def test_lie_degenerates_loudly_rather_than_silently_when_it_cannot_see_a_population():
    ctx = _ctx(n_poison=1, pool=1)
    out = LIE().craft(ctx)
    # sigma over one sample is 0, so the "attack" is the honest update itself.
    assert torch.allclose(_delta(ctx, out[0]), ctx.known_deltas()[0], atol=1e-6)


# --- IPM --------------------------------------------------------------------

def test_ipm_is_the_negatively_scaled_honest_mean():
    ctx = _ctx()
    out = IPM(epsilon=0.3).craft(ctx)
    want = -0.3 * ctx.known_deltas().mean(0)
    assert all(torch.allclose(_delta(ctx, w), want, atol=1e-6) for w in out.values())


# --- Min-Max / Min-Sum ------------------------------------------------------

def test_perturbation_directions_are_the_papers_three():
    known = _ctx().known_deltas()
    mu = known.mean(0)
    assert torch.allclose(perturbation(known, mu, "std"), known.std(0, unbiased=False))
    assert torch.allclose(perturbation(known, mu, "unit_vec"), mu / mu.norm())
    assert torch.allclose(perturbation(known, mu, "sign"), torch.sign(mu))


def test_solve_gamma_finds_a_positive_feasible_step_and_stops_at_the_boundary():
    known = _ctx().known_deltas()
    mu = known.mean(0)
    d2 = pairwise_sq_dists(known)
    limit = float(d2.max())
    gamma, mal = solve_gamma(known, mu, perturbation(known, mu, "std"),
                             lambda x: float(x.max()), limit, gamma0=10.0)
    assert gamma > 0.0
    # Feasible at the solution...
    assert float(((known - mal) ** 2).sum(dim=1).max()) <= limit * (1 + 1e-6)
    # ...and infeasible a good way past it, i.e. the search really is at the boundary.
    over = mu - (gamma * 2.0) * perturbation(known, mu, "std")
    assert float(((known - over) ** 2).sum(dim=1).max()) > limit


def test_min_max_output_respects_its_own_feasibility_constraint():
    ctx = _ctx()
    known = ctx.known_deltas()
    limit = float(pairwise_sq_dists(known).max())
    out = MinMax(perturbation_type="std").craft(ctx)
    mal = _delta(ctx, out[0])
    assert float(((known - mal) ** 2).sum(dim=1).max()) <= limit * (1 + 1e-5)
    assert not torch.allclose(mal, known.mean(0))          # it actually perturbed


def test_min_sum_respects_its_constraint_and_the_min_bound_is_the_weaker_attack():
    ctx = _ctx()
    known = ctx.known_deltas()
    sums = pairwise_sq_dists(known).sum(dim=1)
    mal_max = _delta(ctx, MinSum(bound="max").craft(ctx)[0])
    mal_min = _delta(ctx, MinSum(bound="min").craft(ctx)[0])
    assert float(((known - mal_max) ** 2).sum(dim=1).sum()) <= float(sums.max()) * (1 + 1e-5)
    assert float(((known - mal_min) ** 2).sum(dim=1).sum()) <= float(sums.min()) * (1 + 1e-5)
    mu = known.mean(0)
    assert (mal_max - mu).norm() >= (mal_min - mu).norm()


# --- Fang -------------------------------------------------------------------

def test_outside_range_stays_outside_whatever_the_sign_of_the_bound():
    bound = torch.tensor([2.0, -2.0, 0.0])
    lo, hi = outside_range(bound, 2.0, above=True)
    assert torch.all(hi >= bound) and torch.all(lo >= torch.minimum(bound, hi) - 1e-9)
    assert float(hi[0]) == 4.0 and float(hi[1]) == -1.0     # b*max / max/b
    lo, hi = outside_range(bound, 2.0, above=False)
    assert float(lo[0]) == 1.0 and float(lo[1]) == -4.0     # min/b / b*min


def test_fang_places_every_coordinate_on_the_far_side_of_the_honest_range():
    ctx = _ctx()
    known = ctx.known_deltas()
    s = torch.sign(known.mean(0))
    lo, hi = known.min(0).values, known.max(0).values
    out = FangTrimmedMean(b=2.0, seed=3).craft(ctx)
    assert sorted(out) == ctx.poisoned_ids
    mal = _delta(ctx, out[0])
    rising, falling = s > 0, s < 0
    # Where the honest mean rises the crafted value is at/below the honest minimum,
    # and where it falls it is at/above the honest maximum: the aggregate is dragged
    # against the honest direction in every coordinate that has one.
    assert torch.all(mal[rising] <= lo[rising] + 1e-6)
    assert torch.all(mal[falling] >= hi[falling] - 1e-6)


def test_fang_gives_each_compromised_client_its_own_draw():
    out = FangTrimmedMean(seed=7).craft(_ctx())
    first = out[0]
    assert any(not torch.equal(out[cid]["net.0.weight"], first["net.0.weight"])
               for cid in out if cid != 0)


def test_fang_krum_lands_a_crafted_update_in_krums_selection():
    """The whole point of the AGR-tailored attack: Krum must PICK a malicious update."""
    from benchmark.defenses.multikrum import (
        k_closest_count, krum_scores, pairwise_sq_dists as mk_dists, select_lowest,
    )
    ctx = _ctx()
    known = ctx.known_deltas()
    m = ctx.n_malicious
    out = FangKrum(num_byzantine=m).craft(ctx)
    mal = _delta(ctx, out[0])
    assert all(torch.allclose(_delta(ctx, w), mal, atol=1e-6) for w in out.values())
    cohort = torch.cat([mal.unsqueeze(0).expand(m, -1), known], dim=0)
    n_sim = cohort.shape[0]
    scores = krum_scores(mk_dists(cohort), k_closest_count(n_sim, m))
    assert min(select_lowest(scores, 1)) < m, "Krum did not select a crafted update"


# --- Mimic and the classics -------------------------------------------------

def test_mimic_submits_an_unmodified_honest_update():
    ctx = _ctx(knowledge="full")
    known = ctx.known_deltas()
    out = Mimic(warmup_iters=5, seed=1).craft(ctx)
    mal = _delta(ctx, out[0])
    assert any(torch.allclose(mal, known[i], atol=1e-6) for i in range(known.shape[0]))
    assert all(torch.allclose(_delta(ctx, w), mal, atol=1e-6) for w in out.values())


def test_mimic_picks_the_client_furthest_along_the_top_variance_direction():
    ctx = _ctx(knowledge="full")
    known = ctx.known_deltas()
    attack = Mimic(warmup_iters=30, seed=1)
    mal = _delta(ctx, attack.craft(ctx)[0])
    centered = known - known.mean(0, keepdim=True)
    want = ctx.known_ids[int((centered @ attack._z).argmax())]
    assert torch.allclose(mal, known[ctx.known_ids.index(want)], atol=1e-6)


def test_sign_flip_negates_each_clients_own_update():
    ctx = _ctx()
    out = SignFlip(c=2.0).craft(ctx)
    for i, cid in enumerate(ctx.poisoned_ids):
        assert torch.allclose(_delta(ctx, out[cid]), -2.0 * ctx.deltas_for(ctx.poisoned_ids)[i],
                              atol=1e-6)


def test_scaling_boosts_each_clients_own_update():
    ctx = _ctx()
    out = Scaling(gamma=5.0).craft(ctx)
    d = ctx.deltas_for(ctx.poisoned_ids)
    assert all(torch.allclose(_delta(ctx, out[cid]), 5.0 * d[i], atol=1e-6)
               for i, cid in enumerate(ctx.poisoned_ids))


def test_noise_scales_with_the_honest_spread_and_differs_per_client():
    ctx = _ctx()
    small = GaussianNoise(sigma=1.0, seed=0).craft(ctx)
    big = GaussianNoise(sigma=100.0, seed=0).craft(ctx)
    assert _delta(ctx, big[0]).norm() > 50 * _delta(ctx, small[0]).norm()
    assert not torch.allclose(_delta(ctx, small[0]), _delta(ctx, small[1]))


def test_every_baseline_returns_exactly_the_requested_clients_and_changes_them():
    ctx = _ctx(knowledge="full")
    for name in BASELINES:
        if name in ("label_flip", "clean"):
            continue           # label_flip needs data loaders; clean is a control row
        attack = build_attacks([name], seed=2)[name]
        out = attack.craft(ctx)
        assert sorted(out) == ctx.poisoned_ids, name
        changed = sum(1 for cid, w in out.items()
                      if not all(torch.equal(w[k], ctx.honest[cid][k]) for k in w))
        # Mimic legitimately leaves the client it copies untouched; everything else
        # must actually poison every client it was handed.
        assert changed >= len(out) - (1 if name == "mimic" else 0), name


def test_clean_control_row_submits_the_honest_updates_untouched():
    ctx = _ctx()
    attack = build_attacks(["clean"])["clean"]
    assert attack.poisons is False, "the control row must be scored against no ground truth"
    out = attack.craft(ctx)
    assert sorted(out) == ctx.poisoned_ids
    for cid, w in out.items():
        assert all(torch.equal(w[k], ctx.honest[cid][k]) for k in w)
        # ...but as fresh tensors: a defense writing in place must not reach the
        # honest updates the rest of the panel still has to use this round.
        assert all(w[k] is not ctx.honest[cid][k] for k in w)


# --- registry ---------------------------------------------------------------

def test_registry_contents_and_errors():
    assert DEFAULT[0] == "llm" and set(DEFAULT) <= set(AVAILABLE)
    assert "llm" not in BASELINES and set(BASELINES) < set(AVAILABLE)
    assert list(build_attacks(["ipm", "lie"])) == ["ipm", "lie"]     # order preserved
    for bad, exc in ((["nope"], ValueError), (["llm"], ValueError),
                     (["label_flip"], ValueError)):
        try:
            build_attacks(bad)
        except exc:
            continue
        raise AssertionError(f"build_attacks({bad}) should have raised")


def _all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} benchmark-attack tests passed.")


if __name__ == "__main__":
    _all()
