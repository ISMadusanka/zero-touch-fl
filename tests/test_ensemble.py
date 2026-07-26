"""Tests for the algorithmic ensemble defense (--freeze defender's defense).

Two halves:
  * pure vote logic (torch-free)
  * the end-to-end ensemble over the real member defenses (needs torch)

Run:  python tests/test_ensemble.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.types import ModelUpdate  # noqa: E402
from benchmark.defenses import build_defenses  # noqa: E402
from benchmark.defenses.ensemble import (  # noqa: E402
    EnsembleDefense, combine_votes, resolve_min_votes,
)


# ---------------------------------------------------------------------------
# Vote logic (no torch needed)
# ---------------------------------------------------------------------------

def test_resolve_min_votes():
    assert resolve_min_votes("majority", 4) == 2
    assert resolve_min_votes("majority", 3) == 2      # ceil(3/2)
    assert resolve_min_votes("majority", 1) == 1
    assert resolve_min_votes("any", 4) == 1
    assert resolve_min_votes("all", 4) == 4
    assert resolve_min_votes("unanimous", 4) == 4
    assert resolve_min_votes(3, 4) == 3
    assert resolve_min_votes("3", 4) == 3
    assert resolve_min_votes(99, 4) == 4              # clamped to n
    assert resolve_min_votes(0, 4) == 1               # clamped to >= 1
    try:
        resolve_min_votes("sometimes", 4)
        assert False, "should reject an unknown vote rule"
    except ValueError:
        pass


def test_combine_votes_threshold():
    flags = {"fltrust": {0, 1}, "multikrum": {1}, "dnc": {1, 2}, "defl": set()}
    by_id = {v.client_id: v for v in combine_votes(flags, [0, 1, 2, 3], min_votes=2)}
    assert not by_id[0].is_suspicious      # 1 vote  < 2
    assert by_id[1].is_suspicious          # 3 votes >= 2
    assert not by_id[2].is_suspicious      # 1 vote
    assert not by_id[3].is_suspicious      # 0 votes
    # "any" catches everything at least one member flagged.
    any_ids = {v.client_id for v in combine_votes(flags, [0, 1, 2, 3], min_votes=1)
               if v.is_suspicious}
    assert any_ids == {0, 1, 2}
    # "all" (4 of 4) catches nobody here.
    assert not any(v.is_suspicious for v in combine_votes(flags, [0, 1, 2, 3], min_votes=4))


def test_combine_votes_confidence_is_monotone_in_agreement():
    """The attacker's stealth reward reads `confidence`, so P(malicious) derived
    from it must rise with the number of members that flagged the client —
    otherwise fooling 3 of 4 members would score the same as fooling none."""
    from rl.rewards import _soft_malicious_prob

    def prob(n_flagging, n_members=4, min_votes=2):
        flags = {f"m{i}": ({7} if i < n_flagging else set()) for i in range(n_members)}
        v = combine_votes(flags, [7], min_votes)[0]
        return _soft_malicious_prob(v)

    probs = [prob(k) for k in range(5)]
    assert probs == sorted(probs), probs
    assert probs[0] == 0.0 and probs[4] == 1.0


def test_combine_votes_reason_names_the_voters():
    flags = {"fltrust": {0}, "dnc": {0}, "defl": set()}
    v = combine_votes(flags, [0], min_votes=2)[0]
    assert v.is_suspicious and "2/3 votes" in v.reason
    assert "fltrust" in v.reason and "dnc" in v.reason


# ---------------------------------------------------------------------------
# End-to-end over the real member defenses
# ---------------------------------------------------------------------------

def _world(n_clients=8, seed=0):
    """A global model plus n honest client updates clustered around it."""
    torch.manual_seed(seed)
    global_weights = {"net.2.weight": torch.randn(16, 49) * 0.1,
                      "net.2.bias": torch.zeros(16),
                      "net.4.weight": torch.randn(10, 16) * 0.1,
                      "net.4.bias": torch.zeros(10)}
    updates = []
    for cid in range(n_clients):
        w = {k: v + torch.randn_like(v) * 0.01 for k, v in global_weights.items()}
        updates.append(ModelUpdate(client_id=cid, weights=w, metadata={"poisoned": False}))
    return global_weights, updates


def _ensemble(members=("multikrum", "dnc", "defl"), vote="majority", **kw):
    kw.setdefault("dnc_num_byzantine", 1)
    kw.setdefault("multikrum_num_byzantine", 1)
    return EnsembleDefense(build_defenses(list(members), **kw), vote=vote)


def test_ensemble_catches_a_blatant_poisoner():
    global_weights, updates = _world()
    # Client 3 scales its whole model 20x — loud on distance, spectrum AND per-layer norm.
    updates[3].weights = {k: v * 20.0 for k, v in updates[3].weights.items()}

    ens = _ensemble()
    ens.reset(global_weights)
    verdicts, info = ens.detect(updates, global_weights)
    flagged = {v.client_id for v in verdicts if v.is_suspicious}
    assert flagged == {3}, (flagged, info["per_member"])
    assert info["min_votes"] == 2 and info["n_members"] == 3


def test_ensemble_vote_rule_changes_strictness():
    """`any` is the union of the members' rejects, `all` their intersection, so on
    the same round: any >= majority >= all."""
    global_weights, updates = _world()
    updates[3].weights = {k: v * 20.0 for k, v in updates[3].weights.items()}

    sizes = []
    for vote in ("any", "majority", "all"):
        ens = _ensemble(vote=vote)
        ens.reset(global_weights)
        verdicts, _ = ens.detect(updates, global_weights)
        sizes.append(len({v.client_id for v in verdicts if v.is_suspicious}))
    assert sizes[0] >= sizes[1] >= sizes[2], sizes


def test_scoring_detection_does_not_advance_member_state():
    """Scoring the G candidate attacks of a GRPO round must leave the defense
    exactly as the committed round will find it — otherwise DeFL's Beta counts and
    CLP baseline would be advanced G+1 times per round."""
    global_weights, updates = _world()
    updates[3].weights = {k: v * 20.0 for k, v in updates[3].weights.items()}

    ens = _ensemble(members=("defl", "dnc", "multikrum"))
    ens.reset(global_weights)
    defl = ens.members["defl"]

    for _ in range(5):                       # five scored rollouts
        ens.detect(updates, global_weights, advance_state=False)
    assert defl._prev_total_fgnv is None     # untouched
    assert defl._beta.alpha == {} and defl._beta.beta == {}

    ens.detect(updates, global_weights, advance_state=True)      # the committed round
    assert defl._prev_total_fgnv is not None
    assert len(defl._beta.alpha) + len(defl._beta.beta) > 0


def test_detect_is_deterministic_and_leaves_the_reference_model_alone():
    global_weights, updates = _world()
    updates[2].weights = {k: -v for k, v in updates[2].weights.items()}   # sign flip

    ens = _ensemble()
    ens.reset(global_weights)
    before = {k: v.clone() for k, v in global_weights.items()}
    first, _ = ens.detect(updates, global_weights)
    second, _ = ens.detect(updates, global_weights)
    assert [(v.client_id, v.is_suspicious) for v in first] == \
           [(v.client_id, v.is_suspicious) for v in second]
    for k in before:
        assert torch.equal(before[k], global_weights[k]), f"{k} was mutated"


def test_step_drops_the_rejected_clients_from_the_average():
    global_weights, updates = _world()
    updates[3].weights = {k: v * 20.0 for k, v in updates[3].weights.items()}

    ens = _ensemble()
    ens.reset(global_weights)
    result = ens.step(updates, {3})
    assert {v.client_id for v in result.verdicts if v.is_suspicious} == {3}
    # The new global is the mean of the KEPT clients — client 3's 20x weights are
    # nowhere in it (a plain FedAvg over all 8 would be visibly larger).
    kept = [u for u in updates if u.client_id != 3]
    expected = torch.stack([u.weights["net.4.bias"] for u in kept]).mean(dim=0)
    assert torch.allclose(result.new_global["net.4.bias"], expected, atol=1e-6)
    assert torch.equal(ens.global_weights()["net.4.bias"], result.new_global["net.4.bias"])


def test_forbidden_and_llm_members_are_rejected():
    global_weights, _ = _world()
    for bad in ("oracle", "llm_defender", "ensemble"):
        try:
            EnsembleDefense({bad: object()})
            assert False, f"should refuse member {bad!r}"
        except ValueError:
            pass
    try:
        EnsembleDefense({})
        assert False, "should refuse an empty member set"
    except ValueError:
        pass
    assert global_weights is not None


def test_registry_builds_the_ensemble():
    """`--defenses ...,ensemble` works from the benchmark registry too."""
    built = build_defenses(["ensemble"], ensemble_members=["multikrum", "dnc"],
                           ensemble_vote="all", multikrum_num_byzantine=1,
                           dnc_num_byzantine=1)
    ens = built["ensemble"]
    assert list(ens.members) == ["multikrum", "dnc"] and ens.min_votes == 2
    assert "multikrum,dnc" in ens.describe()


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} ensemble-defense tests passed.")


if __name__ == "__main__":
    _run()
