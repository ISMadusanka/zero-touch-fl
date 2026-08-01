"""Tests for the algorithmic defense ensemble used by ``main.py --freeze defender``.

Covers the union-of-rejections contract, the consensus confidence the attacker's
stealth reward reads, and the rollback that keeps GRPO rollout scoring from
advancing a stateful defense's history.

Needs torch (the real defenses flatten tensors):  python tests/test_defense_ensemble.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.types import DetectionVerdict, ModelUpdate  # noqa: E402
from benchmark.defenses.base import Defense, StepResult  # noqa: E402
from benchmark.defenses.defl import DeFL  # noqa: E402
from benchmark.defenses.multikrum import MultiKrum  # noqa: E402
from server.defense_ensemble import ALGORITHMIC, DefenseEnsemble, build_ensemble  # noqa: E402


def _zeros_global():
    return {
        "net.2.weight": torch.zeros(2, 2),
        "net.2.bias": torch.zeros(2),
        "net.4.weight": torch.zeros(2, 2),
        "net.4.bias": torch.zeros(2),
    }


def _const(value):
    return {k: torch.full_like(v, float(value)) for k, v in _zeros_global().items()}


def _updates(poison_value=10.0, benign_value=0.1, n=5):
    ups = [ModelUpdate(client_id=0, weights=_const(poison_value))]
    ups += [ModelUpdate(client_id=c, weights=_const(benign_value)) for c in range(1, n)]
    return ups


class FakeDefense(Defense):
    """Flags a fixed set of client ids; counts how often it was stepped."""

    def __init__(self, name, flag_ids):
        super().__init__("cpu")
        self.name = name
        self.flag_ids = set(flag_ids)
        self.steps = 0

    def step(self, updates, poisoned_ids):
        self.steps += 1
        verdicts = [DetectionVerdict(u.client_id, u.client_id in self.flag_ids, 0.9, self.name)
                    for u in updates]
        return StepResult({k: v.clone() for k, v in self._global.items()}, verdicts)


# ---------------------------------------------------------------------------
# Union / consensus contract
# ---------------------------------------------------------------------------

def test_union_flags_a_client_any_algorithm_rejects():
    ens = DefenseEnsemble({"a": FakeDefense("a", [0]), "b": FakeDefense("b", [3]),
                           "c": FakeDefense("c", [])}, mode="union")
    verdicts, info = ens.verdicts(_updates(), _zeros_global())
    flagged = {v.client_id for v in verdicts if v.is_suspicious}
    assert flagged == {0, 3}                      # union, not intersection/majority
    assert info["flagged"] == [0, 3]
    assert info["per_defense_flags"] == {"a": [0], "b": [3], "c": []}


def test_confidence_is_the_fraction_of_algorithms_that_flagged():
    ens = DefenseEnsemble({"a": FakeDefense("a", [0, 1]), "b": FakeDefense("b", [0]),
                           "c": FakeDefense("c", [0]), "d": FakeDefense("d", [])},
                          mode="union")
    by_id = {v.client_id: v for v in ens.verdicts(_updates(), _zeros_global())[0]}
    assert by_id[0].is_suspicious and abs(by_id[0].confidence - 0.75) < 1e-9   # 3 of 4
    assert by_id[1].is_suspicious and abs(by_id[1].confidence - 0.25) < 1e-9   # 1 of 4
    # Cleared by everyone -> benign with FULL confidence, so the attacker's stealth
    # term reads a soft P(malicious) of 0 rather than an ambiguous 0.5.
    assert by_id[2].is_suspicious is False and by_id[2].confidence == 1.0


def test_a_failing_algorithm_does_not_kill_the_round():
    """One algorithm raising must not abort training, must flag nobody, and must be
    reported — never silently dropped from the panel."""
    class Boom(FakeDefense):
        def step(self, updates, poisoned_ids):
            raise RuntimeError("svd did not converge")

    ens = DefenseEnsemble({"boom": Boom("boom", [0]), "ok": FakeDefense("ok", [2])},
                          mode="union")
    verdicts, info = ens.verdicts(_updates(), _zeros_global())
    flagged = {v.client_id for v in verdicts if v.is_suspicious}
    assert flagged == {2}, "the surviving algorithm should still decide"
    assert "boom" in info["errors"] and info["per_defense_flags"]["boom"] == []


# ---------------------------------------------------------------------------
# Single-defense mode (the default): ONE algorithm judges each round
# ---------------------------------------------------------------------------

def _panel(**kw):
    return DefenseEnsemble({"a": FakeDefense("a", [0]), "b": FakeDefense("b", [1]),
                            "c": FakeDefense("c", [2])}, **kw)


def test_single_mode_runs_exactly_one_algorithm():
    """Union of four aggregators rejected 14/20 clients on a CLEAN round, which both
    closed every gap an attack could use and swamped the `drop` reward with noise.
    One at a time keeps each defense honest and leaves a learnable frontier."""
    ens = _panel()
    assert ens.mode == "single"                       # the shipped default
    ens.begin_round()
    verdicts, info = ens.verdicts(_updates(), _zeros_global())
    assert {v.client_id for v in verdicts if v.is_suspicious} == {0}   # only "a" ran
    assert info["algorithms"] == ["a"] and info["configured"] == ["a", "b", "c"]
    assert set(info["per_defense_flags"]) == {"a"}
    # The round log merges this into {"mode": "algorithmic"}; a "mode" key here
    # would overwrite that and break consumers branching on who judged the round.
    assert "mode" not in info and info["panel_mode"] == "single"


def test_rotate_advances_one_algorithm_per_round():
    ens = _panel(selection="rotate")
    picks = [ens.begin_round()[0] for _ in range(7)]
    assert picks == ["a", "b", "c", "a", "b", "c", "a"], picks


def test_fixed_selection_never_moves():
    ens = _panel(selection="fixed")
    assert [ens.begin_round()[0] for _ in range(4)] == ["a"] * 4


def test_random_selection_is_seeded_by_the_run_rng():
    import random
    picks = []
    for _ in range(2):
        ens = _panel(selection="random", rng=random.Random(1234))
        picks.append([ens.begin_round()[0] for _ in range(6)])
    assert picks[0] == picks[1], "same seed must replay the same defense schedule"


def test_the_pick_is_frozen_for_the_whole_round():
    """The clean counterfactual, the G rollout scorings and the commit must all be
    judged by the SAME algorithm — otherwise `drop` subtracts two accuracies that
    were filtered by different defenses and stops measuring the attack."""
    ens = _panel(selection="rotate")
    ens.begin_round()
    seen = set()
    for _ in range(5):                       # clean ref + G rollouts, no begin_round
        _v, info = ens.verdicts(_updates(), _zeros_global(), commit=False)
        seen.add(tuple(info["algorithms"]))
    _v, info = ens.verdicts(_updates(), _zeros_global(), commit=True)
    seen.add(tuple(info["algorithms"]))
    assert seen == {("a",)}, seen


def test_single_mode_confidence_is_a_full_flag():
    """With one active algorithm the consensus fraction degenerates to a hard flag:
    1.0 either way, so the stealth term reads a clean P(malicious) of 1 or 0."""
    ens = _panel()
    ens.begin_round()
    by_id = {v.client_id: v for v in ens.verdicts(_updates(), _zeros_global())[0]}
    assert by_id[0].is_suspicious and by_id[0].confidence == 1.0
    assert by_id[1].is_suspicious is False and by_id[1].confidence == 1.0


def test_single_mode_only_snapshots_the_active_defense():
    """Scoring rollbacks must not touch defenses that never ran this round."""
    touched = []

    class Tracker(FakeDefense):
        def state_dict(self):
            touched.append(self.name)
            return {}

    ens = DefenseEnsemble({"a": Tracker("a", [0]), "b": Tracker("b", [1])})
    ens.begin_round()
    ens.verdicts(_updates(), _zeros_global(), commit=False)
    assert touched == ["a"], touched


def test_bad_mode_or_selection_is_rejected_loudly():
    for kw in ({"mode": "majority"}, {"selection": "shuffle"}):
        try:
            _panel(**kw)
        except ValueError:
            continue
        raise AssertionError(f"{kw} should be rejected")


def test_verdicts_cover_every_update_in_order():
    ens = DefenseEnsemble({"a": FakeDefense("a", [0])})
    updates = _updates(n=6)
    verdicts, _ = ens.verdicts(updates, _zeros_global())
    assert [v.client_id for v in verdicts] == [u.client_id for u in updates]


def test_every_algorithm_sees_the_shared_global_not_its_own():
    """The ensemble judges ONE shared model: each defense is re-pointed at the
    supplied global every round instead of evolving a private copy."""
    a = FakeDefense("a", [0])
    ens = DefenseEnsemble({"a": a})
    ens.verdicts(_updates(), _zeros_global())
    later = {k: torch.full_like(v, 7.0) for k, v in _zeros_global().items()}
    ens.verdicts(_updates(), later)
    assert all(torch.allclose(v, torch.full_like(v, 7.0)) for v in a.global_weights().values())


def test_ground_truth_is_never_passed_to_a_defense():
    """These are detectors, not the benchmark's oracle — poisoned_ids must be empty."""
    seen = {}

    class Spy(FakeDefense):
        def step(self, updates, poisoned_ids):
            seen["ids"] = poisoned_ids
            return super().step(updates, poisoned_ids)

    DefenseEnsemble({"spy": Spy("spy", [])}).verdicts(_updates(), _zeros_global())
    assert seen["ids"] == set()


# ---------------------------------------------------------------------------
# Scoring must not advance a stateful defense
# ---------------------------------------------------------------------------

def test_scoring_pass_rolls_back_defl_state_and_commit_advances_it():
    defl = DeFL()
    defl.reset(_zeros_global())
    ens = DefenseEnsemble({"defl": defl})

    ens.verdicts(_updates(), _zeros_global(), commit=False)
    assert defl.state_dict()["prev_total_fgnv"] is None      # rolled back
    assert defl.state_dict()["alpha"] == {}                  # no trust accumulated

    ens.verdicts(_updates(), _zeros_global(), commit=True)
    assert defl.state_dict()["prev_total_fgnv"] is not None   # history advanced
    assert defl.state_dict()["alpha"] != {}


def test_repeated_scoring_is_deterministic():
    """G rollouts against the same state must get the same verdicts, so reward
    differences come from the attack — not from drifting defense state."""
    defl = DeFL()
    defl.reset(_zeros_global())
    ens = DefenseEnsemble({"defl": defl, "multikrum": MultiKrum(num_byzantine=1)},
                          mode="union")
    first = None
    for _ in range(3):
        verdicts, _ = ens.verdicts(_updates(), _zeros_global(), commit=False)
        flags = [(v.client_id, v.is_suspicious, v.confidence) for v in verdicts]
        first = flags if first is None else first
        assert flags == first


# ---------------------------------------------------------------------------
# End to end with the real algorithms
# ---------------------------------------------------------------------------

def test_real_algorithms_catch_a_blatant_poisoner():
    from benchmark.defenses.dnc import DnC
    ens = DefenseEnsemble({
        "multikrum": MultiKrum(num_byzantine=1),
        "dnc": DnC(num_byzantine=1),
        "defl": DeFL(),
    }, mode="union")
    for d in ens.defenses.values():
        d.reset(_zeros_global())
    verdicts, info = ens.verdicts(_updates(poison_value=10.0), _zeros_global(), commit=True)
    by_id = {v.client_id: v for v in verdicts}
    assert by_id[0].is_suspicious, info["per_defense_flags"]


def test_flagged_poisoner_is_dropped_from_the_aggregate():
    """The whole point: one detection => the poisoner never reaches FedAvg."""
    from server.aggregation import FedAvgAggregator
    ens = DefenseEnsemble({"multikrum": MultiKrum(num_byzantine=1)})
    for d in ens.defenses.values():
        d.reset(_zeros_global())
    updates = _updates(poison_value=10.0, benign_value=0.1)
    verdicts, _ = ens.verdicts(updates, _zeros_global(), commit=True)
    agg = FedAvgAggregator().aggregate(updates, verdicts)
    for v in agg.values():                       # pure benign average, no poison leak
        assert torch.allclose(v, torch.full_like(v, 0.1), atol=1e-5)


# ---------------------------------------------------------------------------
# Construction from config
# ---------------------------------------------------------------------------

def _cfg(**defense):
    return {
        "fl": {"n_clients": 20, "device": "cpu", "poison_seed": 0, "lr": 0.002,
               "batch_size": 64, "n_compromisable": 5},
        "attack": {"max_poison_clients": 5},
        "data": {"data_dir": "./data/mnist_raw"},
        "defense": defense,
    }


def test_build_ensemble_rejects_non_algorithmic_defenses():
    for bad in ("oracle", "llm_defender", "fedavg", "nope"):
        try:
            build_ensemble(_cfg(algorithms=[bad]))
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not be accepted as an algorithmic defense")


def test_build_ensemble_rejects_an_empty_panel():
    try:
        build_ensemble(_cfg(algorithms=[]))
    except ValueError:
        return
    raise AssertionError("an empty defense panel should be rejected")


def test_assumed_malicious_defaults_to_the_budget_and_keeps_a_benign_majority():
    ens = build_ensemble(_cfg(algorithms=["multikrum", "dnc"]))
    assert ens.defenses["multikrum"].num_byzantine == 5        # attack.max_poison_clients
    assert ens.defenses["dnc"].num_byzantine == 5
    cfg = _cfg(algorithms=["multikrum"], assumed_malicious=99)
    assert build_ensemble(cfg).defenses["multikrum"].num_byzantine == 9   # (20-1)//2


def test_default_panel_is_every_algorithm():
    assert set(ALGORITHMIC) == {"fltrust", "multikrum", "dnc", "defl"}


def test_build_ensemble_defaults_to_one_defense_per_round():
    ens = build_ensemble(_cfg(algorithms=["multikrum", "dnc"]))
    assert ens.mode == "single" and ens.selection == "rotate"
    assert ens.begin_round() == ["multikrum"]
    assert ens.begin_round() == ["dnc"]


def test_build_ensemble_honours_mode_and_selection():
    ens = build_ensemble(_cfg(algorithms=["multikrum", "dnc"], mode="union"))
    assert ens.mode == "union" and ens.active_names == ["multikrum", "dnc"]
    fixed = build_ensemble(_cfg(algorithms=["multikrum", "dnc"], selection="fixed"))
    assert [fixed.begin_round()[0] for _ in range(3)] == ["multikrum"] * 3


def test_build_ensemble_rejects_an_unknown_mode_or_selection():
    for bad in (dict(mode="majority"), dict(selection="shuffle")):
        try:
            build_ensemble(_cfg(algorithms=["multikrum"], **bad))
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} defense-ensemble tests passed.")


if __name__ == "__main__":
    _run()
