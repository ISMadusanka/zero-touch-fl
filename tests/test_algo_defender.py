"""Tests for the algorithmic (non-LLM) defense path.

Covers ``server/algo_defender.py`` and the env's defense hooks (``rl/env.py``).
This path is what ``--dry-run`` / ``--baseline`` / the benchmark use; the trained
defender LLM (``defense.mode: llm``) is the default for training. Synthetic
tensors shaped like MNIST — no download, no GPU, no LLM:

    python tests/test_algo_defender.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from rl.switch import PhaseController, SwitchConfig  # noqa: E402
from server.algo_defender import (  # noqa: E402
    ALGORITHMS, AlgorithmicDefender, build_algorithmic_defender,
    configured_algorithms, defense_mode,
)

N_CLIENTS = 8
POISONED = [0, 1]


def _loader(seed: int, n: int = 64):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.randn(n, 1, 28, 28, generator=g),
                      torch.randint(0, 10, (n,), generator=g)),
        batch_size=32, shuffle=True)


def _cfg(**defense):
    base = {
        # freeze_global_in_phase2: False — these tests cover the CONTINUING-federation
        # path (commit installs the defense's aggregate, the benign interlude advances
        # the shared model). The frozen simulated rounds that Phase 2 now runs by
        # default are covered by tests/test_frozen_rounds.py.
        "fl": {"n_clients": N_CLIENTS, "device": "cpu", "lr": 0.05, "local_epochs": 1,
               "training_rounds": 5, "freeze_global_in_phase2": False,
               "poison_seed": 0, "batch_size": 32},
        "data": {"data_dir": "./data/mnist_raw"},
        "attack": {"type": "label_flip", "poison_client_ids": list(POISONED),
                   "goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.2}},
        "defense": {"mode": "algorithmic",
                    # FLTrust is excluded by default here: it needs a real clean root
                    # dataset (MNIST download). It is covered by tests/test_fltrust.py.
                    "algorithms": ["defl", "dnc", "multikrum"],
                    "selection": "random"},
    }
    base["defense"].update(defense)
    return base


def _defender(cfg=None):
    return build_algorithmic_defender(cfg or _cfg(), seed=0)


def _env(cfg=None, defense="build"):
    cfg = cfg or _cfg()
    if defense == "build":
        defense = _defender(cfg)
    loaders = [_loader(i) for i in range(N_CLIENTS)]
    env = FLArmsRaceEnv(cfg, loaders, _loader(99, n=128), random.Random(0),
                        defense=defense)
    torch.manual_seed(1234)
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    cw = [{k: v + torch.randn_like(v) * 0.01 for k, v in gw.items()}
          for _ in range(N_CLIENTS)]
    env.reset(gw, cw, 0.5)
    return env


def _wreck(sd):
    """A blatant model-poisoning update every one of these defenses should notice."""
    return {k: v.float() * -40.0 for k, v in sd.items()}


def _wrecked_updates(env, ids=POISONED):
    """This round's cohort with ``ids`` replaced by a blatantly poisoned update.

    Built here rather than through the env because the attack the env produces is
    a LABEL FLIP — a real SGD trajectory that deliberately sits inside the honest
    update distribution. These tests are about whether the DEFENSES flag an obvious
    outlier and whether the env plumbs their verdicts and aggregates correctly, so
    they need an update that is unambiguously anomalous.
    """
    from core.types import ModelUpdate
    out = []
    for u in env.build_updates(include_poison=False):
        w = _wreck(u.weights) if u.client_id in set(ids) else u.weights
        out.append(ModelUpdate(u.client_id, w, dict(u.metadata or {})))
    return out


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def test_defaults_to_the_algorithmic_defender():
    assert defense_mode({}) == "algorithmic"
    assert configured_algorithms({}) == list(ALGORITHMS)


def test_llm_mode_returns_no_algorithmic_defender():
    cfg = _cfg(mode="llm")
    assert defense_mode(cfg) == "llm"
    assert build_algorithmic_defender(cfg, seed=0) is None


def test_ground_truth_and_llm_defenses_are_not_selectable():
    for name in ("oracle", "llm_defender", "fedavg"):
        try:
            configured_algorithms({"defense": {"algorithms": [name]}})
        except ValueError as e:
            assert name in str(e)
        else:
            raise AssertionError(f"'{name}' must be rejected as a rotation member")


def test_unknown_algorithm_is_rejected():
    try:
        configured_algorithms({"defense": {"algorithms": ["trimmed_mean"]}})
    except ValueError as e:
        assert "trimmed_mean" in str(e)
    else:
        raise AssertionError("unknown algorithm must be rejected")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_one_algorithm_is_drawn_per_round_and_all_get_used():
    defender = _defender()
    drawn = {defender.choose() for _ in range(200)}
    assert drawn == set(defender.names), f"rotation never reached {set(defender.names) - drawn}"


def test_round_robin_cycles_in_order():
    defender = _defender(_cfg(selection="round_robin"))
    names = defender.names
    got = [defender.choose() for _ in range(len(names) * 2)]
    assert got == names * 2


def test_the_algorithm_is_fixed_for_the_whole_round():
    """All G rollouts must be graded against the SAME defense, so the draw happens
    once in begin_round() and env.defend() never re-draws."""
    env = _env()
    env.begin_round()
    chosen = env.round_defense
    assert chosen in env.defense.names
    updates = env.build_updates(include_poison=False)
    for _ in range(5):
        env.defend(updates, commit=False)
        assert env.round_defense == chosen


def test_the_draw_does_not_disturb_the_env_rng():
    """The algorithm is drawn from a dedicated stream, so poison/budget sampling
    stays reproducible whether or not a defense is attached."""
    seq_with = [_env().rng.random() for _ in range(1)]
    seq_without = [_env(defense=None).rng.random() for _ in range(1)]
    env_a, env_b = _env(), _env(defense=None)
    for _ in range(3):
        env_a.begin_round()
        env_b.begin_round()
    assert seq_with == seq_without
    assert env_a.rng.random() == env_b.rng.random()


# ---------------------------------------------------------------------------
# Defending
# ---------------------------------------------------------------------------

def test_defend_returns_verdicts_and_its_own_aggregate():
    env = _env()
    env.begin_round()
    updates = env.build_updates(include_poison=False)
    verdicts, state = env.defend(updates, commit=False)
    assert [v.client_id for v in verdicts] == list(range(N_CLIENTS))
    assert state is not None
    assert set(state.keys()) == set(env.global_weights.keys())


def test_every_algorithm_flags_a_blatant_poisoner():
    env = _env()
    for name in env.defense.names:
        env.begin_round()
        env.round_defense = name
        updates = _wrecked_updates(env)
        verdicts, _state = env.defend(updates, commit=False)
        flagged = {v.client_id for v in verdicts if v.is_suspicious}
        assert flagged & set(POISONED), f"{name} missed the wrecking clients (flagged={flagged})"


def test_scoring_does_not_advance_the_defense_memory():
    """DeFL carries Beta counts + S(t-1) across rounds. Scoring candidate rollouts
    must leave that memory untouched, or later rollouts in the same group would
    face a different defense than earlier ones."""
    env = _env(_cfg(algorithms=["defl"]))
    env.begin_round()
    defl = env.defense._defenses["defl"]
    updates = _wrecked_updates(env)

    before = defl.state_snapshot()
    first, _ = env.defend(updates, commit=False)
    after_scoring = defl.state_snapshot()
    assert after_scoring == before, "a scored rollout leaked into the defense's memory"

    # ... and identical scoring calls stay identical.
    second, _ = env.defend(updates, commit=False)
    assert [(v.client_id, v.is_suspicious) for v in first] == \
           [(v.client_id, v.is_suspicious) for v in second]

    # Committing DOES advance it.
    env.defend(updates, commit=True)
    assert defl.state_snapshot() != before


def test_commit_installs_the_defense_aggregate():
    env = _env()
    env.begin_round()
    updates = _wrecked_updates(env)
    _verdicts, state = env.defend(updates, commit=True)
    env.commit_state(state)
    committed = env.global_weights
    for k in state:
        assert torch.allclose(committed[k].float(), state[k].float(), atol=1e-6)


def test_commit_state_keeps_the_global_when_the_defense_declines():
    env = _env()
    env.begin_round()
    before = {k: v.clone() for k, v in env.global_weights.items()}
    acc = env.commit_state(None)
    assert acc == env.current_accuracy
    for k, v in env.global_weights.items():
        assert torch.equal(v, before[k])


def test_clean_reference_goes_through_the_round_defense():
    """The counterfactual must be the DEFENDED unpoisoned aggregate, so the drop
    isolates the attack instead of charging the attacker for the defense's own
    cost on an honest round."""
    env = _env()
    ctx = env.begin_round()
    updates = env.build_updates(include_poison=False)
    _v, state = env.defend(updates, commit=False)
    assert abs(ctx.clean_accuracy - env.evaluate_state(state)) < 1e-12


def test_defend_without_a_defense_is_an_error():
    env = _env(defense=None)
    env.begin_round()
    assert env.round_defense is None
    try:
        env.defend(env.build_updates(include_poison=False))
    except RuntimeError as e:
        assert "algorithmic defense" in str(e)
    else:
        raise AssertionError("env.defend() must refuse to run on the defender-LLM path")


def test_the_llm_path_is_untouched():
    """defense=None must keep the original FedAvg-over-unflagged behaviour."""
    from core.types import DetectionVerdict
    env = _env(defense=None)
    ctx = env.begin_round()
    updates = env.build_updates(include_poison=False)
    clean = [DetectionVerdict(u.client_id, False, 0.0, "") for u in updates]
    assert abs(ctx.clean_accuracy - env.evaluate_updates(updates, clean)) < 1e-12


# ---------------------------------------------------------------------------
# Training refuses to run on this path
# ---------------------------------------------------------------------------

def test_defender_turn_refuses_to_run_without_the_defender_llm():
    from rl.turns import DefenderTurn
    env = _env()
    env.begin_round()
    try:
        DefenderTurn(env, None)
    except RuntimeError as e:
        assert "defense.mode" in str(e)
    else:
        raise AssertionError("DefenderTurn must refuse an algorithmically defended env")


def test_the_driver_refuses_an_algorithmic_defense():
    """The attack is a fixed schedule and the algorithms are not policies, so
    there is nothing left to train — say so instead of running empty rounds."""
    import rl.schedule as sched
    from metrics import MetricsTracker

    cfg = _cfg()
    cfg["fl"]["simulation_rounds"] = 2
    cfg["rl"] = {"G": 2}
    try:
        sched.train(_env(cfg), None, None, cfg, MetricsTracker(0.5, output_dir="logs/_t"),
                    lambda log: None, random.Random(0))
    except RuntimeError as e:
        assert "defense.mode: llm" in str(e), e
    else:
        raise AssertionError("training must refuse to run with an algorithmic defense")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} algorithmic-defender tests passed.")


if __name__ == "__main__":
    _run()
