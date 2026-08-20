"""Tests for the ``--poisoners`` / ``--learn`` command-line overrides.

Both flags rewrite a SET of config keys rather than one (see
``core/config_overrides.py``), and the failure mode they exist to prevent is a
half-applied override — 8 clients poisoned while the defenses still budget for
10, or ``--learn attacker`` quietly training the defender too. So these tests
check the whole set each flag is responsible for, in both poisoning regimes, and
that the downstream consumers actually honour it:

  * ``core/config_overrides.py``  — the key sets and the validation errors;
  * ``rl/curriculum.py``          — the sweep collapses to the requested count;
  * ``server/algo_defender.py``   — DnC / Multi-Krum assume the requested count;
  * ``rl/env.py``                 — the round's quota and pool really are N;
  * ``rl/schedule.py``            — only the requested side gets an optimizer.

No GPU, no LLM, no dataset:

    python tests/test_cli_overrides.py
"""

import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from core.config_overrides import (  # noqa: E402
    apply_learner_choice, apply_poisoner_count, resolve_trainable,
)
from data.feature_spec import DEFAULT_SPEC  # noqa: E402
from rl.curriculum import build_training_curriculum  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from server.algo_defender import build_algorithmic_defender  # noqa: E402

N_CLIENTS = 20


def _loader(seed: int, n: int = 64):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.randn(n, DEFAULT_SPEC.input_dim, generator=g),
                      torch.randint(0, DEFAULT_SPEC.n_classes, (n,), generator=g)),
        batch_size=32, shuffle=True)


def _env(cfg):
    return FLArmsRaceEnv(cfg, [_loader(i) for i in range(N_CLIENTS)],
                         _loader(99, n=128), random.Random(0))


def _cfg(**over):
    """A cut-down configs/base.yaml: the shipped fixed-poison-set regime."""
    cfg = {
        "fl": {
            "n_clients": N_CLIENTS, "n_compromisable": 10, "batch_size": 32,
            "lr": 0.01, "local_epochs": 1, "device": "cpu", "poison_seed": 0,
            "training_rounds": 1, "simulation_rounds": 10,
            "benign_retrain_each_round": False, "freeze_global_in_phase2": False,
        },
        "data": {"source": "synthetic"},
        "attack": {
            "fixed_poison_clients": 10,
            "max_poison_clients": 10,
            "eval_poison_clients": 10,
            "sample_budget_in_training": False,
            "goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.10},
        },
        "defense": {"mode": "algorithmic",
                    "algorithms": ["defl", "dnc", "multikrum"]},
        "curriculum": {"enabled": True, "rounds_per_block": 10,
                       "poisoner_counts": [10]},
        "rl": {"first_learner": "attacker"},
    }
    for section, values in over.items():
        cfg[section] = {**cfg.get(section, {}), **values}
    return cfg


def _selection_cfg():
    """The other regime: no fixed set, so the attacker picks which of its pool."""
    return _cfg(attack={"fixed_poison_clients": None,
                        "sample_budget_in_training": True},
                curriculum={"poisoner_counts": [1, 2, 3, 4, 5]})


# ---------------------------------------------------------------------------
# --poisoners: the fixed-set regime (the shipped default)
# ---------------------------------------------------------------------------

def test_poisoners_rewrites_every_key_that_carries_the_count():
    cfg = _cfg()
    changed = apply_poisoner_count(cfg, 8)

    assert cfg["attack"]["fixed_poison_clients"] == 8
    assert cfg["fl"]["n_compromisable"] == 8         # the pool IS the set here
    assert cfg["attack"]["max_poison_clients"] == 8  # -> assumed_byzantine
    assert cfg["attack"]["eval_poison_clients"] == 8  # -> the benchmark
    assert cfg["curriculum"]["poisoner_counts"] == [8]
    # All five were 10, so all five are reported as changed.
    assert set(changed) == {
        "attack.fixed_poison_clients", "fl.n_compromisable",
        "attack.max_poison_clients", "attack.eval_poison_clients",
        "curriculum.poisoner_counts",
    }, changed


def test_poisoners_matching_the_config_changes_nothing():
    cfg = _cfg()
    before = copy.deepcopy(cfg)
    assert apply_poisoner_count(cfg, 10) == {}
    assert cfg == before


def test_poisoners_reaches_the_curriculum_and_the_defense():
    cfg = _cfg()
    apply_poisoner_count(cfg, 8)

    defense = build_algorithmic_defender(cfg, seed=0)
    # DnC and Multi-Krum size their filtering by the assumed #malicious, which
    # defaults to attack.max_poison_clients — the flag has to move it, or an
    # 8-poisoner run is defended as if 10 were coming.
    for name in ("dnc", "multikrum"):
        assert defense._defenses[name].num_byzantine == 8, name

    curriculum = build_training_curriculum(cfg, algorithms=defense.names)
    assert curriculum.poisoner_counts == [8]
    assert {curriculum.slot_at(i).n_poisoners for i in range(60)} == {8}


def test_poisoners_is_the_quota_and_the_pool_in_the_env():
    cfg = _cfg()
    apply_poisoner_count(cfg, 8)
    env = _env(cfg)
    assert env.n_compromisable == 8
    assert env.fixed_poison_clients == 8
    assert env._round_budget() == 8
    assert env.budget_cap == 8


# ---------------------------------------------------------------------------
# --poisoners: the attacker-selected regime
# ---------------------------------------------------------------------------

def test_poisoners_pins_an_exact_quota_without_shrinking_the_pool():
    cfg = _selection_cfg()
    apply_poisoner_count(cfg, 3)

    # The set is still the policy's to choose — the pool stays at 10 — but the
    # quota is exactly 3, so the per-round draw has to go.
    assert cfg["fl"]["n_compromisable"] == 10
    assert cfg["attack"]["fixed_poison_clients"] is None
    assert cfg["attack"]["sample_budget_in_training"] is False
    assert cfg["attack"]["max_poison_clients"] == 3
    assert cfg["curriculum"]["poisoner_counts"] == [3]

    env = _env(cfg)
    assert env.n_compromisable == 10 and env.budget_cap == 3
    assert env._round_budget() == 3


def test_poisoners_above_the_controllable_pool_is_an_error():
    cfg = _selection_cfg()
    try:
        apply_poisoner_count(cfg, 12)
    except ValueError as exc:
        assert "n_compromisable" in str(exc)
    else:
        raise AssertionError("expected a pool-size error")


def test_poisoners_out_of_range_is_an_error():
    for bad in (0, -1, N_CLIENTS + 1):
        try:
            apply_poisoner_count(_cfg(), bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"--poisoners {bad} should be rejected")


# ---------------------------------------------------------------------------
# --learn
# ---------------------------------------------------------------------------

def test_learn_attacker_trains_only_the_attacker():
    cfg = _cfg(defense={"mode": "llm"}, rl={"first_learner": "defender"})
    assert apply_learner_choice(cfg, "attacker") == ("attacker",)
    assert cfg["rl"]["learners"] == ["attacker"]
    # The sole learner must also be the first one, or the schedule would open on
    # a phase it cannot run.
    assert cfg["rl"]["first_learner"] == "attacker"
    assert resolve_trainable(cfg["rl"], algorithmic_defense=False) == ("attacker",)


def test_learn_defender_trains_only_the_defender():
    cfg = _cfg(defense={"mode": "llm"})
    assert apply_learner_choice(cfg, "defender") == ("defender",)
    assert cfg["rl"]["first_learner"] == "defender"
    assert resolve_trainable(cfg["rl"], algorithmic_defense=False) == ("defender",)


def test_learn_both_keeps_the_configured_first_learner():
    cfg = _cfg(defense={"mode": "llm"}, rl={"first_learner": "defender"})
    assert apply_learner_choice(cfg, "both") == ("attacker", "defender")
    assert cfg["rl"]["first_learner"] == "defender"
    assert resolve_trainable(cfg["rl"], algorithmic_defense=False) == ("attacker", "defender")


def test_learn_defender_under_an_algorithmic_defense_is_refused():
    # There is no defender POLICY to optimize — FLTrust/DeFL/DnC/Multi-Krum have
    # no parameters. Refusing beats silently running attacker-only.
    for choice in ("defender", "both"):
        try:
            apply_learner_choice(_cfg(), choice)      # defense.mode: algorithmic
        except ValueError as exc:
            assert "defense.mode" in str(exc)
        else:
            raise AssertionError(f"--learn {choice} should be refused")


def test_learn_defaults_are_the_historical_rule():
    assert resolve_trainable({}, algorithmic_defense=True) == ("attacker",)
    assert resolve_trainable({}, algorithmic_defense=False) == ("attacker", "defender")


def test_stale_defender_learner_is_dropped_not_fatal():
    # A config that still names the defender after the defense went algorithmic
    # (e.g. resumed from an older run) drops it rather than crashing.
    assert resolve_trainable({"learners": ["attacker", "defender"]},
                             algorithmic_defense=True) == ("attacker",)
    try:
        resolve_trainable({"learners": ["defender"]}, algorithmic_defense=True)
    except ValueError as exc:
        assert "nobody to train" in str(exc)
    else:
        raise AssertionError("a defender-only run under an algorithmic defense "
                             "leaves nothing to train")


def test_schedule_gives_an_optimizer_only_to_the_learner():
    """The end of the wire: rl/schedule.py builds one optimizer per trainable
    adapter and persists only those, so --learn is what keeps the frozen side's
    checkpoint byte-identical."""
    import inspect
    from rl import schedule

    src = inspect.getsource(schedule.train)
    assert "resolve_trainable(rl, algorithmic_defense)" in src
    assert 'for name in state["trainable"]' in inspect.getsource(schedule._save_adapters)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} CLI-override tests passed.")


if __name__ == "__main__":
    _run()
