"""Tests for the algorithmic (non-LLM) defender that replaces the defender LLM.

Covers ``server/algo_defender.py``, the env's defense hooks (``rl/env.py``) and
the attacker-only phase rotation (``rl/switch.py``). Synthetic tensors shaped
like MNIST — no download, no GPU, no LLM:

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
               "benign_retrain_each_round": False, "training_rounds": 5,
               "freeze_global_in_phase2": False,
               "n_compromisable": 2, "poison_seed": 0, "batch_size": 32},
        "data": {"data_dir": "./data/mnist_raw"},
        "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.2},
                   "max_poison_clients": 2, "sample_budget_in_training": False},
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
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    cw = [{k: v + torch.randn_like(v) * 0.01 for k, v in gw.items()}
          for _ in range(N_CLIENTS)]
    env.reset(gw, cw, 0.5)
    return env


def _wreck(sd):
    """A blatant model-poisoning update every one of these defenses should notice."""
    return {k: v.float() * -40.0 for k, v in sd.items()}


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
    updates = env.build_updates({})
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
    updates = env.build_updates({})
    verdicts, state = env.defend(updates, commit=False)
    assert [v.client_id for v in verdicts] == list(range(N_CLIENTS))
    assert state is not None
    assert set(state.keys()) == set(env.global_weights.keys())


def test_every_algorithm_flags_a_blatant_poisoner():
    env = _env()
    for name in env.defense.names:
        env.begin_round()
        env.round_defense = name
        poisoned = {cid: _wreck(env.pool_benign[cid]) for cid in POISONED}
        updates = env.build_updates(poisoned)
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
    updates = env.build_updates({cid: _wreck(env.pool_benign[cid]) for cid in POISONED})

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
    updates = env.build_updates({cid: _wreck(env.pool_benign[cid]) for cid in POISONED})
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
    updates = env.build_updates({})
    _v, state = env.defend(updates, commit=False)
    assert abs(ctx.clean_accuracy - env.evaluate_state(state)) < 1e-12


def test_defend_without_a_defense_is_an_error():
    env = _env(defense=None)
    env.begin_round()
    assert env.round_defense is None
    try:
        env.defend(env.build_updates({}))
    except RuntimeError as e:
        assert "algorithmic defense" in str(e)
    else:
        raise AssertionError("env.defend() must refuse to run on the defender-LLM path")


def test_the_llm_path_is_untouched():
    """defense=None must keep the original FedAvg-over-unflagged behaviour."""
    from core.types import DetectionVerdict
    env = _env(defense=None)
    ctx = env.begin_round()
    updates = env.build_updates({})
    clean = [DetectionVerdict(u.client_id, False, 0.0, "") for u in updates]
    assert abs(ctx.clean_accuracy - env.evaluate_updates(updates, clean)) < 1e-12


# ---------------------------------------------------------------------------
# Attacker-only schedule
# ---------------------------------------------------------------------------

def test_attacker_only_rotation_never_hands_off_to_the_defender():
    ctrl = PhaseController(SwitchConfig(), first_learner="attacker",
                           learners=("attacker",))
    for _ in range(5):
        ctrl.next_phase("success")
        assert ctrl.learner == "attacker"
    assert ctrl.phase_index == 5


def test_two_learner_rotation_still_alternates():
    ctrl = PhaseController(SwitchConfig(), first_learner="attacker")
    ctrl.next_phase("success")
    assert ctrl.learner == "defender"
    ctrl.next_phase("cap")
    assert ctrl.learner == "attacker" and ctrl.capped


def test_resume_coerces_a_defender_phase_to_the_only_trainable_learner():
    """A checkpoint written while the defender LLM was trainable must not resume
    into a defender phase once the defense is algorithmic."""
    ctrl = PhaseController(SwitchConfig(), first_learner="attacker",
                           learners=("attacker",))
    ctrl.load_state_dict({"learner": "defender", "phase_index": 3, "phase_round": 2,
                          "streak": 1, "capped": False})
    assert ctrl.learner == "attacker"
    assert ctrl.phase_index == 3 and ctrl.phase_round == 2


def test_defender_turn_refuses_to_run_without_the_defender_llm():
    from rl.turns import DefenderTurn
    env = _env()
    try:
        DefenderTurn(env, None, None, None)
    except RuntimeError as e:
        assert "defense.mode" in str(e)
    else:
        raise AssertionError("DefenderTurn must refuse an algorithmically defended env")


def test_attacker_turn_requires_a_generator_only_on_the_llm_path():
    from rl.turns import AttackerTurn
    env = _env(defense=None)
    env.begin_round()

    class _Agent:
        def system_prompt(self):
            return "sys"

        def build_user_prompt(self, *a, **k):
            return "user"

    try:
        AttackerTurn(env, _Agent(), _Agent(), None)
    except ValueError as e:
        assert "defender generator" in str(e)
    else:
        raise AssertionError("the LLM path must demand a defender generator")


def test_attacker_turn_needs_no_generator_with_an_algorithmic_defense():
    from rl.turns import AttackerTurn

    class _Agent:
        def system_prompt(self):
            return "sys"

        def build_user_prompt(self, *a, **k):
            return "user"

    env = _env()
    env.begin_round()
    turn = AttackerTurn(env, _Agent(), _Agent(), None)
    assert turn.algorithmic
    assert turn.reference_accuracy == env.clean_reference_accuracy()


def test_the_driver_trains_the_attacker_only_and_never_writes_the_defender():
    """The full GRPO driver with a stubbed round body + fake policy: with an
    algorithmic defense it must build one optimizer, keep every phase on the
    attacker, still run the between-phase FL interlude, and leave the defender
    checkpoint on disk untouched so ``defense.mode: llm`` can be restored."""
    import copy
    import tempfile

    import rl.schedule as sched
    from metrics import MetricsTracker

    class _FakePolicy:
        adapters = ("attacker",)

        def __init__(self):
            self._state = {"attacker": {"w": torch.tensor([1.0])}}
            self._params = {"attacker": [torch.nn.Parameter(torch.zeros(1))]}
            self.saved = {}

        def adapter_parameters(self, name):
            return self._params[name]

        def set_adapter(self, name):
            pass

        def get_adapter_state(self, name):
            return copy.deepcopy(self._state[name])

        def set_adapter_state(self, name, s):
            self._state[name] = copy.deepcopy(s)

        def save_adapter(self, name, path):
            self.saved[path] = copy.deepcopy(self._state[name])

        def save_adapter_state_dict(self, name, state, path):
            self.saved[path] = copy.deepcopy(state)

    cfg = _cfg()
    cfg["fl"]["simulation_rounds"] = 6
    cfg["rl"] = {"G": 2, "save_every": 1, "league_snapshot_every": 1, "league_prob": 1.0,
                 # Deliberately wrong for this mode — the driver must override it.
                 "first_learner": "defender", "switch_mode": "best_response",
                 "min_phase_rounds": 2, "max_phase_rounds": 3, "success_streak": 2,
                 "fl_interlude_between_phases": True, "curriculum_on_cap": True}
    env = _env(cfg)

    learners, interludes = [], {"n": 0}
    orig_interlude = env.run_benign_fl_round

    def counted():
        interludes["n"] += 1
        return orig_interlude()

    env.run_benign_fl_round = counted

    def fake_step(state, learner, opp, opp_gen, pidx, pround):
        learners.append(learner)
        assert opp_gen is None, "no opponent generator exists with an algorithmic defense"
        env.begin_round()
        return {"rewards": [0.0], "completions": ["x"], "loss": 0.0, "mean_reward": 0.0,
                "max_reward": 0.0, "min_reward": 0.0,
                "zero_advantage_fraction": 0.0}, 0.01, True

    saved_step = sched._step_round
    sched._step_round = fake_step
    try:
        with tempfile.TemporaryDirectory() as td:
            cfg["rl"]["adapter_paths"] = {"attacker": os.path.join(td, "att"),
                                          "defender": os.path.join(td, "def")}
            policy = _FakePolicy()
            sched.train(env, policy, None, None, cfg,
                        MetricsTracker(0.5, output_dir=os.path.join(td, "m")),
                        lambda log: None, random.Random(0))
            assert set(learners) == {"attacker"}, f"defender phases ran: {set(learners)}"
            assert len(learners) == 6
            assert interludes["n"] >= 1, "the between-phase FL interlude stopped firing"
            assert os.path.join(td, "att") in policy.saved
            assert os.path.join(td, "def") not in policy.saved, \
                "the defender checkpoint was overwritten while its LLM is disabled"
            assert "defender" not in sched.League(random.Random(0), 1).snapshots
    finally:
        sched._step_round = saved_step


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} algorithmic-defender tests passed.")


if __name__ == "__main__":
    _run()
