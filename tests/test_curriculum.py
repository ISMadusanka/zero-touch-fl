"""Tests for the deterministic (defense, #poisoners) training curriculum.

Covers ``rl/curriculum.py``, its wiring into ``rl/env.py`` /
``server/algo_defender.py``, and the resume path in ``rl/schedule.py``. Torch is
used only for the env-level tests (synthetic MNIST-shaped tensors — no download,
no GPU, no LLM):

    python tests/test_curriculum.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from model.mnist_net import MnistNet  # noqa: E402
from rl.curriculum import (  # noqa: E402
    TrainingCurriculum, build_training_curriculum, resolve_poisoner_counts,
)
from rl.env import FLArmsRaceEnv  # noqa: E402
from rl.schedule import _resume_curriculum  # noqa: E402
from server.algo_defender import build_algorithmic_defender  # noqa: E402

N_CLIENTS = 8
ALGS = ["defl", "dnc", "multikrum"]     # FLTrust needs a real root set (test_fltrust.py)


def _loader(seed: int, n: int = 64):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.randn(n, 1, 28, 28, generator=g),
                      torch.randint(0, 10, (n,), generator=g)),
        batch_size=32, shuffle=True)


def _cfg(**curriculum):
    cur = {"enabled": True, "rounds_per_block": 2, "poisoner_counts": [1, 2, 3]}
    cur.update(curriculum)
    return {
        "fl": {"n_clients": N_CLIENTS, "device": "cpu", "lr": 0.05, "local_epochs": 1,
               "benign_retrain_each_round": False, "training_rounds": 5,
               "n_compromisable": 3, "poison_seed": 0, "batch_size": 32},
        "data": {"data_dir": "./data/mnist_raw"},
        "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.1},
                   "max_poison_clients": 3, "sample_budget_in_training": True,
                   "sample_target_in_training": False},
        "defense": {"mode": "algorithmic", "algorithms": list(ALGS), "selection": "random"},
        "curriculum": cur,
    }


def _env(cfg=None):
    cfg = cfg or _cfg()
    defense = build_algorithmic_defender(cfg, seed=0)
    curriculum = build_training_curriculum(cfg, algorithms=defense.names)
    env = FLArmsRaceEnv(cfg, [_loader(i) for i in range(N_CLIENTS)], _loader(99, n=128),
                        random.Random(0), defense=defense, curriculum=curriculum)
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    cw = [{k: v + torch.randn_like(v) * 0.01 for k, v in gw.items()}
          for _ in range(N_CLIENTS)]
    env.reset(gw, cw, 0.5)
    return env


# --------------------------------------------------------------------------- plan shape
def test_outer_loop_is_the_algorithm_inner_loop_is_the_poisoner_count():
    """One algorithm is held across ALL poisoner counts before the next is picked
    up — the schedule the user asked for, not the transpose."""
    cur = TrainingCurriculum(["fltrust", "multikrum"], [1, 2, 3], rounds_per_block=10)
    seen = [(s.algorithm, s.n_poisoners)
            for s in (cur.advance() for _ in range(cur.rounds_per_cycle))]
    # first 10 rounds: fltrust @1; next 10: fltrust @2; ... then multikrum @1 ...
    assert seen[0:10] == [("fltrust", 1)] * 10
    assert seen[10:20] == [("fltrust", 2)] * 10
    assert seen[20:30] == [("fltrust", 3)] * 10
    assert seen[30:40] == [("multikrum", 1)] * 10
    assert seen[40:50] == [("multikrum", 2)] * 10
    assert seen[50:60] == [("multikrum", 3)] * 10
    assert len(seen) == 60


def test_cycle_repeats_from_the_first_algorithm():
    cur = TrainingCurriculum(["a", "b"], [1, 2], rounds_per_block=3)
    assert cur.rounds_per_cycle == 12
    first = [cur.advance() for _ in range(12)]
    second = [cur.advance() for _ in range(12)]
    assert [(s.algorithm, s.n_poisoners) for s in first] == \
           [(s.algorithm, s.n_poisoners) for s in second]
    assert first[0].cycle == 0 and second[0].cycle == 1
    assert second[0].block == cur.blocks_per_cycle


def test_every_pair_gets_exactly_the_same_number_of_rounds():
    """The whole point: equal opportunity, not equal in expectation."""
    cur = TrainingCurriculum(ALGS, [1, 2, 3, 4, 5], rounds_per_block=10)
    counts = {}
    for _ in range(3 * cur.rounds_per_cycle):
        s = cur.advance()
        counts[(s.algorithm, s.n_poisoners)] = counts.get((s.algorithm, s.n_poisoners), 0) + 1
    assert len(counts) == len(ALGS) * 5
    assert set(counts.values()) == {30}, counts


def test_slot_at_is_pure_and_matches_advance():
    cur = TrainingCurriculum(ALGS, [1, 2], rounds_per_block=4)
    expected = [cur.slot_at(i) for i in range(20)]
    assert cur.step == 0, "slot_at must not consume the schedule"
    assert [cur.advance() for _ in range(20)] == expected
    assert cur.step == 20


def test_peek_does_not_consume():
    cur = TrainingCurriculum(ALGS, [1], rounds_per_block=2)
    assert cur.peek() == cur.peek() == cur.slot_at(0)
    assert cur.step == 0
    assert cur.advance() == cur.slot_at(0)
    assert cur.step == 1


def test_block_round_counts_within_the_block():
    cur = TrainingCurriculum(["a"], [1, 2], rounds_per_block=3)
    rounds = [cur.advance() for _ in range(6)]
    assert [s.block_round for s in rounds] == [0, 1, 2, 0, 1, 2]
    assert [s.block for s in rounds] == [0, 0, 0, 1, 1, 1]
    assert [s.n_poisoners for s in rounds] == [1, 1, 1, 2, 2, 2]


def test_single_algorithm_and_llm_mode_still_sweep_the_counts():
    """Under defense.mode: llm there is no algorithm axis — only the quota sweeps."""
    cur = TrainingCurriculum(None, [1, 2], rounds_per_block=2)
    assert cur.algorithms == [None]
    assert [(s.algorithm, s.n_poisoners) for s in (cur.advance() for _ in range(4))] == \
           [(None, 1), (None, 1), (None, 2), (None, 2)]


# --------------------------------------------------------------------------- validation
def test_rejects_a_degenerate_plan():
    for bad in (dict(poisoner_counts=[]), dict(poisoner_counts=[0]),
                dict(poisoner_counts=[-1]), dict(rounds_per_block=0)):
        kwargs = dict(algorithms=ALGS, poisoner_counts=[1], rounds_per_block=2)
        kwargs.update(bad)
        try:
            TrainingCurriculum(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted a degenerate plan: {bad}")


def test_counts_default_to_the_attack_budget():
    assert resolve_poisoner_counts(None, max_poison_clients=5, n_compromisable=5) == [1, 2, 3, 4, 5]


def test_counts_above_the_pool_are_dropped_not_clamped():
    """Clamping [1..5] onto a 3-client pool would give '3' three blocks per cycle —
    the exact imbalance this curriculum exists to remove."""
    assert resolve_poisoner_counts([1, 2, 3, 4, 5], 5, n_compromisable=3) == [1, 2, 3]
    assert resolve_poisoner_counts([2, 2, 1], 5, n_compromisable=5) == [2, 1]
    try:
        resolve_poisoner_counts([7, 8], 8, n_compromisable=3)
    except ValueError:
        pass
    else:
        raise AssertionError("an entirely un-expressible sweep must not build")


def test_builder_is_off_without_a_config_block_and_when_disabled():
    cfg = _cfg()
    cfg.pop("curriculum")
    assert build_training_curriculum(cfg, algorithms=ALGS) is None
    cfg2 = _cfg(enabled=False)
    assert build_training_curriculum(cfg2, algorithms=ALGS) is None


def test_builder_narrows_and_reorders_the_algorithm_loop():
    cur = build_training_curriculum(_cfg(algorithms=["multikrum", "defl"]), algorithms=ALGS)
    assert cur.algorithms == ["multikrum", "defl"]
    for bad in (["oracle"], ["defl", "defl"]):
        try:
            build_training_curriculum(_cfg(algorithms=bad), algorithms=ALGS)
        except ValueError:
            continue
        raise AssertionError(f"accepted curriculum.algorithms={bad}")


# --------------------------------------------------------------------------- resume
def test_state_round_trips_and_survives_a_plan_change():
    cur = TrainingCurriculum(ALGS, [1, 2], rounds_per_block=3)
    for _ in range(7):
        cur.advance()
    state = cur.state_dict()
    assert state["step"] == 7

    fresh = TrainingCurriculum(ALGS, [1, 2], rounds_per_block=3)
    fresh.load_state_dict(state)
    assert fresh.step == 7 and fresh.peek() == cur.peek()

    # An edited config keeps the POSITION (restarting the sweep would re-train the
    # first block from scratch on every restart) but warns — see load_state_dict.
    changed = TrainingCurriculum(ALGS, [1, 2, 3], rounds_per_block=10)
    changed.load_state_dict(state)
    assert changed.step == 7


def test_resume_falls_back_to_rounds_done_for_old_checkpoints():
    """`rounds_done` counts exactly the rounds that consume a curriculum slot (the
    between-phase FL interlude consumes neither), so it IS the sweep position."""
    class _Env:
        curriculum = TrainingCurriculum(ALGS, [1, 2], rounds_per_block=3)

    env = _Env()
    _resume_curriculum(env, {"rounds_done": 11, "curriculum": None}, start_round=11)
    assert env.curriculum.step == 11
    assert env.curriculum.peek() == env.curriculum.slot_at(11)


def test_resume_prefers_the_saved_position():
    class _Env:
        curriculum = TrainingCurriculum(ALGS, [1, 2], rounds_per_block=3)

    env = _Env()
    _resume_curriculum(env, {"rounds_done": 11, "curriculum": {"step": 4}}, start_round=11)
    assert env.curriculum.step == 4


def test_resume_is_a_noop_without_a_curriculum():
    class _Env:
        curriculum = None

    _resume_curriculum(_Env(), {"rounds_done": 5}, start_round=5)   # must not raise


# --------------------------------------------------------------------------- env wiring
def test_env_rounds_follow_the_curriculum_not_the_rng():
    env = _env()                       # 3 algorithms x [1,2,3] poisoners x 2 rounds
    seen = []
    for _ in range(env.curriculum.rounds_per_cycle):
        ctx = env.begin_round()
        seen.append((env.round_defense, ctx.budget))
        # the RoundContext the agents see must carry the block's quota
        assert ctx.budget == env.round_budget
        assert env.round_curriculum is not None
    expected = [(a, k) for a in ALGS for k in (1, 2, 3) for _ in range(2)]
    assert seen == expected, seen


def test_env_ignores_sample_budget_while_a_curriculum_is_attached():
    cfg = _cfg()
    cfg["attack"]["sample_budget_in_training"] = True   # would randomise [1..3]
    env = _env(cfg)
    budgets = [env.begin_round().budget for _ in range(6)]
    assert budgets == [1, 1, 2, 2, 3, 3], budgets


def test_defender_current_tracks_the_curriculum():
    """`AlgorithmicDefender.run()` falls back to `current` when no algorithm is
    passed, so pinning the round must keep it in sync."""
    env = _env()
    for _ in range(6):
        env.begin_round()
        assert env.defense.current == env.round_defense


def test_env_without_a_curriculum_keeps_the_random_draws():
    cfg = _cfg()
    cfg.pop("curriculum")
    defense = build_algorithmic_defender(cfg, seed=0)
    assert build_training_curriculum(cfg, algorithms=defense.names) is None
    env = FLArmsRaceEnv(cfg, [_loader(i) for i in range(N_CLIENTS)], _loader(99, n=128),
                        random.Random(0), defense=defense)
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    env.reset(gw, [dict(gw) for _ in range(N_CLIENTS)], 0.5)
    env.begin_round()
    assert env.round_curriculum is None
    assert env.round_defense in ALGS
    assert 1 <= env.round_budget <= 3


def test_fl_interlude_does_not_consume_a_curriculum_slot():
    """The honest FL round between phases is not a GRPO training round, so a block
    must still get its full `rounds_per_block` attacker rounds."""
    env = _env()
    env.begin_round()
    step_before = env.curriculum.step
    env.run_benign_fl_round()
    assert env.curriculum.step == step_before
    assert env.begin_round().budget == env.curriculum.slot_at(step_before).n_poisoners


def test_target_drop_is_held_fixed_during_training():
    """sample_target_in_training is off, so every round asks for the same drop —
    otherwise a block's 10 rounds would not be comparable to each other."""
    env = _env()
    goals = [env.begin_round().goal["target_accuracy_drop"] for _ in range(6)]
    assert goals == [0.1] * 6, goals


# --------------------------------------------------------------------------- driver
class _FakePolicy:
    adapters = ("attacker",)

    def __init__(self):
        self._state = {"attacker": {"w": torch.tensor([1.0])}}
        self._params = {"attacker": [torch.nn.Parameter(torch.zeros(1))]}

    def adapter_parameters(self, name):
        return self._params[name]

    def set_adapter(self, name):
        pass

    def get_adapter_state(self, name):
        return {k: v.clone() for k, v in self._state[name].items()}

    def set_adapter_state(self, name, s):
        self._state[name] = {k: v.clone() for k, v in s.items()}

    def save_adapter(self, name, path):
        pass

    def save_adapter_state_dict(self, name, state, path):
        pass


def _drive(env, cfg, sim_rounds, td, resume=None, start_round=0):
    """Run the real GRPO driver with a stubbed round body; return (slots, progress)."""
    import rl.schedule as sched
    from metrics import MetricsTracker

    cfg["fl"]["simulation_rounds"] = sim_rounds
    cfg["rl"] = {"G": 2, "save_every": 1, "league_snapshot_every": 0, "league_prob": 0.0,
                 "switch_mode": "best_response", "min_phase_rounds": 2,
                 "max_phase_rounds": 3, "success_streak": 2,
                 "fl_interlude_between_phases": True,
                 "adapter_paths": {"attacker": os.path.join(td, "att"),
                                   "defender": os.path.join(td, "def")}}
    slots, progress = [], []

    def fake_step(state, learner, opp, opp_gen, pidx, pround):
        env.begin_round()
        slots.append((env.round_defense, env.round_budget))
        return {"rewards": [0.0], "completions": ["x"], "loss": 0.0, "mean_reward": 0.0,
                "max_reward": 0.0, "min_reward": 0.0,
                "zero_advantage_fraction": 0.0}, 0.01, True

    saved_step = sched._step_round
    sched._step_round = fake_step
    try:
        sched.train(env, _FakePolicy(), None, None, cfg,
                    MetricsTracker(0.5, output_dir=os.path.join(td, "m")),
                    lambda log: None, random.Random(0),
                    progress_cb=lambda d, round_index=None, controller=None, curriculum=None:
                        progress.append((d, curriculum)),
                    start_round=start_round, resume=resume)
    finally:
        sched._step_round = saved_step
    return slots, progress


def test_driver_follows_the_sweep_across_phase_switches_and_resumes_mid_block():
    """The arms-race phase machinery (switches + FL interludes) runs on top of the
    sweep without perturbing it, and a restart continues in the same block."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # 3 algorithms x [1,2,3] poisoners x 2 rounds per block.
        first, progress = _drive(_env(), _cfg(), sim_rounds=7, td=td)
        assert first == [("defl", 1), ("defl", 1), ("defl", 2), ("defl", 2),
                         ("defl", 3), ("defl", 3), ("dnc", 1)], first

        # The position is checkpointed with the round count, and they agree.
        done, saved = progress[-1]
        assert done == 7 and saved["step"] == 7, progress[-1]

        # Resume: continue block 3 (dnc @ 1 poisoner) rather than restarting at defl.
        resumed, _ = _drive(_env(), _cfg(), sim_rounds=11, td=td,
                            resume={"rounds_done": 7, "round_index": 7,
                                    "controller": None, "curriculum": saved},
                            start_round=7)
        assert resumed == [("dnc", 1), ("dnc", 2), ("dnc", 2),
                           ("dnc", 3)], resumed


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} curriculum tests passed.")


if __name__ == "__main__":
    _run()
