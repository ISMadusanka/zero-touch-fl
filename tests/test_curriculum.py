"""Tests for the deterministic defense-algorithm training curriculum.

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
    TrainingCurriculum, build_training_curriculum,
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


def _cfg(poison_ids=(0, 1), **curriculum):
    cur = {"enabled": True, "rounds_per_block": 2}
    cur.update(curriculum)
    return {
        "fl": {"n_clients": N_CLIENTS, "device": "cpu", "lr": 0.05, "local_epochs": 1,
               "training_rounds": 5, "poison_seed": 0, "batch_size": 32,
               "freeze_global_in_phase2": True},
        "data": {"data_dir": "./data/mnist_raw"},
        "attack": {"type": "label_flip", "poison_client_ids": list(poison_ids),
                   "goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.1},
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
    torch.manual_seed(1234)
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    cw = [{k: v + torch.randn_like(v) * 0.01 for k, v in gw.items()}
          for _ in range(N_CLIENTS)]
    env.reset(gw, cw, 0.5)
    return env


# --------------------------------------------------------------------------- plan shape
def test_one_algorithm_is_held_for_a_whole_block():
    """The reason the sweep exists: DeFL and DnC are stateful, so their memory must
    advance over consecutive rounds rather than on ~1 round in 4, scattered."""
    cur = TrainingCurriculum(["fltrust", "multikrum"], [2], rounds_per_block=10)
    seen = [s.algorithm for s in (cur.advance() for _ in range(cur.rounds_per_cycle))]
    assert seen[0:10] == ["fltrust"] * 10
    assert seen[10:20] == ["multikrum"] * 10
    assert len(seen) == 20


def test_cycle_repeats_from_the_first_algorithm():
    cur = TrainingCurriculum(["a", "b"], [1], rounds_per_block=3)
    assert cur.rounds_per_cycle == 6
    first = [cur.advance() for _ in range(6)]
    second = [cur.advance() for _ in range(6)]
    assert [s.algorithm for s in first] == [s.algorithm for s in second]
    assert first[0].cycle == 0 and second[0].cycle == 1
    assert second[0].block == cur.blocks_per_cycle


def test_every_algorithm_gets_exactly_the_same_number_of_rounds():
    """The whole point: equal opportunity, not equal in expectation."""
    cur = TrainingCurriculum(ALGS, [2], rounds_per_block=10)
    counts = {}
    for _ in range(3 * cur.rounds_per_cycle):
        s = cur.advance()
        counts[s.algorithm] = counts.get(s.algorithm, 0) + 1
    assert len(counts) == len(ALGS)
    assert set(counts.values()) == {30}, counts


def test_n_poisoners_is_a_constant_not_an_axis():
    """The attack strength is the label-flip ladder's, not the curriculum's. The
    field survives so a round log names the attack size, but it never varies."""
    cur = TrainingCurriculum(ALGS, [3], rounds_per_block=2)
    slots = [cur.advance() for _ in range(cur.rounds_per_cycle)]
    assert {s.n_poisoners for s in slots} == {3}


def test_slot_at_is_pure_and_matches_advance():
    cur = TrainingCurriculum(ALGS, [2], rounds_per_block=4)
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
    cur = TrainingCurriculum(["a", "b"], [1], rounds_per_block=3)
    rounds = [cur.advance() for _ in range(6)]
    assert [s.block_round for s in rounds] == [0, 1, 2, 0, 1, 2]
    assert [s.block for s in rounds] == [0, 0, 0, 1, 1, 1]
    assert [s.algorithm for s in rounds] == ["a", "a", "a", "b", "b", "b"]


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


def test_builder_is_off_without_a_config_block_and_when_disabled():
    cfg = _cfg()
    cfg.pop("curriculum")
    assert build_training_curriculum(cfg, algorithms=ALGS) is None
    cfg2 = _cfg(enabled=False)
    assert build_training_curriculum(cfg2, algorithms=ALGS) is None


def test_builder_is_off_under_the_defender_llm():
    """No algorithm axis to sweep, and the attack strength is the ladder's — so the
    curriculum has nothing left to schedule."""
    assert build_training_curriculum(_cfg(), algorithms=None) is None
    assert build_training_curriculum(_cfg(), algorithms=[]) is None
    # ...but explicitly naming algorithms with no algorithmic defense is a config
    # error, not something to silently ignore.
    try:
        build_training_curriculum(_cfg(algorithms=["defl"]), algorithms=None)
    except ValueError:
        return
    raise AssertionError("curriculum.algorithms under defense.mode: llm must be rejected")


def test_builder_takes_the_poisoner_count_from_the_attack_config():
    cur = build_training_curriculum(_cfg(poison_ids=(0, 1, 2)), algorithms=ALGS)
    assert cur.poisoner_counts == [3]
    assert build_training_curriculum(_cfg(poison_ids=(0,)),
                                     algorithms=ALGS).poisoner_counts == [1]


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
    cur = TrainingCurriculum(ALGS, [2], rounds_per_block=3)
    for _ in range(7):
        cur.advance()
    state = cur.state_dict()
    assert state["step"] == 7

    fresh = TrainingCurriculum(ALGS, [2], rounds_per_block=3)
    fresh.load_state_dict(state)
    assert fresh.step == 7 and fresh.peek() == cur.peek()

    # An edited config keeps the POSITION (restarting the sweep would re-train the
    # first block from scratch on every restart) but warns — see load_state_dict.
    changed = TrainingCurriculum(ALGS, [3], rounds_per_block=10)
    changed.load_state_dict(state)
    assert changed.step == 7


def test_resume_falls_back_to_rounds_done_for_old_checkpoints():
    """`rounds_done` counts exactly the rounds that consume a curriculum slot (the
    between-phase FL interlude consumes neither), so it IS the sweep position."""
    class _Env:
        curriculum = TrainingCurriculum(ALGS, [2], rounds_per_block=3)

    env = _Env()
    _resume_curriculum(env, {"rounds_done": 11, "curriculum": None}, start_round=11)
    assert env.curriculum.step == 11
    assert env.curriculum.peek() == env.curriculum.slot_at(11)


def test_resume_prefers_the_saved_position():
    class _Env:
        curriculum = TrainingCurriculum(ALGS, [2], rounds_per_block=3)

    env = _Env()
    _resume_curriculum(env, {"rounds_done": 11, "curriculum": {"step": 4}}, start_round=11)
    assert env.curriculum.step == 4


def test_resume_is_a_noop_without_a_curriculum():
    class _Env:
        curriculum = None

    _resume_curriculum(_Env(), {"rounds_done": 5}, start_round=5)   # must not raise


# --------------------------------------------------------------------------- env wiring
def test_env_rounds_follow_the_curriculum_not_the_rng():
    env = _env()                       # 3 algorithms x 2 rounds per block
    seen = []
    for _ in range(env.curriculum.rounds_per_cycle):
        env.begin_round()
        seen.append(env.round_defense)
        assert env.round_curriculum is not None
    assert seen == [a for a in ALGS for _ in range(2)], seen


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
    torch.manual_seed(1234)
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    env.reset(gw, [dict(gw) for _ in range(N_CLIENTS)], 0.5)
    env.begin_round()
    assert env.round_curriculum is None
    assert env.round_defense in ALGS


def test_the_attack_is_unaffected_by_the_curriculum():
    """The poisoned set and the ladder level come from the attack config, not the
    block — so two blocks differ only by which defense faced them."""
    env = _env()
    for _ in range(4):
        ctx = env.begin_round()
        assert ctx.poisoned_ids == [0, 1]
        assert ctx.flip_fraction == 1.0     # no verdicts fed back, so it holds


def test_fl_interlude_does_not_consume_a_curriculum_slot():
    """The honest FL round between phases is not a GRPO training round, so a block
    must still get its full `rounds_per_block` rounds."""
    cfg = _cfg()
    cfg["fl"]["freeze_global_in_phase2"] = False   # the interlude is a no-op when frozen
    env = _env(cfg)
    env.begin_round()
    step_before = env.curriculum.step
    env.run_benign_fl_round()
    assert env.curriculum.step == step_before
    env.begin_round()
    assert env.round_defense == env.curriculum.slot_at(step_before).algorithm


def test_target_drop_is_held_fixed_during_training():
    """sample_target_in_training is off, so every round reports against the same
    damage bar — otherwise a block's rounds would not be comparable to each other."""
    env = _env()
    goals = [env.begin_round().goal["target_accuracy_drop"] for _ in range(6)]
    assert goals == [0.1] * 6, goals


# --------------------------------------------------------------------------- driver
class _FakePolicy:
    adapters = ("defender",)

    def __init__(self):
        self._state = {"defender": {"w": torch.tensor([1.0])}}
        self._params = {"defender": [torch.nn.Parameter(torch.zeros(1))]}

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


def _drive(env, cfg, sim_rounds, td, resume=None, start_round=0):
    """Run the real GRPO driver with a stubbed round body; return (slots, progress).

    ``env.defense`` is cleared so ``train()`` accepts the run (it refuses an
    algorithmic defense, which leaves nothing to train) while the curriculum the
    env was built with keeps driving the sweep.
    """
    import rl.schedule as sched
    from metrics import MetricsTracker

    env.defense = None
    cfg["fl"]["simulation_rounds"] = sim_rounds
    cfg["rl"] = {"G": 2, "save_every": 1, "league_snapshot_every": 0,
                 "min_phase_rounds": 2, "max_phase_rounds": 3, "success_streak": 2,
                 "fl_interlude_between_phases": True,
                 "adapter_paths": {"defender": os.path.join(td, "def")}}
    slots, progress = [], []

    def fake_step(state, pidx, pround):
        env.begin_round()
        slots.append(env.round_curriculum.algorithm)
        return {"rewards": [0.0], "completions": ["x"], "loss": 0.0, "mean_reward": 0.0,
                "max_reward": 0.0, "min_reward": 0.0,
                "zero_advantage_fraction": 0.0}, 0.01, True

    saved_step = sched._step_round
    sched._step_round = fake_step
    try:
        sched.train(env, _FakePolicy(), None, cfg,
                    MetricsTracker(0.5, output_dir=os.path.join(td, "m")),
                    lambda log: None, random.Random(0),
                    progress_cb=lambda d, round_index=None, controller=None,
                    curriculum=None, attacker_state=None:
                        progress.append((d, curriculum)),
                    start_round=start_round, resume=resume)
    finally:
        sched._step_round = saved_step
    return slots, progress


def test_driver_follows_the_sweep_across_phase_boundaries_and_resumes_mid_block():
    """The phase machinery (boundaries + FL interludes) runs on top of the sweep
    without perturbing it, and a restart continues in the same block."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # 3 algorithms x 2 rounds per block.
        first, progress = _drive(_env(), _cfg(), sim_rounds=5, td=td)
        assert first == ["defl", "defl", "dnc", "dnc", "multikrum"], first

        # The position is checkpointed with the round count, and they agree.
        done, saved = progress[-1]
        assert done == 5 and saved["step"] == 5, progress[-1]

        # Resume: continue block 2 (multikrum) rather than restarting at defl.
        resumed, _ = _drive(_env(), _cfg(), sim_rounds=8, td=td,
                            resume={"rounds_done": 5, "round_index": 5,
                                    "controller": None, "curriculum": saved},
                            start_round=5)
        assert resumed == ["multikrum", "defl", "defl"], resumed


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} curriculum tests passed.")


if __name__ == "__main__":
    _run()
