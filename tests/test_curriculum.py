"""Tests for the deterministic (defense x poisoner-count) training curriculum.

Covers ``rl/curriculum.py`` (the ordering itself, config construction and the
resume cursor), its two integration points — ``FLArmsRaceEnv.begin_round`` and
``AlgorithmicDefender.select`` — and the guarantee that evaluation is unaffected.

The ordering tests are torch-free; the env tests use synthetic MNIST-shaped
tensors, so there is no download, no GPU and no LLM:

    python tests/test_curriculum.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.curriculum import (  # noqa: E402
    TrainingCurriculum, build_curriculum, curriculum_enabled,
)

ALGOS = ["fltrust", "defl", "dnc", "multikrum"]
COUNTS = [1, 2, 3, 4, 5]


def _curriculum(**kw):
    kw.setdefault("algorithms", ALGOS)
    kw.setdefault("poisoner_counts", COUNTS)
    kw.setdefault("rounds_per_block", 10)
    return TrainingCurriculum(**kw)


def _cfg(**curriculum):
    base = {
        "fl": {"n_clients": 20, "n_compromisable": 5},
        "attack": {"max_poison_clients": 5, "sample_budget_in_training": True},
        "defense": {"mode": "algorithmic", "algorithms": ALGOS},
        "curriculum": {"enabled": True},
    }
    base["curriculum"].update(curriculum)
    return base


class _FakeDefender:
    """Stand-in for AlgorithmicDefender: the pool, the two selectors, and a
    ``run`` that declines to aggregate (so ``clean_reference_accuracy`` short-
    circuits to the current accuracy and no test-set pass is needed)."""

    def __init__(self, names=ALGOS):
        self._names = list(names)
        self.current = self._names[0]
        self.chose_randomly = 0
        self.ran_as = []

    @property
    def names(self):
        return list(self._names)

    def describe(self):
        return f"random over {self._names}"

    def choose(self):
        self.chose_randomly += 1
        return self._names[0]

    def select(self, name):
        if name not in self._names:
            raise KeyError(name)
        self.current = name
        return name

    def run(self, updates, global_weights, *, commit=False, algorithm=None):
        from server.algo_defender import DefenseOutcome
        self.ran_as.append(algorithm or self.current)
        return DefenseOutcome(algorithm or self.current, [], None, {})


# ---------------------------------------------------------------------------
# The sweep itself
# ---------------------------------------------------------------------------

def test_one_algorithm_is_held_while_the_poisoner_count_climbs():
    """The requested shape: 10 rounds of FLTrust@1, then FLTrust@2 ... FLTrust@5,
    and only THEN the next algorithm."""
    cur = _curriculum()
    got = [(s.algorithm, s.n_poisoners) for s in (cur.take() for _ in range(50))]
    expected = [("fltrust", n) for n in COUNTS for _ in range(10)]
    assert got == expected, got[:12]
    # Round 51 is the first round of the SECOND algorithm, back at 1 poisoner.
    nxt = cur.take()
    assert (nxt.algorithm, nxt.n_poisoners) == ("defl", 1)


def test_every_pair_gets_exactly_the_same_rounds_per_cycle():
    cur = _curriculum()
    assert cur.blocks_per_cycle == 20 and cur.rounds_per_cycle == 200
    seen = {}
    for _ in range(cur.rounds_per_cycle):
        s = cur.take()
        seen[(s.algorithm, s.n_poisoners)] = seen.get((s.algorithm, s.n_poisoners), 0) + 1
    assert set(seen) == {(a, n) for a in ALGOS for n in COUNTS}
    assert set(seen.values()) == {10}, seen


def test_the_cycle_repeats_from_the_first_algorithm():
    cur = _curriculum()
    first = [(s.algorithm, s.n_poisoners) for s in (cur.take() for _ in range(200))]
    second = [(s.algorithm, s.n_poisoners) for s in (cur.take() for _ in range(200))]
    assert first == second
    assert cur.slot_at(0).cycle == 0 and cur.slot_at(200).cycle == 1
    assert cur.slot_at(400).cycle == 2


def test_blocks_are_contiguous_and_labelled():
    cur = _curriculum()
    slots = [cur.take() for _ in range(21)]
    assert [s.round_in_block for s in slots[:10]] == list(range(1, 11))
    assert slots[10].round_in_block == 1 and slots[10].n_poisoners == 2
    assert slots[0].label == "fltrust/1p" and slots[10].label == "fltrust/2p"
    assert slots[0].block_index == 0 and slots[10].block_index == 1
    assert [s.step for s in slots[:3]] == [0, 1, 2]


def test_slot_at_is_pure_and_peek_does_not_consume():
    cur = _curriculum()
    assert cur.slot_at(137) == cur.slot_at(137)
    assert cur.step == 0
    assert cur.peek() == cur.slot_at(0) and cur.step == 0
    assert cur.take().step == 0 and cur.step == 1


def test_rounds_per_block_of_one_degenerates_to_round_robin_over_pairs():
    cur = _curriculum(rounds_per_block=1)
    got = [(s.algorithm, s.n_poisoners) for s in (cur.take() for _ in range(6))]
    assert got == [("fltrust", 1), ("fltrust", 2), ("fltrust", 3),
                   ("fltrust", 4), ("fltrust", 5), ("defl", 1)]


def test_no_algorithm_rotation_sweeps_poisoner_counts_only():
    """defense.mode: llm has no algorithm pool — the sweep still paces the quota."""
    cur = TrainingCurriculum(algorithms=[], poisoner_counts=[1, 2], rounds_per_block=2)
    got = [(s.algorithm, s.n_poisoners) for s in (cur.take() for _ in range(5))]
    assert got == [(None, 1), (None, 1), (None, 2), (None, 2), (None, 1)]
    assert cur.rounds_per_cycle == 4


def test_invalid_shapes_are_rejected():
    for kw in ({"poisoner_counts": []}, {"poisoner_counts": [0, 1]},
               {"rounds_per_block": 0}):
        try:
            _curriculum(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} must be rejected")


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def test_cursor_roundtrips_so_a_restart_continues_mid_block():
    cur = _curriculum()
    for _ in range(23):
        cur.take()
    snap = cur.state_dict()
    assert snap["step"] == 23

    restored = _curriculum()
    restored.load_state_dict(snap)
    assert restored.peek() == cur.peek()
    # Round 24 of the sweep is block 3 (fltrust @ 3 poisoners), round 4 of it.
    s = restored.take()
    assert (s.algorithm, s.n_poisoners, s.round_in_block) == ("fltrust", 3, 4)


def test_load_state_dict_keeps_the_current_shape_not_the_saved_one():
    """Editing the curriculum must take effect on resume rather than being
    silently overridden by whatever the checkpoint was written with."""
    old = TrainingCurriculum(algorithms=["defl"], poisoner_counts=[1],
                             rounds_per_block=3, step=7)
    new = _curriculum()
    new.load_state_dict(old.state_dict())
    assert new.step == 7
    assert new.algorithms == ALGOS and new.rounds_per_block == 10


def test_an_empty_or_missing_state_leaves_the_cursor_alone():
    cur = _curriculum(step=5)
    cur.load_state_dict({})
    cur.load_state_dict(None)
    assert cur.step == 5


# ---------------------------------------------------------------------------
# Construction from the config
# ---------------------------------------------------------------------------

def test_disabled_by_default_and_off_switch_returns_none():
    assert curriculum_enabled({}) is False
    assert build_curriculum({}, defender=_FakeDefender()) is None
    assert build_curriculum(_cfg(enabled=False), defender=_FakeDefender()) is None


def test_defaults_come_from_the_defense_pool_and_the_quota_cap():
    cur = build_curriculum(_cfg(), defender=_FakeDefender())
    assert cur.algorithms == ALGOS
    assert cur.poisoner_counts == [1, 2, 3, 4, 5]
    assert cur.rounds_per_block == 10


def test_an_explicit_subset_and_order_is_honoured():
    cur = build_curriculum(_cfg(algorithms=["multikrum", "fltrust"],
                                poisoner_counts=[2, 4], rounds_per_block=3),
                           defender=_FakeDefender())
    got = [(s.algorithm, s.n_poisoners) for s in (cur.take() for _ in range(13))]
    assert got[:3] == [("multikrum", 2)] * 3
    assert got[3:6] == [("multikrum", 4)] * 3
    assert got[6:9] == [("fltrust", 2)] * 3
    assert got[12] == ("multikrum", 2)          # cycle wrapped


def test_an_algorithm_the_defense_cannot_run_is_rejected_at_build_time():
    try:
        build_curriculum(_cfg(algorithms=["fltrust", "trimmed_mean"]),
                         defender=_FakeDefender())
    except ValueError as e:
        assert "trimmed_mean" in str(e)
    else:
        raise AssertionError("an algorithm outside defense.algorithms must be rejected")


def test_more_poisoners_than_the_attacker_can_reach_is_rejected():
    try:
        build_curriculum(_cfg(poisoner_counts=[1, 6]), defender=_FakeDefender())
    except ValueError as e:
        assert "6" in str(e) and "n_compromisable" in str(e)
    else:
        raise AssertionError("a count above fl.n_compromisable must be rejected")


def test_llm_mode_gets_a_poisoner_only_sweep():
    cfg = _cfg()
    cfg["defense"] = {"mode": "llm"}
    cur = build_curriculum(cfg, defender=None)
    assert cur.algorithms == [] and cur.poisoner_counts == [1, 2, 3, 4, 5]
    assert cur.take().algorithm is None


# ---------------------------------------------------------------------------
# Env integration
# ---------------------------------------------------------------------------

def _env(curriculum=None, defense="build", **cfg_over):
    """A tiny synthetic env (8 clients, MNIST-shaped) with the given curriculum."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from model.mnist_net import MnistNet
    from rl.env import FLArmsRaceEnv

    def loader(seed, n=64):
        g = torch.Generator().manual_seed(seed)
        return DataLoader(
            TensorDataset(torch.randn(n, 1, 28, 28, generator=g),
                          torch.randint(0, 10, (n,), generator=g)),
            batch_size=32, shuffle=True)

    cfg = {
        "fl": {"n_clients": 8, "device": "cpu", "lr": 0.05, "local_epochs": 1,
               "benign_retrain_each_round": False, "training_rounds": 5,
               "n_compromisable": 5, "poison_seed": 0, "batch_size": 32},
        "data": {"data_dir": "./data/mnist_raw"},
        "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.10},
                   "max_poison_clients": 5, "sample_budget_in_training": True},
    }
    cfg["attack"].update(cfg_over.pop("attack", {}))
    cfg.update(cfg_over)
    if defense == "build":
        defense = _FakeDefender()
    env = FLArmsRaceEnv(cfg, [loader(i) for i in range(8)], loader(99, n=128),
                        random.Random(0), defense=defense, curriculum=curriculum)
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    cw = [{k: v + torch.randn_like(v) * 0.01 for k, v in gw.items()} for _ in range(8)]
    env.reset(gw, cw, 0.5)
    return env


def test_begin_round_takes_the_defense_and_the_quota_from_the_sweep():
    cur = _curriculum(rounds_per_block=2, poisoner_counts=[1, 3])
    defender = _FakeDefender()
    env = _env(curriculum=cur, defense=defender)
    seen = []
    for _ in range(6):
        ctx = env.begin_round()
        seen.append((env.round_defense, ctx.budget))
    assert seen == [("fltrust", 1), ("fltrust", 1), ("fltrust", 3), ("fltrust", 3),
                    ("defl", 1), ("defl", 1)], seen
    assert defender.chose_randomly == 0, "the random draw must not run under a curriculum"


def test_the_slot_is_exposed_for_logging_and_advances_once_per_round():
    cur = _curriculum(rounds_per_block=2)
    env = _env(curriculum=cur)
    env.begin_round()
    assert env.round_slot.step == 0 and env.round_slot.round_in_block == 1
    # The clean counterfactual + every scored rollout run inside the round and must
    # NOT consume another slot.
    env.clean_reference_accuracy()
    env.build_updates({})
    assert cur.step == 1
    env.begin_round()
    assert env.round_slot.step == 1 and env.round_slot.round_in_block == 2


def test_the_honest_fl_interlude_does_not_consume_a_slot():
    """Between-phase benign rounds are not GRPO rounds — a run with many phase
    switches must not skip blocks."""
    cur = _curriculum(rounds_per_block=2)
    env = _env(curriculum=cur)
    env.begin_round()
    env.run_benign_fl_round()
    env.run_benign_fl_round()
    assert cur.step == 1
    assert env.begin_round().budget == 1 and env.round_slot.round_in_block == 2


def test_without_a_curriculum_the_random_draws_are_untouched():
    defender = _FakeDefender()
    env = _env(curriculum=None, defense=defender)
    budgets = {env.begin_round().budget for _ in range(30)}
    assert defender.chose_randomly == 30
    assert len(budgets) > 1, "sample_budget_in_training should still randomise"
    assert env.round_slot is None


def test_a_curriculum_pins_the_attack_goal():
    """Per-round target sampling is force-disabled: the sweep varies who and how
    many, never how hard, so the reward scale stays fixed."""
    cur = _curriculum()
    env = _env(curriculum=cur,
               attack={"sample_target_in_training": True,
                       "target_choices": [0.05, 0.30]})
    assert env.sample_target is False
    targets = {env.begin_round().goal["target_accuracy_drop"] for _ in range(10)}
    assert targets == {0.10}


def test_a_narrowed_pool_clamps_instead_of_over_poisoning():
    cur = _curriculum(rounds_per_block=1, poisoner_counts=[5])
    env = _env(curriculum=cur)
    env.n_compromisable = 2                      # what the benchmark's widening does in reverse
    assert env.begin_round().budget == 2


# ---------------------------------------------------------------------------
# Driver integration: the cursor survives a checkpoint/resume
# ---------------------------------------------------------------------------

def _fake_policy():
    import copy

    import torch

    class _FakePolicy:
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

    return _FakePolicy()


def _drive(env, td, sim_rounds, start_round=0, resume=None):
    """Run the real GRPO driver with a stubbed round body; return the slots the
    rounds ran under and the last progress payload the driver persisted."""
    import rl.schedule as sched
    from metrics import MetricsTracker

    cfg = {
        "fl": {"simulation_rounds": sim_rounds},
        "rl": {"G": 2, "save_every": 1, "league_snapshot_every": 0, "league_prob": 0.0,
               "switch_mode": "best_response", "min_phase_rounds": 2,
               "max_phase_rounds": 2, "success_streak": 2,
               "fl_interlude_between_phases": False, "curriculum_on_cap": False,
               "adapter_paths": {"attacker": os.path.join(td, "att"),
                                 "defender": os.path.join(td, "def")}},
    }
    slots, saved = [], {}

    def fake_step(state, learner, opp, opp_gen, pidx, pround):
        state["env"].begin_round()
        slots.append(state["env"].round_slot)
        return {"rewards": [0.0], "completions": ["x"], "loss": 0.0, "mean_reward": 0.0,
                "max_reward": 0.0, "min_reward": 0.0,
                "zero_advantage_fraction": 0.0}, 0.01, False

    def progress_cb(done, round_index=None, controller=None, curriculum=None):
        saved.update(rounds_done=done, round_index=round_index,
                     controller=controller, curriculum=curriculum)

    original = sched._step_round
    sched._step_round = fake_step
    try:
        sched.train(env, _fake_policy(), None, None, cfg,
                    MetricsTracker(0.5, output_dir=os.path.join(td, "m")),
                    lambda log: None, random.Random(0), progress_cb=progress_cb,
                    start_round=start_round, resume=resume)
    finally:
        sched._step_round = original
    return slots, saved


def test_the_driver_persists_and_resumes_the_curriculum_cursor():
    """Without the cursor in the progress file, every restart would replay the
    first algorithm's 1-poisoner block and the later blocks would never run."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        env = _env(curriculum=_curriculum(rounds_per_block=2, poisoner_counts=[1, 2]))
        slots, saved = _drive(env, td, sim_rounds=3)
        assert [(s.algorithm, s.n_poisoners) for s in slots] == [
            ("fltrust", 1), ("fltrust", 1), ("fltrust", 2)]
        assert saved["rounds_done"] == 3 and saved["curriculum"]["step"] == 3

        # A fresh process: new env, new curriculum object, resumed from that payload.
        env2 = _env(curriculum=_curriculum(rounds_per_block=2, poisoner_counts=[1, 2]))
        resume = {"rounds_done": 3, "round_index": saved["round_index"],
                  "controller": saved["controller"], "curriculum": saved["curriculum"]}
        slots2, saved2 = _drive(env2, td, sim_rounds=5, start_round=3, resume=resume)
        assert [(s.algorithm, s.n_poisoners) for s in slots2] == [
            ("fltrust", 2), ("defl", 1)], [(s.algorithm, s.n_poisoners) for s in slots2]
        assert saved2["curriculum"]["step"] == 5


def test_the_driver_runs_the_legacy_random_draws_when_the_curriculum_is_off():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        env = _env(curriculum=None)
        slots, saved = _drive(env, td, sim_rounds=3)
        assert slots == [None, None, None]
        assert saved["curriculum"] is None


# ---------------------------------------------------------------------------
# Evaluation must be unaffected
# ---------------------------------------------------------------------------

def test_the_benchmark_clears_any_curriculum():
    from benchmark.run_benchmark import _resolve_eval_budget

    env = _env(curriculum=_curriculum())
    _resolve_eval_budget(env, 3)
    assert env.curriculum is None
    assert env.sample_budget is False and env.budget_cap == 3
    assert env.begin_round().budget == 3
    assert env.begin_round().budget == 3


# ---------------------------------------------------------------------------
# AlgorithmicDefender.select
# ---------------------------------------------------------------------------

def test_select_pins_the_algorithm_and_rejects_unknown_names():
    from server.algo_defender import AlgorithmicDefender

    defenses = {name: object() for name in ("defl", "dnc")}
    defender = AlgorithmicDefender(defenses, random.Random(0), selection="random")
    assert defender.select("DnC") == "dnc" and defender.current == "dnc"
    try:
        defender.select("fltrust")
    except KeyError as e:
        assert "fltrust" in str(e)
    else:
        raise AssertionError("select() must refuse an algorithm outside the pool")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} curriculum tests passed.")


if __name__ == "__main__":
    _run()
