"""Tests for single-learner training (``main.py --freeze <agent>``).

Guards the contract the mode promises:
  * only ONE agent learns and the learner NEVER switches;
  * no between-phase benign FL interlude and no league snapshots (there are no phases);
  * only the learner's adapter is written — the frozen agent's checkpoint is untouched;
  * the arms-race ``PhaseController`` state on disk SURVIVES a frozen run, so going
    back to a plain ``main.py`` run resumes alternating where it left off;
  * with a defense ensemble attached the defender LLM is never called, and the round's
    clean counterfactual is measured under that same defense.

Torch-only (no MNIST/GPU/LLM):  python tests/test_freeze_mode.py
"""

import copy
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from core.types import DetectionVerdict  # noqa: E402
from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from rl.turns import AttackerTurn  # noqa: E402
import rl.schedule as sched  # noqa: E402
import storage.checkpoint as ckpt  # noqa: E402


def _loader(seed, n=64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)


def _cfg():
    return {
        "fl": {"n_clients": 4, "device": "cpu", "benign_retrain_each_round": False,
               "training_rounds": 20, "n_compromisable": 2, "lr": 0.05, "local_epochs": 1,
               "simulation_rounds": 1},
        "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.2},
                   "max_poison_clients": 2, "sample_budget_in_training": True},
        "rl": {"G": 2, "save_every": 1, "league_snapshot_every": 1, "league_prob": 1.0,
               "first_learner": "attacker", "switch_mode": "best_response",
               "min_phase_rounds": 1, "max_phase_rounds": 2, "success_streak": 1,
               "fl_interlude_between_phases": True, "curriculum_on_cap": True,
               "adapter_paths": {"attacker": None, "defender": None}},
    }


def _make_env(defense=None):
    env = FLArmsRaceEnv(_cfg(), [_loader(i) for i in range(4)], _loader(999, 128),
                        random.Random(0), defense=defense)
    net = MnistNet()
    gw = {k: v.clone() for k, v in net.state_dict().items()}
    cw = [{k: v.clone() + 0.01 * (i + 1) for k, v in gw.items()} for i in range(4)]
    env.reset(gw, cw, baseline_accuracy=0.10)
    return env


class _FakePolicy:
    def __init__(self):
        self.adapters = ("attacker", "defender")
        self._state = {"attacker": {"w": torch.tensor([1.0])},
                       "defender": {"w": torch.tensor([2.0])}}
        self._params = {n: [torch.nn.Parameter(torch.zeros(1))] for n in self.adapters}
        self.saved = {}

    def adapter_parameters(self, name): return self._params[name]
    def set_adapter(self, name): pass
    def get_adapter_state(self, name): return copy.deepcopy(self._state[name])
    def set_adapter_state(self, name, s): self._state[name] = copy.deepcopy(s)
    def save_adapter(self, name, path): self.saved[path] = copy.deepcopy(self._state[name])
    def save_adapter_state_dict(self, name, state, path): self.saved[path] = copy.deepcopy(state)
    def generate(self, *a, **kw): raise AssertionError("no LLM should be generated here")


class _NullDefense:
    """Stand-in panel: clears everyone, records every call.

    ``begin_round`` is part of the defense contract — the env calls it once per
    round so ``defense.mode: single`` can pick that round's algorithm BEFORE the
    clean counterfactual is measured (see DefenseEnsemble.begin_round).
    """

    def __init__(self):
        self.calls = []
        self.rounds_begun = 0

    def describe(self): return "null"

    def begin_round(self):
        self.rounds_begun += 1
        return ["null"]

    def verdicts(self, updates, global_weights, *, commit=False):
        self.calls.append(commit)
        return ([DetectionVerdict(u.client_id, False, 1.0, "null") for u in updates],
                {"panel_mode": "single", "algorithms": ["null"], "configured": ["null"],
                 "per_defense_flags": {"null": []}, "flagged": []})


def _run_frozen(freeze, sim_rounds, td, defense=None, resume=None):
    """Drive sched.train in --freeze mode with a stubbed round body. Returns
    (learners seen per round, interlude count, controllers passed to progress_cb)."""
    cfg = _cfg()
    cfg["fl"]["simulation_rounds"] = sim_rounds
    cfg["rl"]["adapter_paths"] = {"attacker": os.path.join(td, "att"),
                                  "defender": os.path.join(td, "def")}
    env = _make_env(defense=defense)
    interlude = {"n": 0}
    _orig = env.run_benign_fl_round

    def counted():
        interlude["n"] += 1
        return _orig()
    env.run_benign_fl_round = counted

    learners = []

    def fake_step(state, learner, opp, opp_gen, pidx, pround):
        learners.append(learner)
        env.begin_round()
        return {"rewards": [0.0], "completions": ["x"], "loss": 0.0, "mean_reward": 0.0,
                "max_reward": 0.0, "min_reward": 0.0, "zero_advantage_fraction": 0.0}, 0.0, True

    policy = _FakePolicy()
    plog = []
    _saved = sched._step_round
    sched._step_round = fake_step
    try:
        sched.train(env, policy, None, None, cfg, object(), lambda log: None,
                    random.Random(0),
                    progress_cb=lambda d, round_index=None, controller=None: plog.append(controller),
                    fl_state_cb=lambda s: None,
                    start_round=(resume or {}).get("rounds_done", 0), resume=resume,
                    freeze=freeze)
    finally:
        sched._step_round = _saved
    return learners, interlude["n"], plog, policy, cfg


# --------------------------------------------------------------------------- schedule
def test_freeze_defender_trains_only_the_attacker():
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        learners, interludes, _, _, _ = _run_frozen("defender", 6, td, defense=_NullDefense())
    assert learners == ["attacker"] * 6, f"learner switched: {learners}"
    assert interludes == 0, "a single-learner run has no phases, so no FL interlude"


def test_freeze_attacker_trains_only_the_defender():
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        learners, interludes, _, _, _ = _run_frozen("attacker", 4, td)
    assert learners == ["defender"] * 4, f"learner switched: {learners}"
    assert interludes == 0


def test_switching_still_happens_without_freeze():
    """Control: the same config DOES alternate when --freeze is not given, so the
    frozen behaviour comes from the flag and not from the stubbed round body."""
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        learners, interludes, _, _, _ = _run_frozen(None, 4, td)
    assert len(set(learners)) == 2, f"expected alternation, got {learners}"
    assert interludes >= 1, "the normal schedule runs an FL interlude between phases"


def test_only_the_learner_adapter_is_written():
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        _, _, _, policy, cfg = _run_frozen("defender", 3, td, defense=_NullDefense())
        assert cfg["rl"]["adapter_paths"]["attacker"] in policy.saved
        assert cfg["rl"]["adapter_paths"]["defender"] not in policy.saved, (
            "the frozen defender's adapter must not be rewritten")


def test_no_league_snapshots_in_frozen_mode():
    """league_snapshot_every=1 + league_prob=1.0 in the config; frozen mode must
    still take none (the opponent never changes, and each snapshot costs ~115 MB)."""
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        cfg = _cfg()
        cfg["fl"]["simulation_rounds"] = 3
        cfg["rl"]["adapter_paths"] = {"attacker": os.path.join(td, "att"),
                                      "defender": os.path.join(td, "def")}
        env = _make_env(defense=_NullDefense())
        taken = []
        _saved_step, _saved_snap = sched._step_round, sched.League.snapshot
        sched._step_round = lambda *a, **k: (
            {"rewards": [0.0], "completions": ["x"], "loss": 0.0, "mean_reward": 0.0,
             "max_reward": 0.0, "min_reward": 0.0, "zero_advantage_fraction": 0.0}, 0.0, True)
        sched.League.snapshot = lambda self, policy, names, states=None: taken.append(list(names))
        try:
            sched.train(env, _FakePolicy(), None, None, cfg, object(), lambda log: None,
                        random.Random(0), progress_cb=None, fl_state_cb=None,
                        start_round=0, freeze="defender")
        finally:
            sched._step_round, sched.League.snapshot = _saved_step, _saved_snap
    assert taken == [], f"frozen mode should take no league snapshots, took {taken}"


# --------------------------------------------------------------------------- resume
def test_frozen_run_preserves_the_saved_arms_race_schedule():
    """A frozen run advances the round counters but must NOT clobber the controller,
    so a later plain run resumes the alternating schedule where it stopped."""
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        saved_ctrl = {"learner": "defender", "phase_index": 7, "phase_round": 4,
                      "streak": 2, "capped": False}
        ckpt.save_progress(100, round_index=100, controller=saved_ctrl)

        # What the frozen schedule does at every checkpoint: controller=None.
        ckpt.save_progress(103, round_index=103, controller=None)

        prog = ckpt.load_progress()
        assert prog["rounds_done"] == 103 and prog["round_index"] == 103
        assert prog["controller"] == saved_ctrl, (
            f"frozen run wiped the arms-race schedule: {prog['controller']}")


def test_progress_file_still_round_trips_a_fresh_start():
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        ckpt.save_progress(5)
        assert ckpt.load_progress() == {"rounds_done": 5, "round_index": None,
                                        "controller": None}
        with open(os.path.join(td, "rl_progress.json")) as f:
            assert json.load(f) == {"rounds_done": 5}


def test_frozen_progress_cb_never_sends_a_controller():
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        _, _, plog, _, _ = _run_frozen("defender", 3, td, defense=_NullDefense())
    assert plog and all(c is None for c in plog), f"controller leaked: {plog}"


# --------------------------------------------------------------------------- turn wiring
def test_attacker_turn_uses_the_defense_and_never_the_defender_llm():
    from agents.attacker_agent import AttackerAgent

    defense = _NullDefense()
    env = _make_env(defense=defense)
    env.begin_round()

    # defender_agent=None and defender_gen=None: if the turn ever tried the LLM path
    # it would raise AttributeError instead of quietly working.
    turn = AttackerTurn(env, AttackerAgent({}), None, None, defense=defense)
    n_before = len(defense.calls)
    turn.reward('{"clients": [{"id": 0, "operations": [{"op": "scale", "factor": 3.0}]}]}')
    assert len(defense.calls) > n_before
    assert defense.calls[-1] is False, "scoring a rollout must not commit defense state"

    info = turn.commit('{"clients": [{"id": 0, "operations": [{"op": "scale", "factor": 3.0}]}]}')
    assert defense.calls[-1] is True, "the committed round must advance defense state"
    assert info["defense_info"]["algorithms"] == ["null"]

    # And the normal path still requires a generator.
    try:
        AttackerTurn(env, AttackerAgent({}), None, None)
    except ValueError:
        return
    raise AssertionError("AttackerTurn with neither a generator nor a defense should raise")


def test_clean_reference_runs_through_the_attached_defense():
    """The clean counterfactual must be measured WITH the defense, otherwise the
    honest clients Multi-Krum/DnC always drop show up as attacker damage."""
    class _DropOne(_NullDefense):
        def verdicts(self, updates, global_weights, *, commit=False):
            self.calls.append(commit)
            return ([DetectionVerdict(u.client_id, u.client_id == 3, 1.0, "drop3")
                     for u in updates],
                    {"algorithms": ["drop3"], "per_defense_flags": {"drop3": [3]},
                     "flagged": [3]})

    defense = _DropOne()
    env = _make_env(defense=defense)
    env.begin_round()
    assert defense.calls, "clean_reference_accuracy did not consult the defense"
    assert all(c is False for c in defense.calls), "the clean reference must not commit"


def test_the_round_defense_is_chosen_before_the_clean_reference_is_measured():
    """With defense.mode=single a different algorithm judges each round, so the pick
    must happen at begin_round — BEFORE the clean counterfactual — or `drop` would
    subtract accuracies measured under two different defenses."""
    defense = _NullDefense()
    env = _make_env(defense=defense)

    order = []
    _begin, _verdicts = defense.begin_round, defense.verdicts
    defense.begin_round = lambda: (order.append("begin_round"), _begin())[1]
    defense.verdicts = lambda *a, **kw: (order.append("verdicts"), _verdicts(*a, **kw))[1]

    env.begin_round()
    assert order and order[0] == "begin_round", order
    assert "verdicts" in order, "the clean reference never ran"
    assert defense.rounds_begun == 1
    env.begin_round()
    assert defense.rounds_begun == 2, "each FL round must re-pick its defense"

    plain = _make_env()          # same state, no defense attached
    plain.begin_round()
    assert plain.defense is None


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} freeze-mode tests passed.")


if __name__ == "__main__":
    _run()
