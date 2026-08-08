"""Resume/switching correctness tests for the Phase-2 arms-race driver.

Guards the four resume bugs fixed in rl/schedule.py + rl/env.py + storage:
  #5  the shared Phase-2 FL state (evolving global model + client weights) round-trips
      through snapshot/restore + save/load, so a resume continues the arms race instead
      of rewinding to the Phase-1 baseline;
  #2  the between-phase benign FL interlude fires at a TRUE phase start but NOT when
      resuming into the middle of a phase (no spurious accuracy-bumping round on restart);
  #3  the PhaseController snapshot persisted at a checkpoint reflects the RECORDED round;
  #1  a checkpoint taken during a borrowed (league/curriculum) phase persists the
      opponent's LIVE weights, never the swapped-in snapshot.

Uses tiny synthetic loaders + a stubbed round body, so no MNIST/GPU/LLM is needed:
    python tests/test_resume.py
"""

import copy
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
import rl.schedule as sched  # noqa: E402
import storage.checkpoint as ckpt  # noqa: E402


def _loader(seed, n=64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)


def _cfg():
    return {
        # freeze_global_in_phase2: False — the FL-state roundtrip and the interlude
        # gating tested here are about the CONTINUING federation, whose shared model
        # a resume has to carry. Frozen rounds re-anchor instead (covered in
        # tests/test_frozen_rounds.py::test_restore_reanchors_a_drifted_checkpoint).
        "fl": {"n_clients": 4, "device": "cpu", "benign_retrain_each_round": False,
               "training_rounds": 20, "n_compromisable": 2, "lr": 0.05, "local_epochs": 1,
               "freeze_global_in_phase2": False, "simulation_rounds": 1},
        "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.2},
                   "max_poison_clients": 2, "sample_budget_in_training": True},
        "rl": {"G": 2, "save_every": 1, "league_snapshot_every": 0, "league_prob": 0.0,
               "first_learner": "attacker", "switch_mode": "best_response",
               "min_phase_rounds": 2, "max_phase_rounds": 50, "success_streak": 2,
               "fl_interlude_between_phases": True, "curriculum_on_cap": False,
               "adapter_paths": {"attacker": None, "defender": None}},
    }


def _make_env():
    env = FLArmsRaceEnv(_cfg(), [_loader(i) for i in range(4)], _loader(999, 128), random.Random(0))
    net = MnistNet()
    gw = {k: v.clone() for k, v in net.state_dict().items()}
    cw = [{k: v.clone() + 0.01 * (i + 1) for k, v in gw.items()} for i in range(4)]
    env.reset(gw, cw, baseline_accuracy=0.10)
    return env, gw, cw


class _FakePolicy:
    """Minimal LLMPolicy stand-in: named adapter states + save hooks that record what
    was actually written to disk (so borrowed-save behaviour is observable)."""

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


def _differs(a, b):
    return any(not torch.allclose(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------- #5
def test_fl_state_roundtrip():
    env, gw, cw = _make_env()
    env.run_benign_fl_round()                    # move the shared model off baseline
    assert _differs(gw, env.global_weights)
    snap = env.snapshot_fl_state()
    deg_global = copy.deepcopy(env.global_weights)
    deg_clients = [copy.deepcopy(w) for w in env.client_weights]
    deg_acc, deg_ri = env.current_accuracy, env.round_index

    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        ckpt.save_fl_state(snap)
        loaded = ckpt.load_fl_state()

    env2 = FLArmsRaceEnv(_cfg(), [_loader(i) for i in range(4)], _loader(999, 128), random.Random(0))
    env2.reset(gw, cw, baseline_accuracy=0.10)   # what resume does BEFORE restore
    assert not _differs(gw, env2.global_weights), "reset should give the Phase-1 baseline"
    env2.restore_fl_state(loaded)
    assert not _differs(deg_global, env2.global_weights), "global not restored"
    for a, b in zip(deg_clients, env2.client_weights):
        assert not _differs(a, b), "client weights not restored"
    assert abs(env2.current_accuracy - deg_acc) < 1e-9, "accuracy not restored"
    assert env2.round_index == deg_ri, "round_index not restored"
    assert _differs(gw, env2.global_weights), "restored global must differ from baseline"


# --------------------------------------------------------------------------- driver harness
def _run_driver(cfg, sim_rounds, successes, resume, td, max_phase=None):
    if max_phase is not None:
        cfg["rl"]["max_phase_rounds"] = max_phase
    env, gw, cw = _make_env()
    cfg["fl"]["simulation_rounds"] = sim_rounds
    cfg["rl"]["adapter_paths"] = {"attacker": os.path.join(td, "att"),
                                  "defender": os.path.join(td, "def")}
    interlude = {"n": 0}
    _orig = env.run_benign_fl_round
    def counted():
        interlude["n"] += 1
        return _orig()
    env.run_benign_fl_round = counted

    q = list(successes)
    def fake_step(state, learner, opp, opp_gen, pidx, pround):
        env.begin_round()
        env.current_accuracy = max(0.0, env.current_accuracy - 0.01)
        s = q.pop(0) if q else False
        return {"rewards": [0.0], "completions": ["x"], "loss": 0.0, "mean_reward": 0.0,
                "max_reward": 0.0, "min_reward": 0.0, "zero_advantage_fraction": 0.0}, 0.01, s

    _saved_step = sched._step_round
    sched._step_round = fake_step
    plog, fl = [], {"n": 0}
    try:
        sched.train(env, _FakePolicy(), None, None, cfg, object(), lambda log: None,
                    random.Random(0),
                    progress_cb=lambda d, round_index=None, controller=None, curriculum=None: plog.append(copy.deepcopy(controller)),
                    fl_state_cb=lambda s: fl.__setitem__("n", fl["n"] + 1),
                    start_round=resume["rounds_done"] if resume else 0, resume=resume)
    finally:
        sched._step_round = _saved_step   # never leak the monkeypatch
    return interlude, plog, fl


# --------------------------------------------------------------------------- #2 + #3
def test_interlude_fires_at_true_phase_start():
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        # attacker wins streak=2 by round 3 -> switch; defender then runs 1 round.
        interlude, plog, fl = _run_driver(_cfg(), 4, [False, True, True, True], None, td)
    assert interlude["n"] == 1, f"expected exactly 1 interlude at the phase start, got {interlude['n']}"
    assert fl["n"] >= 1, "fl_state_cb should fire at checkpoints"
    last_ctrl = [c for c in plog if c is not None][-1]
    assert last_ctrl["phase_round"] >= 1, f"controller persisted before record(): {last_ctrl}"


def test_no_interlude_on_midphase_resume():
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        # Resume mid-defender at phase_round=5; max=7 -> caps after 2 rounds (done 8->10==sim),
        # so the run ends when THIS phase caps and no new phase starts.
        resume = {"rounds_done": 8, "round_index": 8,
                  "controller": {"learner": "defender", "phase_index": 1, "phase_round": 5,
                                 "streak": 1, "capped": False}}
        interlude, _, _ = _run_driver(_cfg(), 10, [False] * 10, resume, td, max_phase=7)
    assert interlude["n"] == 0, f"spurious interlude on mid-phase resume: {interlude['n']}"


# --------------------------------------------------------------------------- #1
def test_borrowed_opponent_saves_live_weights():
    with tempfile.TemporaryDirectory() as td:
        policy = _FakePolicy()
        live_def = policy.get_adapter_state("defender")
        league = sched.League(random.Random(0))
        league.snapshots["defender"] = [{"w": torch.tensor([99.0])}]   # an OLD snapshot
        state = {"policy": policy, "league": league, "max_new_tokens": 8,
                 "borrowed_opponent": None,
                 # Both sides trainable: this is the defender-LLM path, the only
                 # one where an opponent adapter can be borrowed at all.
                 "trainable": ("attacker", "defender"), "algorithmic_defense": False,
                 "adapter_paths": {"attacker": os.path.join(td, "att"),
                                   "defender": os.path.join(td, "def")}}
        _opp_gen, restore = sched._opponent_generator(state, "defender", face_snapshot=True)
        assert torch.allclose(policy._state["defender"]["w"], torch.tensor([99.0])), "snapshot not swapped in"
        assert state["borrowed_opponent"][0] == "defender", "borrowed marker not set"
        sched._save_adapters(state)              # a mid-phase checkpoint
        saved = policy.saved[state["adapter_paths"]["defender"]]
        assert torch.allclose(saved["w"], live_def["w"]), (
            f"borrowed checkpoint clobbered the opponent: saved {saved} want {live_def}")
        restore()
        assert state["borrowed_opponent"] is None and torch.allclose(policy._state["defender"]["w"], live_def["w"])


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} resume/switching tests passed.")


if __name__ == "__main__":
    _run()
