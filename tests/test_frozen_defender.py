"""Tests for `python main.py --env linux --freeze defender`.

Covers the three things that mode changes:
  1. the phase controller no longer alternates learners (rl/switch.py),
  2. the attacker's round takes its verdicts from the algorithmic ensemble
     instead of the defender LLM (rl/turns.py + rl/defenders.py),
  3. the schedule trains ONLY the attacker and never touches a defender adapter
     (rl/schedule.py::_train_attacker_only).

Uses a stub LLM policy (one real torch Parameter, so GRPO's backward/step path is
exercised) against the REAL env, ensemble and reward — no GPU, no model download.

Run:  python tests/test_frozen_defender.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from benchmark.defenses import build_defenses  # noqa: E402
from benchmark.defenses.ensemble import EnsembleDefense  # noqa: E402
from rl.defenders import AlgorithmicDefenderPolicy  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from rl.schedule import train  # noqa: E402
from rl.switch import PhaseController, SwitchConfig  # noqa: E402
from agents.attacker_agent import AttackerAgent  # noqa: E402
from agents.defender_agent import DefenderAgent  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Pinned phase controller
# ---------------------------------------------------------------------------

def test_pinned_controller_never_switches_learner():
    cfg = SwitchConfig(min_phase_rounds=2, max_phase_rounds=4, success_streak=2)
    ctrl = PhaseController(cfg, first_learner="attacker", alternate=False)
    for reason in ("success", "cap", "success"):
        ctrl.next_phase(reason)
        assert ctrl.learner == "attacker", "the learner must stay pinned"
    assert ctrl.phase_index == 3 and ctrl.phase_round == 0 and ctrl.streak == 0
    # The gates themselves are untouched: a sustained win still ends the phase.
    assert ctrl.record(True) == (False, None)
    assert ctrl.record(True) == (True, "success")
    # ...and alternating remains the default.
    assert PhaseController(cfg).alternate is True


# ---------------------------------------------------------------------------
# A tiny end-to-end world: 6 clients, 4 test batches, a stub attacker policy
# ---------------------------------------------------------------------------

N_CLIENTS = 6
CONFIG = {
    "fl": {"n_clients": N_CLIENTS, "n_compromisable": 3, "training_rounds": 0,
           "simulation_rounds": 4, "benign_retrain_each_round": False,
           "lr": 0.01, "local_epochs": 1, "batch_size": 8, "device": "cpu",
           "poison_seed": 0},
    "data": {"n_classes": 10},
    "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.10},
               "max_poison_clients": 1, "sample_budget_in_training": False,
               "sample_target_in_training": False},
    "rl": {"G": 2, "kl_beta": 0.0, "lr": 1e-3, "max_new_tokens": 8, "temperature": 1.0,
           "save_every": 0, "league_snapshot_every": 0, "league_prob": 0.0,
           "switch_mode": "best_response", "first_learner": "attacker",
           "min_phase_rounds": 1, "max_phase_rounds": 2, "success_streak": 1,
           "fl_interlude_between_phases": False,
           "skip_zero_advantage": False, "resample_on_zero_advantage": False},
    "defense": {"members": ["multikrum", "dnc", "defl"], "vote": "majority",
                "num_byzantine": 1},
}

# Two attack plans with clearly different outcomes, so the GRPO group has reward
# spread: a 1% edit that hides inside the honest clients' own spread (Multi-Krum
# and DnC spend their standing drop-quota on an honest client instead), and a 50x
# edit that every member flags.
PLANS = [
    json.dumps({"clients": [{"id": 0, "operations": [
        {"op": "scale", "target": "all", "factor": 1.01}]}]}),
    json.dumps({"clients": [{"id": 1, "operations": [
        {"op": "scale", "target": "all", "factor": 50.0}]}]}),
]


class StubPolicy:
    """Minimal stand-in for rl.policy.LLMPolicy: cycles fixed completions and
    exposes one real Parameter so grpo_step's backward + optimizer step run."""

    adapters = ("attacker",)

    def __init__(self):
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.saved = []
        self.generated_for = []
        self.last_generation_completed = []
        self._i = 0

    def generate(self, adapter, system, user, n=1, temperature=0.0, max_new_tokens=0):
        self.generated_for.append(adapter)
        out = []
        for _ in range(n):
            out.append(PLANS[self._i % len(PLANS)])
            self._i += 1
        self.last_generation_completed = [True] * n
        return out

    def adapter_parameters(self, name):
        assert name == "attacker", f"no adapter {name!r} exists in frozen-defender mode"
        return [self.param]

    def policy_token_logprobs(self, adapter, system, user, completion, append_eos=False):
        return (self.param * float(len(completion) % 7 + 1)).repeat(3)

    def reference_token_logprobs(self, system, user, completion, append_eos=False):
        return torch.zeros(3)

    def save_adapter(self, name, path):
        self.saved.append((name, path))

    def get_adapter_state(self, name):
        raise AssertionError("the league must be disabled in frozen-defender mode")

    def set_adapter_state(self, name, state):
        raise AssertionError("the league must be disabled in frozen-defender mode")


class _Tracker:
    def __init__(self):
        self.rounds = []

    def update(self, round_num, verdicts, accuracy, poisoned):
        self.rounds.append(round_num)


def _test_loader(n_batches=4, batch=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(batch, 1, 28, 28, generator=g),
             torch.randint(0, 10, (batch,), generator=g)) for _ in range(n_batches)]


def _client_weights(seed=0):
    from model.mnist_net import MnistNet
    torch.manual_seed(seed)
    base = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    return base, [{k: v + torch.randn_like(v) * 0.01 for k, v in base.items()}
                  for _ in range(N_CLIENTS)]


def _algorithmic_defender():
    members = build_defenses(CONFIG["defense"]["members"],
                             multikrum_num_byzantine=1, dnc_num_byzantine=1)
    return AlgorithmicDefenderPolicy(EnsembleDefense(members, vote="majority"))


def _env():
    import random
    global_weights, client_weights = _client_weights()
    env = FLArmsRaceEnv(CONFIG, None, _test_loader(), random.Random(0))
    env.reset(global_weights, client_weights, 0.5)
    return env


# ---------------------------------------------------------------------------
# 2. The attacker's round is judged by the ensemble, not by an LLM
# ---------------------------------------------------------------------------

def test_attacker_turn_uses_the_algorithmic_defense():
    from rl.turns import AttackerTurn

    env = _env()
    env.begin_round()
    defense = _algorithmic_defender()
    turn = AttackerTurn(env, AttackerAgent({"n_clients": N_CLIENTS}), DefenderAgent({}), defense)

    # The ensemble — not an LLM — decides, and the attacker's reward moves with
    # it: the 50x client is flagged by every member (stealth ~0), the 1% one
    # evades (full stealth bonus).
    loud = turn.reward(PLANS[1])
    quiet = turn.reward(PLANS[0])
    assert quiet > loud, (quiet, loud)

    info = turn.commit(PLANS[0])
    assert info["poisoned_ids"] == [0]
    assert {v.client_id for v in info["verdicts"]} == set(range(N_CLIENTS))
    assert 0 not in {v.client_id for v in info["verdicts"] if v.is_suspicious}
    assert all("votes" in v.reason for v in info["verdicts"]), \
        "verdicts should carry the ensemble's vote, not an LLM's free text"
    # ...and a blatant attack IS caught.
    env.begin_round()
    turn = AttackerTurn(env, AttackerAgent({"n_clients": N_CLIENTS}), DefenderAgent({}), defense)
    caught = turn.commit(PLANS[1])
    assert caught["poisoned_ids"] == [1]
    assert 1 in {v.client_id for v in caught["verdicts"] if v.is_suspicious}


def test_algorithmic_defender_advances_state_only_on_commit():
    env = _env()
    env.begin_round()
    defense = _algorithmic_defender()
    defl = defense.ensemble.members["defl"]
    updates = env.build_updates({})

    for _ in range(3):
        defense.verdicts(env, updates, commit=False)
    assert defl._prev_total_fgnv is None
    defense.verdicts(env, updates, commit=True)
    assert defl._prev_total_fgnv is not None


# ---------------------------------------------------------------------------
# 3. The schedule trains only the attacker
# ---------------------------------------------------------------------------

def test_train_frozen_defender_only_touches_the_attacker():
    import random

    env = _env()
    policy = StubPolicy()
    tracker = _Tracker()
    logs = []
    saved_progress = []

    train(env, policy, AttackerAgent({"n_clients": N_CLIENTS}), DefenderAgent({}),
          CONFIG, tracker, logs.append, random.Random(0),
          progress_cb=lambda done, ri=None, ctrl=None: saved_progress.append((done, ri, ctrl)),
          algorithmic_defender=_algorithmic_defender(),
          adapter_paths={"attacker": "checkpoints/frozen_defender/attacker_adapter"})

    assert len(tracker.rounds) == CONFIG["fl"]["simulation_rounds"]
    # Only the attacker ever generated, learned and was checkpointed.
    assert set(policy.generated_for) == {"attacker"}
    assert {name for name, _ in policy.saved} == {"attacker"}
    assert all(path.startswith("checkpoints/frozen_defender") for _, path in policy.saved)
    assert {r.learning_agent for r in logs} == {"attacker"}
    # The phase index advances (min_phase_rounds=1 over 4 rounds) but the learner
    # never becomes the defender.
    assert saved_progress and all(c is None or c["learner"] == "attacker"
                                  for _, _, c in saved_progress)
    assert max(r.attack_metadata["phase_index"] for r in logs) >= 1


def test_resume_from_an_arms_race_controller_is_pinned_back_to_attacker():
    """A progress file written by an alternating run can name the defender as the
    learner; frozen-defender mode must not try to train an adapter that is not
    even loaded."""
    import random

    env = _env()
    policy = StubPolicy()
    logs = []
    train(env, policy, AttackerAgent({"n_clients": N_CLIENTS}), DefenderAgent({}),
          CONFIG, _Tracker(), logs.append, random.Random(0),
          resume={"rounds_done": 0, "round_index": 0,
                  "controller": {"learner": "defender", "phase_index": 7,
                                 "phase_round": 0, "streak": 0, "capped": False}},
          algorithmic_defender=_algorithmic_defender(),
          adapter_paths={"attacker": "checkpoints/frozen_defender/attacker_adapter"})
    assert {r.learning_agent for r in logs} == {"attacker"}
    assert logs[0].attack_metadata["phase_index"] == 7   # phase state still resumed


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} frozen-defender tests passed.")


if __name__ == "__main__":
    _run()
