"""Tests for the FIXED poisoner set (``attack.fixed_poison_clients``).

The setting pins the poisoned clients to the first N ids, every round, instead of
drawing a per-round quota and letting the attacker LLM choose which of its pool
fills it. These tests cover the three places that has to hold together:

  * ``rl/env.py``   — the pool IS the poisoned set, and the quota is N whatever
    the curriculum or ``sample_budget_in_training`` say;
  * ``rl/curriculum.py`` — the poisoner sweep collapses to that single count;
  * ``agents/attacker_agent.py`` — a quota equal to the pool size makes
    ``select_and_apply`` poison exactly clients 0..N-1 for ANY model output,
    and the system prompt says so.

Torch is used for synthetic MNIST-shaped tensors — no download, no GPU, no LLM:

    python tests/test_fixed_poison_set.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from agents.attacker_agent import AttackerAgent  # noqa: E402
from rl.curriculum import build_training_curriculum  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from server.algo_defender import build_algorithmic_defender  # noqa: E402

N_CLIENTS = 12
N_POISON = 10
ALGS = ["defl", "dnc", "multikrum"]     # FLTrust needs a real root set (test_fltrust.py)


def _loader(seed: int, n: int = 64):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.randn(n, 1, 28, 28, generator=g),
                      torch.randint(0, 10, (n,), generator=g)),
        batch_size=32, shuffle=True)


def _cfg(fixed=N_POISON, **attack):
    a = {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.1},
         "fixed_poison_clients": fixed,
         # Deliberately WRONG on purpose: the fixed set must override both.
         "max_poison_clients": 3, "sample_budget_in_training": True,
         "sample_target_in_training": False}
    a.update(attack)
    return {
        "fl": {"n_clients": N_CLIENTS, "device": "cpu", "lr": 0.05, "local_epochs": 1,
               "benign_retrain_each_round": False, "training_rounds": 5,
               "n_compromisable": 3, "poison_seed": 0, "batch_size": 32,
               "freeze_global_in_phase2": False},
        "data": {"data_dir": "./data/mnist_raw"},
        "attack": a,
        "defense": {"mode": "algorithmic", "algorithms": list(ALGS), "selection": "random"},
        "curriculum": {"enabled": True, "rounds_per_block": 2,
                       "poisoner_counts": [1, 2, 3]},
    }


def _env(cfg=None):
    cfg = cfg or _cfg()
    defense = build_algorithmic_defender(cfg, seed=0)
    curriculum = build_training_curriculum(cfg, algorithms=defense.names)
    env = FLArmsRaceEnv(cfg, [_loader(i) for i in range(N_CLIENTS)], _loader(99, n=128),
                        random.Random(0), defense=defense, curriculum=curriculum)
    from model.mnist_net import MnistNet
    gw = MnistNet().state_dict()
    env.reset(gw, [{k: v.clone() for k, v in gw.items()} for _ in range(N_CLIENTS)], 0.5)
    return env


# --- env ------------------------------------------------------------------

def test_pool_is_the_first_n_clients_and_quota_matches():
    env = _env()
    assert env.fixed_poison_clients == N_POISON
    # It overrides fl.n_compromisable (3) and attack.max_poison_clients (3).
    assert env.n_compromisable == N_POISON and env.budget_cap == N_POISON
    assert env.sample_budget is False
    ctx = env.begin_round()
    assert ctx.pool_ids == list(range(N_POISON))
    assert ctx.budget == N_POISON


def test_quota_is_identical_every_round_despite_sampling_and_curriculum():
    """The two things that used to vary the poisoner count are both overridden."""
    env = _env()
    budgets = {env.begin_round().budget for _ in range(6)}
    assert budgets == {N_POISON}


def test_disabled_restores_sampling():
    for off in (None, 0, False):
        env = _env(_cfg(fixed=off))
        assert env.fixed_poison_clients is None
        assert env.n_compromisable == 3          # back to fl.n_compromisable
        # The curriculum drives the quota again, so it sweeps its configured counts.
        assert {env.begin_round().budget for _ in range(8)} != {N_POISON}


def test_clamped_to_n_clients():
    env = _env(_cfg(fixed=N_CLIENTS + 5))
    assert env.fixed_poison_clients == N_CLIENTS
    assert env.begin_round().pool_ids == list(range(N_CLIENTS))


# --- curriculum -----------------------------------------------------------

def test_curriculum_sweep_collapses_to_the_fixed_count():
    cfg = _cfg()
    defense = build_algorithmic_defender(cfg, seed=0)
    curriculum = build_training_curriculum(cfg, algorithms=defense.names)
    assert curriculum.poisoner_counts == [N_POISON]
    # The defense axis is untouched: one block per algorithm, still swept.
    assert curriculum.algorithms == list(ALGS)


def test_curriculum_still_sweeps_counts_when_not_fixed():
    cfg = _cfg(fixed=None)
    defense = build_algorithmic_defender(cfg, seed=0)
    curriculum = build_training_curriculum(cfg, algorithms=defense.names)
    assert curriculum.poisoner_counts == [1, 2, 3]


# --- attacker agent -------------------------------------------------------

def _sd(scale=1.0):
    return {"net.2.weight": torch.ones(3, 3) * scale, "net.2.bias": torch.ones(3) * scale,
            "net.4.weight": torch.ones(2, 3) * scale, "net.4.bias": torch.ones(2) * scale}


def _pool(n=N_POISON):
    return {cid: _sd(scale=cid + 1) for cid in range(n)}


def test_every_pool_client_is_poisoned_whatever_the_model_emits():
    """A quota equal to the pool size is what actually forces the fixed set."""
    agent = AttackerAgent({"fixed_poison_set": True})
    pool = _pool()
    for text in (
        '{"clients":[{"id":0,"operations":[{"op":"scale","target":"all","factor":2.0}]}]}',
        '{"clients":[{"id":99,"operations":[{"op":"scale","target":"all","factor":2.0}]}]}',
        '{"operations":[{"op":"sign_flip","target":"all","fraction":1.0}]}',
        '{"clients":[{"id":3,"operations":[{"op":"scale","target":"all","factor":3.0}]},'
        '{"id":3,"operations":[{"op":"scale","target":"all","factor":4.0}]}]}',
    ):
        poisoned, chosen, n_malformed = agent.select_and_apply(text, pool, budget=len(pool))
        assert sorted(chosen) == list(range(N_POISON)), text
        assert n_malformed == 0 and len(poisoned) == N_POISON


def test_unusable_output_poisons_nobody_rather_than_faking_the_set():
    """Ground truth stays honest: no plan means no poison, not 10 untouched 'poisoners'."""
    agent = AttackerAgent({"fixed_poison_set": True})
    pool = _pool()
    poisoned, chosen, n_malformed = agent.select_and_apply("not json at all", pool,
                                                           budget=len(pool))
    assert chosen == [] and poisoned == {} and n_malformed == N_POISON


def test_fixed_mode_prompt_lists_the_ids_and_drops_the_selection_rule():
    agent = AttackerAgent({"fixed_poison_set": True})
    system = agent.system_prompt()
    assert "poison_client_ids" in system
    assert "EVERY id in `poison_client_ids`" in system
    assert "SELECT EXACTLY" not in system

    payload = json.loads(agent.build_user_prompt(1, 0.9, _pool(), _sd(), budget=N_POISON))
    assert payload["poison_client_ids"] == list(range(N_POISON))
    assert payload["n_poison_clients"] == N_POISON
    assert "max_poison_clients" not in payload


def test_selection_mode_prompt_is_unchanged():
    agent = AttackerAgent()                      # default = the original behaviour
    assert "EXACTLY `max_poison_clients`" in agent.system_prompt()
    payload = json.loads(agent.build_user_prompt(1, 0.9, _pool(5), _sd(), budget=2))
    assert payload["controllable_client_ids"] == list(range(5))
    assert payload["max_poison_clients"] == 2


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} fixed-poison-set tests passed.")


if __name__ == "__main__":
    _run()
