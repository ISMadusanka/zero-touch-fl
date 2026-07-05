"""Tests for the between-phase benign FL round (rl.env.run_benign_fl_round).

Uses tiny synthetic client loaders (random tensors shaped like MNIST), so no
MNIST download and no GPU are needed:  python tests/test_fl_interlude.py

Covers the contract the arms-race schedule relies on: a benign FL round advances
BOTH the global model and the per-client benign references, and hands those
refreshed weights to the next begin_round (with benign_retrain_each_round=False)
so the attacker/defender/aggregator consume them.
"""

import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402

N_CLIENTS = 4
N_COMPROMISABLE = 2
TRAINING_ROUNDS = 20


def _loader(seed: int, n: int = 64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)


def _cfg():
    return {
        "fl": {
            "n_clients": N_CLIENTS,
            "device": "cpu",
            "benign_retrain_each_round": False,  # honest updates replay client_weights
            "training_rounds": TRAINING_ROUNDS,
            "n_compromisable": N_COMPROMISABLE,
            "lr": 0.05,
            "local_epochs": 1,
        },
        "attack": {
            "goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.20},
            "max_poison_clients": 2,
            "sample_budget_in_training": True,
        },
    }


def _make_env():
    """Build an env reset from a fresh (untrained) Phase-1-style state."""
    client_loaders = [_loader(i) for i in range(N_CLIENTS)]
    test_loader = _loader(999, n=128)
    env = FLArmsRaceEnv(_cfg(), client_loaders, test_loader, random.Random(0))

    net = MnistNet()
    global_weights = {k: v.clone() for k, v in net.state_dict().items()}
    # Distinct per-client "frozen Phase-1" weights so we can detect a refresh.
    client_weights = []
    for i in range(N_CLIENTS):
        w = {k: v.clone() + 0.01 * (i + 1) for k, v in global_weights.items()}
        client_weights.append(w)
    env.reset(global_weights, client_weights, baseline_accuracy=0.1)
    return env, global_weights, client_weights, test_loader


def _differs(a: dict, b: dict) -> bool:
    """True if any tensor in the two state dicts differs."""
    return any(not torch.allclose(a[k], b[k]) for k in a)


def test_returns_summary_and_advances_round_index():
    env, _, _, _ = _make_env()
    r0 = env.round_index
    info = env.run_benign_fl_round()
    assert info is not None
    assert env.round_index == r0 + 1
    assert info["round_num"] == TRAINING_ROUNDS + env.round_index
    assert info["n_clients"] == N_CLIENTS
    assert len(info["updates"]) == N_CLIENTS
    assert info["post_accuracy"] == env.current_accuracy


def test_refreshes_client_weights_and_global():
    env, global0, client0, _ = _make_env()
    global_before = env.global_weights
    info = env.run_benign_fl_round()
    # Per-client benign references were replaced by freshly trained weights.
    for cid in range(N_CLIENTS):
        assert _differs(client0[cid], env.client_weights[cid]), (
            f"client {cid} weights not refreshed by the FL round"
        )
    # The global model advanced too (FedAvg of the trained clients).
    assert _differs(global_before, env.global_weights), "global model not advanced"


def test_next_begin_round_consumes_refreshed_weights():
    """With benign_retrain_each_round=False the honest updates of the NEXT round
    must be exactly the FL round's refreshed client weights."""
    env, _, _, _ = _make_env()
    info = env.run_benign_fl_round()
    refreshed = [copy.deepcopy(w) for w in env.client_weights]
    ctx = env.begin_round()
    # Round numbers are sequential across the interlude (no collision/reuse).
    assert ctx.round_num == info["round_num"] + 1
    for cid in range(N_CLIENTS):
        hu = env.honest_updates[cid].weights
        for k in hu:
            assert torch.allclose(hu[k], refreshed[cid][k]), (
                f"begin_round did not replay refreshed weights for client {cid}"
            )
    # The attacker's controllable pool now points at the refreshed weights.
    for cid in ctx.pool_ids:
        for k in ctx.pool_benign[cid]:
            assert torch.allclose(ctx.pool_benign[cid][k], refreshed[cid][k])


def test_sequential_round_numbers_across_multiple_interludes():
    env, _, _, _ = _make_env()
    nums = []
    nums.append(env.begin_round().round_num)          # 21
    nums.append(env.run_benign_fl_round()["round_num"])  # 22
    nums.append(env.begin_round().round_num)          # 23
    nums.append(env.run_benign_fl_round()["round_num"])  # 24
    assert nums == list(range(TRAINING_ROUNDS + 1, TRAINING_ROUNDS + 1 + len(nums)))
    assert len(set(nums)) == len(nums)  # all unique — no round_data collision


def test_noop_without_client_loaders():
    net = MnistNet()
    gw = {k: v.clone() for k, v in net.state_dict().items()}
    cw = [copy.deepcopy(gw) for _ in range(N_CLIENTS)]
    env = FLArmsRaceEnv(_cfg(), None, _loader(1), random.Random(0))
    env.reset(gw, cw, baseline_accuracy=0.1)
    assert env.run_benign_fl_round() is None
    assert env.round_index == 0  # no round consumed


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} FL-interlude tests passed.")


if __name__ == "__main__":
    _run()
