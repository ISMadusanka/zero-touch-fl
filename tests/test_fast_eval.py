"""Tests for the cheap-eval path: FedServer.preload_test_set / evaluate(n_samples)
and the FLArmsRaceEnv wiring (preload on reset, subsample for rollout rewards,
full test set on commit).

CPU only — uses a tiny synthetic MNIST-shaped test set and the real MnistNet.
No GPU, no LLM.

Run on any box with torch:  python tests/test_fast_eval.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from server.fed_server import FedServer  # noqa: E402

torch.manual_seed(0)
_N = 64
_X = torch.randn(_N, 1, 28, 28)
_Y = torch.randint(0, 10, (_N,))


def _loader(batch_size=16):
    # shuffle=False mirrors the real test loader → the first-N subsample is deterministic.
    return DataLoader(TensorDataset(_X, _Y), batch_size=batch_size, shuffle=False)


def _manual_acc(model, x, y):
    model.eval()
    with torch.no_grad():
        return model(x).argmax(dim=1).eq(y).float().mean().item()


def test_preloaded_matches_loader_iteration():
    srv = FedServer(device="cpu")
    acc_legacy = srv.evaluate(_loader())          # legacy path (explicit loader)
    srv.preload_test_set(_loader())
    acc_pre = srv.evaluate()                        # fast path (preloaded, full set)
    assert abs(acc_legacy - acc_pre) < 1e-9, (acc_legacy, acc_pre)
    assert abs(acc_pre - _manual_acc(srv.model, _X, _Y)) < 1e-6


def test_subsample_uses_first_n_deterministically():
    srv = FedServer(device="cpu")
    srv.preload_test_set(_loader())
    k = 10
    expected = _manual_acc(srv.model, _X[:k], _Y[:k])
    assert abs(srv.evaluate(n_samples=k) - expected) < 1e-6
    # Repeated calls are identical (deterministic subset).
    assert srv.evaluate(n_samples=k) == srv.evaluate(n_samples=k)


def test_n_samples_ge_total_is_full_set():
    srv = FedServer(device="cpu")
    srv.preload_test_set(_loader())
    full = srv.evaluate()
    assert abs(srv.evaluate(n_samples=10 * _N) - full) < 1e-9
    assert abs(srv.evaluate(n_samples=None) - full) < 1e-9


def test_evaluate_without_preload_or_loader_raises():
    srv = FedServer(device="cpu")
    try:
        srv.evaluate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no loader and no preload")


def _env_config(reward_eval_samples=1500):
    return {
        "fl": {
            "n_clients": 3, "device": "cpu", "training_rounds": 0, "lr": 0.01,
            "local_epochs": 1, "benign_retrain_each_round": False, "n_compromisable": 2,
        },
        "attack": {"max_poison_clients": 2, "sample_budget_in_training": False},
        "rl": {"reward_eval_samples": reward_eval_samples},
    }


def test_env_preloads_on_reset_and_reads_reward_samples():
    import random
    env = FLArmsRaceEnv(_env_config(reward_eval_samples=20), None, _loader(), random.Random(0))
    assert env.reward_eval_samples == 20
    gw = MnistNet().state_dict()
    env.reset(gw, [copy.deepcopy(gw) for _ in range(3)], baseline_accuracy=0.5)
    # reset() should have preloaded the test set into the server.
    assert env.server._eval_x is not None
    assert env.server._eval_x.shape[0] == _N


def test_env_eval_state_subsample_vs_full():
    import random
    env = FLArmsRaceEnv(_env_config(reward_eval_samples=8), None, _loader(), random.Random(0))
    gw = MnistNet().state_dict()
    env.reset(gw, [copy.deepcopy(gw) for _ in range(3)], baseline_accuracy=0.5)
    # A candidate state (reuse the global weights) — subsample vs full both valid accuracies.
    sub = env._eval_state(env.server.get_global_weights(), n_samples=env.reward_eval_samples)
    full = env._eval_state(env.server.get_global_weights(), n_samples=None)
    for a in (sub, full):
        assert 0.0 <= a <= 1.0
    # _eval_state must not mutate the committed global model (it restores the backup).
    assert abs(env.server.evaluate() - full) < 1e-9


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} fast-eval tests passed.")


if __name__ == "__main__":
    _run()
