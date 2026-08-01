"""Locks in the budget-conditioned target ladder (rl/rewards.py::target_for_budget)
and proves the reward path, the schedule's win gate, and the attacker prompt
cannot disagree about a round's target.

Torch-only (needs torch for ``FLArmsRaceEnv``/``MnistNet``, no MNIST download and
no GPU):  python tests/test_target_ladder.py
"""

import copy
import json
import os
import random
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from agents.attacker_agent import AttackerAgent  # noqa: E402
from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from rl.rewards import DEFAULT_TARGET_LADDER, goal_target, target_for_budget  # noqa: E402
from rl.switch import SwitchConfig, attacker_succeeded  # noqa: E402


@dataclass
class V:
    """Minimal DetectionVerdict stand-in (see tests/test_switch.py)."""
    client_id: int
    is_suspicious: bool


def _loader(seed, n=64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _cfg(**overrides) -> dict:
    """Minimal env config: 4-client pool, budget fixed at the cap (no sampling)
    so a test can pin the round's poison budget deterministically. Overrides are
    deep-merged so a test can swap the ladder, the goal, or the budget cap
    without repeating the whole dict."""
    base = {
        "fl": {"n_clients": 4, "device": "cpu", "benign_retrain_each_round": False,
               "training_rounds": 20, "n_compromisable": 3, "lr": 0.05, "local_epochs": 1},
        "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.20},
                   "max_poison_clients": 3, "sample_budget_in_training": False},
    }
    return _deep_merge(base, overrides)


def _make_env(cfg: dict) -> FLArmsRaceEnv:
    n_clients = cfg["fl"]["n_clients"]
    loaders = [_loader(i) for i in range(n_clients)]
    env = FLArmsRaceEnv(cfg, loaders, _loader(999, 128), random.Random(0))
    net = MnistNet()
    gw = {k: v.clone() for k, v in net.state_dict().items()}
    cw = [{k: v.clone() + 0.01 * (i + 1) for k, v in gw.items()} for i in range(n_clients)]
    env.reset(gw, cw, baseline_accuracy=0.10)
    return env


# ---------------------------------------------------------------------------
# End-to-end: budget 3 resolves 0.06 through every real consumer
# ---------------------------------------------------------------------------

def test_budget_three_resolves_one_target_end_to_end():
    """Budget 3, no explicit ladder: begin_round() -> goal_target -> the win gate
    -> the attacker prompt all resolve 0.06. Every consumer is invoked for real
    (prohibition P-03) — none of this compares target_for_budget to itself."""
    cfg = _cfg()
    env = _make_env(cfg)
    ctx = env.begin_round()

    assert ctx.budget == 3
    assert ctx.goal["target_accuracy_drop"] == 0.06
    assert goal_target(ctx.goal) == 0.06

    prompt = AttackerAgent().build_user_prompt(
        ctx.round_num, ctx.global_accuracy, ctx.pool_benign, env.global_weights,
        ctx.budget, goal=ctx.goal)
    recovered = json.loads(prompt)["attack_goal"]["target_accuracy_drop"]
    assert recovered == 0.06

    switch_cfg = SwitchConfig(win_fraction=0.6)
    # Bar = 0.6 * 0.06 = 0.036: a drop of 0.036 with an evading poisoner wins;
    # one representable step below does not.
    assert attacker_succeeded(0.036, [V(0, False)], [0], switch_cfg, ctx.goal)
    assert not attacker_succeeded(0.035, [V(0, False)], [0], switch_cfg, ctx.goal)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} target-ladder tests passed.")


if __name__ == "__main__":
    _run()
