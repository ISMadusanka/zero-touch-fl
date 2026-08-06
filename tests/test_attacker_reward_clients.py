"""Tests that client count is not penalized and that the collaboration (diversity)
bonus works in the attacker reward. Needs torch (perturbation_diversity).

Run on any box with torch:  python tests/test_attacker_reward_clients.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core.types import DetectionVerdict  # noqa: E402
from rl.rewards import attacker_reward, perturbation_diversity  # noqa: E402

GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}


def _benign_verdicts(ids):
    # Confidently-benign verdicts -> full stealth for each poisoned client.
    return [DetectionVerdict(cid, False, 1.0, "") for cid in ids]


def test_more_clients_do_not_lower_reward():
    # Same drop and mean stealth; there is no fewer-clients/client-cost term.
    r1 = attacker_reward(0.9, 0.8, GOAL, [0], _benign_verdicts([0]), 0,
                         alpha=1.0, beta=0.0, gamma=0.0, zeta=0.0)
    r2 = attacker_reward(0.9, 0.8, GOAL, [0, 1], _benign_verdicts([0, 1]), 0,
                         alpha=1.0, beta=0.0, gamma=0.0, zeta=0.0)
    assert abs(r2 - r1) < 1e-9


def test_perturbation_diversity_orthogonal_vs_identical():
    refs = {0: {"w": torch.zeros(4)}, 1: {"w": torch.zeros(4)}}
    orthogonal = {0: {"w": torch.tensor([1.0, 0, 0, 0])},
                  1: {"w": torch.tensor([0.0, 1, 0, 0])}}
    identical = {0: {"w": torch.tensor([1.0, 0, 0, 0])},
                 1: {"w": torch.tensor([1.0, 0, 0, 0])}}
    assert abs(perturbation_diversity(orthogonal, refs) - 1.0) < 1e-6
    assert abs(perturbation_diversity(identical, refs) - 0.0) < 1e-6
    assert perturbation_diversity({0: {"w": torch.ones(4)}}, {0: {"w": torch.zeros(4)}}) == 0.0


def test_collab_bonus_rewards_diverse_multiclient():
    v = _benign_verdicts([0, 1])
    diverse = attacker_reward(0.9, 0.8, GOAL, [0, 1], v, 0,
                              alpha=1.0, beta=0.0, gamma=0.0, zeta=1.0,
                              diversity=1.0)
    redundant = attacker_reward(0.9, 0.8, GOAL, [0, 1], v, 0,
                                alpha=1.0, beta=0.0, gamma=0.0, zeta=1.0,
                                diversity=0.0)
    assert abs((diverse - redundant) - 1.0) < 1e-6  # zeta=1.0 * (1.0 - 0.0)


def test_collab_bonus_ignored_for_single_client():
    r = attacker_reward(0.9, 0.8, GOAL, [0], _benign_verdicts([0]), 0,
                        alpha=1.0, beta=0.0, gamma=0.0, zeta=1.0, diversity=1.0)
    base = attacker_reward(0.9, 0.8, GOAL, [0], _benign_verdicts([0]), 0,
                           alpha=1.0, beta=0.0, gamma=0.0, zeta=0.0)
    assert abs(r - base) < 1e-9                      # no collaboration with one client


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} attacker-reward (client) tests passed.")


if __name__ == "__main__":
    _run()
