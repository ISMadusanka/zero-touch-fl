"""Tests for AttackerTurn.reward_batch — the batched frozen-defender scoring.

Verifies that scoring the G rollouts with ONE batched defender generation
(generate_many) yields byte-for-byte the same rewards as the old path of one
defender generate per rollout, and that it really issues a single batched call.

Uses fakes for the env and the defender generator (deterministic on the prompt),
the REAL AttackerAgent/DefenderAgent (prompt build + parse + attack-op apply), and
the real reward math. CPU only, no LLM/GPU.

Run on any box with torch:  python tests/test_batched_reward.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from agents.attacker_agent import AttackerAgent  # noqa: E402
from agents.defender_agent import DefenderAgent  # noqa: E402
from core.types import ModelUpdate  # noqa: E402
from rl.rewards import attacker_reward, perturbation_diversity  # noqa: E402
from rl.turns import AttackerTurn  # noqa: E402

REWARD_CFG = {"alpha": 1.0, "beta": 0.5, "gamma": 1.0, "delta": 0.3, "zeta": 0.2}


def _sd(scale=1.0):
    return {
        "net.2.weight": torch.ones(4, 4) * scale,
        "net.2.bias": torch.ones(4) * scale,
        "net.4.weight": torch.ones(2, 4) * scale,
        "net.4.bias": torch.ones(2) * scale,
    }


class FakeEnv:
    """Minimal stand-in for FLArmsRaceEnv exposing what AttackerTurn needs."""

    def __init__(self):
        self.n_clients = 3
        self.n_compromisable = 3
        self.current_accuracy = 0.9
        self.round_index = 1
        self.training_rounds = 0
        self.goal = {"type": "untargeted_degrade", "target_accuracy_drop": 0.2}
        self._global = _sd(0.5)                       # non-zero → delta stats well-defined
        self.pool_benign = {0: _sd(1.0), 1: _sd(2.0), 2: _sd(3.0)}
        self.round_budget = 3

    @property
    def global_weights(self):
        return self._global

    def build_updates(self, poisoned):
        ups = []
        for cid in range(self.n_clients):
            if cid in poisoned:
                ups.append(ModelUpdate(cid, poisoned[cid], {"poisoned": True}))
            else:
                ups.append(ModelUpdate(cid, self.pool_benign[cid], {"poisoned": False}))
        return ups

    def features(self, updates):
        # JSON-serializable per-client features (shape irrelevant to this test).
        return {u.client_id: {"whole": {"l2_norm": float(u.client_id)}} for u in updates}

    def evaluate_updates(self, updates, verdicts):
        # Deterministic, depends on the verdicts (which depend on the prompt) so
        # different rollouts get genuinely different post-accuracies.
        n_flagged = sum(1 for v in verdicts if v.is_suspicious)
        return 0.9 - 0.05 * n_flagged


class FakeDefenderGen:
    """Deterministic on the prompt: generate() and generate_many() return the SAME
    verdict text for the same user prompt, so batched == sequential."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.gen_calls = 0
        self.many_calls = 0
        self.last_prompt_count = 0

    @staticmethod
    def _verdict_text(user):
        cids = json.loads(user)["client_ids"]
        return json.dumps({"clients": [
            {"client_id": c, "is_suspicious": (c % 2 == 0), "confidence": 0.8} for c in cids
        ]})

    def generate(self, system, user, n=1, temperature=0.0):
        self.gen_calls += 1
        return [self._verdict_text(user)] * n

    def generate_many(self, prompts, temperature=0.0):
        self.many_calls += 1
        self.last_prompt_count = len(prompts)
        return [self._verdict_text(u) for (_s, u) in prompts]


COMPLETIONS = [
    '{"clients":[{"id":0,"operations":[{"op":"scale","target":"all","factor":2.0}]}]}',
    ('{"clients":['
     '{"id":1,"operations":[{"op":"sign_flip","target":"all"}]},'
     '{"id":2,"operations":[{"op":"scale","target":"net.4","factor":3.0}]}]}'),
    'total garbage, no json at all',
]


def _make_turn(gen):
    return AttackerTurn(
        FakeEnv(), AttackerAgent(), DefenderAgent(), gen,
        reward_cfg=REWARD_CFG, opponent_temperature=0.0, scoring_opponent_temperature=0.0,
    )


def _reference_rewards_sequential(turn, gen, comps):
    """Recompute rewards the OLD way: one defender generate() per rollout."""
    d_sys = turn.defender_agent.system_prompt()
    out = []
    for text in comps:
        poisoned, chosen_ids, n_malformed = turn.attacker_agent.select_and_apply(
            text, turn.pool_references, turn.budget)
        updates = turn.env.build_updates(poisoned)
        d_user = turn.defender_agent.build_user_prompt(turn.env.features(updates))
        d_text = gen.generate(d_sys, d_user, n=1, temperature=turn.scoring_opp_temp)[0]
        client_ids = [u.client_id for u in updates]
        verdicts = turn.defender_agent.parse(d_text, client_ids)
        post_acc = turn.env.evaluate_updates(updates, verdicts)
        diversity = perturbation_diversity(
            poisoned, {cid: turn.pool_references[cid] for cid in chosen_ids})
        out.append(attacker_reward(
            turn.prev_accuracy, post_acc, turn.env.goal, chosen_ids, verdicts, n_malformed,
            alpha=REWARD_CFG["alpha"], beta=REWARD_CFG["beta"], gamma=REWARD_CFG["gamma"],
            delta=REWARD_CFG["delta"], zeta=REWARD_CFG["zeta"],
            pool_size=turn.pool_size, diversity=diversity))
    return out


def test_reward_batch_matches_sequential_single_generate():
    gen = FakeDefenderGen()
    turn = _make_turn(gen)

    gen.reset()
    ref = _reference_rewards_sequential(turn, gen, COMPLETIONS)
    assert gen.gen_calls == len(COMPLETIONS) and gen.many_calls == 0

    gen.reset()
    batched = turn.reward_batch(COMPLETIONS)
    # ONE batched defender call covering all rollouts (no per-rollout generate()).
    assert gen.many_calls == 1
    assert gen.last_prompt_count == len(COMPLETIONS)
    assert gen.gen_calls == 0

    assert len(batched) == len(COMPLETIONS)
    for a, b in zip(batched, ref):
        assert abs(a - b) < 1e-9, (a, b)


def test_single_reward_delegates_to_batch():
    gen = FakeDefenderGen()
    turn = _make_turn(gen)
    ref = _reference_rewards_sequential(turn, gen, COMPLETIONS)

    gen.reset()
    r0 = turn.reward(COMPLETIONS[0])
    assert gen.many_calls == 1 and gen.last_prompt_count == 1
    assert abs(r0 - ref[0]) < 1e-9


def test_batch_preserves_per_rollout_variation():
    # The whole point of scoring at nonzero opponent temp is a spread of rewards;
    # here the spread comes from different plans → different verdicts/post-acc.
    gen = FakeDefenderGen()
    turn = _make_turn(gen)
    rewards = turn.reward_batch(COMPLETIONS)
    assert len({round(r, 6) for r in rewards}) > 1, rewards


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} batched-reward tests passed.")


if __name__ == "__main__":
    _run()
