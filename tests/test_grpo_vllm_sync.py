"""grpo_step <-> vLLM weight-sync coupling test.

The vLLM backend only samples from fresh LoRA weights because grpo_step flags the
learner adapter dirty AFTER it steps the optimizer. This test pins that contract
with a tiny fake policy (one real torch Parameter so the backward/step actually
run) and a fake turn:

  * a group with reward spread -> the optimizer steps -> mark_adapter_dirty(adapter)
    is called exactly once with the learner's name;
  * a degenerate zero-advantage group with skip_zero_advantage -> NO step ->
    mark_adapter_dirty is NOT called.

Needs torch (CPU is fine). Run: python tests/test_grpo_vllm_sync.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from rl.grpo import grpo_step  # noqa: E402


class _FakePolicy:
    """Minimal LLMPolicy-shaped stand-in with a single trainable parameter."""
    def __init__(self):
        self.w = torch.nn.Parameter(torch.zeros(()))
        self.dirty_calls = []

    def generate(self, adapter, system, user, n=1, temperature=0.0, max_new_tokens=1024):
        return [f"c{i}" for i in range(n)]

    def policy_token_logprobs(self, adapter, system, user, completion):
        # Differentiable, length-3 log-probs that depend on w (so backward reaches it).
        return self.w + torch.zeros(3)

    def reference_token_logprobs(self, system, user, completion):
        return torch.zeros(3)

    def adapter_parameters(self, adapter):
        return [self.w]

    def mark_adapter_dirty(self, name):
        self.dirty_calls.append(name)


class _SpreadTurn:
    """Distinct reward per completion -> non-zero group advantage -> a real step."""
    def messages(self):
        return "sys", "user"

    def reward(self, completion):
        return {"c0": 0.0, "c1": 0.25, "c2": 0.75, "c3": 1.0}[completion]


class _FlatTurn:
    """Equal reward for every completion -> zero advantage -> skipped step."""
    def messages(self):
        return "sys", "user"

    def reward(self, completion):
        return 0.5


def test_step_marks_learner_dirty():
    policy = _FakePolicy()
    opt = torch.optim.SGD([policy.w], lr=0.1)
    stats = grpo_step(
        policy, "attacker", opt, _SpreadTurn(),
        G=4, kl_beta=0.02, temperature=1.0, max_new_tokens=16,
        skip_zero_advantage=True, resample_on_zero_advantage=False,
    )
    assert stats["stepped"] is True
    assert policy.dirty_calls == ["attacker"], policy.dirty_calls
    print("ok: optimizer step marks learner adapter dirty")


def test_skipped_step_does_not_mark_dirty():
    policy = _FakePolicy()
    opt = torch.optim.SGD([policy.w], lr=0.1)
    stats = grpo_step(
        policy, "defender", opt, _FlatTurn(),
        G=4, kl_beta=0.02, temperature=1.0, max_new_tokens=16,
        skip_zero_advantage=True, resample_on_zero_advantage=False,
    )
    assert stats["stepped"] is False
    assert policy.dirty_calls == [], policy.dirty_calls
    print("ok: skipped (zero-advantage) step leaves vLLM copy untouched")


if __name__ == "__main__":
    test_step_marks_learner_dirty()
    test_skipped_step_does_not_mark_dirty()
    print("\nAll grpo<->vLLM sync tests passed.")
