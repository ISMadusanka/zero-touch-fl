"""Tests for the per-round CLEAN counterfactual (rl.env.clean_reference_accuracy)
and the round-budget / league-cap plumbing in rl.schedule.

Synthetic tensors shaped like MNIST — no download, no GPU:
    python tests/test_clean_reference.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from core.types import DetectionVerdict  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from rl.schedule import League, resolve_round_budget  # noqa: E402

N_CLIENTS = 6
TRAINING_ROUNDS = 10


def _loader(seed: int, n: int = 64):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.randn(n, 1, 28, 28, generator=g),
                      torch.randint(0, 10, (n,), generator=g)),
        batch_size=32, shuffle=True)


def _env(benign_retrain=False):
    cfg = {
        # freeze_global_in_phase2: False — the counterfactual's interaction with a
        # MOVING global (it must be invalidated when the shared model advances) is
        # what several of these tests are about. Its behaviour on frozen simulated
        # rounds is covered by tests/test_frozen_rounds.py.
        "fl": {"n_clients": N_CLIENTS, "device": "cpu",
               "benign_retrain_each_round": benign_retrain,
               "freeze_global_in_phase2": False,
               "training_rounds": TRAINING_ROUNDS, "n_compromisable": 2,
               "lr": 0.05, "local_epochs": 1},
        "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.20},
                   "max_poison_clients": 2, "sample_budget_in_training": False},
    }
    loaders = [_loader(i) for i in range(N_CLIENTS)]
    env = FLArmsRaceEnv(cfg, loaders, _loader(99, n=128), random.Random(0))
    from model.mnist_net import MnistNet
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    cw = [{k: v + torch.randn_like(v) * 0.01 for k, v in gw.items()}
          for _ in range(N_CLIENTS)]
    env.reset(gw, cw, 0.5)
    return env


def _wreck(sd):
    return {k: v * -50.0 for k, v in sd.items()}


def _use_deterministic_eval(env):
    """Replace the test-set evaluation with a deterministic function of the global
    weights (accuracy falls as the model blows up).

    The synthetic loaders here carry random labels, so a real evaluation sits at
    chance no matter what the aggregate looks like — useless for asserting that a
    poisoned round scores lower than a clean one. This keeps the test about the
    env's REFERENCE BOOKKEEPING, which is what is under test, and makes it exact.
    """
    def evaluate(_loader):
        flat = torch.cat([v.flatten().float()
                          for v in env.server.get_global_weights().values()])
        return 1.0 / (1.0 + float(flat.norm()))
    env.server.evaluate = evaluate


def test_context_exposes_the_clean_counterfactual():
    env = _env()
    ctx = env.begin_round()
    assert ctx.clean_accuracy is not None
    # It equals the accuracy of the all-honest aggregate.
    updates = env.build_updates({})
    clean = [DetectionVerdict(u.client_id, False, 0.0, "") for u in updates]
    assert abs(ctx.clean_accuracy - env.evaluate_updates(updates, clean)) < 1e-12


def test_clean_reference_is_cached_within_a_round():
    env = _env()
    env.begin_round()
    assert env.clean_reference_accuracy() is env.clean_reference_accuracy()


def test_repeated_identical_attack_gives_a_stable_drop():
    """The regression that motivated the change: with frozen benign replay the
    env is memoryless, so an identical attack must yield an identical drop every
    round. Measured against current_accuracy it collapsed to 0 after round 1."""
    env = _env(benign_retrain=False)
    _use_deterministic_eval(env)
    env.current_accuracy = env.server.evaluate(None)
    drops, legacy_drops = [], []
    for _ in range(3):
        ctx = env.begin_round()
        poisoned = {0: _wreck(ctx.pool_benign[0])}
        updates = env.build_updates(poisoned)
        verdicts = [DetectionVerdict(u.client_id, False, 0.9, "") for u in updates]
        env.set_committed_poison([0])
        post = env.commit(updates, verdicts)
        drops.append(round(ctx.clean_accuracy - post, 10))
        legacy_drops.append(round(ctx.global_accuracy - post, 10))
    assert len(set(drops)) == 1, f"clean-reference drop drifted: {drops}"
    assert drops[0] > 0.0, "the wrecking attack should register damage"
    # The old reference decayed to ~0 once the model was already wrecked.
    assert legacy_drops[-1] < drops[-1]


def test_clean_reference_refreshes_after_a_benign_fl_round():
    env = _env()
    env.begin_round()
    before = env.clean_reference_accuracy()
    env.run_benign_fl_round()
    assert env._clean_ref_acc is None          # invalidated
    assert not env.clean_reference_measured    # and so is the "was it measured" flag
    after = env.clean_reference_accuracy()
    assert isinstance(after, float) and after == env.clean_reference_accuracy()
    assert before is not None


# --- an unmeasurable counterfactual must be reported, not faked ---------------

class _RefusingDefense:
    """A defense that declines to aggregate (every client removed).

    FLTrust does this whenever every trust score is 0, DeFL whenever a CLP removes
    everyone. It happened on roughly a quarter of the rounds in a recorded run.
    """

    def __init__(self):
        self.name = "refusing"

    def run(self, updates, global_weights, *, commit=False, algorithm=None):
        from server.algo_defender import DefenseOutcome
        verdicts = [DetectionVerdict(u.client_id, True, 1.0, "rejected", p_malicious=1.0)
                    for u in updates]
        return DefenseOutcome("refusing", verdicts, None, {})

    def select(self, name):
        return name

    def choose(self):
        return "refusing"


def test_unmeasurable_clean_reference_is_flagged_not_silently_faked():
    """The bug: with no clean aggregate, ``clean_reference_accuracy`` returned
    ``current_accuracy`` — the PREVIOUS round's post-attack accuracy, which is exactly
    what this method exists not to be. When the poisoned round also produced no
    aggregate the post accuracy was that same number, so ``drop`` was identically
    +0.0000 by construction and looked like a clean measurement of "the attack achieved
    nothing"."""
    env = _env()
    env.defense = _RefusingDefense()
    ctx = env.begin_round()

    # The value is still usable as a placeholder...
    assert ctx.clean_accuracy == env.current_accuracy
    # ...but it is explicitly marked as NOT measured, on both the env and the context.
    assert env.clean_reference_measured is False
    assert ctx.clean_measured is False

    # And the pathology it produced is reproducible: a round that also fails to
    # aggregate reports a drop of exactly zero.
    updates = env.build_updates({})
    assert env.commit_state(None) == env.current_accuracy
    assert ctx.clean_accuracy - env.current_accuracy == 0.0
    assert len(updates) == N_CLIENTS


def test_a_measurable_round_reports_measured():
    """The flag must not be pessimistic: an ordinary round is measured."""
    env = _env()
    ctx = env.begin_round()
    assert env.clean_reference_measured is True
    assert ctx.clean_measured is True


def test_unmeasured_rounds_skip_the_policy_update():
    """``grpo_step(skip_update=True)`` still scores the group (the environment needs a
    rollout to advance) but applies no gradient, so a structurally-zero damage term
    cannot move the policy."""
    from rl.grpo import grpo_step

    class _Policy:
        def generate(self, adapter, system, user, n, temperature, max_new_tokens):
            # Deliberately DIFFERENT lengths so the rewards below separate: the point
            # of this test is that only the caller's veto stops the step, not degeneracy.
            return ["plan" + "!" * i for i in range(n)]

        def policy_token_logprobs(self, *a, **k):
            raise AssertionError("skip_update must not run the log-prob pass")

        def adapter_parameters(self, adapter):
            raise AssertionError("skip_update must not touch the optimizer")

    class _Turn:
        def messages(self):
            return "sys", "user"

        def reward(self, text):
            return 0.1 * len(text)

    class _Optimizer:
        def zero_grad(self):
            raise AssertionError("skip_update must not touch the optimizer")

        def step(self):
            raise AssertionError("skip_update must not touch the optimizer")

    stats = grpo_step(_Policy(), "attacker", _Optimizer(), _Turn(), G=4,
                      skip_update=True, skip_reason="test")
    assert stats["stepped"] is False
    assert stats["skipped_by_caller"] is True
    assert stats["reward_spread"] > 0.0        # the group WAS separated...
    assert stats["zero_advantage_fraction"] == 0.0   # ...and not degenerate...
    assert len(stats["completions"]) == 4      # ...and the rollouts are available
    assert stats["loss"] == 0.0                # ...but nothing was optimized


def test_league_is_a_bounded_ring_buffer():
    class _FakePolicy:
        adapters = ("attacker", "defender")

        def __init__(self):
            self.n = 0

        def get_adapter_state(self, name):
            self.n += 1
            return {"w": self.n}

    league = League(random.Random(0), max_snapshots=3)
    policy = _FakePolicy()
    for _ in range(10):
        league.snapshot(policy, ["attacker"])
    assert len(league.snapshots["attacker"]) == 3
    # Oldest evicted: only the three most recent states survive.
    assert [s["w"] for s in league.snapshots["attacker"]] == [8, 9, 10]
    assert league.has("attacker") and not league.has("defender")
    assert league.sample("attacker") in league.snapshots["attacker"]


def test_league_cap_is_at_least_one():
    league = League(random.Random(0), max_snapshots=0)
    assert league.max_snapshots == 1


def test_round_budget_defaults_to_the_config():
    assert resolve_round_budget(2_000_000) == 2_000_000
    assert resolve_round_budget(2_000_000, start_round=17) == 2_000_000


def test_rounds_flag_overrides_the_config_budget():
    """--rounds N used to be computed in main.py and then dropped, so training
    silently ran the whole fl.simulation_rounds budget."""
    assert resolve_round_budget(2_000_000, total_rounds=8) == 8


def test_debug_cap_is_relative_to_rounds_already_done():
    # Fresh run: 3 rounds.
    assert resolve_round_budget(2_000_000, max_new_rounds=3) == 3
    # Resumed at round 500: 3 MORE rounds, not "exit, 500 > 3".
    assert resolve_round_budget(2_000_000, max_new_rounds=3, start_round=500) == 503


def test_explicit_rounds_still_wins_over_the_config():
    assert resolve_round_budget(2_000_000, total_rounds=10, max_new_rounds=3) == 3
    assert resolve_round_budget(50, total_rounds=10, start_round=0) == 10


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} clean-reference / league tests passed.")


if __name__ == "__main__":
    _run()
