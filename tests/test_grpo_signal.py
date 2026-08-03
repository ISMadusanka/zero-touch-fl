"""Regression tests for the GRPO learning signal (rl/grpo.py, rl/rewards.py).

These lock in five fixes, each of which was SILENT — the run trained happily and
produced plausible logs while the gradient pointed somewhere other than intended.

1. ``_soft_malicious_prob`` reads a calibrated ``p_malicious`` when the detector
   supplies one. The algorithmic defenses report a suspicion score, not certainty
   in their verdict, so reconstructing P(malicious) from ``(is_suspicious,
   confidence)`` ran BACKWARDS over every un-flagged client — the attacker's
   stealth reward paid it for creeping up to the detection boundary.
2. FLTrust computes its trusted reference ``g0`` once per global model. It is the
   output of SGD over a shuffled root loader, so re-running it per call gave every
   scored rollout in a group a different defense, and GRPO fit that noise.
3. ``group_advantages`` declares a group degenerate by absolute reward SPREAD, not
   by ``std < 1e-6``, and floors the z-score denominator. Accuracy is measured on a
   finite test set, so identical rollouts differ by ~2e-3 of reward; that used to
   be amplified to full-magnitude advantages.
4. The log-prob pass is told the temperature the rollouts were SAMPLED at,
   including the hotter zero-advantage re-roll.
5. The log-prob pass scores the EXACT sampled token ids, not a re-tokenization of
   the decoded text (a BPE round-trip is not a guaranteed inverse, and any mismatch
   voids the ``ratio == 1`` identity the single-iteration loss is built on).

Needs torch (for the grpo_step stubs), but no GPU and no LLM:
    python tests/test_grpo_signal.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from benchmark.defenses.base import rank_normalized_scores  # noqa: E402
from core.types import DetectionVerdict  # noqa: E402
from rl.grpo import grpo_step  # noqa: E402
from rl.rewards import (  # noqa: E402
    DEFAULT_MIN_REWARD_SPREAD, _soft_malicious_prob, attacker_reward,
    group_advantages,
)

GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}


# --- 1. the soft evasion signal points the right way -------------------------

def test_p_malicious_overrides_the_confidence_reconstruction():
    """A calibrated p_malicious wins over (is_suspicious, confidence)."""
    v = DetectionVerdict(0, False, 0.99, "barely trusted", p_malicious=0.97)
    assert abs(_soft_malicious_prob(v) - 0.97) < 1e-9
    # ...and is clipped into [0, 1] rather than trusted blindly.
    assert _soft_malicious_prob(DetectionVerdict(0, True, 0.0, "x", p_malicious=7.5)) == 1.0


def test_confidence_reconstruction_still_used_for_the_llm_defender():
    """The LLM defender reports certainty in its own verdict and sets no
    p_malicious, so the (is_suspicious, confidence) mapping must still apply."""
    assert _soft_malicious_prob(DetectionVerdict(0, True, 1.0, "sure it's bad")) == 1.0
    assert _soft_malicious_prob(DetectionVerdict(0, False, 1.0, "sure it's fine")) == 0.0
    assert _soft_malicious_prob(DetectionVerdict(0, False, 0.0, "no idea")) == 0.5


def test_stealth_increases_as_a_surviving_client_looks_more_honest():
    """THE bug this suite exists for. Two un-flagged poisoned clients: one the
    defense nearly dropped, one it comfortably kept. Stealth must reward the
    comfortable one. Under the old confidence reconstruction the ordering was
    inverted, so the attacker was trained to sit ON the detection boundary."""
    nearly_caught = [DetectionVerdict(0, False, 0.98, "trust=0.01", p_malicious=0.99)]
    comfortable = [DetectionVerdict(0, False, 0.20, "trust=0.60", p_malicious=0.40)]
    r_nearly = attacker_reward(0.9, 0.7, GOAL, [0], nearly_caught, 0, pool_size=5)
    r_comfy = attacker_reward(0.9, 0.7, GOAL, [0], comfortable, 0, pool_size=5)
    assert r_comfy > r_nearly, (r_comfy, r_nearly)

    # The inverted mapping the fix removed, shown explicitly: it ranked the
    # nearly-caught client as the STEALTHIER of the two.
    def old_soft_p(v):
        c = max(0.0, min(1.0, v.confidence))
        return 0.5 + 0.5 * c if v.is_suspicious else 0.5 - 0.5 * c
    assert old_soft_p(nearly_caught[0]) < old_soft_p(comfortable[0])


def test_unbounded_scores_are_rank_normalized_not_clipped():
    """Multi-Krum/DnC scores are unbounded sums of squares (and +inf for a
    non-finite client). Clipping them to [0, 1] collapsed stealth to a binary;
    ranking keeps it continuous inside the surviving set."""
    p = rank_normalized_scores([0.1, 5.0, 5.0, 1e9, float("inf")])
    assert p == [0.0, 0.25, 0.25, 0.75, 1.0]          # bounded, ties equal, inf worst
    assert rank_normalized_scores([float("nan"), 1.0]) == [1.0, 0.0]   # NaN = worst
    assert rank_normalized_scores([]) == [] and rank_normalized_scores([9.9]) == [0.0]
    # The old path clipped every score > 1 to exactly 1.0 -> no spread at all.
    assert len(set(p)) > 1


# --- 2. one defense per round, identical for every scored rollout ------------

def test_fltrust_reference_is_stable_across_scored_rollouts():
    """All G rollouts in a GRPO group must be graded by the same defense. FLTrust's
    g0 comes from SGD over a shuffled root loader, so without caching each rollout
    got its own trust direction AND its own aggregate scale (Eq. 3 rescales to
    ||g0||) — reward noise that GRPO cannot distinguish from the attacker's own
    effect on the outcome."""
    import random

    from benchmark.defenses.fltrust import FLTrust
    from core.types import ModelUpdate
    from model.mnist_net import MnistNet
    from server.algo_defender import AlgorithmicDefender
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    gw = MnistNet().state_dict()
    root = TensorDataset(torch.randn(128, 1, 28, 28), torch.randint(0, 10, (128,)))
    ft = FLTrust(DataLoader(root, batch_size=32, shuffle=True), lr=0.05, local_epochs=1)
    ft.reset(gw)
    defender = AlgorithmicDefender({"fltrust": ft}, random.Random(0))

    ups = []
    for cid in range(6):
        w = {k: v.clone() for k, v in gw.items()}
        w["net.2.weight"] += 0.02 * torch.randn_like(w["net.2.weight"])
        ups.append(ModelUpdate(client_id=cid, weights=w, metadata={"train_samples": 100}))

    runs = [defender.run(ups, gw, commit=False, algorithm="fltrust") for _ in range(5)]
    assert len({round(r.info["g0_norm"], 12) for r in runs}) == 1
    assert len({tuple(round(v.p_malicious, 12) for v in r.verdicts) for r in runs}) == 1

    # A genuinely NEW global must redraw the reference (the cache is keyed on the
    # global's contents, so it cannot stale across rounds).
    moved = {k: v.clone() for k, v in gw.items()}
    moved["net.2.weight"] += 0.5
    after = defender.run(ups, moved, commit=False, algorithm="fltrust")
    assert round(after.info["g0_norm"], 12) != round(runs[0].info["g0_norm"], 12)


# --- 3. measurement noise is not a learning signal --------------------------

def test_test_set_quantization_noise_is_treated_as_degenerate():
    """10k test examples -> accuracy quantized to 1e-4 -> ~2e-3 of reward per single
    flipped example at the smallest sampled target. The old `std < 1e-6` gate let
    that through and z-scored it to A = +-1.2."""
    posts = [0.7000, 0.7001, 0.6999, 0.7000]        # one example apart
    rs = [attacker_reward(0.9, p, GOAL, [0],
                          [DetectionVerdict(0, False, 0.0, "x", p_malicious=0.2)], 0,
                          pool_size=5) for p in posts]
    assert 0 < (max(rs) - min(rs)) < DEFAULT_MIN_REWARD_SPREAD, rs   # real but tiny
    adv, zero_frac = group_advantages(rs)
    assert zero_frac == 1.0 and adv == [0.0] * 4
    # Textbook GRPO would have amplified it to full magnitude.
    raw, raw_zero = group_advantages(rs, min_spread=0.0, std_floor=0.0)
    assert raw_zero == 0.0 and max(abs(a) for a in raw) > 1.0


def test_advantage_magnitude_tracks_real_reward_differences():
    """With the denominator floored, a barely-separated group produces a small
    update and a well-separated one a full-size update. Raw z-scoring gave both
    the identical +-1.34, so the step size carried no information."""
    small = group_advantages([0.50, 0.52, 0.54, 0.56])[0]
    large = group_advantages([0.0, 0.7, 1.4, 2.1])[0]
    assert max(abs(a) for a in small) < 0.7
    assert max(abs(a) for a in large) > 1.3
    # Above the floor this IS standard GRPO z-scoring.
    assert large == group_advantages([0.0, 0.7, 1.4, 2.1], std_floor=0.0)[0]


def test_a_genuinely_separated_group_still_learns():
    """The noise gate must not swallow real signal: a 1% accuracy difference at the
    default target clears it comfortably."""
    rs = [attacker_reward(0.9, p, GOAL, [0],
                          [DetectionVerdict(0, False, 0.0, "x", p_malicious=0.2)], 0,
                          pool_size=5) for p in (0.75, 0.74, 0.73, 0.72)]
    adv, zero_frac = group_advantages(rs)
    assert zero_frac == 0.0 and max(abs(a) for a in adv) > 1.0


# --- 4 + 5. the loss differentiates what was actually sampled ---------------

class _StubPolicy:
    """Minimal LLMPolicy surface, recording what grpo_step asks it to score."""

    def __init__(self, ids_per_rollout, rewards):
        self._ids = ids_per_rollout
        self._rewards = rewards
        self.param = torch.zeros(3, requires_grad=True)
        self.last_generation_ids = []
        self.last_generation_completed = []
        self.calls = []          # (kind, completion_ids, temperature)
        self.n_generate = 0

    def generate(self, adapter, system, user, n=1, temperature=0.0, max_new_tokens=0):
        self.n_generate += 1
        # A DIFFERENT id set per generate call, so a stale capture is detectable.
        self.last_generation_ids = [ids + 100 * (self.n_generate - 1)
                                    for ids in self._ids]
        self.last_generation_completed = [True] * n
        return [f"gen{self.n_generate}-rollout{i}" for i in range(n)]

    def _logprobs(self, kind, completion_ids, temperature):
        self.calls.append((kind, completion_ids, temperature))
        return self.param * 1.0

    def policy_token_logprobs(self, adapter, system, user, completion,
                              append_eos=False, completion_ids=None, temperature=1.0):
        return self._logprobs("policy", completion_ids, temperature)

    def reference_token_logprobs(self, system, user, completion, append_eos=False,
                                 completion_ids=None, temperature=1.0):
        return self._logprobs("ref", completion_ids, temperature).detach()

    def adapter_parameters(self, name):
        return [self.param]


class _StubTurn:
    """Serves one reward per scored rollout, advancing through ``groups`` so a
    re-rolled group can be given different rewards than the first."""

    def __init__(self, *groups):
        self.groups = [list(g) for g in groups]
        self._i = 0

    def messages(self):
        return "sys", "user"

    def reward(self, text):
        group = self.groups[min(self._i // len(self.groups[0]), len(self.groups) - 1)]
        r = group[self._i % len(group)]
        self._i += 1
        return r


class _StubOptimizer:
    def __init__(self):
        self.steps = 0

    def zero_grad(self):
        pass

    def step(self):
        self.steps += 1


def _run(ids, *reward_groups, **kw):
    policy = _StubPolicy(ids, reward_groups[0])
    opt = _StubOptimizer()
    stats = grpo_step(policy, "attacker", opt, _StubTurn(*reward_groups),
                      G=len(reward_groups[0]), max_new_tokens=8, **kw)
    return policy, opt, stats


def test_logprobs_score_the_sampled_ids_not_a_retokenization():
    """grpo_step must hand the exact sampled token ids to BOTH the policy and the
    reference pass. Re-tokenizing the decoded text is not a guaranteed inverse of
    sampling; when it differs, GRPO differentiates a sequence the policy never
    produced and the ratio==1 identity behind this loss no longer holds."""
    ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]),
           torch.tensor([6]), torch.tensor([7, 8, 9, 10])]
    policy, opt, stats = _run(ids, [0.0, 0.5, 1.0, 1.5])
    assert stats["stepped"] and opt.steps == 1

    scored = [c[1] for c in policy.calls]
    assert len(scored) == 8                      # 4 rollouts x (policy + reference)
    assert all(t is not None for t in scored), "fell back to re-tokenizing the text"
    for i, expected in enumerate(policy.last_generation_ids):
        assert torch.equal(scored[2 * i], expected)      # policy pass
        assert torch.equal(scored[2 * i + 1], expected)  # reference pass, same tokens


def test_logprobs_use_the_sampling_temperature_including_the_resample():
    """The policy GRPO optimizes is softmax(logits / T) for the T the rollouts were
    drawn at. A degenerate group is re-rolled at resample_temperature, so the
    log-prob pass has to follow it there."""
    ids = [torch.tensor([1, 2]), torch.tensor([3, 4])]

    _p, _o, _s = _run(ids, [0.2, 1.4], temperature=0.8)
    assert {c[2] for c in _p.calls} == {0.8}

    # First group degenerate -> one hotter re-roll, which recovers spread and steps.
    policy, opt, stats = _run(ids, [0.5, 0.5], [0.1, 1.1], temperature=0.8,
                              resample_on_zero_advantage=True,
                              resample_temperature=1.3)
    assert stats["resampled"] and policy.n_generate == 2
    assert stats["stepped"] and opt.steps == 1
    assert stats["temperature"] == 1.3
    assert {c[2] for c in policy.calls} == {1.3}, "scored at the pre-resample temperature"
    # ...and against the RE-ROLLED ids, not the discarded first group.
    for i, expected in enumerate(policy.last_generation_ids):
        assert torch.equal(policy.calls[2 * i][1], expected)
        assert int(expected[0]) >= 100, "captured the first group's ids"


def test_degenerate_group_applies_no_gradient():
    ids = [torch.tensor([1]), torch.tensor([2])]
    _p, opt, stats = _run(ids, [0.5, 0.5], resample_on_zero_advantage=False,
                          skip_zero_advantage=True)
    assert stats["zero_advantage_fraction"] == 1.0
    assert stats["stepped"] is False and opt.steps == 0
    assert stats["reward_spread"] == 0.0


def test_stub_generators_without_ids_fall_back_to_the_text_path():
    """rl/inference.py's frozen backends cannot return token ids; grpo_step must
    degrade to re-tokenizing rather than crash."""
    class _NoIds(_StubPolicy):
        def generate(self, *a, **kw):
            texts = super().generate(*a, **kw)
            del self.last_generation_ids          # attribute absent entirely
            return texts

    policy = _NoIds([torch.tensor([1]), torch.tensor([2])], [0.0, 1.0])
    stats = grpo_step(policy, "attacker", _StubOptimizer(), _StubTurn([0.0, 1.0]),
                      G=2, max_new_tokens=8)
    assert stats["temperature"] == 1.0
    assert stats["stepped"]
    assert all(c[1] is None for c in policy.calls)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
