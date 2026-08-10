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

from benchmark.defenses.base import (  # noqa: E402
    boundary_calibrated_p, selection_boundary,
)
from core.types import DetectionVerdict  # noqa: E402
from data.feature_spec import DEFAULT_SPEC  # noqa: E402
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
    r_nearly = attacker_reward(0.9, 0.7, GOAL, [0], nearly_caught, 0)
    r_comfy = attacker_reward(0.9, 0.7, GOAL, [0], comfortable, 0)
    assert r_comfy > r_nearly, (r_comfy, r_nearly)

    # The inverted mapping the fix removed, shown explicitly: it ranked the
    # nearly-caught client as the STEALTHIER of the two.
    def old_soft_p(v):
        c = max(0.0, min(1.0, v.confidence))
        return 0.5 + 0.5 * c if v.is_suspicious else 0.5 - 0.5 * c
    assert old_soft_p(nearly_caught[0]) < old_soft_p(comfortable[0])


def test_unbounded_scores_are_boundary_calibrated_not_clipped_or_ranked():
    """Multi-Krum/DnC scores are unbounded sums of squares (and +inf for a
    non-finite client). Clipping them to [0, 1] collapsed stealth to a binary, and
    rank-normalizing them made p purely relative (a fixed 0..1 spread every round,
    so p moved when OTHER clients moved). Calibrating against the actual keep/drop
    boundary keeps it continuous AND meaningful."""
    scores = [0.1, 5.0, 5.0, 1e9, float("inf")]
    kept = {0, 1, 2}                                   # keep the 3 lowest
    thr = selection_boundary(scores, kept)
    assert 5.0 < thr < 1e9                             # midway between kept and dropped
    p = boundary_calibrated_p(scores, thr, flags=[i not in kept for i in range(5)])

    # THE invariant: p >= 0.5 exactly for the dropped clients.
    assert [pi >= 0.5 for pi in p] == [False, False, False, True, True]
    assert all(0.0 <= pi <= 1.0 for pi in p)
    assert p[1] == p[2]                                # equal scores -> equal p
    assert p[0] < p[1]                                 # monotone inside the kept set
    assert p[4] == 1.0                                 # +inf saturates, stays serializable
    assert len(set(p)) > 1                             # real spread, not a binary

    # NaN is the worst score, so it must saturate on the dropped side while the
    # accepted client keeps a real (non-zero, sub-0.5) probability.
    p_nan = boundary_calibrated_p([float("nan"), 1.0], 1.5, flags=[True, False])
    assert p_nan[0] == 1.0
    assert 0.0 < p_nan[1] < 0.5
    assert boundary_calibrated_p([], 0.0) == []


def test_calibration_survives_a_tie_across_the_boundary():
    """A keep-the-lowest-k rule breaks ties by index, so two clients with identical
    scores can land on opposite sides of the cut and NO threshold separates them. The
    defense's own flags must win — otherwise the invariant fails exactly on the
    boundary, which is where the attacker's stealth gradient lives."""
    scores = [1.0, 1.0, 1.0, 1.0]
    kept = {0, 1}                                      # tie broken by index
    p = boundary_calibrated_p(scores, selection_boundary(scores, kept),
                              flags=[i not in kept for i in range(4)])
    assert [pi >= 0.5 for pi in p] == [False, False, True, True]


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
    from model.nidd_net import NiddNet
    from server.algo_defender import AlgorithmicDefender
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    gw = NiddNet().state_dict()
    root = TensorDataset(torch.randn(128, DEFAULT_SPEC.input_dim),
                         torch.randint(0, DEFAULT_SPEC.n_classes, (128,)))
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
                          [DetectionVerdict(0, False, 0.0, "x", p_malicious=0.2)], 0)
          for p in posts]
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
                          [DetectionVerdict(0, False, 0.0, "x", p_malicious=0.2)], 0)
          for p in (0.75, 0.74, 0.73, 0.72)]
    adv, zero_frac = group_advantages(rs)
    assert zero_frac == 0.0 and max(abs(a) for a in adv) > 1.0


# --- 4 + 5. the loss differentiates what was actually sampled ---------------

class _StubPolicy:
    """Minimal LLMPolicy surface, recording what grpo_step asks it to score.

    Each rollout's log-probs are ``params[i] * ones(len(ids_i))``, so ``params[i].grad``
    isolates that rollout's contribution to the update and the token-count weighting
    is directly observable."""

    def __init__(self, ids_per_rollout, rewards):
        self._ids = ids_per_rollout
        self._rewards = rewards
        self.param = torch.zeros(3, requires_grad=True)
        self.params = [torch.zeros(1, requires_grad=True) for _ in ids_per_rollout]
        self.per_rollout_lengths = None    # set to a list to use the params above
        self._call_i = 0
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
        if self.per_rollout_lengths is None:
            return self.param * 1.0
        i = self._call_i
        if kind == "ref":                     # policy then ref for each rollout
            self._call_i += 1
        return self.params[i] * torch.ones(self.per_rollout_lengths[i])

    def policy_token_logprobs(self, adapter, system, user, completion,
                              append_eos=False, completion_ids=None, temperature=1.0):
        return self._logprobs("policy", completion_ids, temperature)

    def reference_token_logprobs(self, system, user, completion, append_eos=False,
                                 completion_ids=None, temperature=1.0):
        return self._logprobs("ref", completion_ids, temperature).detach()

    def adapter_parameters(self, name):
        return [self.param] if self.per_rollout_lengths is None else list(self.params)


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


def _length_weighted_grads(lengths, rewards):
    """Run one step where rollout i's log-probs are ``w_i * ones(L_i)``, and return
    ``(per-rollout grad, advantages, total_tokens)``. With ``grad_clip`` disabled so
    the raw normalization is observable."""
    ids = [torch.arange(L) for L in lengths]
    policy = _StubPolicy(ids, rewards)
    policy.per_rollout_lengths = list(lengths)
    stats = grpo_step(policy, "attacker", _StubOptimizer(), _StubTurn(rewards),
                      G=len(rewards), max_new_tokens=64, kl_beta=0.0,
                      grad_clip=1e9)
    return [float(p.grad) for p in policy.params], stats["advantages"], stats


def test_gradient_weights_every_token_equally_not_every_rollout():
    """THE length-bias fix. Rollout 1 sampled 4x as many tokens as rollout 0, so it
    must carry 4x the gradient at equal |advantage|. The old sequence-mean form
    divided each rollout by its OWN length, giving both the same weight — i.e. the
    short rollout's tokens counted 4x each. Output length here is tied to the action
    (a 5-client plan is much longer than a 1-client plan), so that bias pushed on
    client-count selection independently of the reward."""
    grads, adv, stats = _length_weighted_grads([2, 8], [0.0, 1.0])
    assert adv == [-1.0, 1.0], adv
    assert stats["total_tokens"] == 10 and stats["completion_tokens"] == [2, 8]

    # d/dw_i of (1/ΣL) * Σ_i (-A_i * w_i * L_i)  =  -A_i * L_i / ΣL
    assert abs(grads[0] - (-adv[0] * 2 / 10)) < 1e-6, grads
    assert abs(grads[1] - (-adv[1] * 8 / 10)) < 1e-6, grads
    assert abs(abs(grads[1] / grads[0]) - 4.0) < 1e-6, "gradient is not length-weighted"

    # The old form would have given both the same magnitude (-A_i / G).
    assert abs(abs(grads[0]) - abs(adv[0]) / 2) > 1e-3


def test_equal_length_rollouts_reproduce_the_sequence_mean_update():
    """Token-level normalization must not silently rescale the effective learning
    rate: when all rollouts are the same length the two forms coincide exactly."""
    for L in (1, 3, 17):
        grads, adv, stats = _length_weighted_grads([L] * 4, [0.0, 0.4, 0.8, 1.2])
        assert stats["total_tokens"] == 4 * L
        for g, a in zip(grads, adv):
            assert abs(g - (-a / 4)) < 1e-6, (L, g, a)   # == old (pg = -A*mean_t lp)/G


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


# --- 6. the defense is strong enough to produce a measurable outcome ---------

def test_root_epochs_match_the_clients_iteration_count_but_are_capped():
    """FLTrust rescales EVERY accepted delta to ||g0|| and applies w <- w + eta*g, so
    the SERVER's reference update sets how far the global can move per round. At
    root_epochs=1 the root took ~2 SGD steps (100 examples / batch 64) against a
    client's ~144: ||g0|| was ~60x too small, the global froze, and the damage term
    was ~0 for every candidate in the group — no gradient on FLTrust's ~1/4 of rounds.

    Matching iterations UN-CAPPED overshoots the other way: 144/2 = 72 epochs over 100
    root examples memorises them, so g0 stops being a descent direction the clients
    share. Trust is ReLU(cos(delta_i, g0)), so honest clients then score cos <= 0 and
    are DROPPED — the recorded run showed FLTrust at FPR 0.9-1.0 for whole blocks, and
    rounds where the only accepted clients were the poisoned ones. Hence the cap."""
    from server.algo_defender import DEFAULT_MAX_ROOT_EPOCHS, resolve_root_epochs

    # null -> match the honest client's ITERATION count (not its epoch count)...
    assert resolve_root_epochs(None, root_batches=48, client_iterations=144) == 3
    # ...but never past the overfit ceiling.
    assert resolve_root_epochs(None, root_batches=2, client_iterations=144) == \
        DEFAULT_MAX_ROOT_EPOCHS
    assert resolve_root_epochs(None, root_batches=2, client_iterations=144,
                               max_epochs=4) == 4
    # The cap only ever lowers the matched value, never raises it.
    assert resolve_root_epochs(None, root_batches=48, client_iterations=144,
                               max_epochs=999) == 3
    # An explicit value is always honoured, is never allowed below 1, and is NOT capped
    # (the operator asked for it).
    assert resolve_root_epochs(5, root_batches=2, client_iterations=144) == 5
    assert resolve_root_epochs(0, root_batches=2, client_iterations=144) == 1
    assert resolve_root_epochs(70, root_batches=2, client_iterations=144,
                               max_epochs=10) == 70
    # Unknown client cost (no loaders, e.g. a unit test) degrades to the old default.
    assert resolve_root_epochs(None, root_batches=2, client_iterations=None) == 1
    assert resolve_root_epochs(None, root_batches=0, client_iterations=144) == 1


def test_matching_iterations_makes_the_global_actually_move():
    """The measurable consequence: ||g0|| goes from a rounding error to a real step."""
    import torch as _torch
    from benchmark.defenses.fltrust import FLTrust, _flatten
    from clients.benign_client import BenignClient
    from core.types import ModelUpdate
    from model.nidd_net import NiddNet
    from server.fed_server import FedServer
    from torch.utils.data import DataLoader, TensorDataset

    _torch.manual_seed(0)
    srv = FedServer(device="cpu")
    gw = srv.get_global_weights()
    keys = list(gw.keys())

    def loader(n, batch):
        return DataLoader(TensorDataset(_torch.randn(n, DEFAULT_SPEC.input_dim),
                                        _torch.randint(0, DEFAULT_SPEC.n_classes, (n,))),
                          batch_size=batch, shuffle=False)

    client_loader = loader(3072, 64)                       # 48 batches
    client = BenignClient(0, client_loader, lr=0.002, local_epochs=3, device="cpu")
    client_delta = (_flatten(client.train(srv.model).weights, keys)
                    - _flatten(gw, keys)).norm()
    ups = [ModelUpdate(client_id=i, weights=client.train(srv.model).weights,
                       metadata={"train_samples": 100}) for i in range(4)]

    norms = {}
    for label, epochs in (("old", 1), ("matched", 72)):    # 2 batches x 72 = 144 iters
        ft = FLTrust(loader(100, 64), lr=0.002, local_epochs=epochs, device="cpu")
        ft.reset(gw)
        norms[label] = ft.step(ups, set()).info["g0_norm"]

    old_frac = norms["old"] / float(client_delta)
    new_frac = norms["matched"] / float(client_delta)
    # `root_epochs: 1` gives the server 2 SGD iterations against a client's 144, so
    # ||g0|| lands at a small fraction of an honest step — and since FLTrust rescales
    # EVERY accepted delta to ||g0|| (Eq. 3), that fraction is how far the global is
    # allowed to move per round. The bound is loose because it is architecture- and
    # seed-dependent (0.042-0.053 measured across seeds on the 681-parameter NiddNet,
    # smaller on the 970-parameter image model this replaced); what the test pins is
    # that it is a small fraction, not that it is one specific number.
    assert old_frac < 0.08, old_frac              # was a rounding error
    # Iteration-matching is what buys the real step: measured 70-77x, not the bare 10x.
    assert new_frac > 10 * old_frac, (old_frac, new_frac)


# --- 7. the environment advances on-policy, not on best-of-G -----------------

def test_committed_rollout_is_an_on_policy_draw_not_the_argmax():
    """Committing the argmax made the committed round — and therefore every logged
    `learner_success` and every phase switch — a best-of-G result that does not
    reproduce at eval, where the benchmark samples once."""
    import random

    from rl.schedule import _committed_index

    rewards = [0.1, 0.9, 0.2, 0.3]        # index 1 is the best
    assert _committed_index(rewards, "argmax", random.Random(0)) == 1

    picks = {_committed_index(rewards, "sample", random.Random(s)) for s in range(50)}
    assert picks == {0, 1, 2, 3}, picks              # every rollout is reachable
    # Uniform, so the best is chosen ~1/G of the time rather than always.
    rng = random.Random(0)
    chose_best = sum(_committed_index(rewards, "sample", rng) == 1 for _ in range(2000))
    assert 300 < chose_best < 700, chose_best
    assert _committed_index([], "sample", random.Random(0)) == 0


def test_commit_selection_is_validated():
    """A typo must fail loudly rather than silently falling back to sampling."""
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "rl" / "schedule.py"
    text = src.read_text(encoding="utf-8")
    assert re.search(r"commit_selection must be sample\|argmax", text)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
