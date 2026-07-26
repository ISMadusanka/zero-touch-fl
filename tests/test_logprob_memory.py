"""Regression tests for LLMPolicy._completion_token_logprobs.

The function used to run ``log_softmax(logits.float())`` over the WHOLE sequence
and then keep only the completion's rows. With Qwen2.5's ~152k vocab that
intermediate is ``seq_len x 152k x 4`` bytes — ~2.8 GB on a 4.8k-token prompt —
allocated and immediately discarded, which OOM'd a 31 GB GPU mid-training.

It now slices to the completion's positions BEFORE the fp32 ``log_softmax``.
That is an exact identity, not an approximation (``log_softmax`` normalizes over
the vocab dimension independently at each position), and these tests pin it:
the sliced implementation must reproduce the original's numbers bit-for-bit.

No GPU, no model download — a stub model stands in for the LM.
    python tests/test_logprob_memory.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from rl.policy import LLMPolicy  # noqa: E402

VOCAB = 71
N_IDS = 50


class _FakeTok:
    """Character-level tokenizer — enough to exercise the id plumbing."""
    eos_token_id = VOCAB - 1

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        # Empty text -> a [1, 0] tensor, matching a real tokenizer.
        ids = [(ord(c) % N_IDS) for c in text]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


class _FakeModel:
    """Deterministic stand-in LM. Records whether a tail-only kwarg was used."""

    def __init__(self, seed=0, supports_keep_kw=None):
        g = torch.Generator().manual_seed(seed)
        # One logits row per token id — sized to the vocab so the EOS id indexes.
        self.table = torch.randn(VOCAB, VOCAB, generator=g)
        self.supports = supports_keep_kw      # None = no tail kwarg accepted
        self.last_keep = None

    def __call__(self, ids, **kw):
        for k in ("logits_to_keep", "num_logits_to_keep"):
            if k in kw:
                if self.supports != k:
                    raise TypeError(f"forward() got an unexpected keyword argument '{k}'")
                self.last_keep = kw[k]
                full = self.table[ids[0]].unsqueeze(0)
                return SimpleNamespace(logits=full[:, -kw[k]:, :])
        return SimpleNamespace(logits=self.table[ids[0]].unsqueeze(0))


def _policy(model, max_seq_len=4096):
    """An LLMPolicy with just the attributes the log-prob path touches."""
    p = object.__new__(LLMPolicy)
    p.torch = torch
    p._tok = _FakeTok()
    p.device = torch.device("cpu")
    p.max_seq_len = max_seq_len
    p.model = model
    p._logits_kw = None
    p._prompt_ids = lambda system, user: torch.tensor(
        [[(ord(c) % N_IDS) for c in (system + user)]])
    return p


def _original_impl(model, full, comp_len):
    """The pre-fix computation, verbatim: full-sequence log_softmax, then slice."""
    logits = model(full).logits[:, :-1, :]
    logp = torch.log_softmax(logits.float(), dim=-1)
    targets = full[:, 1:]
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    return tok_logp[-comp_len:]


def _full_ids(p, system, user, completion, append_eos=False):
    prompt = p._prompt_ids(system, user)
    comp = p._tok(completion).input_ids
    if append_eos:
        comp = torch.cat([comp, torch.tensor([[_FakeTok.eos_token_id]])], dim=1)
    full = torch.cat([prompt, comp], dim=1)
    if full.shape[1] > p.max_seq_len:
        full = full[:, -p.max_seq_len:]
    return full, comp.shape[1]


SYSTEM = "you are the adversary in a federated learning system" * 8
USER = '{"round": 12, "client_update_stats": {"0": {"rel_update": 0.21}}}' * 12
COMPLETION = '{"clients":[{"id":0,"operations":[{"op":"scale","rows":[2]}]}]}'


def test_matches_the_original_full_sequence_computation():
    model = _FakeModel(seed=1)
    p = _policy(model)
    got = p._completion_token_logprobs(SYSTEM, USER, COMPLETION, with_grad=False)
    full, comp_len = _full_ids(p, SYSTEM, USER, COMPLETION)
    want = _original_impl(model, full, comp_len)
    assert got.shape == want.shape == (comp_len,)
    assert torch.equal(got, want), (got - want).abs().max()


def test_matches_the_original_with_appended_eos():
    model = _FakeModel(seed=2)
    p = _policy(model)
    got = p._completion_token_logprobs(SYSTEM, USER, COMPLETION,
                                       with_grad=False, append_eos=True)
    full, comp_len = _full_ids(p, SYSTEM, USER, COMPLETION, append_eos=True)
    want = _original_impl(model, full, comp_len)
    assert torch.equal(got, want)


def test_matches_when_the_prompt_is_truncated_to_the_context_window():
    """Truncation keeps the TAIL, so the completion survives and the end-relative
    slices must still line up."""
    model = _FakeModel(seed=3)
    p = _policy(model, max_seq_len=64)
    got = p._completion_token_logprobs(SYSTEM, USER, COMPLETION, with_grad=False)
    full, comp_len = _full_ids(p, SYSTEM, USER, COMPLETION)
    want = _original_impl(model, full, comp_len)
    assert full.shape[1] == 64
    assert torch.equal(got, want)


def test_tail_only_forward_gives_identical_numbers():
    """With `logits_to_keep` honored the model returns only the tail of the logits;
    because both slices are end-relative the result must be unchanged."""
    plain = _FakeModel(seed=4)
    tail = _FakeModel(seed=4, supports_keep_kw="logits_to_keep")
    got_plain = _policy(plain)._completion_token_logprobs(SYSTEM, USER, COMPLETION, with_grad=False)
    p_tail = _policy(tail)
    got_tail = p_tail._completion_token_logprobs(SYSTEM, USER, COMPLETION, with_grad=False)
    assert torch.equal(got_plain, got_tail)
    assert p_tail._logits_kw == "logits_to_keep"
    # Asked for exactly comp_len+1 positions, not the whole sequence.
    assert tail.last_keep == got_tail.shape[0] + 1


def test_legacy_kwarg_name_is_probed_too():
    m = _FakeModel(seed=5, supports_keep_kw="num_logits_to_keep")
    p = _policy(m)
    p._completion_token_logprobs(SYSTEM, USER, COMPLETION, with_grad=False)
    assert p._logits_kw == "num_logits_to_keep"


def test_unsupported_kwarg_falls_back_and_is_probed_once():
    m = _FakeModel(seed=6, supports_keep_kw=None)
    p = _policy(m)
    a = p._completion_token_logprobs(SYSTEM, USER, COMPLETION, with_grad=False)
    assert p._logits_kw == ""            # probed, unsupported -> cached
    b = p._completion_token_logprobs(SYSTEM, USER, COMPLETION, with_grad=False)
    assert torch.equal(a, b)


def test_empty_completion_returns_empty():
    p = _policy(_FakeModel(seed=7))
    out = p._completion_token_logprobs(SYSTEM, USER, "", with_grad=False)
    assert out.numel() == 0


def test_gradients_still_flow_through_the_sliced_path():
    """The policy term needs a differentiable result; slicing must not detach it."""
    model = _FakeModel(seed=8)
    model.table.requires_grad_(True)
    p = _policy(model)
    lp = p._completion_token_logprobs(SYSTEM, USER, COMPLETION, with_grad=True)
    assert lp.requires_grad
    lp.mean().backward()
    assert model.table.grad is not None and torch.count_nonzero(model.table.grad) > 0


# ---------------------------------------------------------------------------
# generate()'s OOM handling
#
# The fallback from KV-cached generation to the manual no-cache decoder is STICKY
# (it disables fast generate for the rest of the run). That is right for a broken
# kernel, but wrong for an OOM: the manual decoder re-runs the full forward per
# generated token, so it needs MORE memory and fails harder. An OOM must retry the
# same path, not downgrade to a costlier one.
# ---------------------------------------------------------------------------

def _gen_policy(fast_side_effects):
    """Policy whose _fast_generate raises/returns per ``fast_side_effects``."""
    p = object.__new__(LLMPolicy)
    p.torch = torch
    p._use_fast_generate = True
    p.calls = {"fast": 0, "manual": 0}
    pending = list(fast_side_effects)

    def _fast(*a, **k):
        p.calls["fast"] += 1
        out = pending.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out

    def _manual(*a, **k):
        p.calls["manual"] += 1
        return ["manual"]

    p._fast_generate = _fast
    p._manual_generate = _manual
    p.set_adapter = lambda name: None
    p._prompt_ids = lambda system, user: torch.zeros((1, 8), dtype=torch.long)
    return p


def _oom(msg="CUDA out of memory. Tried to allocate 332.00 MiB"):
    exc = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    return exc(msg)


def test_oom_retries_the_same_path_and_does_not_downgrade():
    p = _gen_policy([_oom(), ["ok"]])
    assert LLMPolicy.generate(p, "a", "sys", "usr") == ["ok"]
    assert p.calls == {"fast": 2, "manual": 0}      # retried, never fell back
    assert p._use_fast_generate is True             # and stayed on the fast path


def test_repeated_oom_surfaces_instead_of_hiding_behind_a_costlier_path():
    p = _gen_policy([_oom(), _oom()])
    try:
        LLMPolicy.generate(p, "a", "sys", "usr")
    except Exception as e:
        assert p._is_oom(e)
    else:
        raise AssertionError("a second OOM should propagate, not fall back")
    assert p.calls["manual"] == 0


def test_non_oom_failure_still_falls_back_stickily():
    """The original behaviour, preserved: a broken kernel is a permanent property
    of the install, so switching decoders once is correct there."""
    p = _gen_policy([RuntimeError("paged-KV kernel not supported")])
    assert LLMPolicy.generate(p, "a", "sys", "usr") == ["manual"]
    assert p.calls == {"fast": 1, "manual": 1}
    assert p._use_fast_generate is False


def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} log-prob memory/equivalence tests passed.")


if __name__ == "__main__":
    _run()
