"""Numerical-equivalence tests for the batched GRPO log-prob path.

The speedup in ``rl/grpo.py`` replaces 2·G sequential log-prob forwards with two
batched forwards. The risky part is purely tensor bookkeeping — right-padding G
completions that share one prompt and slicing each row's completion-token
log-probs back out. This test locks that bookkeeping to the ORIGINAL
single-sequence computation, with NO GPU / unsloth / real model:

    python tests/test_batch_logprobs.py

It uses a genuinely *causal* fake "LM" (logits at position t are a running sum of
per-token vectors, so they depend only on tokens[:t+1]). That is exactly the
property the batched path relies on: with right-padding a real token never sees a
later pad token, so its logits — and thus its log-probs — are identical to the
unpadded forward. If the slicing in ``_gather_completion_logprobs`` were off by a
position, or mixed up rows, these asserts would fail.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from rl.policy import _gather_completion_logprobs  # noqa: E402


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _FakeCausalModel:
    """Causal 'LM': logits[b, t] = sum_{s<=t} table[tokens[b, s]] (a running sum),
    so position t depends only on tokens up to t. Right padding therefore cannot
    change the logits of earlier real tokens — the invariant the batched log-prob
    path assumes. Returns an object with a ``.logits`` attribute like a HF model."""

    def __init__(self, vocab, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.table = torch.randn(vocab, vocab, generator=g)

    def __call__(self, input_ids, attention_mask=None):
        contrib = self.table[input_ids]           # (B, T, V)
        return _Out(torch.cumsum(contrib, dim=1))  # causal running sum


def _seq_reference(model, prompt, comp):
    """The ORIGINAL single-sequence completion-logprob math (see
    LLMPolicy._completion_token_logprobs): forward [prompt|comp], take the last
    ``len(comp)`` per-token log-probs."""
    full = torch.cat([prompt, comp]).unsqueeze(0)          # (1, P+C)
    logits = model(full).logits[:, :-1, :]
    logp = logits.float().log_softmax(dim=-1)
    targets = full[:, 1:]
    tok = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    return tok[-comp.shape[0]:] if comp.shape[0] > 0 else tok[:0]


def _build_batch(prompt, comps, pad_id=0):
    """Right-pad [prompt | comp_i | pad] into a rectangular (B, P+maxC) batch."""
    P = prompt.shape[0]
    lens = [int(c.shape[0]) for c in comps]
    total = P + (max(lens) if lens else 0)
    full = torch.full((len(comps), total), pad_id, dtype=torch.long)
    for i, c in enumerate(comps):
        full[i, :P] = prompt
        if lens[i]:
            full[i, P:P + lens[i]] = c
    return full, P, lens


def test_batched_matches_sequential():
    """Every row's batched completion log-probs equal the unpadded computation."""
    torch.manual_seed(1)
    V = 40
    model = _FakeCausalModel(V)
    prompt = torch.randint(0, V, (6,))
    comps = [torch.randint(0, V, (c,)) for c in (3, 7, 1, 5)]

    full, P, lens = _build_batch(prompt, comps)
    batched = _gather_completion_logprobs(model(full).logits, full, P, lens)

    assert len(batched) == len(comps)
    for i, c in enumerate(comps):
        ref = _seq_reference(model, prompt, c)
        assert batched[i].shape == ref.shape, (i, batched[i].shape, ref.shape)
        assert torch.allclose(batched[i], ref, atol=1e-5), \
            (i, float((batched[i] - ref).abs().max()))


def test_padding_amount_is_irrelevant():
    """A row's result must not depend on how much padding other rows force. Score
    a short completion alone vs. in a batch with a much longer one."""
    torch.manual_seed(2)
    V = 32
    model = _FakeCausalModel(V)
    prompt = torch.randint(0, V, (4,))
    short = torch.randint(0, V, (2,))
    long = torch.randint(0, V, (9,))

    full_a, P, lens_a = _build_batch(prompt, [short])          # no extra padding
    alone = _gather_completion_logprobs(model(full_a).logits, full_a, P, lens_a)[0]

    full_b, P, lens_b = _build_batch(prompt, [short, long])    # short now padded by +7
    padded = _gather_completion_logprobs(model(full_b).logits, full_b, P, lens_b)[0]

    assert torch.allclose(alone, padded, atol=1e-6), float((alone - padded).abs().max())


def test_empty_completion_returns_empty():
    """A zero-length completion yields an empty log-prob vector (matches the
    sequential path's ``numel() == 0`` skip in grpo_step)."""
    torch.manual_seed(3)
    V = 20
    model = _FakeCausalModel(V)
    prompt = torch.randint(0, V, (5,))
    comps = [torch.randint(0, V, (4,)), torch.zeros(0, dtype=torch.long)]

    full, P, lens = _build_batch(prompt, comps)
    out = _gather_completion_logprobs(model(full).logits, full, P, lens)

    assert out[1].numel() == 0
    assert torch.allclose(out[0], _seq_reference(model, prompt, comps[0]), atol=1e-5)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} batched-logprob tests passed.")


if __name__ == "__main__":
    _run()
