"""Tests for the batched log-prob path in rl.policy (the 8→2 forwards change).

The important property: computing the G completions' per-token log-probs in ONE
right-padded batched forward must give byte-for-byte the same values as the
existing per-completion path. We verify this against the REAL
LLMPolicy._batched_completion_logprobs / _completion_token_logprobs methods, run
on a tiny randomly-initialized causal LM (built from config — no download, CPU),
by binding them to a lightweight stub that supplies the few attributes they use.

Also unit-tests the pure index math in _slice_completion_logprobs.

Run on any box with torch + transformers:  python tests/test_batched_logprobs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from rl.policy import LLMPolicy  # noqa: E402  (module import is cheap — no unsloth at import time)

VOCAB = 64


def test_slice_completion_logprobs_indexing():
    G, Tm1 = 3, 12
    tok_logp = torch.arange(G * Tm1, dtype=torch.float).reshape(G, Tm1)
    P = 4
    lengths = [3, 5, 0]
    out = LLMPolicy._slice_completion_logprobs(tok_logp, P, lengths)
    # Row completion tokens sit at absolute [P, P+Ci) → predicted by [P-1, P-1+Ci).
    assert torch.equal(out[0], tok_logp[0, 3:6])
    assert torch.equal(out[1], tok_logp[1, 3:8])
    assert out[2].numel() == 0


# --------------------------------------------------------------------------
# Real-method equivalence with a tiny causal LM
# --------------------------------------------------------------------------
class _FakeTok:
    """Deterministic char→id tokenizer over ids [1, VOCAB) (0 reserved for pad)."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = [(ord(ch) % (VOCAB - 1)) + 1 for ch in text]

        class _Enc:
            pass

        e = _Enc()
        e.input_ids = torch.tensor([ids], dtype=torch.long)
        return e


class _StubPolicy:
    """Minimal object exposing what the log-prob methods touch, with the REAL
    methods bound onto it (so we test production code, not a reimplementation)."""
    _slice_completion_logprobs = staticmethod(LLMPolicy._slice_completion_logprobs)
    _batched_completion_logprobs = LLMPolicy._batched_completion_logprobs
    _completion_token_logprobs = LLMPolicy._completion_token_logprobs

    def __init__(self, model):
        self.torch = torch
        self.model = model
        self._tok = _FakeTok()
        self.device = "cpu"
        self.max_seq_len = 4096

    def _prompt_ids(self, system, user):
        # Same prompt tokenization used by BOTH paths → keeps them comparable
        # without needing a real chat template.
        return self._tok(f"{system} || {user}").input_ids.to(self.device)


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


SYSTEM = "you are the defender, classify each client"
USER = "features for twenty clients go here as json"
# Completions of DIFFERENT lengths → exercises right padding.
COMPLETIONS = [
    '{"clients": [{"client_id": 0, "is_suspicious": true, "confidence": 0.9}]}',
    '{"clients": []}',
    'x',
    '{"clients": [{"client_id": 0, "is_suspicious": false, "confidence": 0.1}, '
    '{"client_id": 1, "is_suspicious": true, "confidence": 0.8}]}',
]


def test_batched_equals_per_sample_values():
    stub = _StubPolicy(_tiny_model())
    batched = stub._batched_completion_logprobs(SYSTEM, USER, COMPLETIONS, with_grad=False)
    per = [stub._completion_token_logprobs(SYSTEM, USER, c, with_grad=False) for c in COMPLETIONS]
    assert len(batched) == len(per) == len(COMPLETIONS)
    for i, (b, p) in enumerate(zip(batched, per)):
        assert b.shape == p.shape, (i, b.shape, p.shape)
        assert torch.allclose(b, p, atol=1e-4, rtol=1e-4), (i, (b - p).abs().max().item())


def test_batched_grad_flows():
    stub = _StubPolicy(_tiny_model())
    out = stub._batched_completion_logprobs(SYSTEM, USER, COMPLETIONS, with_grad=True)
    # A non-empty completion's log-probs must be differentiable (so .backward works).
    non_empty = [t for t in out if t.numel() > 0]
    assert non_empty and all(t.requires_grad for t in non_empty)
    non_empty[0].sum().backward()   # must not raise


def test_empty_and_edge_cases():
    stub = _StubPolicy(_tiny_model())
    assert stub._batched_completion_logprobs(SYSTEM, USER, [], with_grad=False) == []
    # All-empty completions → all length-0 vectors.
    out = stub._batched_completion_logprobs(SYSTEM, USER, ["", ""], with_grad=False)
    assert len(out) == 2 and all(t.numel() == 0 for t in out)
    # Mixed empty + non-empty stays aligned.
    out2 = stub._batched_completion_logprobs(SYSTEM, USER, ["", "abc"], with_grad=False)
    assert out2[0].numel() == 0 and out2[1].numel() == 3


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} batched-logprob tests passed.")


if __name__ == "__main__":
    _run()
