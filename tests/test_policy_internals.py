"""Tests for the LLMPolicy sampling / log-prob internals (rl/policy.py).

``rl/policy.py`` keeps its heavy dependencies (unsloth, peft, transformers) inside
``__init__``, so the module imports on a CPU box and the interesting methods can be
exercised against a stub model built with ``object.__new__``. That is worth doing:
these are the functions that decide WHICH DISTRIBUTION GRPO differentiates, and a
mistake here is silent — training proceeds and the logs look fine.

Locks in:

* ``_split_generated`` cuts each rollout at its FIRST EOS, so the recorded ids are
  the exact sampled sequence. Qwen2.5's pad token IS one of its EOS ids, so
  "contains an EOS" was trivially true for any row HF had padded, and no
  pad-stripping heuristic can tell a real ``<|endoftext|>`` stop from filler.
* ``_sampling_config`` neutralizes every logits warper, so the model's chat-tuned
  generation defaults (``top_k: 20``, ``repetition_penalty: 1.05``) cannot make the
  behaviour policy differ from the one in the loss.
* ``_completion_ids`` prefers the sampled ids over re-tokenizing decoded text.
* ``_completion_token_logprobs`` scores only the completion positions, requests just
  those logits from the model, and applies the sampling temperature.

    python tests/test_policy_internals.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from rl.policy import LLMPolicy  # noqa: E402

VOCAB = 41
IM_END = 30          # stands in for <|im_end|>  (tokenizer eos)
ENDOFTEXT = 31       # stands in for <|endoftext|> (pad AND a generation_config eos)


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _StubModel:
    """Enough of an HF causal LM for the log-prob path.

    ``supports_keep`` mimics a Transformers version that accepts (or rejects) the
    last-N-positions kwarg, so both branches of ``_forward_logits`` are covered.
    """

    def __init__(self, supports_keep=True, kwarg="logits_to_keep"):
        self.supports_keep = supports_keep
        self.kwarg = kwarg
        self.calls = []                     # (seq_len, keep_or_None)
        self.generation_config = type("gc", (), {"eos_token_id": [IM_END, ENDOFTEXT]})()

    def __call__(self, ids, **kw):
        keep = None
        for k in ("logits_to_keep", "num_logits_to_keep"):
            if k in kw:
                if not (self.supports_keep and k == self.kwarg):
                    raise TypeError(f"forward() got an unexpected keyword argument '{k}'")
                keep = int(kw[k])
        self.calls.append((int(ids.shape[1]), keep))
        n = ids.shape[1] if keep is None else keep
        g = torch.Generator().manual_seed(0)
        # Deterministic logits that depend on position, so slicing errors show up.
        full = torch.randn(1, ids.shape[1], VOCAB, generator=g)
        return _Out(full if keep is None else full[:, -keep:, :])


class _StubTok:
    eos_token_id = IM_END
    pad_token_id = ENDOFTEXT

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        # One id per character, offset into the "normal token" range. Supports both
        # attribute and mapping access, like a real BatchEncoding.
        ids = torch.tensor([[ord(c) % 20 + 1 for c in text]])

        class _Enc(dict):
            input_ids = ids
        return _Enc(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        vals = [int(i) for i in ids.tolist()]
        if skip_special_tokens:
            vals = [v for v in vals if v not in (IM_END, ENDOFTEXT)]
        return "".join(chr(v + 96) for v in vals)


def _policy(model=None, max_seq_len=64):
    p = object.__new__(LLMPolicy)
    p.torch = torch
    p.model = model if model is not None else _StubModel()
    p.tokenizer = p._tok = _StubTok()
    p.device = "cpu"
    p.max_seq_len = max_seq_len
    p._logits_kw = None
    p._use_system_role = True
    p.last_generation_ids = []
    p.last_generation_completed = []
    return p


# --- sampled ids are the exact sampled sequence ------------------------------

def test_split_generated_cuts_at_the_first_eos():
    p = _policy()
    gen = torch.tensor([
        [5, 6, IM_END, ENDOFTEXT, ENDOFTEXT],   # finished, then HF padding
        [7, 8, 9, 10, 11],                      # hit the token cap, no EOS
        [ENDOFTEXT, 3, 4, 5, 6],                # genuine <|endoftext|> stop, first token
    ])
    ids, completed = p._split_generated(gen)
    assert [t.tolist() for t in ids] == [[5, 6, IM_END], [7, 8, 9, 10, 11], [ENDOFTEXT]]
    assert completed == [True, False, True]
    # The stop token is KEPT (so the policy gets gradient on the decision to stop)
    # while the padding after it is dropped -- the two are the same VALUE in row 0's
    # case only by luck of the id choice, which is exactly why "contains an EOS"
    # was not a usable test.
    assert int(ids[0][-1]) == IM_END


def test_split_generated_handles_empty_and_detaches():
    p = _policy()
    ids, completed = p._split_generated(torch.zeros(2, 0, dtype=torch.long))
    assert [t.numel() for t in ids] == [0, 0] and completed == [False, False]
    src = torch.tensor([[1, 2, IM_END, 9]])
    out, _ = p._split_generated(src)
    src[0, 0] = 99                                  # mutating the source must not leak
    assert out[0].tolist() == [1, 2, IM_END]


def test_completion_ids_prefers_the_sampled_ids():
    p = _policy()
    sampled = torch.tensor([11, 12, IM_END])
    got = p._completion_ids("ignored text", sampled, append_eos=True)
    assert got.shape == (1, 3) and got[0].tolist() == [11, 12, IM_END]
    # No second EOS is appended: the sampled ids already carry the real one.
    assert got[0].tolist().count(IM_END) == 1


def test_completion_ids_text_fallback_reattaches_the_stop_token():
    """Frozen inference backends cannot return ids, so the text path survives -- and
    there the stop token must be re-added, since decoding strips it."""
    p = _policy()
    with_eos = p._completion_ids("abc", None, append_eos=True)
    without = p._completion_ids("abc", None, append_eos=False)
    assert with_eos.shape[1] == without.shape[1] + 1
    assert int(with_eos[0, -1]) == IM_END


# --- the sampling distribution -----------------------------------------------

def test_sampling_config_disables_every_warper():
    """Qwen2.5-Instruct ships top_k=20 / top_p=0.8 / repetition_penalty=1.05. A fresh
    GenerationConfig with all warpers neutralized is what keeps the behaviour policy
    equal to the policy whose log-probs enter the loss."""
    p = _policy()
    cfg = p._sampling_config(n=4, do_sample=True, temperature=0.9, max_new_tokens=128)
    assert cfg.do_sample and cfg.num_return_sequences == 4
    assert cfg.temperature == 0.9                  # the ONLY shaping left
    assert cfg.top_k == 0 and cfg.top_p == 1.0 and cfg.typical_p == 1.0
    assert cfg.repetition_penalty == 1.0 and cfg.no_repeat_ngram_size == 0
    assert cfg.min_p is None
    assert cfg.max_new_tokens == 128 and cfg.use_cache is True
    assert cfg.pad_token_id == ENDOFTEXT and set(cfg.eos_token_id) == {IM_END, ENDOFTEXT}


def test_sampling_config_leaves_greedy_clean():
    """Greedy ignores warpers, and HF warns once per call about non-default sampling
    values there, so they are simply not set on the greedy (frozen-opponent) path.
    Asserted against whatever this Transformers version defaults to (4.x uses
    top_k=50, 5.x leaves these None) rather than a pinned number."""
    p = _policy()
    ref = type(p._sampling_config(1, False, 0.0, 32))()      # library defaults
    cfg = p._sampling_config(n=1, do_sample=False, temperature=0.0, max_new_tokens=32)
    assert cfg.do_sample is False and cfg.num_return_sequences == 1
    for knob in ("temperature", "top_k", "top_p", "typical_p", "repetition_penalty"):
        assert getattr(cfg, knob) == getattr(ref, knob), knob
    # Still fully specified where it matters for correctness.
    assert cfg.max_new_tokens == 32 and cfg.pad_token_id == ENDOFTEXT


# --- log-probs: what is scored, and how much is computed ---------------------

def _reference_logprobs(model, full, n_pred, temperature):
    """The obvious full-sequence implementation, as an oracle."""
    logits = model(full).logits[:, -(n_pred + 1):-1, :]
    if temperature != 1.0:
        logits = logits / temperature
    lp = torch.log_softmax(logits.float(), -1)
    return lp.gather(-1, full[:, -n_pred:].unsqueeze(-1)).squeeze(-1)[0]


def test_logprobs_only_ask_the_model_for_the_completion_positions():
    """The memory fix: Qwen2.5's vocab is ~152k, so projecting every position of
    prompt+completion built gigabytes of logits per rollout (retained for the
    backward) and then threw the prompt part away."""
    model = _StubModel(supports_keep=True)
    p = _policy(model)
    comp = torch.tensor([3, 4, 5, IM_END])
    out = p._completion_token_logprobs("sys", "user", None, with_grad=False,
                                       completion_ids=comp)
    assert out.shape == (4,)
    seq_len, keep = model.calls[-1]
    assert keep == 5, "did not request just the completion window"
    assert keep < seq_len, (keep, seq_len)
    assert p._logits_kw == "logits_to_keep"          # resolved and cached

    # Same numbers as the naive full-sequence implementation.
    prompt = p._prompt_ids("sys", "user")
    full = torch.cat([prompt, comp.reshape(1, -1)], dim=1)
    assert torch.allclose(out, _reference_logprobs(_StubModel(), full, 4, 1.0))


def test_logprobs_fall_back_when_the_kwarg_is_unsupported():
    """Older Transformers rejects the kwarg; the RESULT must be identical, only the
    memory saving is lost."""
    supported = _policy(_StubModel(supports_keep=True))
    legacy = _policy(_StubModel(supports_keep=False))
    comp = torch.tensor([6, 7, 8])
    a = supported._completion_token_logprobs("s", "u", None, with_grad=False,
                                            completion_ids=comp)
    b = legacy._completion_token_logprobs("s", "u", None, with_grad=False,
                                          completion_ids=comp)
    assert legacy._logits_kw == ""                   # probed, unsupported, cached
    assert torch.allclose(a, b)
    assert legacy.model.calls[-1][1] is None         # full logits, sliced by us

    # The alternate kwarg spelling is also found.
    alt = _policy(_StubModel(supports_keep=True, kwarg="num_logits_to_keep"))
    alt._completion_token_logprobs("s", "u", None, with_grad=False, completion_ids=comp)
    assert alt._logits_kw == "num_logits_to_keep"


def test_logprobs_apply_the_sampling_temperature():
    p = _policy()
    comp = torch.tensor([3, 4, 5])
    at_one = p._completion_token_logprobs("s", "u", None, with_grad=False,
                                          completion_ids=comp, temperature=1.0)
    hot = p._completion_token_logprobs("s", "u", None, with_grad=False,
                                       completion_ids=comp, temperature=1.3)
    assert not torch.allclose(at_one, hot), "temperature was ignored"
    # A hotter distribution is flatter, so the sampled tokens are less peaked.
    assert hot.mean() > at_one.mean()

    prompt = p._prompt_ids("s", "u")
    full = torch.cat([prompt, comp.reshape(1, -1)], dim=1)
    assert torch.allclose(hot, _reference_logprobs(_StubModel(), full, 3, 1.3))


def test_logprobs_are_differentiable_and_empty_safe():
    p = _policy()
    assert p._completion_token_logprobs("s", "u", None, with_grad=False,
                                        completion_ids=torch.zeros(0, dtype=torch.long)
                                        ).numel() == 0

    class _GradModel(_StubModel):
        def __init__(self):
            super().__init__()
            self.w = torch.zeros(VOCAB, requires_grad=True)

        def __call__(self, ids, **kw):
            out = super().__call__(ids, **kw)
            return _Out(out.logits + self.w)

    gp = _policy(_GradModel())
    lp = gp._completion_token_logprobs("s", "u", None, with_grad=True,
                                       completion_ids=torch.tensor([2, 3]))
    assert lp.requires_grad
    lp.sum().backward()
    assert gp.model.w.grad is not None and gp.model.w.grad.abs().sum() > 0


def test_context_truncation_keeps_the_indices_consistent():
    """A prompt+completion longer than max_seq_len is left-truncated; the scored
    positions must still line up with the tokens they predict."""
    p = _policy(max_seq_len=12)
    comp = torch.tensor([2, 3, 4, 5, 6])
    out = p._completion_token_logprobs("a long enough system prompt", "and a user turn",
                                       None, with_grad=False, completion_ids=comp)
    seq_len, keep = p.model.calls[-1]
    assert seq_len == 12, seq_len                    # truncated to the context
    assert out.shape[0] == keep - 1 <= comp.numel()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
