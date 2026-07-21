"""Unit tests for rl/vllm_backend.py (the vLLM generation backend plumbing).

These validate the parts that must be correct regardless of the real vLLM: the
online LoRA weight-sync (dirty -> save + bumped LoRARequest id, clean -> reuse),
greedy replication, sampling fan-out, and byte-exact token-id pass-through. vLLM
itself is FAKED via sys.modules, so this runs on any box — no GPU, no torch, no
vllm install.

Run:  python tests/test_vllm_backend.py   (or: pytest tests/test_vllm_backend.py)
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fake vLLM: records what the backend asks of it, returns canned completions.
# ---------------------------------------------------------------------------
class _FakeCompletion:
    def __init__(self, text):
        self.text = text


class _FakeRequestOutput:
    def __init__(self, completions):
        self.outputs = [_FakeCompletion(t) for t in completions]


class _FakeSamplingParams:
    def __init__(self, n=1, temperature=0.0, top_p=1.0, max_tokens=16):
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens


class _FakeLoRARequest:
    def __init__(self, name, int_id, path):
        self.lora_name = name
        self.lora_int_id = int_id
        self.lora_path = path


class _FakeTokensPrompt:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class _FakeLLM:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []  # one dict per generate() call

    def generate(self, prompts=None, sampling_params=None, lora_request=None,
                 use_tqdm=True, prompt_token_ids=None):
        # Capture the token ids however they arrived (TokensPrompt vs legacy kwarg).
        if prompts is not None:
            ids = list(prompts[0].prompt_token_ids)
        else:
            ids = list(prompt_token_ids[0])
        self.calls.append({
            "ids": ids,
            "n": sampling_params.n,
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "lora_id": lora_request.lora_int_id,
            "lora_path": lora_request.lora_path,
        })
        # Return sampling_params.n canned completions.
        return [_FakeRequestOutput([f"gen-{i}" for i in range(sampling_params.n)])]


def _install_fake_vllm(with_tokens_prompt=True):
    vllm = types.ModuleType("vllm")
    vllm.LLM = _FakeLLM
    vllm.SamplingParams = _FakeSamplingParams
    if with_tokens_prompt:
        vllm.TokensPrompt = _FakeTokensPrompt
    lora_mod = types.ModuleType("vllm.lora")
    req_mod = types.ModuleType("vllm.lora.request")
    req_mod.LoRARequest = _FakeLoRARequest
    lora_mod.request = req_mod
    sys.modules["vllm"] = vllm
    sys.modules["vllm.lora"] = lora_mod
    sys.modules["vllm.lora.request"] = req_mod
    # Ensure a clean import of the backend under the fake modules.
    sys.modules.pop("rl.vllm_backend", None)


def _make_generator(tmpdir, adapters=("attacker", "defender"), max_seq_len=64):
    from rl.vllm_backend import VLLMGenerator
    return VLLMGenerator(
        base_model="fake/Qwen",
        adapters=adapters,
        adapter_dir=tmpdir,
        max_seq_len=max_seq_len,
        lora_rank=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        seed=0,
    )


class _SaveRecorder:
    """Stand-in for LLMPolicy._save_adapter_for_vllm; records (name, path) and
    creates the dir so the flow matches a real save."""
    def __init__(self):
        self.calls = []

    def __call__(self, name, path):
        os.makedirs(path, exist_ok=True)
        self.calls.append((name, path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_first_generate_syncs_then_reuses(tmp_path):
    _install_fake_vllm()
    gen = _make_generator(str(tmp_path))
    save = _SaveRecorder()

    # Dirty on construction -> first call saves + assigns lora id 1.
    out = gen.generate("attacker", [1, 2, 3], save, n=2, temperature=1.0, max_new_tokens=8)
    assert out == ["gen-0", "gen-1"]
    assert save.calls == [("attacker", os.path.join(str(tmp_path), "attacker"))]
    assert gen.llm.calls[-1]["lora_id"] == 1
    assert gen.llm.calls[-1]["ids"] == [1, 2, 3]

    # Clean now -> second call must NOT save and must reuse the same lora id.
    gen.generate("attacker", [4, 5], save, n=2, temperature=1.0, max_new_tokens=8)
    assert len(save.calls) == 1                     # no extra save
    assert gen.llm.calls[-1]["lora_id"] == 1        # same version reused

    # mark_dirty -> next call re-saves and bumps the id.
    gen.mark_dirty("attacker")
    gen.generate("attacker", [6], save, n=1, temperature=1.0, max_new_tokens=8)
    assert len(save.calls) == 2
    assert gen.llm.calls[-1]["lora_id"] == 2
    print("ok: sync/dirty/reuse")


def test_greedy_replicates_and_requests_one(tmp_path):
    _install_fake_vllm()
    gen = _make_generator(str(tmp_path))
    save = _SaveRecorder()

    # temperature 0 -> greedy: ask vLLM for exactly ONE completion, replicate to n.
    out = gen.generate("defender", [1, 2], save, n=4, temperature=0.0, max_new_tokens=8)
    assert gen.llm.calls[-1]["n"] == 1
    assert gen.llm.calls[-1]["temperature"] == 0.0
    assert out == ["gen-0", "gen-0", "gen-0", "gen-0"]
    print("ok: greedy replicate")


def test_sampling_fans_out(tmp_path):
    _install_fake_vllm()
    gen = _make_generator(str(tmp_path))
    save = _SaveRecorder()

    out = gen.generate("defender", [1, 2], save, n=3, temperature=0.8, max_new_tokens=8)
    assert gen.llm.calls[-1]["n"] == 3
    assert abs(gen.llm.calls[-1]["temperature"] - 0.8) < 1e-9
    assert out == ["gen-0", "gen-1", "gen-2"]
    print("ok: sampling fan-out")


def test_two_adapters_have_independent_paths_and_versions(tmp_path):
    _install_fake_vllm()
    gen = _make_generator(str(tmp_path))
    save = _SaveRecorder()

    gen.generate("attacker", [1], save, n=1, temperature=1.0)
    gen.generate("defender", [2], save, n=1, temperature=1.0)
    # Each adapter saved once, to its own subdir, each with lora id 1.
    paths = {name: path for name, path in save.calls}
    assert os.path.basename(paths["attacker"]) == "attacker"
    assert os.path.basename(paths["defender"]) == "defender"
    assert paths["attacker"] != paths["defender"]
    ids = [c["lora_id"] for c in gen.llm.calls]
    assert ids == [1, 1]                            # independent version counters
    print("ok: independent adapters")


def test_prompt_truncated_to_context(tmp_path):
    _install_fake_vllm()
    gen = _make_generator(str(tmp_path), max_seq_len=8)
    save = _SaveRecorder()

    # Prompt longer than max_model_len -> keep the TAIL, leave room for tokens.
    gen.generate("attacker", list(range(20)), save, n=1, temperature=0.0, max_new_tokens=100)
    call = gen.llm.calls[-1]
    assert len(call["ids"]) <= 8
    assert call["ids"][-1] == 19                    # tail preserved
    assert call["max_tokens"] >= 1
    assert len(call["ids"]) + call["max_tokens"] <= 8 or call["max_tokens"] == 1
    print("ok: context truncation")


def test_legacy_prompt_token_ids_path(tmp_path):
    # Older vLLM without TokensPrompt -> backend uses the prompt_token_ids kwarg.
    _install_fake_vllm(with_tokens_prompt=False)
    gen = _make_generator(str(tmp_path))
    save = _SaveRecorder()
    out = gen.generate("attacker", [7, 8, 9], save, n=1, temperature=0.0)
    assert gen.llm.calls[-1]["ids"] == [7, 8, 9]
    assert out == ["gen-0"]
    print("ok: legacy prompt_token_ids path")


def test_unknown_adapter_raises(tmp_path):
    _install_fake_vllm()
    gen = _make_generator(str(tmp_path))
    save = _SaveRecorder()
    try:
        gen.generate("nope", [1], save, n=1, temperature=1.0)
    except KeyError:
        print("ok: unknown adapter raises")
        return
    raise AssertionError("expected KeyError for unknown adapter")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            passed += 1
    print(f"\nAll {passed} vllm_backend tests passed.")
