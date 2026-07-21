"""VLLMGenerator — fast, paged-KV generation for the GRPO rollouts + scoring.

This is an OPTIONAL generation backend for :class:`rl.policy.LLMPolicy`. When
``rl.use_vllm: true`` the policy routes every ``generate()`` call (the learner's
``G`` rollouts AND the frozen opponent's per-rollout scoring passes) through an
in-process vLLM engine instead of ``transformers.generate``. Only *generation*
moves — the differentiable log-prob passes and the KL-reference pass stay on the
HF/PEFT model, because vLLM cannot return gradients.

Why this is correct for online GRPO
-----------------------------------
The LoRA adapters are trained online, so vLLM's copy of an adapter goes stale the
moment the optimizer steps. We keep vLLM in lock-step with the HF weights:

* Each adapter carries a ``dirty`` flag. It starts dirty (never synced yet) and is
  re-flagged dirty whenever its HF weights change — after an optimizer step
  (:func:`rl.grpo.grpo_step`) and whenever the league swaps a snapshot in/out
  (``LLMPolicy.set_adapter_state``).
* Right before generating with an adapter we lazily *sync* it: if dirty, dump the
  current HF LoRA to disk (the standard PEFT ``adapter_config.json`` +
  ``adapter_model.safetensors`` layout ``LLMPolicy.save_adapter`` already writes)
  and hand vLLM a fresh ``LoRARequest`` with a bumped integer id so it reloads the
  weights. Generation happens BEFORE the optimizer step, so the weights vLLM
  samples from are byte-identical to the ones the HF model then scores log-probs
  with — the single-iteration ``ratio == 1`` assumption GRPO relies on holds (up
  to the usual, accepted vLLM-vs-HF forward-numerics noise shared by every
  vLLM-accelerated on-policy RL setup).

Everything vLLM-specific lives here so importing the rest of the package stays
cheap and CPU-only. All vLLM imports are deferred to construction time.
"""

import logging
import os

logger = logging.getLogger(__name__)


class VLLMGenerator:
    """In-process vLLM engine that serves multiple hot-swappable LoRA adapters.

    Parameters mirror the knobs :class:`rl.policy.LLMPolicy` already owns; the
    policy stays the single source of truth for the base model name, rank, dtype
    and sequence length so the two engines never disagree.
    """

    def __init__(
        self,
        base_model: str,
        adapters: tuple[str, ...],
        adapter_dir: str,
        *,
        max_seq_len: int = 8192,
        lora_rank: int = 16,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.30,
        enforce_eager: bool = True,
        seed: int = 0,
    ):
        # Deferred, so the module imports on a CPU box and only a real training
        # run with use_vllm=true pays the (heavy) vLLM import cost.
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        self._LoRARequest = LoRARequest
        # TokensPrompt is the modern way to feed pre-tokenized input; fall back to
        # the legacy ``prompt_token_ids=`` kwarg if this vLLM predates it. Feeding
        # token ids (rather than a text prompt) guarantees vLLM tokenizes the
        # prompt byte-identically to the HF path (which uses add_special_tokens
        # =False on an already chat-templated string).
        self._TokensPrompt = None
        try:
            from vllm import TokensPrompt
            self._TokensPrompt = TokensPrompt
        except Exception:
            try:
                from vllm.inputs import TokensPrompt
                self._TokensPrompt = TokensPrompt
            except Exception:
                self._TokensPrompt = None

        self.adapters = tuple(adapters)
        self.adapter_dir = adapter_dir
        self.max_model_len = int(max_seq_len)
        os.makedirs(adapter_dir, exist_ok=True)

        logger.info(
            f"Building vLLM engine — base={base_model} dtype={dtype} "
            f"max_lora_rank={lora_rank} gpu_mem_util={gpu_memory_utilization} "
            f"enforce_eager={enforce_eager} max_model_len={self.max_model_len}"
        )
        self.llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=int(lora_rank),
            max_loras=1,                 # one adapter per generate() call
            max_cpu_loras=8,             # cache a few (both agents + a couple versions)
            max_model_len=self.max_model_len,
            gpu_memory_utilization=float(gpu_memory_utilization),
            enforce_eager=bool(enforce_eager),
            dtype=dtype,
            seed=int(seed),
        )

        # Per-adapter sync bookkeeping. Every adapter starts dirty (never synced).
        self._version: dict[str, int] = {n: 0 for n in self.adapters}
        self._dirty: dict[str, bool] = {n: True for n in self.adapters}
        self._request: dict = {n: None for n in self.adapters}
        self._path: dict[str, str] = {
            n: os.path.join(adapter_dir, n) for n in self.adapters
        }
        logger.info(f"vLLM engine ready — adapters={self.adapters}, lora_dir={adapter_dir}")

    # ------------------------------------------------------------------
    def mark_dirty(self, name: str) -> None:
        """Flag ``name``'s vLLM copy as stale so the next generate() re-syncs it."""
        if name in self._dirty:
            self._dirty[name] = True

    def _ensure_synced(self, name: str, save_fn):
        """Return an up-to-date ``LoRARequest`` for ``name``, dumping the current HF
        weights and bumping the vLLM lora id first if the adapter is dirty.

        ``save_fn(name, path)`` writes the adapter's current PEFT files into
        ``path`` (``LLMPolicy.save_adapter`` — flat adapter_config.json +
        adapter_model.safetensors, exactly what vLLM's PEFT loader consumes).
        """
        if name not in self._request:
            raise KeyError(f"VLLMGenerator has no adapter '{name}' (known: {self.adapters})")
        if self._dirty[name] or self._request[name] is None:
            path = self._path[name]
            save_fn(name, path)                     # dump current HF LoRA to disk
            self._version[name] += 1                # new id forces vLLM to reload
            # Positional (name, int_id, path) is the signature every vLLM version
            # accepts. int_id must be > 0 (0 is reserved for "no adapter"), which
            # holds because we increment before building the request.
            self._request[name] = self._LoRARequest(
                name, self._version[name], path
            )
            self._dirty[name] = False
            logger.debug(f"vLLM: synced adapter '{name}' -> id={self._version[name]} ({path})")
        return self._request[name]

    # ------------------------------------------------------------------
    def generate(
        self, name: str, prompt_token_ids, save_fn,
        n: int = 1, temperature: float = 0.0, max_new_tokens: int = 1024,
    ) -> list[str]:
        """Generate ``n`` completions for one pre-tokenized prompt under adapter ``name``.

        Semantics match ``LLMPolicy.generate``: sampling when ``temperature > 0``
        (``n`` independent rollouts), and deterministic greedy otherwise — in which
        case we generate ONE completion and replicate it ``n`` times (all greedy
        rollouts are identical, so this saves compute and mirrors the HF path).
        """
        ids = list(int(t) for t in prompt_token_ids)
        # Keep prompt + generation within the engine's context. Prefer trimming the
        # prompt HEAD (keep the most recent tokens) — same policy as the HF log-prob
        # path — and always leave room for at least a few new tokens. ``keep`` is
        # floored at 1 so this stays correct even for a tiny max_model_len.
        if len(ids) >= self.max_model_len:
            keep = max(1, self.max_model_len - 16)
            ids = ids[-keep:]
        budget = self.max_model_len - len(ids)
        max_tok = max(1, min(int(max_new_tokens), budget))

        do_sample = bool(temperature and temperature > 0)
        n_gen = int(n) if do_sample else 1
        sp = self._SamplingParams(
            n=n_gen,
            temperature=float(temperature) if do_sample else 0.0,
            top_p=0.95 if do_sample else 1.0,
            max_tokens=max_tok,
        )

        req = self._ensure_synced(name, save_fn)
        outs = self._raw_generate(ids, sp, req)
        completions = [o.text for o in outs[0].outputs]

        # Greedy is deterministic → one row was produced; replicate to n.
        if not do_sample:
            completions = completions * int(n)
        return completions

    def _raw_generate(self, ids, sampling_params, lora_request):
        """Call ``llm.generate`` with pre-tokenized input, tolerant of the input
        API change (TokensPrompt object vs the legacy ``prompt_token_ids`` kwarg)."""
        if self._TokensPrompt is not None:
            prompts = [self._TokensPrompt(prompt_token_ids=ids)]
            return self.llm.generate(
                prompts, sampling_params=sampling_params,
                lora_request=lora_request, use_tqdm=False,
            )
        return self.llm.generate(
            prompt_token_ids=[ids], sampling_params=sampling_params,
            lora_request=lora_request, use_tqdm=False,
        )
