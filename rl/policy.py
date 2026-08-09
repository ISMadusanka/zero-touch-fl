"""LLMPolicy — one frozen Qwen2.5-3B-Instruct base, two trainable LoRA adapters.

This is the only module with heavy GPU dependencies (unsloth / peft /
transformers / torch + a CUDA box). It is imported only on the training path;
the dry-run path uses ``rl/inference.py`` instead, so the rest of the package
stays importable on a CPU machine.

Design — "separate checkpoints on the same LLM":
  * One (optionally 4-bit / QLoRA) Qwen2.5-3B-Instruct base, loaded once and frozen.
  * Two LoRA adapters over it: ``"attacker"`` and ``"defender"``. Each is an
    independent set of low-rank deltas → two separate checkpoints
    (``adapter_model.safetensors`` + ``adapter_config.json``) sharing one base.
  * ``set_adapter(name)`` activates one adapter for both generation and the
    log-prob forward pass; ``disable_adapter()`` exposes the base as the KL
    reference policy.

The class exposes exactly what GRPO needs: sample completions (no grad),
recompute per-token log-probs of a completion (with grad), and a no-grad
reference log-prob under the base model.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Default LoRA target modules — the attention/MLP projection names shared by
# Llama 3.x, Gemma, Qwen, Mistral, etc.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class LLMPolicy:
    def __init__(
        self,
        base_model: str = "unsloth/Qwen2.5-3B-Instruct",
        max_seq_len: int = 8192,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        load_in_4bit: bool = True,
        target_modules: list[str] | None = None,
        seed: int = 0,
        adapters: tuple[str, ...] = ("attacker", "defender"),
        attn_implementation: str = "eager",
        use_fast_generate: bool = True,
    ):
        # Heavy imports kept local so importing this module is cheap.
        import torch
        # Disable Unsloth's fused fast-generate wrapper BEFORE importing unsloth:
        # its fused paged-KV inference kernel (e.g. the *Attention_fast_forward_inference
        # path) can be incompatible with the installed Transformers and crash on a RoPE
        # cos/sin broadcast. With it disabled, model.generate() falls through to
        # standard Transformers generation, which uses a normal KV cache via the
        # regular (working) forward — fast AND correct, no version downgrade.
        os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")
        from unsloth import FastLanguageModel
        # Unsloth is now imported FIRST (before transformers), so its optimizations
        # patch cleanly. Only now is it safe to touch transformers — lower its log
        # verbosity here (moved out of main.quiet_noisy_warnings, which runs at
        # startup and would otherwise import transformers ahead of unsloth).
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
        except Exception:
            pass
        from peft import LoraConfig

        self.torch = torch
        self.adapters = tuple(adapters)
        self.max_seq_len = max_seq_len
        target_modules = target_modules or DEFAULT_TARGET_MODULES

        logger.info(f"Loading base model {base_model} (4bit={load_in_4bit}) ...")
        # Eager attention is the broadly-compatible choice (avoids SDPA-dispatch
        # errors some architectures hit). Qwen2.5 also works with "sdpa".
        # Override via configs/base.yaml -> rl.attn_implementation.
        base, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_len,
            load_in_4bit=load_in_4bit,
            dtype=None,
            attn_implementation=attn_implementation,
        )

        # Inject LoRA (Unsloth optimizations) — this creates the "default" adapter.
        self.model = FastLanguageModel.get_peft_model(
            base,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )

        # Add the two named adapters we actually train; "default" stays unused.
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", target_modules=target_modules, task_type="CAUSAL_LM",
        )
        for name in self.adapters:
            self.model.add_adapter(name, lora_cfg)

        # Multimodal models (e.g. Gemma 3) return a Processor whose chat template
        # is used to RENDER prompts, with the raw text tokenizer at ``.tokenizer``.
        # Text-only models (e.g. Qwen2.5) return the tokenizer directly. Keep
        # both handles: ``self.tokenizer`` renders, ``self._tok`` tokenizes — for
        # a plain tokenizer the getattr fallback makes them the same object.
        self._tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token

        self.device = next(self.model.parameters()).device
        self.model.train()
        self._active = None
        self._logits_kw = None   # resolved on first use: which "last-token-only" kwarg works
        self._use_system_role = None  # resolved on first use: does the chat template accept a system role?
        self._use_fast_generate = bool(use_fast_generate)  # KV-cached generate; auto-falls back on failure
        # Per-rollout "did this finish on its own (EOS) rather than hit the token
        # cap?", set by every generate() call. See generate().
        self.last_generation_completed: list[bool] = []
        # The EXACT completion token ids sampled by the last generate() call, one
        # 1-D tensor per returned text (EOS included when the rollout ended on it,
        # trailing padding stripped). GRPO feeds these straight back into the
        # log-prob pass so it differentiates the sequence that was actually
        # sampled — see generate() and _completion_token_logprobs.
        self.last_generation_ids: list = []
        self.set_adapter(self.adapters[0])
        logger.info(f"LLMPolicy ready — adapters={self.adapters}, device={self.device}")

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------
    def set_adapter(self, name: str):
        if name != self._active:
            self.model.set_adapter(name)
            self._active = name

    def adapter_parameters(self, name: str):
        """Trainable LoRA parameters belonging to one adapter (for an optimizer)."""
        params = [
            p for n, p in self.model.named_parameters()
            if name in n and "lora_" in n.lower()
        ]
        for p in params:
            p.requires_grad_(True)
        return params

    def get_adapter_state(self, name: str) -> dict:
        """CPU copy of one adapter's LoRA tensors (for the opponent league)."""
        from peft import get_peft_model_state_dict
        sd = get_peft_model_state_dict(self.model, adapter_name=name)
        return {k: v.detach().to("cpu").clone() for k, v in sd.items()}

    def set_adapter_state(self, name: str, state: dict):
        from peft import set_peft_model_state_dict
        on_device = {k: v.to(self.device) for k, v in state.items()}
        set_peft_model_state_dict(self.model, on_device, adapter_name=name)

    def save_adapter(self, name: str, path: str):
        """Save one adapter to ``path`` with a flat, version-stable layout.

        Writes ``adapter_model.safetensors`` + ``adapter_config.json`` directly
        in ``path`` (no per-adapter subfolder), so ``load_adapter`` and
        ``storage.checkpoint.adapter_exists`` agree on where files live.
        """
        from safetensors.torch import save_file
        os.makedirs(path, exist_ok=True)
        save_file(self.get_adapter_state(name), os.path.join(path, "adapter_model.safetensors"))
        self.model.peft_config[name].save_pretrained(path)
        logger.info(f"Saved adapter '{name}' -> {path}")

    def save_adapter_state_dict(self, name: str, state: dict, path: str):
        """Save a PROVIDED adapter ``state`` (not the in-memory one) with the same
        on-disk layout as :meth:`save_adapter`.

        Used to persist the LIVE opponent weights while an older league snapshot is
        temporarily swapped into that adapter (curriculum/league phase), so a
        mid-phase checkpoint never overwrites the real opponent adapter with the
        snapshot. ``name`` still selects the ``adapter_config.json`` to write."""
        from safetensors.torch import save_file
        os.makedirs(path, exist_ok=True)
        cpu_state = {k: v.detach().to("cpu").clone() for k, v in state.items()}
        save_file(cpu_state, os.path.join(path, "adapter_model.safetensors"))
        self.model.peft_config[name].save_pretrained(path)
        logger.info(f"Saved adapter '{name}' (live state) -> {path}")

    def load_adapter(self, name: str, path: str):
        """Load saved LoRA weights INTO the already-created ``name`` adapter.

        Resume-safe: we don't call PEFT ``load_adapter`` (which errors if the
        adapter already exists) — we set the existing adapter's state instead.
        """
        from safetensors.torch import load_file
        sd = load_file(os.path.join(path, "adapter_model.safetensors"))
        self.set_adapter_state(name, sd)
        logger.info(f"Loaded adapter '{name}' <- {path}")

    # ------------------------------------------------------------------
    # Generation + log-probs
    # ------------------------------------------------------------------
    def _prompt_ids(self, system: str, user: str):
        # Deliver the instructions in the model's NATIVE system role (Qwen2.5,
        # Llama, Mistral all support it) so they carry system-level authority and
        # the model adopts the role cleanly — this matters for a small instruct
        # model and, for the attacker, keeps the adversarial framing from being
        # overridden by the template's default "helpful assistant" system persona.
        # Templates without a system role (e.g. Gemma) fall back to folding
        # system+user into one user turn — resolved once in _render_chat.
        # Render to text (tokenize=False → str), then tokenize with the raw
        # tokenizer. The template already injects BOS + turn tokens, so
        # add_special_tokens=False avoids a duplicate BOS.
        text = self._render_chat(system, user)
        enc = self._tok(text, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"].to(self.device)

    def count_prompt_tokens(self, system: str, user: str) -> int:
        """Exact prompt length in tokens, chat template included.

        This is what the agents' ``PromptBudget`` measures the context fill
        against (``AttackerAgent.bind_tokenizer``), so it deliberately goes
        through the same ``_render_chat`` path generation uses — an estimate off
        by the template's own scaffolding would mis-size the budget. It only
        tokenizes (no model forward), so it is cheap enough to call per round.
        """
        text = self._render_chat(system, user)
        return len(self._tok(text, add_special_tokens=False).input_ids)

    def _render_chat(self, system: str, user: str) -> str:
        """Render the chat template, preferring a native system role.

        Whether this model's template accepts a ``system`` message is probed ONCE
        and cached in ``self._use_system_role`` (a cheap string render, no model
        forward); afterwards this is a plain dispatch. Generation and the log-prob
        passes both go through here, so the two stay byte-identical.
        """
        if self._use_system_role is None:
            try:
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                    add_generation_prompt=True, tokenize=False,
                )
                self._use_system_role = True
            except Exception:
                self._use_system_role = False
                logger.info("Chat template has no system role — folding system into the user turn.")
        if self._use_system_role:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        else:
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )

    def _forward_logits(self, ids, keep: int):
        """Logits for the LAST ``keep`` positions only — shape ``(B, keep, V)``.

        Asks the model to skip the vocab projection over every earlier position
        (``logits_to_keep`` / ``num_logits_to_keep``, whichever this Transformers
        version accepts — resolved once and cached). Falls back to slicing the full
        logits if neither kwarg exists, so the RESULT is identical either way and
        callers never have to care.

        This matters most in the GRPO log-prob pass, where ``ids`` is
        ``prompt + completion``: Qwen2.5's vocab is ~152k, so a full-sequence
        forward materializes ``(1, L, 152k)`` logits (~0.85 GB in bf16 at L=2800)
        of which only the last ``len(completion)+1`` positions are ever read — and
        that tensor is retained for the backward. Requesting just the tail cuts it
        by roughly ``L / (comp_len + 1)``.
        """
        keep = max(1, int(keep))
        if self._logits_kw is None:
            for kw in ("logits_to_keep", "num_logits_to_keep"):
                try:
                    out = self.model(ids, **{kw: keep})
                except TypeError:
                    continue          # this Transformers version lacks the kwarg
                except Exception as e:
                    # Something else went wrong (e.g. an Unsloth patch that does not
                    # forward the kwarg cleanly). Don't let a memory optimization take
                    # the run down — record "unsupported" and use the full-logits path.
                    logger.warning(
                        f"{kw} probe failed ({type(e).__name__}: {e}); using full logits."
                    )
                    break
                self._logits_kw = kw
                return out.logits[:, -keep:, :]
            self._logits_kw = ""      # unsupported -> full logits
        if self._logits_kw:
            out = self.model(ids, **{self._logits_kw: keep})
        else:
            out = self.model(ids)
        return out.logits[:, -keep:, :]

    def _last_logits(self, ids):
        """Logits for the LAST position only, as ``(B, V)`` float32 — the decode-loop
        entry point into :meth:`_forward_logits`."""
        return self._forward_logits(ids, 1)[:, -1, :].float()

    def _eos_ids(self) -> set:
        ids = set()
        if self._tok.eos_token_id is not None:
            ids.add(int(self._tok.eos_token_id))
        try:
            ge = self.model.generation_config.eos_token_id
            if isinstance(ge, (list, tuple)):
                ids.update(int(x) for x in ge)
            elif ge is not None:
                ids.add(int(ge))
        except Exception:
            pass
        return ids

    def _split_generated(self, gen):
        """Turn a batch of generated spans into per-row ``(ids, completed)``.

        ``gen`` is the ``(rows, max_new_tokens)`` slice AFTER the prompt. Each row is
        cut at (and including) its FIRST end-of-sequence token; a row with no EOS ran
        into the token cap and is kept whole with ``completed=False``.

        Cutting at the first EOS — rather than stripping "trailing pad" — is what
        makes the ids usable as the exact sampled sequence. Qwen2.5's pad token
        (``<|endoftext|>``) is itself one of the model's EOS ids, so the two are
        indistinguishable by value: the previous "does the span contain an EOS?" test
        was satisfied by HF's own padding of already-finished rows, and any
        pad-stripping heuristic would have been unable to tell a genuine
        ``<|endoftext|>`` stop from filler.
        """
        torch = self.torch
        eos_ids = self._eos_ids()
        eos_t = torch.tensor(sorted(eos_ids), device=gen.device) if eos_ids else None
        ids, completed = [], []
        for i in range(gen.shape[0]):
            row = gen[i]
            cut = None
            if eos_t is not None:
                hit = torch.nonzero(torch.isin(row, eos_t), as_tuple=False)
                if hit.numel():
                    cut = int(hit[0].item()) + 1      # keep the stop token itself
            ids.append((row if cut is None else row[:cut]).detach().clone())
            completed.append(cut is not None)
        return ids, completed

    def _sampling_config(self, n, do_sample, temperature, max_new_tokens):
        """A FRESH ``GenerationConfig`` for one generate call.

        Built from scratch — never merged with ``model.generation_config`` — because
        Qwen2.5-Instruct ships chat-tuned defaults (``top_k: 20``, ``top_p: 0.8``,
        ``temperature: 0.7``, ``repetition_penalty: 1.05``) and HF applies every one
        of them that the caller does not override. Passing only ``temperature`` and
        ``top_p`` therefore left ``top_k=20`` and a repetition penalty silently
        active, so rollouts were drawn from a truncated, history-dependent
        distribution while GRPO differentiated the untruncated softmax. The policy
        gradient's ``ratio == 1`` assumption (see rl/grpo.py) requires the sampler
        and the log-prob pass to be the SAME distribution, and a repetition penalty
        is especially damaging here because the attacker's output is repetitive JSON
        (``op``/``target``/``id`` once per client).

        So: every logits warper is explicitly neutralized and temperature is the only
        shaping left — matched by ``_completion_token_logprobs(temperature=...)``.
        """
        from transformers import GenerationConfig
        eos = sorted(self._eos_ids())
        kw = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            use_cache=True,
            num_return_sequences=int(n),
            pad_token_id=self._tok.pad_token_id,
            eos_token_id=(eos if eos else None),
        )
        if do_sample:
            # UNTRUNCATED sampling: the behaviour policy must equal the policy whose
            # log-probs enter the loss, so temperature is the only shaping applied.
            # 0 / 1.0 disable each warper in HF. (Set only under do_sample; greedy
            # ignores warpers and HF warns about non-default values there.)
            kw.update(
                temperature=float(temperature),
                top_k=0, top_p=1.0, typical_p=1.0, min_p=None,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
            )
        return GenerationConfig(**kw)

    def generate(self, adapter, system, user, n=1, temperature=0.0, max_new_tokens=2048) -> list[str]:
        """Sample ``n`` completions.

        Also sets, for the rollouts just produced:

        * ``self.last_generation_ids`` — the exact completion token ids, which GRPO
          feeds back into the log-prob pass instead of re-tokenizing the decoded
          text (a BPE round-trip is not guaranteed to reproduce them).
        * ``self.last_generation_completed`` — one bool per rollout: did it emit EOS
          on its own, or hit ``max_new_tokens``?

        Both are overwritten by the next ``generate`` call, including the opponent's
        during reward scoring, so a caller must read them immediately."""
        self.set_adapter(adapter)
        prompt_ids = self._prompt_ids(system, user)
        plen = prompt_ids.shape[1]
        do_sample = bool(temperature and temperature > 0)

        # Prefer KV-cached generation (much faster). With Unsloth's fused fast
        # wrapper disabled (see __init__), this uses standard Transformers
        # generate + a normal KV cache, avoiding the broken paged-KV kernel. If
        # it still fails on this Unsloth/Transformers combo, fall back ONCE to
        # the manual no-cache decoder and stay there for the rest of the run.
        if self._use_fast_generate:
            try:
                return self._fast_generate(prompt_ids, plen, n, do_sample, temperature, max_new_tokens)
            except Exception as e:
                logger.warning(
                    f"KV-cached generate failed ({type(e).__name__}: {e}); "
                    "falling back to manual no-cache decode for the rest of the run."
                )
                self._use_fast_generate = False
        return self._manual_generate(prompt_ids, plen, n, do_sample, temperature, max_new_tokens)

    def _fast_generate(self, prompt_ids, plen, n, do_sample, temperature, max_new_tokens):
        """KV-cached generation via STANDARD Transformers generate.

        Unsloth's fused fast-generate wrapper is disabled
        (UNSLOTH_DISABLE_FAST_GENERATION, see __init__), so this uses the regular
        forward + a normal KV cache. We switch to eval() for the call so
        gradient checkpointing doesn't force use_cache=False, then restore
        train() for the GRPO backward. We deliberately do NOT call for_inference()
        — that re-enables the broken paged-KV inference kernel.

        The whole group is sampled in ONE batched call (``num_return_sequences=n``)
        instead of an n-iteration Python loop: n independent rollouts share a
        single prefill + decode loop, which is the dominant per-round cost. The
        single prompt is expanded internally, so every output row carries the
        identical prompt prefix and ``[plen:]`` is its completion. Greedy decoding
        is deterministic — the n rollouts would be identical — so we generate one
        and replicate (this also sidesteps HF's "num_return_sequences>1 requires
        sampling" error for the n>1 greedy case).

        Sampling shape comes from ``_sampling_config`` — a FRESH GenerationConfig, so
        the model's chat-tuned generation defaults cannot leak into the behaviour
        policy.
        """
        torch = self.torch
        was_training = self.model.training
        try:
            self.model.eval()
            gen_cfg = self._sampling_config(
                n if do_sample else 1, do_sample, temperature, max_new_tokens)
            with torch.no_grad():
                out = self.model.generate(prompt_ids, generation_config=gen_cfg)
            # Exact sampled ids per row, cut at the first EOS; texts are decoded FROM
            # those ids so text and ids always describe the same sequence.
            ids, completed = self._split_generated(out[:, plen:])
            texts = [self._tok.decode(row, skip_special_tokens=True) for row in ids]
            if not do_sample:
                # deterministic greedy -> replicate the single completion
                texts, ids, completed = texts * n, ids * n, completed * n
            self.last_generation_ids = ids
            self.last_generation_completed = completed
            # No empty_cache() here. Freeing `out` returns the KV cache to PyTorch's
            # caching allocator, which reuses those blocks for the GRPO backward
            # within this same process; empty_cache() only hands them back to the
            # driver, forcing fresh cudaMallocs, and it synchronizes — a real cost
            # G+ times per round. If fragmentation ever does bite, the lever is
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, not this.
            del out
            return texts
        finally:
            if was_training:
                self.model.train()   # restore training mode for the GRPO backward

    def _manual_generate(self, prompt_ids, plen, n, do_sample, temperature, max_new_tokens):
        """Decode WITHOUT a KV cache (full forward each step). Slower (O(L^2))
        but uses the same forward path as the log-prob pass, so it works even
        when Unsloth's fused inference kernel is incompatible.

        Batched: all rollouts decode together as an ``(n_gen, L)`` tensor — one
        forward per step instead of one per (rollout × step). ``_last_logits``
        already returns the last-position logits for every row, so the sampling
        math is unchanged, just vectorised over the batch. Per-row EOS is tracked
        with a ``finished`` mask; once a row finishes it emits ``pad`` so the batch
        stays rectangular and the trailing pad is stripped on decode. The loop
        ends when every row has hit EOS. Greedy is deterministic → generate one
        row and replicate.
        """
        torch = self.torch
        eos_ids = self._eos_ids()
        eos_tensor = (
            torch.tensor(sorted(eos_ids), device=self.device) if eos_ids else None
        )
        pad_id = self._tok.pad_token_id
        n_gen = n if do_sample else 1   # greedy is deterministic → 1 then replicate

        ids = prompt_ids.repeat(n_gen, 1)                          # (n_gen, plen)
        finished = torch.zeros(n_gen, dtype=torch.bool, device=self.device)
        for _step in range(max_new_tokens):
            with torch.no_grad():
                logits = self._last_logits(ids)                    # (n_gen, vocab)
            if do_sample:
                # UNTRUNCATED sampling from the tempered softmax. The old top-k=50
                # truncation made this path sample from a different distribution than
                # both _fast_generate and the log-prob pass, so which sampler ran (a
                # silent runtime fallback) changed the gradient. Now every path is the
                # plain tempered softmax that _completion_token_logprobs scores.
                probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
                nxt = torch.multinomial(probs, 1)                   # (n_gen, 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)           # (n_gen, 1)
            if eos_tensor is not None:
                # Rows already done keep emitting pad → rectangular + stripped on decode.
                nxt = nxt.masked_fill(finished.unsqueeze(1), pad_id)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_tensor is not None:
                finished = finished | torch.isin(nxt.squeeze(1), eos_tensor)
                if bool(finished.all()):
                    break
        # Exact sampled ids per row, cut at the first EOS (which also drops the pad
        # filler this loop emits after a row finishes); texts decode FROM those ids.
        gen_ids, completed = self._split_generated(ids[:, plen:])
        texts = [self._tok.decode(row, skip_special_tokens=True) for row in gen_ids]
        if not do_sample:
            # deterministic greedy -> replicate the single completion
            texts, gen_ids, completed = texts * n, gen_ids * n, completed * n
        self.last_generation_ids = gen_ids
        self.last_generation_completed = completed
        del ids                      # see _fast_generate: no empty_cache() needed
        return texts

    def _completion_ids(self, completion, completion_ids, append_eos):
        """The completion's token ids as a ``(1, L)`` tensor.

        ``completion_ids`` — the ids ``generate`` actually sampled — is the CORRECT
        input and is what GRPO passes. Re-tokenizing the decoded text (the fallback,
        for generators that cannot supply ids, e.g. the frozen inference backends)
        is only approximately inverse: BPE round-tripping is not guaranteed to
        reproduce the sampled ids, and these completions are JSON full of digits,
        decimals and whitespace runs — exactly where re-merge differs. When it does
        differ, GRPO computes log-probs for a sequence the policy never generated,
        which silently voids the ``ratio == 1`` identity the loss is built on.

        ``append_eos`` applies to the TEXT path only: decoding strips special tokens,
        so the stop token has to be re-attached or the policy gets no gradient on the
        decision to stop (and drifts toward rambling past ``max_new_tokens``). Callers
        pass it only for rollouts that terminated on their own. The ids path needs
        none of this — ``_split_generated`` keeps the real stop token in place.
        """
        torch = self.torch
        if completion_ids is not None:
            return completion_ids.reshape(1, -1).to(self.device)
        comp_ids = self._tok(
            completion, add_special_tokens=False, return_tensors="pt",
        ).input_ids.to(self.device)
        if append_eos and self._tok.eos_token_id is not None:
            eos = torch.tensor([[int(self._tok.eos_token_id)]], device=self.device)
            comp_ids = torch.cat([comp_ids, eos], dim=1)
        return comp_ids

    def _completion_token_logprobs(self, system, user, completion, with_grad,
                                   append_eos=False, completion_ids=None,
                                   temperature=1.0):
        """Per-token log-probs of a completion given the prompt.

        Returns a 1-D tensor (one entry per completion token). Differentiable
        when ``with_grad`` and an adapter is active; used both for the policy
        term (grad) and, under ``disable_adapter`` + no_grad, the KL reference.

        ``temperature`` MUST be the temperature the rollout was sampled at. The
        policy GRPO optimizes is ``softmax(logits / T)`` — the distribution the
        sampler drew from — so scoring at a fixed T=1 while sampling at any other
        temperature makes the gradient an estimate of the wrong objective. This
        happened to be harmless while ``rl.temperature`` was exactly 1.0, but the
        zero-advantage resample path re-draws at ``resample_temperature`` (1.3), and
        lowering ``rl.temperature`` would have quietly biased every update with no
        error anywhere. Applied to the reference pass too, so the KL compares two
        distributions at the same temperature.
        """
        torch = self.torch
        prompt_ids = self._prompt_ids(system, user)
        comp_ids = self._completion_ids(completion, completion_ids, append_eos)

        if comp_ids.shape[1] == 0:
            # Empty completion -> zero-length logprob vector.
            return torch.zeros(0, device=self.device)

        full = torch.cat([prompt_ids, comp_ids], dim=1)
        # Guard against exceeding context.
        if full.shape[1] > self.max_seq_len:
            full = full[:, -self.max_seq_len:]
        n_pred = min(comp_ids.shape[1], full.shape[1] - 1)
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            # Only the positions that PREDICT the completion matter, so ask the model
            # for just those (n_pred + 1, then drop the final position which predicts
            # past the end). Previously this ran the vocab projection over the whole
            # prompt+completion and took log_softmax over all of it in fp32 — at
            # Qwen2.5's ~152k vocab that is gigabytes of logits per rollout, retained
            # for the backward, of which the prompt part was immediately discarded.
            logits = self._forward_logits(full, n_pred + 1)[:, :-1, :]
            if float(temperature) != 1.0:
                logits = logits / max(float(temperature), 1e-6)
            logp = torch.log_softmax(logits.float(), dim=-1)
            targets = full[:, -n_pred:]
            tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        return tok_logp

    def policy_token_logprobs(self, adapter, system, user, completion,
                              append_eos=False, completion_ids=None,
                              temperature=1.0):
        """Differentiable per-token log-probs under ``adapter``."""
        self.set_adapter(adapter)
        return self._completion_token_logprobs(
            system, user, completion, with_grad=True, append_eos=append_eos,
            completion_ids=completion_ids, temperature=temperature)

    def reference_token_logprobs(self, system, user, completion, append_eos=False,
                                 completion_ids=None, temperature=1.0):
        """No-grad per-token log-probs under the BASE model (KL reference).

        ``append_eos`` / ``completion_ids`` / ``temperature`` must match the policy
        pass so the two sequences line up token-for-token and the KL is computed over
        the same positions of the same distribution family.
        """
        with self.model.disable_adapter():
            return self._completion_token_logprobs(
                system, user, completion, with_grad=False, append_eos=append_eos,
                completion_ids=completion_ids, temperature=temperature).detach()


class PolicyGenerator:
    """Adapt an ``LLMPolicy`` + fixed adapter to the turn-generator interface."""

    def __init__(self, policy: LLMPolicy, adapter: str, max_new_tokens: int = 2048):
        self.policy = policy
        self.adapter = adapter
        self.max_new_tokens = max_new_tokens

    def generate(self, system: str, user: str, n: int = 1, temperature: float = 0.0) -> list[str]:
        return self.policy.generate(
            self.adapter, system, user, n=n, temperature=temperature,
            max_new_tokens=self.max_new_tokens,
        )
