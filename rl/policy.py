"""LLMPolicy — one frozen Llama-3.2-3B-Instruct base, two trainable LoRA adapters.

This is the only module with heavy GPU dependencies (unsloth / peft /
transformers / torch + a CUDA box). It is imported only on the training path;
the dry-run path uses ``rl/inference.py`` instead, so the rest of the package
stays importable on a CPU machine.

Design — "separate checkpoints on the same LLM":
  * One 4-bit (QLoRA) Llama-3.2-3B-Instruct base, loaded once and frozen.
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
# Llama 3.x, Gemma, Qwen, Mistral, etc. Qwen2.5 uses these exact names, so the
# LoRA setup carries over unchanged from the Llama base.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Default base model. Qwen2.5-1.5B-Instruct is ~2x smaller than Llama-3.2-3B
# (faster per token for the short JSON actions both agents emit) and shares the
# Llama-style architecture, so the LoRA target modules, chat-template folding,
# and eos/pad handling below all apply without change.
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"


def first_json_object_end(text: str) -> int | None:
    """Index just past the end of the FIRST balanced top-level ``{...}`` object.

    Returns the character index one past the closing brace of the first complete
    JSON object in ``text``, or ``None`` if no complete object is present yet.
    String literals (and their escapes) are respected so that braces appearing
    inside a string value do not affect the depth count. Text before the first
    ``{`` (e.g. a ```json fence or a "Here is:" preamble) and text after the
    closing ``}`` are ignored — both attacker and defender emit a single
    top-level object, so this marks exactly where generation can stop.

    Pure function (no torch / no model) so the structured-decoding stop condition
    is unit-testable on any machine.
    """
    depth = 0
    in_str = False
    escape = False
    started = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
            started = True
        elif ch == "}":
            depth -= 1
            if started and depth == 0:
                return i + 1
    return None


class LLMPolicy:
    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
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
        unsloth_fast_generation: bool = True,
        stop_on_json: bool = True,
        gradient_checkpointing: bool = True,
    ):
        # Heavy imports kept local so importing this module is cheap.
        import torch
        # Unsloth's fused fast-generation kernel (for_inference) is the fastest
        # sampling path. It was previously force-disabled because an older Unsloth
        # (2026.3.11) shipped a broken Llama paged-KV kernel (RoPE cos/sin
        # broadcast crash) vs Transformers 5.3; requirements now pin
        # unsloth>=2026.6.9, which fixes it. The env var MUST be set before
        # importing unsloth, so gate it on the config flag here. When disabled we
        # fall back to standard Transformers generate + a normal KV cache (still
        # correct, just slower).
        os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "0" if unsloth_fast_generation else "1"
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
        # errors some architectures hit). Llama 3.2 also works with "sdpa".
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
            # Gradient checkpointing trades compute (a recomputed forward in the
            # backward) for memory. On a small model with VRAM headroom, disabling
            # it speeds up the GRPO backward. Does NOT affect adapter saving.
            use_gradient_checkpointing=("unsloth" if gradient_checkpointing else False),
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
        # Text-only models (e.g. Llama 3.2) return the tokenizer directly. Keep
        # both handles: ``self.tokenizer`` renders, ``self._tok`` tokenizes — for
        # a plain tokenizer the getattr fallback makes them the same object.
        self._tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token

        self.device = next(self.model.parameters()).device
        self.model.train()
        self._active = None
        self._logits_kw = None   # resolved on first use: which "last-token-only" kwarg works
        self._use_fast_generate = bool(use_fast_generate)  # KV-cached generate; auto-falls back on failure
        self._FLM = FastLanguageModel        # for for_inference()/for_training() toggles
        self._unsloth_fast = bool(unsloth_fast_generation)  # engage Unsloth's fused inference kernel
        self._stop_on_json = bool(stop_on_json)             # stop at the first complete JSON object
        self.set_adapter(self.adapters[0])
        logger.info(
            f"LLMPolicy ready — base={base_model}, adapters={self.adapters}, device={self.device}, "
            f"unsloth_fast={self._unsloth_fast}, stop_on_json={self._stop_on_json}, "
            f"grad_checkpointing={gradient_checkpointing}"
        )

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
    def _render_text(self, system: str, user: str) -> str:
        # Fold the system instructions into a single user turn. Llama 3.2 and
        # Qwen2.5 both have a native system role, but some templates (e.g. Gemma)
        # don't — folding is the universally-safe approach and keeps behaviour
        # identical across models. Render to text via the processor's chat
        # template (tokenize=False → str). The template injects BOS + turn tokens.
        messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )

    def _prompt_ids(self, system: str, user: str):
        # Tokenize the rendered prompt with the raw tokenizer. The template already
        # injects BOS + turn tokens, so add_special_tokens=False avoids a duplicate BOS.
        text = self._render_text(system, user)
        enc = self._tok(text, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"].to(self.device)

    def _last_logits(self, ids):
        """Logits for the LAST position only — skips the 128k-vocab projection
        over every position, a big saving in the no-KV-cache decode loop. Falls
        back gracefully if the model's forward doesn't accept the kwarg."""
        if self._logits_kw is None:
            for kw in ("logits_to_keep", "num_logits_to_keep"):
                try:
                    out = self.model(ids, **{kw: 1})
                    self._logits_kw = kw
                    return out.logits[:, -1, :].float()
                except TypeError:
                    continue
            self._logits_kw = ""  # unsupported → full logits
        if self._logits_kw:
            out = self.model(ids, **{self._logits_kw: 1})
        else:
            out = self.model(ids)
        return out.logits[:, -1, :].float()

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

    def generate(self, adapter, system, user, n=1, temperature=0.0, max_new_tokens=2048) -> list[str]:
        self.set_adapter(adapter)
        prompt_ids = self._prompt_ids(system, user)
        plen = prompt_ids.shape[1]
        do_sample = bool(temperature and temperature > 0)

        # Prefer KV-cached generation (much faster): Unsloth's fused fast-inference
        # kernel when unsloth_fast_generation is on, else standard Transformers
        # generate + a normal KV cache. If it fails on this Unsloth/Transformers
        # combo, fall back ONCE to the manual no-cache decoder and stay there for
        # the rest of the run.
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
        """KV-cached generation.

        When ``unsloth_fast_generation`` is on we engage Unsloth's fused
        fast-inference kernel via ``for_inference()`` (restored with
        ``for_training()`` in ``finally`` for the GRPO backward); otherwise we just
        switch to eval() so gradient checkpointing doesn't force use_cache=False and
        let standard Transformers generate + a normal KV cache run. If
        ``for_inference()`` itself errors we degrade to the plain eval() path.

        The whole group is sampled in ONE batched call (``num_return_sequences=n``)
        instead of an n-iteration Python loop: n independent rollouts share a
        single prefill + decode loop, which is the dominant per-round cost. The
        single prompt is expanded internally, so every output row carries the
        identical prompt prefix and ``[plen:]`` is its completion. Greedy decoding
        is deterministic — the n rollouts would be identical — so we generate one
        and replicate (this also sidesteps HF's "num_return_sequences>1 requires
        sampling" error for the n>1 greedy case).
        """
        torch = self.torch
        was_training = self.model.training
        n_rows = n if do_sample else 1
        used_unsloth = False
        try:
            # Engage Unsloth's fused fast-inference kernel when enabled; otherwise
            # just switch to eval() so gradient checkpointing doesn't force
            # use_cache=False. Either way we restore training mode in `finally` for
            # the GRPO backward.
            if self._unsloth_fast:
                try:
                    self._FLM.for_inference(self.model)
                    used_unsloth = True
                except Exception:
                    self.model.eval()
            else:
                self.model.eval()
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens, do_sample=do_sample, use_cache=True,
                pad_token_id=self._tok.pad_token_id,
                num_return_sequences=n_rows,
            )
            if do_sample:
                gen_kwargs.update(temperature=float(temperature), top_p=0.95)
            if self._stop_on_json:
                gen_kwargs["stopping_criteria"] = self._json_stopping_criteria(plen, n_rows)
            with torch.no_grad():
                out = self.model.generate(prompt_ids, **gen_kwargs)
            texts = [
                self._tok.decode(out[i, plen:], skip_special_tokens=True)
                for i in range(out.shape[0])
            ]
            if not do_sample:
                texts = texts * n   # deterministic greedy → replicate the single completion
            del out
            return texts
        finally:
            if used_unsloth:
                try:
                    self._FLM.for_training(self.model)
                except Exception:
                    pass
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
                logits = logits / max(float(temperature), 1e-6)
                k = min(50, logits.shape[-1])
                top = torch.topk(logits, k, dim=-1)
                probs = torch.softmax(top.values, dim=-1)
                nxt = top.indices.gather(-1, torch.multinomial(probs, 1))  # (n_gen, 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)           # (n_gen, 1)
            # Rows already done keep emitting pad → rectangular + stripped on decode.
            nxt = nxt.masked_fill(finished.unsqueeze(1), pad_id)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_tensor is not None:
                finished = finished | torch.isin(nxt.squeeze(1), eos_tensor)
            if self._stop_on_json:
                # Structured stop: end a row once its completion holds a complete
                # top-level JSON object (matches the fast-path StoppingCriteria).
                finished = self._json_finished(ids, plen, finished)
            if bool(finished.all()):
                break
        texts = [
            self._tok.decode(ids[i, plen:], skip_special_tokens=True)
            for i in range(n_gen)
        ]
        if not do_sample:
            texts = texts * n   # deterministic greedy → replicate the single completion
        del ids
        return texts

    def _json_stopping_criteria(self, plen: int, n_rows: int):
        """A StoppingCriteria that stops generation once EVERY row's completion
        (the tokens after ``plen``) contains a complete top-level JSON object.

        We track per-row completion but return a scalar ``bool`` (True only when
        all rows are done) rather than a per-row BoolTensor: the scalar form is
        accepted by every transformers version, whereas a per-row return that a
        given version rejected would raise and force the whole run onto the slow
        no-cache decoder. Rows finish at similar lengths here (same-shaped verdict
        lists / plans), so stopping at the batch max costs little."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        torch = self.torch
        tok = self._tok

        class _JsonObjectStop(StoppingCriteria):
            def __init__(self):
                self.done = torch.zeros(n_rows, dtype=torch.bool)

            def __call__(self, input_ids, scores=None, **kwargs):
                b = input_ids.shape[0]
                if self.done.shape[0] != b:                    # be robust to batch expansion
                    self.done = torch.zeros(b, dtype=torch.bool)
                # Cheap gate: only fully decode a row when its newest token has '}'.
                last_texts = tok.batch_decode(input_ids[:, -1:], skip_special_tokens=True)
                for r in range(b):
                    if bool(self.done[r]) or "}" not in last_texts[r]:
                        continue
                    comp = tok.decode(input_ids[r, plen:], skip_special_tokens=True)
                    if first_json_object_end(comp) is not None:
                        self.done[r] = True
                return bool(self.done.all())

        return StoppingCriteriaList([_JsonObjectStop()])

    def _json_finished(self, ids, plen, finished):
        """Manual-decode analogue of the StoppingCriteria: mark rows whose
        completion now holds a complete top-level JSON object."""
        torch = self.torch
        last_texts = self._tok.batch_decode(ids[:, -1:], skip_special_tokens=True)
        newly = []
        for r in range(ids.shape[0]):
            if bool(finished[r]) or "}" not in last_texts[r]:
                continue
            comp = self._tok.decode(ids[r, plen:], skip_special_tokens=True)
            if first_json_object_end(comp) is not None:
                newly.append(r)
        if newly:
            finished = finished.index_fill(0, torch.tensor(newly, device=finished.device), True)
        return finished

    def generate_many(self, adapter, prompts, temperature=0.0, max_new_tokens=2048) -> list[str]:
        """Generate ONE completion for EACH ``(system, user)`` prompt in a single
        left-padded batch.

        Used to score all G attacker rollouts' frozen-defender responses in one
        forward/decode loop instead of G sequential ``generate`` calls — the main
        cost on attacker-learning rounds. Uses standard Transformers generation
        with an attention mask (correct for left-padded, unequal-length prompts);
        on any failure it falls back to sequential single-prompt generation so the
        run never dies on a batching edge case.
        """
        if not prompts:
            return []
        self.set_adapter(adapter)
        texts_in = [self._render_text(s, u) for (s, u) in prompts]
        tok = self._tok
        prev_side = tok.padding_side
        tok.padding_side = "left"   # decoder-only batched generation needs left padding
        try:
            enc = tok(texts_in, return_tensors="pt", add_special_tokens=False, padding=True)
        finally:
            tok.padding_side = prev_side
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        if input_ids.shape[1] > self.max_seq_len:            # guard context length
            input_ids = input_ids[:, -self.max_seq_len:]
            attn = attn[:, -self.max_seq_len:]
        do_sample = bool(temperature and temperature > 0)
        try:
            return self._batched_generate(input_ids, attn, do_sample, temperature, max_new_tokens)
        except Exception as e:
            logger.warning(
                f"Batched generate_many failed ({type(e).__name__}: {e}); "
                "falling back to sequential single-prompt generation."
            )
            return [
                self.generate(adapter, s, u, n=1, temperature=temperature,
                              max_new_tokens=max_new_tokens)[0]
                for (s, u) in prompts
            ]

    def _batched_generate(self, input_ids, attn, do_sample, temperature, max_new_tokens):
        """One batched generate over a left-padded prompt batch (num_return_sequences=1
        per row). All rows share the padded prompt length, so each completion is
        ``row[plen:]``."""
        torch = self.torch
        was_training = self.model.training
        plen = input_ids.shape[1]
        n_rows = input_ids.shape[0]
        try:
            self.model.eval()
            gen_kwargs = dict(
                attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=do_sample,
                use_cache=True, pad_token_id=self._tok.pad_token_id, num_return_sequences=1,
            )
            if do_sample:
                gen_kwargs.update(temperature=float(temperature), top_p=0.95)
            if self._stop_on_json:
                gen_kwargs["stopping_criteria"] = self._json_stopping_criteria(plen, n_rows)
            with torch.no_grad():
                out = self.model.generate(input_ids, **gen_kwargs)
            texts = [
                self._tok.decode(out[i, plen:], skip_special_tokens=True)
                for i in range(out.shape[0])
            ]
            del out
            return texts
        finally:
            if was_training:
                self.model.train()

    def _completion_token_logprobs(self, system, user, completion, with_grad):
        """Per-token log-probs of ``completion`` given the prompt.

        Returns a 1-D tensor (one entry per completion token). Differentiable
        when ``with_grad`` and an adapter is active; used both for the policy
        term (grad) and, under ``disable_adapter`` + no_grad, the KL reference.
        """
        torch = self.torch
        prompt_ids = self._prompt_ids(system, user)
        comp_ids = self._tok(
            completion, add_special_tokens=False, return_tensors="pt",
        ).input_ids.to(self.device)

        if comp_ids.shape[1] == 0:
            # Empty completion → zero-length logprob vector.
            return torch.zeros(0, device=self.device)

        full = torch.cat([prompt_ids, comp_ids], dim=1)
        # Guard against exceeding context.
        if full.shape[1] > self.max_seq_len:
            full = full[:, -self.max_seq_len:]
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            logits = self.model(full).logits[:, :-1, :]
            logp = torch.log_softmax(logits.float(), dim=-1)
            targets = full[:, 1:]
            tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        comp_len = comp_ids.shape[1]
        return tok_logp[-comp_len:]

    def policy_token_logprobs(self, adapter, system, user, completion):
        """Differentiable per-token log-probs under ``adapter``."""
        self.set_adapter(adapter)
        return self._completion_token_logprobs(system, user, completion, with_grad=True)

    def reference_token_logprobs(self, system, user, completion):
        """No-grad per-token log-probs under the BASE model (KL reference)."""
        with self.model.disable_adapter():
            return self._completion_token_logprobs(system, user, completion, with_grad=False).detach()

    # ------------------------------------------------------------------
    # Batched log-probs (all G completions in one forward)
    # ------------------------------------------------------------------
    @staticmethod
    def _slice_completion_logprobs(tok_logp, P, lengths):
        """From per-position next-token log-probs ``(G, T-1)`` pull each row's
        completion tokens. For row ``i`` the completion occupies absolute positions
        ``[P, P+Ci)``, whose log-probs are ``tok_logp[i, P-1 : P-1+Ci]`` (the token
        at position ``k`` is predicted by the logits at ``k-1``). Right padding sits
        AFTER the completion, so we slice from ``P-1`` — never the tail."""
        out = []
        for i, Ci in enumerate(lengths):
            if Ci == 0:
                out.append(tok_logp.new_zeros(0))
            else:
                out.append(tok_logp[i, P - 1:P - 1 + Ci])
        return out

    def _batched_completion_logprobs(self, system, user, completions, with_grad):
        """Per-token completion log-probs for a GROUP of completions in ONE forward.

        Every completion shares the same prompt, so we build a single right-padded
        ``(G, P + maxC)`` batch and run one forward instead of one per completion —
        the policy + KL passes go from ``2*G`` forwards to ``2``. Right padding plus
        an attention mask keeps each completion's log-probs identical to the
        unbatched result: a causal model's real-token logits never depend on the
        pad tokens that follow them. Returns a list of 1-D tensors (one per
        completion, variable length; empty completion → length 0), differentiable
        when ``with_grad``.
        """
        torch = self.torch
        if not completions:
            return []
        prompt_ids = self._prompt_ids(system, user)                        # (1, P)
        P = prompt_ids.shape[1]
        comp_ids = [
            self._tok(c, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            for c in completions
        ]
        lengths = [c.shape[1] for c in comp_ids]
        maxC = max(lengths)
        if maxC == 0:
            return [torch.zeros(0, device=self.device) for _ in completions]
        T = P + maxC
        if T > self.max_seq_len:
            # Rare: prompt + longest completion exceeds context. Signal the caller
            # to use the per-sample path (which left-truncates safely).
            raise RuntimeError(f"batched logprob length {T} > max_seq_len {self.max_seq_len}")

        G = len(completions)
        pad_id = self._tok.pad_token_id if self._tok.pad_token_id is not None else 0
        full = torch.full((G, T), pad_id, dtype=prompt_ids.dtype, device=self.device)
        attn = torch.zeros((G, T), dtype=torch.long, device=self.device)
        full[:, :P] = prompt_ids                                           # same prompt for every row
        for i, c in enumerate(comp_ids):
            Ci = lengths[i]
            if Ci:
                full[i, P:P + Ci] = c[0]
            attn[i, :P + Ci] = 1

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            logits = self.model(full, attention_mask=attn).logits[:, :-1, :]   # (G, T-1, V)
            targets = full[:, 1:]                                              # (G, T-1)
            # log p(target) = logit[target] - logsumexp(logits), i.e. exactly
            # log_softmax(logits)[target], computed without materializing a second
            # full (G, T-1, V) tensor.
            tgt = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float() # (G, T-1)
            lse = torch.logsumexp(logits.float(), dim=-1)                      # (G, T-1)
            tok_logp = tgt - lse                                               # (G, T-1)
        return self._slice_completion_logprobs(tok_logp, P, lengths)

    def batched_policy_logprobs(self, adapter, system, user, completions):
        """Differentiable per-token log-probs for ALL completions under ``adapter``
        in one forward. Falls back to the per-sample path on any runtime error
        (e.g. CUDA OOM), so a batching edge case can't kill the run."""
        self.set_adapter(adapter)
        try:
            return self._batched_completion_logprobs(system, user, completions, with_grad=True)
        except RuntimeError as e:
            logger.warning(
                f"Batched policy logprobs failed ({type(e).__name__}: {e}); per-sample fallback."
            )
            return [
                self._completion_token_logprobs(system, user, c, with_grad=True)
                for c in completions
            ]

    def batched_reference_logprobs(self, system, user, completions):
        """No-grad per-token log-probs for ALL completions under the BASE model
        (the KL reference) in one forward."""
        with self.model.disable_adapter():
            try:
                lps = self._batched_completion_logprobs(system, user, completions, with_grad=False)
            except RuntimeError as e:
                logger.warning(
                    f"Batched reference logprobs failed ({type(e).__name__}: {e}); per-sample fallback."
                )
                lps = [
                    self._completion_token_logprobs(system, user, c, with_grad=False)
                    for c in completions
                ]
        return [lp.detach() for lp in lps]


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

    def generate_many(self, prompts, temperature: float = 0.0) -> list[str]:
        """One completion per (system, user) prompt, generated in a single batch."""
        return self.policy.generate_many(
            self.adapter, prompts, temperature=temperature, max_new_tokens=self.max_new_tokens,
        )
