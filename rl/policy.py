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
# Llama 3.x, Gemma, Qwen, Mistral, etc.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class LLMPolicy:
    def __init__(
        self,
        base_model: str = "unsloth/Llama-3.2-3B-Instruct",
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
        from unsloth import FastLanguageModel
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
        # Fold the system instructions into a single user turn. Llama 3.2 has a
        # native system role, but some templates (e.g. Gemma) don't — folding is
        # the universally-safe approach and keeps behaviour identical across models.
        # Render to text via the processor (tokenize=False → str), then tokenize
        # with the raw tokenizer. The template already injects BOS + turn tokens,
        # so add_special_tokens=False avoids a duplicate BOS.
        messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
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

        # Prefer Unsloth's KV-cached fast generation (much faster). It requires
        # switching the model into inference mode first — which also initializes
        # the per-token RoPE buffers whose absence caused the cos/sin broadcast
        # crash when calling generate() straight from training mode. If it still
        # fails on this Unsloth/Transformers combo, fall back ONCE to the manual
        # no-cache decoder and stay there for the rest of the run.
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
        """Unsloth KV-cached generation; toggles inference<->training mode."""
        from unsloth import FastLanguageModel
        torch = self.torch
        try:
            FastLanguageModel.for_inference(self.model)   # enable cached fast path
            texts = []
            for _ in range(n):
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens, do_sample=do_sample, use_cache=True,
                    num_return_sequences=1, pad_token_id=self._tok.pad_token_id,
                )
                if do_sample:
                    gen_kwargs.update(temperature=float(temperature), top_p=0.95)
                with torch.no_grad():
                    out = self.model.generate(prompt_ids, **gen_kwargs)
                texts.append(self._tok.decode(out[0, plen:], skip_special_tokens=True))
                del out
                torch.cuda.empty_cache()
            return texts
        finally:
            # ALWAYS restore training mode so the GRPO log-prob/backward works.
            FastLanguageModel.for_training(self.model)

    def _manual_generate(self, prompt_ids, plen, n, do_sample, temperature, max_new_tokens):
        """Decode WITHOUT a KV cache (full forward each step). Slower (O(L^2))
        but uses the same forward path as the log-prob pass, so it works even
        when Unsloth's fused inference kernel is incompatible."""
        torch = self.torch
        eos_ids = self._eos_ids()
        texts = []
        for _ in range(n):
            ids = prompt_ids.clone()
            for _step in range(max_new_tokens):
                with torch.no_grad():
                    logits = self._last_logits(ids)
                if do_sample:
                    logits = logits / max(float(temperature), 1e-6)
                    k = min(50, logits.shape[-1])
                    top = torch.topk(logits, k, dim=-1)
                    probs = torch.softmax(top.values, dim=-1)
                    nxt = top.indices.gather(-1, torch.multinomial(probs, 1))
                else:
                    nxt = logits.argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, nxt], dim=1)
                if int(nxt.item()) in eos_ids:
                    break
            texts.append(self._tok.decode(ids[0, plen:], skip_special_tokens=True))
            del ids
            torch.cuda.empty_cache()
        return texts

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
