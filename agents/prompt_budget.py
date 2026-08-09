"""PromptBudget — how much of the LLM's context one call is allowed to fill.

A small instruct model degrades well before it runs out of context: the longer
the prompt, the more the instructions at the top compete with the data in the
middle, and the attacker's prompt is mostly *data* (per-layer update statistics
for every client in the controllable pool). With the shipped 8k ``rl.max_seq_len``
a 10-client pool in the old verbose JSON encoding measured ~5.4k prompt tokens,
which with ``rl.max_new_tokens: 1024`` reserved is ~79% of the window — the
regime where the attacker starts emitting truncated or malformed plans.

So every attacker call is held to a **context-fill cap** (``rl.max_context_fill``,
0.5 by default)::

    prompt_tokens + reserved_output_tokens  <=  max_context_fill * context_window

``reserved_output_tokens`` is ``rl.max_new_tokens``: the completion shares the
same window, so a prompt that fits only because nothing was generated is not
actually within budget. Callers that exceed the cap COMPACT the payload (see
``AttackerAgent._COMPACTION``) rather than truncating it — a truncated JSON
observation is worse than a coarser complete one.

A second, optional cap bounds the **prompt on its own** (``max_prompt_fill``)::

    prompt_tokens  <=  max_prompt_fill * context_window

Both caps apply; ``prompt_limit`` is whichever binds first. The total cap alone
cannot express "the observation gets at most 30% of the window and the whole
call at most 60%", because it trades prompt room against generation room — a
tiny ``max_new_tokens`` would silently let the observation grow to fill the
entire call budget. Splitting them is what makes the defender's input size an
independent knob from its output size (see ``configs/base.yaml``'s
``rl.defender_max_prompt_fill`` / ``rl.defender_max_context_fill``).

``min_prompt_fill`` is the other side of the same band: a TARGET FLOOR, never
enforced. Under it, the observation is leaving usable context unspent, which is
worth saying out loud — the compaction ladder's job is to pick the richest
payload that fits, and a prompt far under the floor means the richest level is
poorer than it needs to be.

Token counting uses the real tokenizer when one is available (``bind`` it to
``LLMPolicy.count_prompt_tokens``, which renders the chat template exactly as
generation does). On the CPU / dry-run path there is no local tokenizer, so a
character-class heuristic stands in: dense JSON numerics tokenize at ~1.9
chars/token and English prose at ~3.2, both measured on this project's own
prompts. The heuristic is only ever used to DECIDE HOW MUCH TO COMPACT, never to
truncate, so an error of a few percent costs at worst one compaction level.
"""

import logging

logger = logging.getLogger(__name__)

#: Fraction of the context window one call may occupy (prompt + reserved output).
DEFAULT_MAX_FILL = 0.5

# Heuristic token densities, measured on this project's own attacker prompts
# (GPT-2 BPE; Qwen2.5's is within a few percent on the same text):
#   the JSON stats payload  -> 2.09 chars/token
#   the prose system prompt -> 3.69 chars/token
# Both are rounded DOWN here so the estimate errs toward "too many tokens".
_DENSE_CHARS_PER_TOKEN = 1.75
_PROSE_CHARS_PER_TOKEN = 3.0
_DENSE_CHARS = set('0123456789,.:;[]{}"-+_/\\|()*')

# Chat-template scaffolding (role headers, BOS/EOS) the real tokenizer includes
# and a raw character count does not.
_CHAT_OVERHEAD_TOKENS = 40


def estimate_tokens(*parts: str) -> int:
    """Tokenizer-free token estimate for one chat prompt built from ``parts``.

    Blends two densities by character class, because the attacker's prompt is a
    prose system message plus a numeric JSON payload and a single average over
    the two is wrong for both. Includes the chat template's own scaffolding.
    """
    dense = prose = 0
    for part in parts:
        if not part:
            continue
        d = sum(1 for ch in part if ch in _DENSE_CHARS)
        dense += d
        prose += len(part) - d
    if not (dense or prose):
        return 0
    return int(dense / _DENSE_CHARS_PER_TOKEN
               + prose / _PROSE_CHARS_PER_TOKEN) + _CHAT_OVERHEAD_TOKENS


class PromptBudget:
    """The context-fill cap for one agent's LLM calls.

    ``context_window`` is ``rl.max_seq_len`` (0/None = no known window, in which
    case the budget never binds and only reports). ``reserved_output`` is
    ``rl.max_new_tokens``. ``max_fill`` is the fraction of the window a single
    call may occupy, counting the reserved completion.

    ``max_prompt_fill`` optionally caps the PROMPT on its own (``None`` = only
    the total cap applies, which is the attacker's configuration).
    ``min_prompt_fill`` is a reporting-only floor marking the bottom of the
    intended prompt band; it never rejects a prompt.
    """

    def __init__(self, context_window: int | None = None,
                 max_fill: float = DEFAULT_MAX_FILL,
                 reserved_output: int = 0,
                 token_counter=None,
                 label: str = "prompt",
                 max_prompt_fill: float | None = None,
                 min_prompt_fill: float = 0.0):
        self.context_window = int(context_window or 0)
        self.max_fill = max(0.0, min(1.0, float(max_fill)))
        self.reserved_output = max(0, int(reserved_output))
        self.max_prompt_fill = (None if max_prompt_fill is None
                                else max(0.0, min(1.0, float(max_prompt_fill))))
        self.min_prompt_fill = max(0.0, min(1.0, float(min_prompt_fill or 0.0)))
        self.label = label
        self._token_counter = token_counter
        self._counter_failed = False
        self._degenerate_warned = False

    # ------------------------------------------------------------------
    def bind(self, token_counter) -> None:
        """Attach the real tokenizer: ``token_counter(system, user) -> int``.

        Pass ``LLMPolicy.count_prompt_tokens`` so the count covers the same
        chat-template rendering ``generate`` will use. Until this is called (or
        when it is never called, e.g. the Ollama/OpenAI dry-run path) the
        character heuristic is used instead.
        """
        self._token_counter = token_counter
        self._counter_failed = False

    @property
    def active(self) -> bool:
        """Is there a window to budget against at all?"""
        return self.context_window > 0

    @property
    def call_limit(self) -> int:
        """Max tokens (prompt + reserved completion) one call may occupy."""
        return int(self.context_window * self.max_fill) if self.active else 0

    @property
    def prompt_cap(self) -> int:
        """Max PROMPT tokens from ``max_prompt_fill`` alone (0 = no such cap)."""
        if not self.active or self.max_prompt_fill is None:
            return 0
        return int(self.context_window * self.max_prompt_fill)

    @property
    def prompt_floor(self) -> int:
        """Prompt tokens below which the call is leaving context unspent (0 = no floor)."""
        if not self.active or self.min_prompt_fill <= 0:
            return 0
        return int(self.context_window * self.min_prompt_fill)

    @property
    def prompt_limit(self) -> int:
        """Max PROMPT tokens: the tighter of the two caps.

        That is the call limit less the reserved completion, and — when
        ``max_prompt_fill`` is set — the prompt-only cap.

        A configuration that leaves no room for a prompt at all is a config
        error, not something to silently swallow, so it is reported once and
        floored at 1 (which just means "compact as hard as you can").
        """
        if not self.active:
            return 0
        limit = self.call_limit - self.reserved_output
        cap = self.prompt_cap
        if cap:
            limit = min(limit, cap)
        if limit < 1:
            if not self._degenerate_warned:
                self._degenerate_warned = True
                logger.error(
                    f"{self.label}: no room left for a prompt in a "
                    f"{self.context_window}-token context — rl.max_new_tokens "
                    f"({self.reserved_output}) against a {self.max_fill:.0%} call budget "
                    f"({self.call_limit} tokens)"
                    + (f", and a {self.max_prompt_fill:.0%} prompt cap ({cap} tokens)"
                       if cap else "")
                    + ". Raise rl.max_seq_len, lower rl.max_new_tokens, or raise the "
                      "fill caps."
                )
            return 1
        return limit

    # ------------------------------------------------------------------
    def count(self, system: str, user: str) -> int:
        """Prompt tokens for this (system, user) pair."""
        if self._token_counter is not None and not self._counter_failed:
            try:
                return int(self._token_counter(system, user))
            except Exception as e:                      # noqa: BLE001 - never break a round
                self._counter_failed = True
                logger.warning(
                    f"{self.label}: tokenizer count failed ({type(e).__name__}: {e}); "
                    f"falling back to the character estimate for the rest of the run."
                )
        return estimate_tokens(system, user)

    def fill(self, prompt_tokens: int) -> float:
        """Fraction of the context window this call occupies, completion included."""
        if not self.active:
            return 0.0
        return (int(prompt_tokens) + self.reserved_output) / self.context_window

    def prompt_fill(self, prompt_tokens: int) -> float:
        """Fraction of the context window the PROMPT alone occupies."""
        if not self.active:
            return 0.0
        return int(prompt_tokens) / self.context_window

    def fits(self, prompt_tokens: int) -> bool:
        """Is this prompt within the cap(s)? Always true with no known window."""
        return (not self.active) or int(prompt_tokens) <= self.prompt_limit

    def underfilled(self, prompt_tokens: int) -> bool:
        """Is this prompt BELOW the intended band, i.e. leaving context unspent?

        Reporting only — an under-filled prompt is always emitted as-is. It means
        the richest rung of the caller's detail ladder is poorer than the budget
        allows, which is a signal to put more into the observation, not an error.
        """
        floor = self.prompt_floor
        return bool(floor) and int(prompt_tokens) < floor

    def exact(self) -> bool:
        """Is the count coming from a real tokenizer rather than the heuristic?"""
        return self._token_counter is not None and not self._counter_failed

    def describe(self, prompt_tokens: int) -> str:
        """One-line fill report for a log."""
        n = int(prompt_tokens)
        how = "measured" if self.exact() else "estimated"
        if not self.active:
            return f"{self.label}: {n} tokens ({how}; no context window configured)"
        band = ""
        if self.max_prompt_fill is not None or self.prompt_floor:
            lo = f"{self.min_prompt_fill:.0%}" if self.prompt_floor else "0%"
            hi = (f"{self.max_prompt_fill:.0%}" if self.max_prompt_fill is not None
                  else f"{self.max_fill:.0%}")
            band = f", prompt {self.prompt_fill(n):.0%} of window (band {lo}-{hi})"
        return (f"{self.label}: {n} {how} prompt tokens + {self.reserved_output} reserved "
                f"for output = {self.fill(n):.0%} of {self.context_window} "
                f"(cap {self.max_fill:.0%}){band}")


def build_prompt_budget(rl_cfg: dict | None, label: str = "attacker prompt",
                        token_counter=None, prefix: str = "") -> PromptBudget:
    """Construct the budget described by the ``rl:`` config block.

    ``prefix`` selects a per-agent override family: ``prefix="defender_"`` reads
    ``defender_max_context_fill``, ``defender_max_new_tokens``,
    ``defender_max_prompt_fill`` and ``defender_min_prompt_fill``, each falling
    back to the un-prefixed key when unset. ``max_seq_len`` is never prefixed —
    the context window belongs to the shared base model, not to one agent.
    """
    rl_cfg = rl_cfg or {}

    def get(key, default=None):
        if prefix:
            v = rl_cfg.get(f"{prefix}{key}")
            if v is not None:
                return v
        v = rl_cfg.get(key)
        return default if v is None else v

    return PromptBudget(
        context_window=int(rl_cfg.get("max_seq_len", 0) or 0),
        max_fill=float(get("max_context_fill", DEFAULT_MAX_FILL)),
        reserved_output=int(get("max_new_tokens", 0) or 0),
        max_prompt_fill=(None if get("max_prompt_fill") is None
                         else float(get("max_prompt_fill"))),
        min_prompt_fill=float(get("min_prompt_fill", 0.0)),
        token_counter=token_counter,
        label=label,
    )
