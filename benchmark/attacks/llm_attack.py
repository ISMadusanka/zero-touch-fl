"""The trained attacker LLM as a benchmark attack — the system under test.

Wraps the existing inference path (``rl.policy.LLMPolicy`` + ``AttackerAgent``)
in the same :class:`~benchmark.attacks.base.Attack` interface the published
baselines implement, so one harness can put the learned policy and the literature
side by side over identical rounds.

It differs from the baselines in one structural way: the LLM attacker *chooses
which* clients to poison as part of its action, while a baseline is simply handed
a set. The harness therefore resolves this attack FIRST each round
(:meth:`LLMAttack.propose`) and feeds the ids it committed to every other attack —
that is what makes "the same 10 clients are poisoned in every row" true rather
than merely likely. (Under the shipped ``attack.fixed_poison_clients`` regime the
set is pinned to clients ``0..N-1`` anyway, and this simply keeps the guarantee
when it is not.)

The unusable-generation handling lives here too (``_sample_attack``), since it is
a property of sampling from an LLM rather than of the round loop; ``benchmark.harness``
re-exports it.
"""
import logging

from benchmark.attacks.base import Attack

logger = logging.getLogger("benchmark")

# Retries of an unusable attacker action are sampled at (at least) this
# temperature: a greedy redraw would reproduce the same unusable text verbatim.
_RETRY_TEMPERATURE_FLOOR = 0.7


def _snippet(text, limit: int = 240) -> str:
    """One-line, length-capped view of a model output, for a log line."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else f"{s[:limit]}... [{len(s)} chars total]"


def _hit_token_cap(policy) -> bool:
    """Did the last generation stop at ``max_new_tokens`` instead of emitting EOS?

    ``LLMPolicy.generate`` records this per rollout in
    ``last_generation_completed``, which the next ``generate`` call overwrites —
    including the LLM defender's — so this must be read immediately after the
    attacker's own generation. Generators that don't expose it report ``False``.
    """
    flags = getattr(policy, "last_generation_completed", None)
    if isinstance(flags, list) and flags:
        return not bool(flags[0])
    return False


def _sample_attack(policy, attacker_agent, ctx, system, user, *, adapter,
                   temperature, max_new_tokens, retries):
    """Draw this round's attacker action, redrawing an unusable one.

    Returns ``(poisoned, chosen_ids, n_malformed, attempts)`` from the first
    attempt that filled the exact poison quota, or from the last attempt if none
    did (``len(chosen_ids) != ctx.budget`` is how the caller detects that).

    A single generation can come back unusable in ways that are pure sampling
    noise: truncated mid-JSON at the token cap, or a syntactically fine plan
    whose every operation is an arithmetic no-op (``scale factor=1.0``, an
    unknown operator, a target no layer matches). ``select_and_apply`` rightly
    refuses to call byte-identical benign weights poison, so such a round yields
    fewer effective plans than the quota. Aborting the benchmark there threw away
    every round already measured (tens of rounds x every defense in the panel, on
    the GPU) over a condition the same prompt almost always survives on a redraw,
    so the action is resampled a bounded number of times first.

    This is resampling for *validity*, not best-of-N on attack strength: the
    first action that fills the quota is used, whatever it does to accuracy, so
    the measured attack is not quietly strengthened on the rounds that needed a
    retry. Retries do force sampling (``_RETRY_TEMPERATURE_FLOOR``) even when the
    caller asked for greedy decoding, since re-running greedy returns the same
    unusable text.
    """
    result = ({}, [], ctx.budget)
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        temp = (temperature if attempt == 1
                else max(float(temperature), _RETRY_TEMPERATURE_FLOOR))
        text = policy.generate(adapter, system, user, n=1, temperature=temp,
                               max_new_tokens=max_new_tokens)[0]
        truncated = _hit_token_cap(policy)
        result = attacker_agent.select_and_apply(text, ctx.pool_benign, ctx.budget)
        poisoned, chosen_ids, n_malformed = result
        if len(chosen_ids) == ctx.budget:
            if attempt > 1:
                logger.info(f"round {ctx.round_num}: usable attacker action on "
                            f"attempt {attempt}/{attempts}")
            return poisoned, chosen_ids, n_malformed, attempt
        logger.warning(
            f"round {ctx.round_num}: unusable attacker action on attempt "
            f"{attempt}/{attempts} — {len(chosen_ids)}/{ctx.budget} effective plan(s), "
            f"{n_malformed} malformed quota slot(s)"
            + (f"; the generation hit the {max_new_tokens}-token cap mid-output "
               f"(raise rl.max_new_tokens)" if truncated else "")
            + f". attacker output: {_snippet(text)}"
        )
    return (*result, attempts)


class LLMAttack(Attack):
    """The trained attacker adapter, evaluated like any other attack."""

    name = "llm"
    citation = "this work (GRPO-trained attacker)"
    is_llm = True

    def __init__(self, policy, attacker_agent, *, adapter: str = "attacker",
                 temperature: float = 0.7, max_new_tokens: int = 512,
                 retries: int = 3):
        self.policy = policy
        self.agent = attacker_agent
        self.adapter = adapter
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.retries = int(retries)
        self._pending = None            # this round's ({cid: weights}, ids)
        self.last_malformed = 0
        self.last_attempts = 0
        self.last_output = ""

    def reset(self) -> None:
        self._pending = None

    def propose(self, round_ctx, global_weights, reference_accuracy: float):
        """Generate this round's action and return the client ids it committed to.

        Returns ``(chosen_ids, usable)``. ``usable`` is False when no generation
        filled the exact poison quota after every retry — the harness then skips
        the round for the WHOLE panel, so every attack is still compared over an
        identical set of rounds.
        """
        system = self.agent.system_prompt()
        user = self.agent.build_user_prompt(
            round_ctx.round_num, reference_accuracy, round_ctx.pool_benign,
            global_weights, round_ctx.budget)
        poisoned, chosen_ids, n_malformed, attempts = _sample_attack(
            self.policy, self.agent, round_ctx, system, user, adapter=self.adapter,
            temperature=self.temperature, max_new_tokens=self.max_new_tokens,
            retries=self.retries)
        self.last_malformed = int(n_malformed)
        self.last_attempts = int(attempts)
        usable = len(chosen_ids) == round_ctx.budget
        self._pending = (poisoned, [int(c) for c in chosen_ids])
        return list(self._pending[1]), usable

    def craft(self, ctx) -> dict:
        """Return the plan :meth:`propose` already produced for this round."""
        if self._pending is None:
            raise RuntimeError("LLMAttack.craft called before propose() for this round")
        poisoned, ids = self._pending
        if sorted(ids) != sorted(int(c) for c in ctx.poisoned_ids):
            # The harness derives the shared poisoned set FROM this attack, so a
            # mismatch means the round was assembled wrong — and silently returning
            # a differently-poisoned cohort would break the one invariant the whole
            # comparison rests on.
            raise RuntimeError(
                f"LLM attack committed to clients {sorted(ids)} but the round's "
                f"shared poisoned set is {sorted(ctx.poisoned_ids)}")
        return poisoned
