"""TrainingCurriculum — the deterministic (defense algorithm x poisoner count)
sweep the attacker's GRPO rounds follow.

**What it replaces.** Both knobs used to be drawn i.i.d. per round:
``AlgorithmicDefender.choose()`` picked a uniformly random defense and
``FLArmsRaceEnv._round_budget()`` a uniformly random quota in
``[1, attack.max_poison_clients]``. Over an infinite run that is balanced *on
average*, but it is never balanced over any window the policy actually learns
from, and the two draws are independent so the PAIRS are what get starved:

* With 4 algorithms x 5 quotas there are **20 regimes**. A 200-round stretch gives
  each one ~10 rounds — scattered as singletons, so the attacker essentially never
  gets two consecutive rounds in the same regime to fit.
* Random pairing has a heavy tail. Long runs routinely leave "FLTrust with 5
  poisoners" unseen for hundreds of rounds while "DnC with 1" comes up three times
  running. The policy then reads its own reward history as noise: the same plan
  scores very differently from round to round for reasons it cannot observe.
* Both quantities also move the reward SCALE (a 5-poisoner round can do far more
  damage than a 1-poisoner round; FLTrust's rescaling caps damage in a way
  Multi-Krum's selection does not). Shuffling them per round injects variance into
  exactly the signal GRPO normalises within a group.

**What it does instead.** One fixed, repeating sweep — with the shipped settings::

    fltrust    1 poisoner  x10 rounds,  2 x10,  3 x10,  4 x10,  5 x10   (50 rounds)
    defl       1 poisoner  x10 rounds,  2 x10,  3 x10,  4 x10,  5 x10   (50 rounds)
    dnc        ... same five blocks ...                                 (50 rounds)
    multikrum  ... same five blocks ...                                 (50 rounds)
    -> cycle 1 ends; wrap back to (fltrust, 1 poisoner) and repeat

Every (defense, poisoner-count) pair gets exactly ``rounds_per_block`` CONSECUTIVE
rounds per cycle, and every algorithm gets exactly the same number of rounds at
every difficulty — "fair opportunities" by construction rather than in
expectation. The sweep is a pure function of a single counter, so it is trivially
reproducible and resumable (see :meth:`TrainingCurriculum.state_dict`).

**What it deliberately does NOT touch.**

* ``defense.assumed_byzantine`` (DnC / Multi-Krum's assumed #malicious) stays a
  fixed hyperparameter. It is the server's *assumption*, never per-round truth, so
  it must not track the curriculum's current poisoner count — a defense that knew
  the real count each round would not be a defense.
* The attack goal. The curriculum sweeps *who* and *how many*, not *how hard*; the
  target accuracy drop stays pinned at ``attack.goal.target_accuracy_drop`` (see
  ``FLArmsRaceEnv.begin_round``, which force-disables per-round target sampling
  while a curriculum is active).
* Which clients the attacker poisons. The curriculum fixes the round's exact
  quota; the policy still chooses which members of its pool fill it.

This module is torch-free and depends on nothing but the config, so the ordering
logic is unit-testable without a GPU (``tests/test_curriculum.py``).
"""

import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

#: Consecutive GRPO rounds one (defense, poisoner-count) pair holds before the
#: sweep advances.
DEFAULT_ROUNDS_PER_BLOCK = 10


@dataclass(frozen=True)
class CurriculumSlot:
    """The (defense, poisoner-count) regime governing ONE GRPO round."""

    step: int              # 0-based global curriculum round counter
    algorithm: str | None  # None only when there is no algorithm rotation to sweep
    n_poisoners: int
    cycle: int             # 0-based: how many complete sweeps precede this round
    algo_index: int        # position in the curriculum's algorithm list
    count_index: int       # position in the curriculum's poisoner-count list
    block_index: int       # 0-based index of this block WITHIN the cycle
    round_in_block: int    # 1-based round within the block (1..rounds_per_block)

    @property
    def label(self) -> str:
        return f"{self.algorithm or 'llm'}/{self.n_poisoners}p"

    def as_dict(self) -> dict:
        """Plain dict for the round log (``attack_metadata.curriculum``)."""
        return asdict(self)


class TrainingCurriculum:
    """The repeating ``algorithms x poisoner_counts`` sweep, one round at a time.

    ``algorithms`` is swept in the OUTER loop and ``poisoner_counts`` in the inner
    one, so an algorithm is held fixed while the poisoner count climbs — which is
    what makes each block a controlled comparison ("same defense, one more
    poisoner") rather than two things changing at once.

    ``algorithms`` may be empty: the curriculum then sweeps poisoner counts only
    and leaves the round's defense alone (``slot.algorithm is None``). That is the
    ``defense.mode: llm`` case, where there is no algorithm pool to rotate.

    The cursor (:attr:`step`) counts GRPO rounds consumed, NOT FL rounds: the
    honest between-phase FL interlude does not call ``env.begin_round()`` and so
    never advances the sweep.
    """

    def __init__(self, algorithms, poisoner_counts,
                 rounds_per_block: int = DEFAULT_ROUNDS_PER_BLOCK, step: int = 0):
        self.algorithms = [str(a).strip().lower() for a in (algorithms or [])]
        self.poisoner_counts = [int(c) for c in (poisoner_counts or [])]
        self.rounds_per_block = int(rounds_per_block)
        if not self.poisoner_counts:
            raise ValueError("curriculum.poisoner_counts must list at least one count")
        if any(c < 1 for c in self.poisoner_counts):
            raise ValueError(f"curriculum.poisoner_counts must all be >= 1, got "
                             f"{self.poisoner_counts}")
        if self.rounds_per_block < 1:
            raise ValueError(f"curriculum.rounds_per_block must be >= 1, got "
                             f"{self.rounds_per_block}")
        self.step = max(0, int(step))

    # ------------------------------------------------------------------ shape
    @property
    def n_algorithms(self) -> int:
        """Number of algorithms swept — 1 ("no rotation") when the list is empty,
        so the block arithmetic below has no special case."""
        return max(1, len(self.algorithms))

    @property
    def blocks_per_cycle(self) -> int:
        return self.n_algorithms * len(self.poisoner_counts)

    @property
    def rounds_per_cycle(self) -> int:
        return self.blocks_per_cycle * self.rounds_per_block

    # ------------------------------------------------------------------ lookup
    def slot_at(self, step: int) -> CurriculumSlot:
        """The slot governing global round ``step`` (0-based). Pure — no cursor
        movement — so callers can preview or replay any point in the sweep."""
        step = max(0, int(step))
        n_counts = len(self.poisoner_counts)
        block = step // self.rounds_per_block
        cycle, block_in_cycle = divmod(block, self.blocks_per_cycle)
        algo_index, count_index = divmod(block_in_cycle, n_counts)
        return CurriculumSlot(
            step=step,
            algorithm=(self.algorithms[algo_index] if self.algorithms else None),
            n_poisoners=self.poisoner_counts[count_index],
            cycle=cycle,
            algo_index=algo_index,
            count_index=count_index,
            block_index=block_in_cycle,
            round_in_block=(step % self.rounds_per_block) + 1,
        )

    def peek(self) -> CurriculumSlot:
        """The slot the NEXT round will run under, without consuming it."""
        return self.slot_at(self.step)

    def take(self) -> CurriculumSlot:
        """Consume one round: return its slot and advance the cursor.

        Call exactly once per GRPO round, from ``env.begin_round()``, BEFORE
        anything in the round is scored — the whole round (clean counterfactual,
        all G rollouts, the commit) must run under one slot.
        """
        slot = self.slot_at(self.step)
        self.step += 1
        if slot.round_in_block == 1:
            logger.info(
                f"Curriculum: entering block {slot.block_index + 1}/"
                f"{self.blocks_per_cycle} of cycle {slot.cycle + 1} — "
                f"defense={slot.algorithm or 'llm (no rotation)'} with "
                f"{slot.n_poisoners} poisoner(s) for the next "
                f"{self.rounds_per_block} GRPO round(s)"
            )
        return slot

    # ------------------------------------------------------------------ resume
    def state_dict(self) -> dict:
        """Serializable cursor + the shape it indexes.

        The shape is stored alongside the cursor so a resume can detect that the
        sweep was RECONFIGURED between runs (different algorithms, counts or block
        length), in which case the old cursor points at a different regime than it
        did when it was written — see :meth:`load_state_dict`.
        """
        return {
            "step": int(self.step),
            "algorithms": list(self.algorithms),
            "poisoner_counts": list(self.poisoner_counts),
            "rounds_per_block": int(self.rounds_per_block),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the cursor from :meth:`state_dict`.

        Only the cursor is restored — the shape always comes from the CURRENT
        config, so editing the curriculum takes effect on resume instead of being
        silently overridden by the checkpoint. A shape change is reported, because
        it means the resumed cursor lands in a different block than the one the
        previous run was in.
        """
        if not state:
            return
        saved_shape = (list(state.get("algorithms") or []),
                       [int(c) for c in (state.get("poisoner_counts") or [])],
                       int(state.get("rounds_per_block", self.rounds_per_block)))
        if saved_shape != (self.algorithms, self.poisoner_counts, self.rounds_per_block):
            logger.warning(
                f"Curriculum was reconfigured since the checkpoint "
                f"(saved {saved_shape}, now "
                f"{(self.algorithms, self.poisoner_counts, self.rounds_per_block)}). "
                f"Keeping the saved cursor (step={state.get('step')}), so the resumed "
                f"round starts wherever that step falls in the NEW sweep."
            )
        self.step = max(0, int(state.get("step", self.step)))

    # ------------------------------------------------------------------ display
    def describe(self) -> str:
        slot = self.peek()
        algos = self.algorithms or ["(defense rotation disabled)"]
        return (
            f"{self.rounds_per_block} round(s) per (defense, poisoner-count) block; "
            f"algorithms {algos} x poisoners {self.poisoner_counts} = "
            f"{self.blocks_per_cycle} blocks = {self.rounds_per_cycle} rounds per cycle; "
            f"resuming at step {self.step} (cycle {slot.cycle + 1}, "
            f"block {slot.block_index + 1}, round {slot.round_in_block} of it: "
            f"{slot.label})"
        )


# ---------------------------------------------------------------------------
# Construction from configs/base.yaml
# ---------------------------------------------------------------------------

def _pool_size(cfg: dict) -> int:
    """How many clients the attacker can actually reach.

    Mirrors ``FLArmsRaceEnv.__init__``'s clamp of ``fl.n_compromisable`` — the
    curriculum's poisoner counts are validated against the SAME bound the env will
    enforce, so a count that the env would silently clip is rejected here instead.
    """
    fl = cfg.get("fl") or {}
    n_clients = int(fl.get("n_clients", 1))
    return max(1, min(int(fl.get("n_compromisable", n_clients)), n_clients))


def curriculum_enabled(cfg: dict) -> bool:
    """``curriculum.enabled`` — defaults to False so an untouched config keeps the
    legacy random-draw behaviour."""
    return bool((cfg.get("curriculum") or {}).get("enabled", False))


def build_curriculum(cfg: dict, *, defender=None) -> TrainingCurriculum | None:
    """Build the sweep described by ``cfg`` (the whole base config), or ``None``.

    Returns ``None`` when ``curriculum.enabled`` is false — the env then falls back
    to the per-round random draws.

    ``defender`` is the :class:`server.algo_defender.AlgorithmicDefender` for this
    run (``None`` under ``defense.mode: llm``). Its pool is both the DEFAULT
    algorithm list and the validation set: a curriculum may only name algorithms
    the defender can actually run, or the first round of that block would die with
    a ``KeyError`` several hours into training.
    """
    if not curriculum_enabled(cfg):
        return None

    ccfg = cfg.get("curriculum") or {}
    attack = cfg.get("attack") or {}
    pool = _pool_size(cfg)

    available = list(getattr(defender, "names", []) or [])
    requested = ccfg.get("algorithms")
    if requested is None:
        algorithms = available
    else:
        algorithms = [str(a).strip().lower() for a in requested if str(a).strip()]
        unknown = [a for a in algorithms if a not in available]
        if unknown:
            raise ValueError(
                f"curriculum.algorithms names {unknown}, which the round defense "
                f"cannot run (defense.algorithms = "
                f"{available or 'none — defense.mode is llm, so there is no rotation'}). "
                f"Add them to defense.algorithms, or drop them from the curriculum."
            )

    # Default the poisoner sweep to "1 up to the training quota cap" — the exact
    # range the random draw used to cover, now visited in order.
    counts = ccfg.get("poisoner_counts")
    if counts is None:
        cap = max(1, min(int(attack.get("max_poison_clients", pool)), pool))
        counts = list(range(1, cap + 1))
    else:
        counts = [int(c) for c in counts]
        over = sorted({c for c in counts if c > pool})
        if over:
            raise ValueError(
                f"curriculum.poisoner_counts asks for {over} poisoner(s) but the "
                f"attacker can only reach {pool} client(s) "
                f"(fl.n_compromisable). Lower the counts or raise fl.n_compromisable."
            )

    curriculum = TrainingCurriculum(
        algorithms=algorithms,
        poisoner_counts=counts,
        rounds_per_block=int(ccfg.get("rounds_per_block", DEFAULT_ROUNDS_PER_BLOCK)),
    )

    # Say plainly which config knobs the curriculum now overrides, so nobody reads
    # `defense.selection` / `sample_budget_in_training` in the config and believes
    # they are still in effect.
    ignored = []
    if defender is not None:
        ignored.append(f"defense.selection ({defender.describe()})")
    if attack.get("sample_budget_in_training"):
        ignored.append("attack.sample_budget_in_training (random per-round quota)")
    if attack.get("sample_target_in_training"):
        ignored.append("attack.sample_target_in_training (random per-round target drop)")
    logger.info(f"Training curriculum ENABLED: {curriculum.describe()}")
    if ignored:
        logger.info(f"Curriculum overrides {', '.join(ignored)} — "
                    f"the sweep decides the defense and the poison quota, and the "
                    f"attack goal stays fixed at attack.goal.target_accuracy_drop")
    return curriculum
