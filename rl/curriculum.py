"""TrainingCurriculum — the deterministic defense-algorithm sweep.

Only relevant under ``defense.mode: algorithmic``, where the round's defense used
to be drawn uniformly by ``AlgorithmicDefender.choose()``. A uniform draw gives
every algorithm the same *expected* share of rounds but not an even one, and it
re-rolls EVERY round, so no algorithm ever gets a contiguous stretch — which
matters because several of them are stateful.

This module replaces that draw with a fixed sweep::

    for algorithm in defense.algorithms:          # e.g. fltrust, defl, dnc, multikrum
        run `rounds_per_block` (default 10) consecutive rounds defended by it
    # ... then the cycle repeats from the first algorithm.

**Stateful defenses advance contiguously.** DeFL carries Beta reputation counts
and S(t-1) across rounds and DnC carries a subsampling RNG. Under the random
rotation a given algorithm's memory advanced on ~1 round in 4, scattered; here it
advances over 10 consecutive rounds, much closer to how it behaves when deployed
alone.

**The attack strength is NOT swept.** It used to be the second axis (a per-round
poison quota the attacker LLM filled), but the attack is now a fixed set of
label-flipping clients whose poison level is set by its own detection-adaptive
ladder (:mod:`agents.label_flip_attacker`). ``n_poisoners`` therefore survives on
:class:`CurriculumSlot` as a constant — how many clients flip labels — so round
logs stay self-describing, and there is nothing left for the curriculum to vary
on that axis.

Under ``defense.mode: llm`` (the default) there is no algorithm axis either, so
:func:`build_training_curriculum` returns ``None`` and the defense is simply the
defender LLM every round.

The curriculum is a pure function of ONE integer — the number of rounds it has
handed out — so the whole schedule position is ``{"step": n}``. That is persisted
with the rest of the Phase-2 resume state (``checkpoints/rl_progress.json``) and
restored on resume, so a restart continues in the middle of the block it was in
rather than restarting the sweep from the first algorithm every time.

This module is deliberately torch-free and dependency-free so the schedule logic
is unit-testable without a GPU or a dataset.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Consecutive rounds spent on one algorithm.
DEFAULT_ROUNDS_PER_BLOCK = 10


@dataclass(frozen=True)
class CurriculumSlot:
    """What one Phase-2 round is scheduled to face.

    ``algorithm`` is the defense for this block. ``n_poisoners`` is a CONSTANT —
    how many clients flip labels every round (``len(attack.poison_client_ids)``) —
    carried here so a round log names the attack size without re-deriving it from
    the config; it is not an axis the curriculum varies.
    """

    algorithm: str | None
    n_poisoners: int
    cycle: int          # how many full sweeps completed before this round
    block: int          # absolute block index (never resets)
    position: int       # this block's index WITHIN the cycle (0 .. blocks_per_cycle-1)
    block_round: int    # 0-based round within the block (0 .. rounds_per_block-1)

    def as_log_dict(self) -> dict:
        """Compact form for ``RoundLog.attack_metadata`` so analysis can slice the
        run by block without re-deriving it from the round number."""
        return {
            "algorithm": self.algorithm,
            "n_poisoners": self.n_poisoners,
            "cycle": self.cycle,
            "block": self.block,
            "position": self.position,
            "block_round": self.block_round,
        }


class TrainingCurriculum:
    """Deterministic round-by-round sweep over defense-algorithm blocks.

    ``algorithms`` is the ordered defense pool, ``rounds_per_block`` how many
    consecutive rounds each holds, and ``poisoner_counts`` the constant number of
    label-flipping clients (normally a single-element list — see
    :class:`CurriculumSlot`). The generic two-axis machinery is retained so a
    future experiment can sweep a second axis without rewriting the sweep, but
    with one count the schedule is simply one algorithm per block.
    """

    def __init__(self, algorithms, poisoner_counts,
                 rounds_per_block: int = DEFAULT_ROUNDS_PER_BLOCK):
        algs = list(algorithms) if algorithms else [None]
        counts = [int(k) for k in poisoner_counts]
        if not counts:
            raise ValueError("TrainingCurriculum needs at least one poisoner count")
        if any(k < 1 for k in counts):
            raise ValueError(f"poisoner counts must be >= 1, got {counts}")
        if int(rounds_per_block) < 1:
            raise ValueError(f"rounds_per_block must be >= 1, got {rounds_per_block}")
        self._algorithms = algs
        self._counts = counts
        self._rounds_per_block = int(rounds_per_block)
        self._step = 0

    # ------------------------------------------------------------------
    @property
    def algorithms(self) -> list:
        return list(self._algorithms)

    @property
    def poisoner_counts(self) -> list[int]:
        return list(self._counts)

    @property
    def rounds_per_block(self) -> int:
        return self._rounds_per_block

    @property
    def step(self) -> int:
        """Rounds handed out so far (the whole schedule position)."""
        return self._step

    @property
    def blocks_per_cycle(self) -> int:
        return len(self._algorithms) * len(self._counts)

    @property
    def rounds_per_cycle(self) -> int:
        return self.blocks_per_cycle * self._rounds_per_block

    def fingerprint(self) -> dict:
        """The plan itself (not the position) — persisted so a resume can tell the
        user when the config changed underneath a saved ``step``."""
        return {
            "algorithms": list(self._algorithms),
            "poisoner_counts": list(self._counts),
            "rounds_per_block": self._rounds_per_block,
        }

    def describe(self) -> str:
        algs = [a if a is not None else "llm" for a in self._algorithms]
        return (f"{self._rounds_per_block} round(s) per block, "
                f"algorithms={algs} x poisoners={self._counts} "
                f"= {self.blocks_per_cycle} blocks / {self.rounds_per_cycle} rounds per cycle")

    # ------------------------------------------------------------------
    def slot_at(self, step: int) -> CurriculumSlot:
        """The slot the ``step``-th round (0-based) faces. Pure — no state change."""
        step = max(0, int(step))
        block, block_round = divmod(step, self._rounds_per_block)
        cycle, position = divmod(block, self.blocks_per_cycle)
        alg_i, cnt_i = divmod(position, len(self._counts))
        return CurriculumSlot(
            algorithm=self._algorithms[alg_i],
            n_poisoners=self._counts[cnt_i],
            cycle=cycle,
            block=block,
            position=position,
            block_round=block_round,
        )

    def peek(self) -> CurriculumSlot:
        """The slot the NEXT round will face, without consuming it."""
        return self.slot_at(self._step)

    def advance(self) -> CurriculumSlot:
        """Hand out the next slot and consume it. Called exactly once per round,
        from :meth:`FLArmsRaceEnv.begin_round`."""
        slot = self.slot_at(self._step)
        self._step += 1
        if slot.block_round == 0:
            logger.info(
                f"Curriculum block {slot.block} (cycle {slot.cycle}, "
                f"{slot.position + 1}/{self.blocks_per_cycle}): next "
                f"{self._rounds_per_block} round(s) face "
                f"defense={slot.algorithm or 'llm'} with exactly "
                f"{slot.n_poisoners} poisoner(s)"
            )
        return slot

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Serializable schedule position, for resume."""
        return {"step": self._step, "plan": self.fingerprint()}

    def load_state_dict(self, state: dict) -> None:
        """Restore a snapshot from :meth:`state_dict`.

        If the saved plan differs from this run's (the config was edited between
        runs) the ``step`` is still honoured — throwing it away would restart the
        sweep and re-train the first block from scratch — but the change is
        logged loudly, because the same step now maps to a different block.
        """
        saved_plan = state.get("plan")
        if saved_plan and saved_plan != self.fingerprint():
            logger.warning(
                f"Curriculum plan changed since the last run "
                f"(saved={saved_plan}, now={self.fingerprint()}). Keeping the saved "
                f"position (step={state.get('step')}), so the sweep continues rather "
                f"than restarting — but that step now maps to a DIFFERENT block."
            )
        self._step = max(0, int(state.get("step", self._step)))


# ---------------------------------------------------------------------------
# Construction from configs/base.yaml
# ---------------------------------------------------------------------------


def build_training_curriculum(cfg: dict, algorithms=None) -> TrainingCurriculum | None:
    """Build the defense sweep described by ``cfg`` (the whole base config), or ``None``.

    Returns ``None`` when the config has no ``curriculum:`` block, when it sets
    ``enabled: false``, or when there is no algorithm axis to sweep — which is the
    case under ``defense.mode: llm`` (``algorithms`` is empty/``None``), where the
    defender LLM defends every round and the attack strength is set by its own
    ladder rather than by a schedule.

    ``algorithms`` is the defender's VALIDATED rotation pool in listed order — pass
    ``AlgorithmicDefender.names``. ``curriculum.algorithms`` may narrow/reorder that
    pool; every name it lists must be one the defender actually built.
    """
    ccfg = cfg.get("curriculum")
    if not ccfg:
        return None
    if not ccfg.get("enabled", True):
        logger.info("Training curriculum disabled (curriculum.enabled: false) — "
                    "the defense algorithm is drawn at random per round")
        return None

    available = list(algorithms) if algorithms else []
    requested = ccfg.get("algorithms")
    if requested and not available:
        raise ValueError(
            "curriculum.algorithms was given but no algorithmic defense is active "
            "(defense.mode: llm) — remove it, or set defense.mode: algorithmic"
        )
    if not available:
        logger.info(
            "Training curriculum inactive: the defender LLM defends every round "
            "(defense.mode: llm), so there is no algorithm to sweep. The attack "
            "strength is the label-flip ladder's, not the curriculum's."
        )
        return None
    if requested:
        names, seen = [], set()
        for item in requested:
            name = str(item).strip().lower()
            if name not in available:
                raise ValueError(f"curriculum.algorithms: '{name}' is not in the "
                                 f"active defense pool {available}")
            if name in seen:
                raise ValueError(f"curriculum.algorithms: duplicate '{name}' — a "
                                 f"repeated algorithm would get extra rounds")
            names.append(name)
            seen.add(name)
        available = names

    # The number of label-flipping clients is fixed by attack.poison_client_ids, so
    # it is a constant carried on every slot rather than an axis to sweep.
    poison_ids = (cfg.get("attack") or {}).get("poison_client_ids", [0])
    if isinstance(poison_ids, int):
        poison_ids = [poison_ids]
    n_poisoners = max(1, len(list(poison_ids or [0])))

    curriculum = TrainingCurriculum(
        algorithms=available,
        poisoner_counts=[n_poisoners],
        rounds_per_block=int(ccfg.get("rounds_per_block", DEFAULT_ROUNDS_PER_BLOCK)),
    )
    logger.info(f"Training curriculum ACTIVE: {curriculum.describe()}")
    logger.info(
        f"  curriculum supersedes defense.selection "
        f"({(cfg.get('defense') or {}).get('selection', 'random')!r}): the round's "
        f"algorithm is the block's, not a draw"
    )
    return curriculum
