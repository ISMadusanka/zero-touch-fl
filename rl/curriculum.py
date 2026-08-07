"""TrainingCurriculum — the deterministic (defense algorithm, #poisoners) sweep.

Phase-2 rounds used to draw their two hardest knobs INDEPENDENTLY AT RANDOM: the
defense algorithm came from ``AlgorithmicDefender.choose()`` (uniform over
``defense.algorithms``) and the exact poison quota from
``FLArmsRaceEnv._round_budget`` (uniform over ``[1, attack.max_poison_clients]``).
Two uniform draws give every (algorithm, quota) pair the same *expected* share of
rounds, but not an even one: over 200 rounds with 4 algorithms x 5 quotas each
cell is Binomial(200, 1/20) — mean 10, sd ~3.1 — so cells routinely differ by
2-3x and the attacker's gradient budget lands wherever the RNG happened to point.
It also re-rolled the pair EVERY round, so the policy never got a contiguous
stretch against one defense at one attack strength, which is precisely the
regime it has to learn to exploit.

This module replaces both draws with a fixed sweep::

    for algorithm in defense.algorithms:          # e.g. fltrust, defl, dnc, multikrum
        for k in curriculum.poisoner_counts:      # e.g. 1, 2, 3, 4, 5
            run `rounds_per_block` (default 10) consecutive GRPO rounds,
            defended by `algorithm`, poisoning exactly `k` clients
    # ... then the cycle repeats from the first algorithm.

With the shipped settings one cycle is 4 x 5 x 10 = 200 rounds and every
algorithm gets exactly 50 rounds, 10 at each attack strength — the fair-and-equal
opportunity the random draw only gave in expectation. The training round budget
is millions of rounds, so the cycle simply repeats; a partial final cycle is the
only source of imbalance, and it is bounded by one cycle.

Two side effects worth knowing:

* **Stateful defenses advance contiguously.** DeFL carries Beta reputation counts
  and S(t-1) across rounds and DnC carries a subsampling RNG. Under the random
  rotation a given algorithm's memory advanced on ~1 round in 4, scattered; here
  it advances over 10 consecutive rounds, which is much closer to how it behaves
  when deployed alone.
* **It supersedes ``defense.selection`` and ``attack.sample_budget_in_training``.**
  Those knobs describe the random draws this curriculum replaces; when a
  curriculum is attached they are simply not consulted (and the builder says so).

The curriculum is a pure function of ONE integer — the number of rounds it has
handed out — so the whole schedule position is ``{"step": n}``. That is persisted
with the rest of the Phase-2 resume state (``checkpoints/rl_progress.json``) and
restored on resume, so a restart continues in the middle of the block it was in
rather than restarting the sweep and re-doing FLTrust-with-1-poisoner forever.

This module is deliberately torch-free and dependency-free so the schedule logic
is unit-testable without a GPU or a dataset.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Consecutive rounds spent on one (algorithm, #poisoners) pair.
DEFAULT_ROUNDS_PER_BLOCK = 10


@dataclass(frozen=True)
class CurriculumSlot:
    """What one Phase-2 round is scheduled to face.

    ``algorithm`` is ``None`` when the defender LLM defends (``defense.mode:
    llm``) — there is no algorithm axis to sweep there, so the curriculum drives
    only the poisoner count.
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
    """Deterministic round-by-round sweep over (algorithm, #poisoners) blocks.

    ``algorithms`` is the ordered defense pool (``[None]`` when the defender LLM
    defends), ``poisoner_counts`` the ordered attack strengths, and
    ``rounds_per_block`` how many consecutive rounds each pair holds. The outer
    loop is the ALGORITHM and the inner loop the poisoner count: one algorithm is
    faced at 1, 2, ... poisoners before the next algorithm is picked up.
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

def resolve_poisoner_counts(configured, max_poison_clients: int,
                            n_compromisable: int) -> list[int]:
    """The ordered attack strengths the sweep cycles through.

    ``configured is None`` means "every count the attack budget allows", i.e.
    ``1 .. attack.max_poison_clients``. Counts above the attacker's controllable
    pool are dropped with a warning rather than clamped: clamping would silently
    turn ``[1, 2, 3, 4, 5]`` with a 3-client pool into ``[1, 2, 3, 3, 3]`` and
    hand the largest quota three times the rounds of every other one, which is
    exactly the imbalance this curriculum exists to remove.
    """
    pool = max(1, int(n_compromisable))
    if configured is None:
        raw = list(range(1, max(1, int(max_poison_clients)) + 1))
    else:
        raw = [int(k) for k in configured]
    counts, seen = [], set()
    for k in raw:
        if k < 1:
            raise ValueError(f"curriculum.poisoner_counts must be >= 1, got {k}")
        if k > pool:
            logger.warning(
                f"curriculum.poisoner_counts: dropping {k} — the attacker controls "
                f"only {pool} client(s) (fl.n_compromisable), so a {k}-poisoner round "
                f"is not expressible. Raise fl.n_compromisable to sweep it."
            )
            continue
        if k in seen:
            logger.warning(f"curriculum.poisoner_counts: dropping duplicate {k} "
                           f"(a repeated count would get extra rounds)")
            continue
        counts.append(k)
        seen.add(k)
    if not counts:
        raise ValueError(
            f"curriculum.poisoner_counts is empty after validation (pool is "
            f"{pool} client(s)) — list at least one count in 1..{pool}"
        )
    return counts


def build_training_curriculum(cfg: dict, algorithms=None) -> TrainingCurriculum | None:
    """Build the sweep described by ``cfg`` (the whole base config), or ``None``.

    Returns ``None`` when the config has no ``curriculum:`` block (so older
    configs keep the original random draws) or when it sets ``enabled: false``.

    ``algorithms`` is the defender's VALIDATED rotation pool in listed order —
    pass ``AlgorithmicDefender.names``, or ``None`` under ``defense.mode: llm``
    where there is no algorithm axis and only the poisoner count is swept.
    ``curriculum.algorithms`` may narrow/reorder that pool; every name it lists
    must be one the defender actually built.
    """
    ccfg = cfg.get("curriculum")
    if not ccfg:
        return None
    if not ccfg.get("enabled", True):
        logger.info("Training curriculum disabled (curriculum.enabled: false) — "
                    "the defense algorithm and poison quota are drawn at random per round")
        return None

    fl = cfg.get("fl", {})
    attack = cfg.get("attack", {})
    n_clients = int(fl.get("n_clients", 1))
    n_compromisable = max(1, min(int(fl.get("n_compromisable", n_clients)), n_clients))

    available = list(algorithms) if algorithms else []
    requested = ccfg.get("algorithms")
    if requested:
        if not available:
            raise ValueError(
                "curriculum.algorithms was given but no algorithmic defense is "
                "active (defense.mode: llm) — remove it, or set "
                "defense.mode: algorithmic"
            )
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

    counts = resolve_poisoner_counts(
        ccfg.get("poisoner_counts"),
        max_poison_clients=int(attack.get("max_poison_clients", n_compromisable)),
        n_compromisable=n_compromisable,
    )
    curriculum = TrainingCurriculum(
        algorithms=available,
        poisoner_counts=counts,
        rounds_per_block=int(ccfg.get("rounds_per_block", DEFAULT_ROUNDS_PER_BLOCK)),
    )

    logger.info(f"Training curriculum ACTIVE: {curriculum.describe()}")
    # Say plainly which knobs it takes over, so a config that still sets them
    # doesn't read as if it were in charge.
    if available:
        logger.info(
            f"  curriculum supersedes defense.selection "
            f"({(cfg.get('defense') or {}).get('selection', 'random')!r}): the round's "
            f"algorithm is the block's, not a draw"
        )
    if attack.get("sample_budget_in_training", True):
        logger.info(
            "  curriculum supersedes attack.sample_budget_in_training: the round's "
            "poison quota is the block's, not a draw in [1, max_poison_clients]"
        )
    if attack.get("sample_target_in_training", False):
        logger.warning(
            "attack.sample_target_in_training is ON while the curriculum is active. "
            "The curriculum holds the defense and the poisoner count fixed for a whole "
            "block so those blocks are comparable; a target_accuracy_drop that keeps "
            "moving round-to-round re-introduces exactly the variance it removes. Set "
            "attack.sample_target_in_training: false and pin attack.goal."
            "target_accuracy_drop (0.10 is the shipped training default)."
        )
    return curriculum
