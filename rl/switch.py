"""Success-gated phase controller for the defender's training schedule.

The defender is the only learner (the attack is the deterministic label-flip
ladder in :mod:`agents.label_flip_attacker`), so this no longer alternates
between two agents. What survives — and is still load-bearing — is the PHASE
structure: a phase runs until the defender **reliably catches** the current
attack, then ends, freezes a checkpoint, and a fresh phase starts.

That still matters because the opponent is not static. The ladder backs the
attack off every time the defender catches it, so a defender that wins a phase
has, by construction, driven the attack down to a weaker level (or all the way
to the floor and back to full strength). Ending the phase on that sustained win
is what snapshots the defender at each rung it has actually mastered.

Two safety rails keep it well-behaved:

* ``min_phase_rounds`` / ``success_streak`` — a win must be *sustained* (repeat
  for ``success_streak`` consecutive committed rounds, and only after at least
  ``min_phase_rounds``) before we freeze, so a single lucky rollout does not
  trigger a handoff.
* ``max_phase_rounds`` — if the win never comes, the phase still ends, which
  bounds how long the run can sit against an attack level it cannot handle.

This module is deliberately torch-free so the control logic is unit-testable
without a GPU. It consumes only plain numbers and ``DetectionVerdict``-like
objects (anything exposing ``.client_id`` and ``.is_suspicious``).
"""

import logging
from dataclasses import dataclass

from rl.rewards import goal_target

logger = logging.getLogger(__name__)


@dataclass
class SwitchConfig:
    """Thresholds for the success-gated best-response schedule."""

    min_phase_rounds: int = 8        # earliest round a phase may end on success
    max_phase_rounds: int = 200      # hard cap: end the phase even without a sustained win
    success_streak: int = 3          # consecutive winning rounds needed to freeze

    # The DAMAGE bar. No longer a win gate (nothing on the attack side is trained),
    # but still the threshold the metrics tracker's `attack_success` reports against,
    # i.e. "did this round's flipped labels actually cost the model something".
    attacker_min_drop: float = 0.02  # absolute drop bar, used only when the round's goal/target
                                     # is unknown (fallback); normally the RELATIVE bar applies.
    win_fraction: float = 0.6        # RELATIVE damage bar: fraction of the round's requested
                                     # target drop the attack must achieve to count as damaging.

    # Defender "the defense succeeded against this round's attack".
    defender_min_tpr: float = 0.99   # caught the poisoned client(s)
    defender_max_fpr: float = 0.10   # without over-flagging honest clients

    @classmethod
    def from_cfg(cls, rl: dict) -> "SwitchConfig":
        return cls(
            min_phase_rounds=int(rl.get("min_phase_rounds", 8)),
            max_phase_rounds=int(rl.get("max_phase_rounds", 200)),
            success_streak=int(rl.get("success_streak", 3)),
            attacker_min_drop=float(rl.get("attacker_min_drop", 0.02)),
            win_fraction=float(rl.get("win_fraction", 0.6)),
            defender_min_tpr=float(rl.get("defender_min_tpr", 0.99)),
            defender_max_fpr=float(rl.get("defender_max_fpr", 0.10)),
        )


def success_drop_bar(goal: dict | None, cfg: SwitchConfig | None = None) -> float:
    """The accuracy drop a round's attack must achieve to be called damaging.

    ``win_fraction * goal_target(goal)``, falling back to the absolute
    ``attacker_min_drop`` when the round's goal is unknown. Single source of truth
    for the metrics tracker's ``attack_success`` across the training, dry-run and
    baseline paths, so all three mean the same thing by it.

    ``cfg=None`` uses the dataclass defaults, which is what the evaluation and
    baseline paths want — they have no schedule to read from.
    """
    cfg = cfg or SwitchConfig()
    if goal is None:
        return float(cfg.attacker_min_drop)
    return float(cfg.win_fraction) * goal_target(goal)


def _tpr_fpr(verdicts, poisoned_ids) -> tuple[float, float]:
    poisoned = set(poisoned_ids)
    tp = sum(1 for v in verdicts if v.client_id in poisoned and v.is_suspicious)
    fn = sum(1 for v in verdicts if v.client_id in poisoned and not v.is_suspicious)
    fp = sum(1 for v in verdicts if v.client_id not in poisoned and v.is_suspicious)
    tn = sum(1 for v in verdicts if v.client_id not in poisoned and not v.is_suspicious)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr, fpr


def attack_was_damaging(drop: float, poisoned_ids, cfg: SwitchConfig,
                        goal: dict | None = None) -> bool:
    """Did this round's flipped labels cost the model enough accuracy to matter?

    The bar is RELATIVE to the round's requested target when ``goal`` is given —
    ``win_fraction * target`` (``target`` via :func:`rl.rewards.goal_target`) — so a
    per-round-sampled target is judged on its own scale. Without a goal it falls
    back to the absolute ``attacker_min_drop``.

    Purely a REPORTED measurement: the attack is a fixed schedule, so nothing is
    trained or scheduled on this. It answers "was this round's ladder level
    actually an attack?", which is what separates a defender that learned to catch
    poison from one that learned to catch nothing in particular.
    """
    if not poisoned_ids:
        return False
    return drop >= success_drop_bar(goal, cfg)


def defender_succeeded(verdicts, poisoned_ids, cfg: SwitchConfig) -> bool:
    """True when the committed verdicts catch the poisoned client(s) (TPR high)
    without over-flagging honest clients (FPR low).

    ``poisoned_ids`` may be empty — a ladder level that rounds to zero flipped
    labels leaves every update honest. TPR is undefined there, and treating it as
    0 would make a flawless clean round a loss. On a clean round the defender wins
    by staying quiet, i.e. by keeping FPR in bounds.
    """
    tpr, fpr = _tpr_fpr(verdicts, poisoned_ids)
    if not poisoned_ids:
        return fpr <= cfg.defender_max_fpr
    return tpr >= cfg.defender_min_tpr and fpr <= cfg.defender_max_fpr


class PhaseController:
    """Tracks one defender phase and decides when to freeze it.

    Call :meth:`record` once per *committed* round with whether the defender won.
    It returns ``(switch, reason)``: when ``switch`` is True the driver should
    snapshot+freeze the defender, then call :meth:`next_phase(reason)` to start a
    fresh phase.

    ``learners`` exists so the rotation stays expressible, but with the attack
    reduced to a fixed schedule there is exactly one trainable agent, so
    :meth:`next_phase` simply starts the defender on a new phase.
    """

    def __init__(self, cfg: SwitchConfig, first_learner: str = "defender",
                 learners: tuple = ("defender",)):
        self.learners = tuple(learners)
        if not self.learners:
            raise ValueError("learners must contain at least one agent")
        bad = [n for n in self.learners if n != "defender"]
        if bad:
            raise ValueError(
                f"the defender is the only trainable agent (the attack is a fixed "
                f"label-flip schedule), got {bad}"
            )
        if first_learner not in self.learners:
            raise ValueError(f"first_learner must be one of {self.learners}, "
                             f"got {first_learner!r}")
        self.cfg = cfg
        self.learner = first_learner
        self.phase_index = 0
        self.phase_round = 0
        self.streak = 0
        self.capped = False   # did the most recently *completed* phase hit the cap?

    def state_dict(self) -> dict:
        """Serializable snapshot of the schedule state, for resume."""
        return {
            "learner": self.learner,
            "phase_index": self.phase_index,
            "phase_round": self.phase_round,
            "streak": self.streak,
            "capped": self.capped,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a snapshot from :meth:`state_dict` (missing keys keep current)."""
        learner = state.get("learner", self.learner)
        if learner not in self.learners:
            # A checkpoint from before the attacker LLM was removed will say
            # ``"attacker"``. Continue as the defender rather than refusing to
            # resume into a phase that no longer exists.
            logger.warning(
                f"Resumed controller says learner={learner!r}, but only {self.learners} "
                f"is trainable in this run — continuing as {self.learners[0]!r}."
            )
            learner = self.learners[0]
        self.learner = learner
        self.phase_index = int(state.get("phase_index", self.phase_index))
        self.phase_round = int(state.get("phase_round", self.phase_round))
        self.streak = int(state.get("streak", self.streak))
        self.capped = bool(state.get("capped", self.capped))

    def record(self, success: bool) -> tuple[bool, str | None]:
        """Register one committed round. Returns (should_switch, reason)."""
        self.phase_round += 1
        self.streak = self.streak + 1 if success else 0
        if self.phase_round >= self.cfg.min_phase_rounds and self.streak >= self.cfg.success_streak:
            return True, "success"
        if self.phase_round >= self.cfg.max_phase_rounds:
            return True, "cap"
        return False, None

    def next_phase(self, reason: str) -> None:
        """Start a fresh phase. ``reason`` is why the phase just completed ended;
        ``capped`` records that it ran out of rounds without a sustained win, which
        the driver logs as a stalled phase. With the defender as the only learner
        the rotation is a no-op and it simply begins the next phase."""
        self.capped = (reason == "cap")
        i = self.learners.index(self.learner)
        self.learner = self.learners[(i + 1) % len(self.learners)]
        self.phase_index += 1
        self.phase_round = 0
        self.streak = 0
