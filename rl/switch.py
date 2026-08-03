"""Success-gated iterated best-response controller for the arms-race schedule.

Instead of switching the learner on a fixed clock (the old ``K_a``/``K_d`` round
blocks), we train ONE agent until it *beats the frozen opponent* — the attacker
until an attack **passes** (poison evades detection AND degrades the model), the
defender until it **reliably catches** the frozen attacker — then freeze that
agent and hand off. This is a double-oracle / iterated best-response ratchet:
each side makes one concrete win, freezes at that checkpoint, and the other side
then has a real, beatable opponent to climb against.

Two safety rails keep it well-behaved:

* ``min_phase_rounds`` / ``success_streak`` — a win must be *sustained* (repeat
  for ``success_streak`` consecutive committed rounds, and only after at least
  ``min_phase_rounds``) before we freeze-and-switch, so a single lucky rollout
  does not trigger a handoff.
* ``max_phase_rounds`` — if the win never comes, the phase still ends (we switch
  anyway). The schedule may then let the next learner face an *earlier* opponent
  snapshot (curriculum) so it can find a foothold.

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

    min_phase_rounds: int = 8        # earliest round a phase may switch on success
    max_phase_rounds: int = 200      # hard cap: switch even without a sustained win
    success_streak: int = 3          # consecutive winning rounds needed to freeze+switch

    # Attacker "an attack passed through the defender".
    attacker_min_drop: float = 0.02  # absolute drop bar, used only when the round's goal/target
                                     # is unknown (fallback); normally the RELATIVE gate applies.
    win_fraction: float = 0.6        # RELATIVE damage bar: fraction of the round's requested
                                     # target drop the attack must achieve to count as a win.
    attacker_min_evaded: float = 1.0 # fraction of poisoned clients that must evade detection

    # Defender "defense succeeded against the frozen attacker".
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
            attacker_min_evaded=float(rl.get("attacker_min_evaded", 1.0)),
            defender_min_tpr=float(rl.get("defender_min_tpr", 0.99)),
            defender_max_fpr=float(rl.get("defender_max_fpr", 0.10)),
        )


def _tpr_fpr(verdicts, poisoned_ids) -> tuple[float, float]:
    poisoned = set(poisoned_ids)
    tp = sum(1 for v in verdicts if v.client_id in poisoned and v.is_suspicious)
    fn = sum(1 for v in verdicts if v.client_id in poisoned and not v.is_suspicious)
    fp = sum(1 for v in verdicts if v.client_id not in poisoned and v.is_suspicious)
    tn = sum(1 for v in verdicts if v.client_id not in poisoned and not v.is_suspicious)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr, fpr


def attacker_succeeded(drop: float, verdicts, poisoned_ids, cfg: SwitchConfig,
                       goal: dict | None = None) -> bool:
    """True when the committed attack 'passed': enough poisoned clients evaded
    detection AND the round lost enough accuracy.

    The damage bar is RELATIVE to the round's requested target when ``goal`` is
    given — ``win_fraction * target`` (``target`` via :func:`rl.rewards.goal_target`)
    — so a per-round-sampled target is judged on its own scale (a 0.05-target round
    and a 0.30-target round need proportional drops, not one absolute floor). Without
    a goal it falls back to the absolute ``attacker_min_drop``. A flagged client is
    dropped from FedAvg, so meaningful ``drop`` already implies evasion — we keep the
    explicit evasion check for the multi-poisoner case."""
    if not poisoned_ids:
        return False
    flagged = {v.client_id for v in verdicts if v.is_suspicious}
    evaded = sum(1 for cid in poisoned_ids if cid not in flagged)
    evaded_frac = evaded / len(poisoned_ids)
    min_drop = (cfg.win_fraction * goal_target(goal) if goal is not None
                else cfg.attacker_min_drop)
    return evaded_frac >= cfg.attacker_min_evaded and drop >= min_drop


def defender_succeeded(verdicts, poisoned_ids, cfg: SwitchConfig) -> bool:
    """True when the committed verdicts catch the poisoned client(s) (TPR high)
    without over-flagging honest clients (FPR low).

    ``poisoned_ids`` may be empty — the frozen attacker can select clients whose
    plans turn out to be no-ops, leaving every update honest. TPR is undefined
    there, and treating it as 0 would make a flawless clean round a loss (and
    stall the defender's phase whenever its opponent degenerates). On a clean
    round the defender wins by staying quiet, i.e. by keeping FPR in bounds.
    """
    tpr, fpr = _tpr_fpr(verdicts, poisoned_ids)
    if not poisoned_ids:
        return fpr <= cfg.defender_max_fpr
    return tpr >= cfg.defender_min_tpr and fpr <= cfg.defender_max_fpr


def committed_success(learner: str, drop: float, verdicts, poisoned_ids,
                      cfg: SwitchConfig, goal: dict | None = None) -> bool:
    """Did the learner win on this committed round? ``goal`` (this round's attack
    goal) enables the attacker's relative win-gate; it is ignored for the defender."""
    if learner == "attacker":
        return attacker_succeeded(drop, verdicts, poisoned_ids, cfg, goal)
    return defender_succeeded(verdicts, poisoned_ids, cfg)


class PhaseController:
    """Tracks one learner phase and decides when to freeze-and-switch.

    Call :meth:`record` once per *committed* round with whether the learner won.
    It returns ``(switch, reason)``: when ``switch`` is True the driver should
    snapshot+freeze the current learner, then call :meth:`next_phase(reason)`.

    ``learners`` is the rotation of TRAINABLE agents. It is both roles by default;
    when the defender LLM is disabled (the defense is algorithmic, so there is no
    defender policy) the driver passes ``("attacker",)`` and every phase is an
    attacker phase — the phase boundary still fires on a sustained win or the cap,
    which is what triggers the between-phase honest FL interlude.
    """

    def __init__(self, cfg: SwitchConfig, first_learner: str = "attacker",
                 learners: tuple = ("attacker", "defender")):
        self.learners = tuple(learners)
        if not self.learners:
            raise ValueError("learners must contain at least one agent")
        bad = [n for n in self.learners if n not in ("attacker", "defender")]
        if bad:
            raise ValueError(f"learners must be attacker|defender, got {bad}")
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
        if learner not in ("attacker", "defender"):
            raise ValueError(f"controller learner must be attacker|defender, got {learner!r}")
        if learner not in self.learners:
            # Resuming a checkpoint written when the other side was still trainable
            # (e.g. the defender LLM has since been disabled). Fall back to this
            # run's rotation instead of resuming into a phase that cannot run.
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

    @property
    def opponent(self) -> str:
        """The frozen side of this phase. With a single trainable learner the
        opponent is not an agent at all (it is the fixed algorithmic defense) —
        the name is then only a label for logs."""
        return "defender" if self.learner == "attacker" else "attacker"

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
        """Advance to the next learner in the rotation. ``reason`` is the switch
        reason of the phase just completed; ``capped`` flags that the NEXT phase
        may want to face an earlier opponent snapshot (curriculum) because this
        one stalled. With a single trainable learner the rotation is a no-op and
        the same agent simply starts a fresh phase."""
        self.capped = (reason == "cap")
        i = self.learners.index(self.learner)
        self.learner = self.learners[(i + 1) % len(self.learners)]
        self.phase_index += 1
        self.phase_round = 0
        self.streak = 0
