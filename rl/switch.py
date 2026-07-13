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

from dataclasses import dataclass


@dataclass
class SwitchConfig:
    """Thresholds for the success-gated best-response schedule."""

    min_phase_rounds: int = 8        # earliest round a phase may switch on success
    max_phase_rounds: int = 200      # hard cap: switch even without a sustained win
    success_streak: int = 3          # consecutive winning rounds needed to freeze+switch

    # Attacker "an attack passed through the defender".
    attacker_min_drop: float = 0.02  # committed accuracy drop (prev-post) to count as damage
    attacker_min_class_drop: float = 0.10  # targeted_label: min per-class drop to count as damage
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
            attacker_min_class_drop=float(rl.get("attacker_min_class_drop", 0.10)),
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
                       class_drop: float | None = None) -> bool:
    """True when the committed attack 'passed': enough poisoned clients evaded
    detection AND the round caused damage.

    For targeted attacks, ``class_drop`` (the most-damaged class's accuracy drop)
    is checked against ``attacker_min_class_drop`` instead of global ``drop``.
    A flagged client is dropped from FedAvg, so meaningful damage already implies
    evasion — we keep the explicit evasion check for the multi-poisoner case."""
    import logging
    _logger = logging.getLogger("rl.switch")

    if not poisoned_ids:
        return False
    flagged = {v.client_id for v in verdicts if v.is_suspicious}
    evaded = sum(1 for cid in poisoned_ids if cid not in flagged)
    evaded_frac = evaded / len(poisoned_ids)

    if class_drop is not None:
        # Targeted mode: check class-level damage
        success = evaded_frac >= cfg.attacker_min_evaded and class_drop >= cfg.attacker_min_class_drop
        _logger.info(
            f"Targeted attacker_succeeded: class_drop={class_drop:.4f} "
            f"(min={cfg.attacker_min_class_drop}), evaded={evaded_frac:.2f} "
            f"(min={cfg.attacker_min_evaded}) → {success}"
        )
        return success

    return evaded_frac >= cfg.attacker_min_evaded and drop >= cfg.attacker_min_drop


def defender_succeeded(verdicts, poisoned_ids, cfg: SwitchConfig) -> bool:
    """True when the committed verdicts catch the poisoned client(s) (TPR high)
    without over-flagging honest clients (FPR low)."""
    tpr, fpr = _tpr_fpr(verdicts, poisoned_ids)
    return tpr >= cfg.defender_min_tpr and fpr <= cfg.defender_max_fpr


def committed_success(learner: str, drop: float, verdicts, poisoned_ids,
                      cfg: SwitchConfig, class_drop: float | None = None) -> bool:
    """Did the learner win on this committed round?"""
    if learner == "attacker":
        return attacker_succeeded(drop, verdicts, poisoned_ids, cfg, class_drop)
    return defender_succeeded(verdicts, poisoned_ids, cfg)


class PhaseController:
    """Tracks one learner phase and decides when to freeze-and-switch.

    Call :meth:`record` once per *committed* round with whether the learner won.
    It returns ``(switch, reason)``: when ``switch`` is True the driver should
    snapshot+freeze the current learner, then call :meth:`next_phase(reason)`.
    """

    def __init__(self, cfg: SwitchConfig, first_learner: str = "attacker"):
        if first_learner not in ("attacker", "defender"):
            raise ValueError(f"first_learner must be attacker|defender, got {first_learner!r}")
        self.cfg = cfg
        self.learner = first_learner
        self.phase_index = 0
        self.phase_round = 0
        self.streak = 0
        self.capped = False   # did the most recently *completed* phase hit the cap?

    @property
    def opponent(self) -> str:
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
        """Advance to the opponent's phase. ``reason`` is the switch reason of the
        phase just completed; ``capped`` flags that the NEXT phase may want to
        face an earlier opponent snapshot (curriculum) because this one stalled."""
        self.capped = (reason == "cap")
        self.learner = self.opponent
        self.phase_index += 1
        self.phase_round = 0
        self.streak = 0
