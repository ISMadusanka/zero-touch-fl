"""Per-defense metric accumulation for the benchmark.

Two families of metrics, gathered per defense over all rounds:

* Detection (per-client accept/reject vs the ground-truth poisoned set):
  detection-rate (= recall = TPR), FPR, precision, F1.
* Robustness (the resulting model under attack):
  final & mean test accuracy, mean accuracy drop vs the clean baseline, and two
  distinct "attack success" rates:
    - evasion (``attack_success_rate``): fraction of rounds a poisoned client
      slipped through detection (fn > 0);
    - goal-met (``goal_success_rate``): fraction of rounds the defended model's
      accuracy fell to/below ``baseline - target_accuracy_drop`` — i.e. the attack
      actually achieved its requested degradation (only when a goal target is set).

* Targeted (only when the run's goal is ``targeted_label``, and the caller feeds
  per-class evaluations in): the target class's recall vs every other class's, per
  defense. This is the pair of numbers the targeted experiment lives or dies on —
  the attack worked iff the target class collapsed AND the others did not.

Detection counts reuse the existing ``metrics.compute.confusion_counts`` so the
benchmark scores detection exactly like the live system.
"""
from metrics.compute import confusion_counts
from rl.rewards import goal_label, targeted_terms


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


class DefenseMetrics:
    """Accumulates one defense's per-round results into a summary."""

    def __init__(self, name: str, baseline_accuracy: float, target_drop: float | None = None,
                 goal: dict | None = None, win_fraction: float = 0.6):
        self.name = name
        self.baseline = float(baseline_accuracy)
        # --- targeted bookkeeping (inert unless the goal is targeted_label) ---
        self.goal = goal
        self.target_label = goal_label(goal)
        self.win_fraction = float(win_fraction)
        # Per-class recall of the CLEAN (unpoisoned) counterfactual — the reference
        # every per-class number below is compared against. Supplied by the harness
        # once the first round has produced it (:meth:`set_clean_eval`).
        self.clean_eval = None
        self.class_sum: list[float] = []
        self.last_per_class: list[float] = []
        self.target_success_rounds = 0   # rounds the TARGETED goal was met
        self.collateral_sum = 0.0
        self.class_rounds = 0            # rounds a per-class evaluation was supplied
        # The attack's accuracy-degradation GOAL is met in a round when the defended
        # model's accuracy falls to/below ``baseline - target_drop`` (e.g. a 0.10
        # target on a 0.782 baseline -> the attack "succeeds" whenever acc <= 0.682).
        # ``None`` when no goal target is supplied (the goal column then shows n/a).
        self.target_drop = float(target_drop) if target_drop is not None else None
        self.goal_threshold = (self.baseline - self.target_drop
                               if self.target_drop is not None else None)
        self.tp = self.fn = self.fp = self.tn = 0
        self.rounds = 0
        self.acc_sum = 0.0
        self.last_acc = float(baseline_accuracy)
        self.attack_success_rounds = 0   # evasion: rounds a poisoned client slipped through (fn>0)
        self.goal_success_rounds = 0     # goal met: rounds acc fell to/below the goal threshold
        self.skipped = 0                 # rounds the defense produced no new global (kept prior)
        self.history: list[dict] = []

    def set_clean_eval(self, clean_eval) -> None:
        """Install the clean per-class reference (``core.types.ClassEval``).

        Called once by the harness with the round-1 clean counterfactual — the
        accuracy the aggregate reaches with no poison at all — so "the target class
        dropped from X to Y" is measured against an unpoisoned model rather than
        against the previous, already-damaged round.
        """
        self.clean_eval = clean_eval
        if clean_eval is not None and not self.class_sum:
            self.class_sum = [0.0] * len(clean_eval.per_class)
            self.last_per_class = list(clean_eval.per_class)

    def record(self, round_num: int, verdicts, poisoned_ids, accuracy: float, skipped: bool = False,
               class_eval=None):
        tp, fn, fp, tn = confusion_counts(verdicts, set(poisoned_ids))
        self.tp += tp; self.fn += fn; self.fp += fp; self.tn += tn
        self.rounds += 1
        self.acc_sum += float(accuracy)
        self.last_acc = float(accuracy)
        if fn > 0:                       # a poisoned client was NOT flagged -> got through
            self.attack_success_rounds += 1
        # Accuracy-degradation goal met this round? (only when a target was given)
        goal_hit = (self.goal_threshold is not None
                    and float(accuracy) <= self.goal_threshold + 1e-9)
        if goal_hit:
            self.goal_success_rounds += 1
        if skipped:
            self.skipped += 1

        # --- per-class / targeted bookkeeping -----------------------------
        entry_targeted = None
        if class_eval is not None:
            if not self.class_sum:
                self.class_sum = [0.0] * len(class_eval.per_class)
            for i, v in enumerate(class_eval.per_class[:len(self.class_sum)]):
                self.class_sum[i] += float(v)
            self.last_per_class = list(class_eval.per_class)
            self.class_rounds += 1
            terms = targeted_terms(self.goal, self.clean_eval, class_eval)
            if terms is not None:
                self.collateral_sum += terms["collateral"]
                # Same bar the training win-gate uses (rl/switch.attacker_succeeded):
                # enough of the target class destroyed AND collateral within tolerance.
                hit = (terms["target_drop"] >= self.win_fraction * terms["effective_target"]
                       and terms["collateral"] <= terms["max_collateral"])
                if hit:
                    self.target_success_rounds += 1
                entry_targeted = {
                    "label": terms["label"],
                    "target_recall": round(terms["post_recall"], 6),
                    "target_drop": round(terms["target_drop"], 6),
                    "collateral": round(terms["collateral"], 6),
                    "others_recall": round(terms["others_post"], 6),
                    "hit": bool(hit),
                }

        self.history.append({
            "round": round_num, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": float(accuracy), "skipped": bool(skipped), "goal_hit": bool(goal_hit),
            "flagged": sorted(v.client_id for v in verdicts if v.is_suspicious),
            "poisoned": sorted(poisoned_ids),
            "per_class": ([round(float(v), 6) for v in class_eval.per_class]
                          if class_eval is not None else None),
            "targeted": entry_targeted,
        })

    def summary(self) -> dict:
        tpr = _safe_div(self.tp, self.tp + self.fn)
        fpr = _safe_div(self.fp, self.fp + self.tn)
        precision = _safe_div(self.tp, self.tp + self.fp)
        recall = tpr
        f1 = _safe_div(2 * precision * recall, precision + recall)
        mean_acc = _safe_div(self.acc_sum, self.rounds)
        out = {
            "defense": self.name,
            "rounds": self.rounds,
            "malicious_total": self.tp + self.fn,     # poisoned-client instances seen
            "detected": self.tp,                      # of those, how many were caught
            "detection_rate": tpr,                    # = recall = "how much of the attack detected"
            "tpr": tpr,
            "fpr": fpr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarms": self.fp,
            "final_accuracy": self.last_acc,
            "mean_accuracy": mean_acc,
            "mean_acc_drop": self.baseline - mean_acc,
            "attack_success_rate": _safe_div(self.attack_success_rounds, self.rounds),
            "goal_success_rate": (_safe_div(self.goal_success_rounds, self.rounds)
                                  if self.goal_threshold is not None else None),
            "goal_threshold": self.goal_threshold,
            "target_drop": self.target_drop,
            "skipped_rounds": self.skipped,
            "baseline_accuracy": self.baseline,
        }
        out.update(self._targeted_summary())
        return out

    def _targeted_summary(self) -> dict:
        """Per-class + target-class fields. Empty unless this is a targeted run with
        per-class evaluations recorded, so the untargeted report is unchanged."""
        if self.target_label is None or not self.class_rounds or self.clean_eval is None:
            return {}
        L = self.target_label
        mean_per_class = [s / self.class_rounds for s in self.class_sum]
        clean = self.clean_eval.per_class
        others = [i for i in range(len(self.last_per_class)) if i != L]
        return {
            "target_label": L,
            "per_class_clean": [round(v, 6) for v in clean],
            "per_class_final": [round(v, 6) for v in self.last_per_class],
            "per_class_mean": [round(v, 6) for v in mean_per_class],
            # The headline pair: did the target class collapse, and did the rest hold?
            "target_recall_clean": self.clean_eval.recall(L),
            "target_recall_final": (self.last_per_class[L]
                                    if L < len(self.last_per_class) else 0.0),
            "target_recall_mean": mean_per_class[L] if L < len(mean_per_class) else 0.0,
            "target_recall_drop": (self.clean_eval.recall(L)
                                   - (self.last_per_class[L]
                                      if L < len(self.last_per_class) else 0.0)),
            "others_recall_clean": self.clean_eval.others_mean(L),
            "others_recall_final": _mean(self.last_per_class[i] for i in others),
            "others_recall_mean": _mean(mean_per_class[i] for i in others),
            "mean_collateral": _safe_div(self.collateral_sum, self.class_rounds),
            # Fraction of rounds the TARGETED goal was met (target class destroyed
            # enough AND collateral within tolerance) — the targeted analogue of
            # ``goal_success_rate``, scored with the same bar training used.
            "targeted_success_rate": _safe_div(self.target_success_rounds, self.class_rounds),
        }
