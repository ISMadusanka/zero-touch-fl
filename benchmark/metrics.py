"""Per-defense metric accumulation for the benchmark.

Two families of metrics, gathered per defense over all rounds:

* Detection (per-client accept/reject vs the ground-truth poisoned set):
  detection-rate (= recall = TPR), FPR, precision, F1.
* Robustness (the resulting model under attack):
  final & mean test accuracy, mean accuracy drop vs the clean baseline, and the
  attack-success rate (fraction of rounds in which a poisoned client slipped
  through, i.e. fn > 0).

Detection counts reuse the existing ``metrics.compute.confusion_counts`` so the
benchmark scores detection exactly like the live system.

``baseline_class_accuracy`` / ``class_accuracy`` (both optional): when the
attack goal is ``targeted_label``, global accuracy alone can look fine while
one class is quietly destroyed — that is the whole point of the dual-objective
targeted reward (Section on ``rl.rewards.attacker_reward``). Passing per-class
accuracy each round lets ``summary()`` additionally report the worst-hit
class's accuracy and drop, independent of which class that turns out to be
(the attacker may pick a different one each round under ``label: "menu"``).
Left ``None`` for the untargeted goal, where this tracking is meaningless.
"""
from metrics.compute import confusion_counts


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


class DefenseMetrics:
    """Accumulates one defense's per-round results into a summary."""

    def __init__(self, name: str, baseline_accuracy: float,
                baseline_class_accuracy: dict | None = None):
        self.name = name
        self.baseline = float(baseline_accuracy)
        self.baseline_class_accuracy = (
            {str(k): float(v) for k, v in baseline_class_accuracy.items()}
            if baseline_class_accuracy else None
        )
        self.tp = self.fn = self.fp = self.tn = 0
        self.rounds = 0
        self.acc_sum = 0.0
        self.last_acc = float(baseline_accuracy)
        self.last_class_accuracy: dict | None = None
        self.attack_success_rounds = 0
        self.skipped = 0                 # rounds the defense produced no new global (kept prior)
        self.history: list[dict] = []

    def record(self, round_num: int, verdicts, poisoned_ids, accuracy: float,
              skipped: bool = False, class_accuracy: dict | None = None):
        tp, fn, fp, tn = confusion_counts(verdicts, set(poisoned_ids))
        self.tp += tp; self.fn += fn; self.fp += fp; self.tn += tn
        self.rounds += 1
        self.acc_sum += float(accuracy)
        self.last_acc = float(accuracy)
        if fn > 0:                       # a poisoned client was NOT flagged -> got through
            self.attack_success_rounds += 1
        if skipped:
            self.skipped += 1
        class_acc_row = None
        if class_accuracy is not None:
            class_acc_row = {str(k): float(v) for k, v in class_accuracy.items()}
            self.last_class_accuracy = class_acc_row
        self.history.append({
            "round": round_num, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": float(accuracy), "skipped": bool(skipped),
            "flagged": sorted(v.client_id for v in verdicts if v.is_suspicious),
            "poisoned": sorted(poisoned_ids),
            "class_accuracy": class_acc_row,
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
            "skipped_rounds": self.skipped,
            "baseline_accuracy": self.baseline,
            "worst_class": None,
            "worst_class_accuracy": None,
            "worst_class_drop": None,
        }
        if self.baseline_class_accuracy and self.last_class_accuracy is not None:
            drops = {c: self.baseline_class_accuracy.get(c, 0.0) - self.last_class_accuracy.get(c, 0.0)
                    for c in self.last_class_accuracy}
            if drops:
                worst_class = max(drops, key=drops.get)
                out["worst_class"] = worst_class
                out["worst_class_accuracy"] = self.last_class_accuracy.get(worst_class)
                out["worst_class_drop"] = drops[worst_class]
        return out
