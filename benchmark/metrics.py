"""Per-defense metric accumulation for the benchmark.

Two families of metrics, gathered per defense over all rounds:

* Detection (per-client accept/reject vs the ground-truth poisoned set):
  detection-rate (= recall = TPR), FPR, precision, F1.
* Robustness (the resulting model under attack):
  final & mean test accuracy, mean accuracy drop vs the clean baseline, and two
  distinct "attack success" measures:
    - evasion (``attack_success_rate``): fraction of rounds a poisoned client
      slipped through detection (fn > 0);
    - goal-met (``goal_success_rate``): the attack's **weighted** success against
      its requested degradation — per round ``min(1, acc_drop / target_drop)``,
      averaged over rounds (only when a goal target is set). A round that achieved
      the full requested drop scores 1.0, one that achieved half of it 0.5, one that
      cost the model nothing 0.0 — so a partially effective attack is credited
      partially instead of being written off as a failure.
      ``goal_full_success_rate`` keeps the older all-or-nothing view alongside it:
      the fraction of rounds that reached the target in full.

Detection counts reuse the existing ``metrics.compute.confusion_counts`` so the
benchmark scores detection exactly like the live system.
"""
from metrics.compute import confusion_counts


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class DefenseMetrics:
    """Accumulates one defense's per-round results into a summary."""

    def __init__(self, name: str, baseline_accuracy: float, target_drop: float | None = None):
        self.name = name
        self.baseline = float(baseline_accuracy)
        # The attack's accuracy-degradation GOAL asks for ``target_drop`` of accuracy,
        # i.e. it is met IN FULL in a round when the defended model's accuracy falls
        # to/below ``baseline - target_drop`` (e.g. a 0.10 target on a 0.782 baseline ->
        # fully met whenever acc <= 0.682). Rounds in between are scored proportionally
        # -- see :meth:`goal_score`. ``None`` when no goal target is supplied (the goal
        # column then shows n/a).
        self.target_drop = float(target_drop) if target_drop is not None else None
        self.goal_threshold = (self.baseline - self.target_drop
                               if self.target_drop is not None else None)
        self.tp = self.fn = self.fp = self.tn = 0
        self.rounds = 0
        self.acc_sum = 0.0
        self.last_acc = float(baseline_accuracy)
        self.attack_success_rounds = 0   # evasion: rounds a poisoned client slipped through (fn>0)
        self.goal_success_sum = 0.0      # weighted goal success in [0,1], summed over rounds
        self.goal_success_rounds = 0     # goal met IN FULL: rounds acc fell to/below the threshold
        self.skipped = 0                 # rounds the defense produced no new global (kept prior)
        self.history: list[dict] = []

    def goal_score(self, accuracy: float) -> float | None:
        """Weighted attack success for ONE round, in [0, 1] (``None`` with no target).

        The attack asked to cost the model ``target_drop`` of accuracy; this round it
        actually cost ``baseline - accuracy``. The score is the ratio of the two, so a
        partially effective attack gets partial credit rather than being counted as a
        flat failure: at a target of 0.10, a realised drop of 0.10 scores 100%, 0.05
        scores 50%, 0.02 scores 20%.

        Overshoot is capped at 1.0 — the metric answers "how much of its goal did the
        attack achieve?", which keeps it a readable percentage and keeps the ceiling
        comparable across defenses. (Rewarding overshoot is the training signal's job,
        not the report's; see ``rl.rewards.drop_term``.) A round that left the model no
        worse than the clean baseline — or improved it — scores 0.0.
        """
        if self.target_drop is None:
            return None
        drop = self.baseline - float(accuracy)
        if self.target_drop <= 0.0:      # degenerate target: any loss at all is "the goal"
            return 1.0 if drop >= 0.0 else 0.0
        return _clip(drop / self.target_drop, 0.0, 1.0)

    def record(self, round_num: int, verdicts, poisoned_ids, accuracy: float, skipped: bool = False):
        tp, fn, fp, tn = confusion_counts(verdicts, set(poisoned_ids))
        self.tp += tp; self.fn += fn; self.fp += fp; self.tn += tn
        self.rounds += 1
        self.acc_sum += float(accuracy)
        self.last_acc = float(accuracy)
        if fn > 0:                       # a poisoned client was NOT flagged -> got through
            self.attack_success_rounds += 1
        # How much of the requested accuracy drop this round achieved, in [0,1]
        # (``None`` when no goal target was given). Full credit == the goal was met.
        score = self.goal_score(accuracy)
        goal_hit = score is not None and score >= 1.0 - 1e-9
        if score is not None:
            self.goal_success_sum += score
            if goal_hit:
                self.goal_success_rounds += 1
        if skipped:
            self.skipped += 1
        self.history.append({
            "round": round_num, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": float(accuracy), "skipped": bool(skipped),
            "goal_success": score, "goal_hit": bool(goal_hit),
            "flagged": sorted(v.client_id for v in verdicts if v.is_suspicious),
            "poisoned": sorted(poisoned_ids),
        })

    def summary(self) -> dict:
        tpr = _safe_div(self.tp, self.tp + self.fn)
        fpr = _safe_div(self.fp, self.fp + self.tn)
        precision = _safe_div(self.tp, self.tp + self.fp)
        recall = tpr
        f1 = _safe_div(2 * precision * recall, precision + recall)
        mean_acc = _safe_div(self.acc_sum, self.rounds)
        return {
            "defense": self.name,
            "rounds": self.rounds,
            "malicious_total": self.tp + self.fn,     # poisoned-client instances seen
            # Mean clients actually poisoned per round. The eval budget
            # (--max-poison-clients) is a CEILING, not a quota: how many of its pool the
            # attacker recruits is part of its action, and it was trained with
            # rl.reward.attacker.delta charging it for each extra client. Report the
            # realised count so "budget 10" is never mistaken for "10 were poisoned".
            "mean_poisoned": _safe_div(self.tp + self.fn, self.rounds),
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
            # atk_succ — WEIGHTED: mean per-round min(1, acc_drop / target_drop), so a
            # round that achieved half the requested degradation counts as 50% success.
            "goal_success_rate": (_safe_div(self.goal_success_sum, self.rounds)
                                  if self.target_drop is not None else None),
            # ...and the all-or-nothing view kept alongside it: rounds that hit the
            # target in full (what atk_succ used to report on its own).
            "goal_full_success_rate": (_safe_div(self.goal_success_rounds, self.rounds)
                                       if self.target_drop is not None else None),
            "goal_threshold": self.goal_threshold,
            "target_drop": self.target_drop,
            "skipped_rounds": self.skipped,
            "baseline_accuracy": self.baseline,
        }
