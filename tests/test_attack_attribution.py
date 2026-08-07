"""Tests for attack ATTRIBUTION and attack STRENGTH.

These lock in the fixes for a failure mode where every individual number in the
benchmark table was computed correctly and the table as a whole said the opposite
of the truth:

1. **Attribution.** Every defense's accuracy loss was measured against ONE global
   clean baseline, so the accuracy a defense costs *itself* on an honest round was
   credited to the attacker. A defense could exclude every poisoner in 80% of
   rounds and still be reported as suffering a 43%-successful attack. Each defense
   is now probed on the unpoisoned updates each round (``Defense.probe``) and the
   loss splits into ``mean_defense_cost`` and ``mean_attack_drop``, with
   ``goal_success_rate`` scored on the latter.

2. **Strength.** Nothing measured how large the attacker's perturbation actually
   was, so a policy emitting near-no-ops was indistinguishable from one beating a
   real defense — both read as "0% detected, 100% throughput".
   ``perturbation_size`` measures it against the honest update it replaces, and
   anything below ``AttackerAgent.min_perturbation`` is no longer written into the
   ground truth at all.

Needs torch.  Run:  python tests/test_attack_attribution.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from agents.attack_ops import perturbation_size, perturbation_sizes  # noqa: E402
from agents.attacker_agent import AttackerAgent  # noqa: E402
from benchmark.defenses.base import Defense, StepResult  # noqa: E402
from benchmark.metrics import DefenseMetrics  # noqa: E402
from benchmark import report  # noqa: E402
from core.types import DetectionVerdict, ModelUpdate  # noqa: E402


def _sd(v=1.0, n=4):
    return {"w": torch.full((n,), float(v)), "b": torch.full((2,), float(v))}


def _verdicts(flagged, all_ids):
    return [DetectionVerdict(c, c in flagged, 1.0, "") for c in all_ids]


# --- 1. perturbation size ----------------------------------------------------

def test_perturbation_size_scales_are_dimensionless():
    benign, glob = _sd(2.0), _sd(1.0)
    # Poison adds exactly the honest update again -> rel_update == 1.0.
    poisoned = {k: v + (benign[k] - glob[k]) for k, v in benign.items()}
    s = perturbation_size(benign, poisoned, glob)
    assert abs(s["rel_update"] - 1.0) < 1e-6
    # ||P-B|| == ||B-G|| == sqrt(6); ||B|| == 2*sqrt(6) -> rel_weights == 0.5
    assert abs(s["rel_weights"] - 0.5) < 1e-6
    assert s["abs"] > 0.0


def test_perturbation_size_reports_none_without_a_global():
    s = perturbation_size(_sd(1.0), _sd(2.0))
    assert s["rel_update"] is None and s["rel_weights"] > 0.0


def test_perturbation_size_is_inf_when_the_honest_update_is_zero():
    same = _sd(1.0)
    s = perturbation_size(same, _sd(2.0), same)   # benign == global -> honest update 0
    assert s["rel_update"] == float("inf")


def test_perturbation_sizes_skips_clients_without_a_reference():
    sizes = perturbation_sizes({0: _sd(2.0), 9: _sd(2.0)}, {0: _sd(1.0)}, _sd(1.0))
    assert list(sizes) == [0]


# --- 2. the no-op floor is a MAGNITUDE, not byte-equality --------------------

def test_negligible_edit_is_not_counted_as_poison():
    """A plan that technically changes the tensors but is numerically invisible must
    be charged as a wasted quota slot, not written into the ground truth."""
    agent = AttackerAgent()
    pool = {0: _sd(1.0), 1: _sd(1.0)}
    # scale by 1 + 1e-12: every tensor differs, the update does not.
    text = '{"clients":[{"id":0,"operations":[{"op":"scale","factor":1.000000000001}]}]}'
    poisoned, ids, n_malformed = agent.select_and_apply(text, pool, budget=1)
    assert ids == [] and poisoned == {} and n_malformed == 1


def test_a_real_edit_is_still_counted_as_poison():
    agent = AttackerAgent()
    pool = {0: _sd(1.0), 1: _sd(1.0)}
    text = '{"clients":[{"id":0,"operations":[{"op":"scale","factor":1.5}]}]}'
    poisoned, ids, n_malformed = agent.select_and_apply(text, pool, budget=1)
    assert ids == [0] and n_malformed == 0
    assert perturbation_size(pool[0], poisoned[0])["rel_weights"] > 0.4


def test_the_floor_is_configurable():
    loose = AttackerAgent({"min_perturbation": 1.0})   # demand a huge edit
    pool = {0: _sd(1.0)}
    text = '{"clients":[{"id":0,"operations":[{"op":"scale","factor":1.5}]}]}'
    _poisoned, ids, n_malformed = loose.select_and_apply(text, pool, budget=1)
    assert ids == [] and n_malformed == 1             # 0.5 relative < the 1.0 floor


# --- 3. Defense.probe must not advance the defense's world ------------------

class _Counter(Defense):
    """Minimal stateful defense: averages everything and counts its own steps."""

    name = "counter"

    def __init__(self):
        super().__init__("cpu")
        self.steps = 0

    def state_snapshot(self):
        return {"steps": self.steps}

    def state_restore(self, snapshot):
        self.steps = snapshot["steps"]

    def step(self, updates, poisoned_ids):
        self.steps += 1
        new_global = {k: torch.stack([u.weights[k].float() for u in updates]).mean(0)
                      for k in self._global}
        self._global = new_global
        return StepResult(new_global, _verdicts(set(), [u.client_id for u in updates]))


def test_probe_rolls_back_the_global_model_and_the_memory():
    d = _Counter()
    d.reset(_sd(0.0))
    before = {k: v.clone() for k, v in d.global_weights().items()}
    updates = [ModelUpdate(0, _sd(4.0)), ModelUpdate(1, _sd(6.0))]

    probe = d.probe(updates, set())
    # The probe RETURNS the aggregate it would have installed...
    assert abs(float(probe.new_global["w"][0]) - 5.0) < 1e-6
    # ...but leaves neither the model nor the cross-round state advanced.
    assert d.steps == 0
    assert all(torch.equal(d.global_weights()[k], before[k]) for k in before)

    # A real step still advances both, so the probe is not disabling anything.
    d.step(updates, set())
    assert d.steps == 1
    assert abs(float(d.global_weights()["w"][0]) - 5.0) < 1e-6


# --- 4. metrics: the loss splits into defense cost + attack damage ----------

def test_defense_cost_and_attack_drop_are_reported_separately():
    # A defense that costs 0.10 by itself on every honest round, and lets the
    # attack take a further 0.02.
    m = DefenseMetrics("fltrust", baseline_accuracy=0.80, target_drop=0.10)
    for r in range(5):
        m.record(r, _verdicts({0}, [0, 1, 2]), {0}, accuracy=0.68, clean_accuracy=0.70)
    s = m.summary()
    assert abs(s["mean_acc_drop"] - 0.12) < 1e-9        # combined, vs the baseline
    assert abs(s["mean_defense_cost"] - 0.10) < 1e-9    # the defense's own price
    assert abs(s["mean_attack_drop"] - 0.02) < 1e-9     # what the attack added
    assert abs(s["mean_clean_accuracy"] - 0.70) < 1e-9
    assert s["counterfactual_rounds"] == 5


def test_goal_success_is_scored_on_the_attack_not_the_defense_cost():
    """The regression this exists to prevent: a defense whose own cost equals the
    attacker's whole target must NOT read as a fully successful attack."""
    m = DefenseMetrics("fltrust", baseline_accuracy=0.80, target_drop=0.10)
    for r in range(4):
        # The defense drops the model to 0.70 on its own; the attack adds nothing.
        m.record(r, _verdicts({0}, [0, 1, 2]), {0}, accuracy=0.70, clean_accuracy=0.70)
    s = m.summary()
    assert s["goal_success_rate"] == 0.0            # the attack achieved nothing
    assert s["goal_full_success_rate"] == 0.0
    assert abs(s["mean_acc_drop"] - 0.10) < 1e-9   # ...even though 0.10 was lost
    assert abs(s["mean_defense_cost"] - 0.10) < 1e-9


def test_without_a_counterfactual_the_old_baseline_behaviour_is_kept():
    m = DefenseMetrics("d", baseline_accuracy=0.80, target_drop=0.10)
    for r in range(3):
        m.record(r, _verdicts(set(), [0, 1]), {0}, accuracy=0.70)
    s = m.summary()
    assert s["mean_defense_cost"] is None and s["mean_attack_drop"] is None
    assert s["counterfactual_rounds"] == 0
    assert s["goal_success_rate"] == 1.0           # 0.10 drop vs the 0.10 target


def test_history_records_the_per_round_attribution():
    m = DefenseMetrics("d", baseline_accuracy=0.80, target_drop=0.10)
    m.record(1, _verdicts(set(), [0]), {0}, accuracy=0.60, clean_accuracy=0.75,
             poison_ratios=[0.5, 1.5])
    h = m.history[0]
    assert abs(h["clean_accuracy"] - 0.75) < 1e-9
    assert abs(h["attack_drop"] - 0.15) < 1e-9
    assert h["poison_ratios"] == [0.5, 1.5]


def test_mean_acc_drop_never_prints_as_negative_zero():
    m = DefenseMetrics("d", baseline_accuracy=0.655)
    m.record(1, _verdicts(set(), [0]), {0}, accuracy=0.655)
    assert report._cell(m.summary()["mean_acc_drop"], "{:+.3f}") == "+0.000"


# --- 5. attack strength is surfaced in the report ---------------------------

def test_mean_poison_ratio_is_accumulated_over_clients():
    m = DefenseMetrics("d", baseline_accuracy=0.8)
    m.record(1, _verdicts(set(), [0]), {0}, 0.8, poison_ratios=[1.0, 2.0])
    m.record(2, _verdicts(set(), [0]), {0}, 0.8, poison_ratios=[3.0])
    assert abs(m.mean_poison_ratio() - 2.0) < 1e-9
    assert abs(m.summary()["mean_poison_ratio"] - 2.0) < 1e-9


def test_report_warns_when_the_attack_is_too_small_to_mean_anything():
    weak = DefenseMetrics("fedavg", baseline_accuracy=0.655, target_drop=0.1)
    weak.record(1, _verdicts(set(), [0, 1]), {0}, 0.655, clean_accuracy=0.655,
                poison_ratios=[0.001])
    text = report.render([weak.summary()], n_rounds=1, baseline_accuracy=0.655)
    assert "Attack strength" in text and "WARNING" in text
    assert "nothing to detect" in text


def test_report_states_strength_without_warning_for_a_real_attack():
    strong = DefenseMetrics("fedavg", baseline_accuracy=0.655, target_drop=0.1)
    strong.record(1, _verdicts(set(), [0, 1]), {0}, 0.4, clean_accuracy=0.655,
                  poison_ratios=[1.2])
    text = report.render([strong.summary()], n_rounds=1, baseline_accuracy=0.655)
    assert "Attack strength" in text and "WARNING" not in text


def test_report_table_shows_both_attribution_columns():
    m = DefenseMetrics("fltrust", baseline_accuracy=0.8, target_drop=0.1)
    m.record(1, _verdicts({0}, [0, 1]), {0}, 0.68, clean_accuracy=0.70,
             poison_ratios=[1.0])
    table = report.format_table([m.summary()])
    assert "def_cost" in table and "atk_drop" in table
    assert "+0.100" in table and "+0.020" in table


def test_report_table_marks_attribution_na_without_the_counterfactual():
    m = DefenseMetrics("d", baseline_accuracy=0.8, target_drop=0.1)
    m.record(1, _verdicts(set(), [0]), {0}, 0.7)
    assert "n/a" in report.format_table([m.summary()])


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} attribution/strength tests passed.")


if __name__ == "__main__":
    _run()
