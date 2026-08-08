"""Canonical attack-goal success and detection-evasion metrics."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.types import DetectionVerdict
from metrics.compute import compute_round_metrics
from metrics.tracker import MetricsTracker


def _verdict(client_id: int, suspicious: bool) -> DetectionVerdict:
    return DetectionVerdict(client_id, suspicious, 1.0, "test")


def test_evasion_without_target_damage_is_not_attack_success():
    metrics = compute_round_metrics(
        round_num=1,
        verdicts=[_verdict(0, False), _verdict(1, False)],
        malicious_ids={0},
        current_accuracy=0.79,
        baseline_accuracy=0.90,
        reference_accuracy=0.80,
        target_accuracy_drop=0.10,
    )

    assert metrics.evasion_success is True
    assert metrics.attack_success is False
    assert metrics.induced_drop == pytest.approx(0.01)


def test_target_damage_is_success_even_when_poisoner_is_caught():
    metrics = compute_round_metrics(
        round_num=2,
        verdicts=[_verdict(0, True), _verdict(1, False)],
        malicious_ids={0},
        current_accuracy=0.70,
        baseline_accuracy=0.90,
        reference_accuracy=0.80,
        target_accuracy_drop=0.10,
    )

    assert metrics.attack_success is True
    assert metrics.evasion_success is False
    assert metrics.reference_accuracy == 0.80
    assert metrics.target_accuracy_drop == 0.10


def test_success_uses_clean_reference_not_phase_one_baseline():
    metrics = compute_round_metrics(
        round_num=3,
        verdicts=[_verdict(0, False)],
        malicious_ids={0},
        current_accuracy=0.75,
        baseline_accuracy=0.90,
        reference_accuracy=0.80,
    )

    # Baseline loss is 0.15, but the attack itself induced only 0.05.
    assert metrics.induced_drop == pytest.approx(0.05)
    assert metrics.attack_success is False
    assert metrics.evasion_success is True


def test_no_attack_cannot_receive_goal_success():
    metrics = compute_round_metrics(
        round_num=4,
        verdicts=[_verdict(0, False)],
        malicious_ids=set(),
        current_accuracy=0.60,
        baseline_accuracy=0.80,
        reference_accuracy=0.80,
    )

    assert metrics.induced_drop == pytest.approx(0.20)
    assert metrics.attack_success is False
    assert metrics.evasion_success is False


def test_default_reference_is_backwards_compatible_and_target_is_validated():
    metrics = compute_round_metrics(
        round_num=5,
        verdicts=[_verdict(0, False)],
        malicious_ids={0},
        current_accuracy=0.70,
        baseline_accuracy=0.80,
    )
    assert metrics.reference_accuracy == 0.80
    assert metrics.attack_success is True

    with pytest.raises(ValueError, match="target_accuracy_drop"):
        compute_round_metrics(
            round_num=6,
            verdicts=[_verdict(0, False)],
            malicious_ids={0},
            current_accuracy=0.70,
            baseline_accuracy=0.80,
            target_accuracy_drop=0.0,
        )


def test_tracker_aggregates_goal_success_and_evasion_separately(caplog):
    caplog.set_level(logging.INFO, logger="metrics.tracker")
    with patch.object(MetricsTracker, "_save_round", return_value=None):
        tracker = MetricsTracker(0.90, output_dir=".")

        # Goal hit, caught by the detector.
        tracker.update(
            1,
            [_verdict(0, True), _verdict(1, False)],
            0.70,
            {0},
            reference_accuracy=0.80,
            target_accuracy_drop=0.10,
        )
        # Evaded, but harmless.
        tracker.update(
            2,
            [_verdict(0, False), _verdict(1, False)],
            0.80,
            {0},
            reference_accuracy=0.80,
            target_accuracy_drop=0.10,
        )

    aggregate = tracker.aggregate()
    assert aggregate.attack_success_rate == 0.5
    assert aggregate.evasion_success_rate == 0.5

    saved = [round_metrics.to_dict() for round_metrics in tracker.rounds]
    assert saved[0]["attack_success"] is True
    assert saved[0]["evasion_success"] is False
    assert saved[1]["attack_success"] is False
    assert saved[1]["evasion_success"] is True
    assert "induced_drop=+0.1000 target=0.1000" in caplog.text


def test_schedule_logs_exact_reward_components_and_soft_probabilities(caplog):
    from rl.schedule import _log_round

    goal = {"type": "untargeted_degrade", "target_accuracy_drop": 0.10}
    verdicts = [
        DetectionVerdict(0, False, 0.80, "calibrated", p_malicious=0.25),
        DetectionVerdict(1, False, 0.60, "derived"),
    ]
    env = SimpleNamespace(goal=goal, pool_benign={}, baseline_accuracy=0.90)
    ctx = SimpleNamespace(
        round_num=7,
        goal=goal,
        clean_accuracy=0.80,
        global_accuracy=0.81,
        budget=1,
        pool_ids=[0, 1],
    )
    info = {
        "verdicts": verdicts,
        "post_accuracy": 0.70,
        "n_malformed": 0,
        "poisoned_ids": [0],
        "poisoned_by_client": {},
        "perturbation_ratios": {0: 1.0},
    }
    stats = {
        "loss": 0.1,
        "mean_reward": 1.0,
        "max_reward": 1.0,
        "rewards": [1.0],
        "zero_advantage_fraction": 0.0,
        "stepped": True,
    }
    saved_logs = []
    caplog.set_level(logging.INFO, logger="rl.schedule")

    with patch.object(MetricsTracker, "_save_round", return_value=None):
        tracker = MetricsTracker(0.90, output_dir=".")
        _log_round(
            env,
            ctx,
            info,
            "attacker",
            stats,
            tracker,
            saved_logs.append,
            success=True,
            poisoned_ids=[0],
        )

    logged = saved_logs[0]
    parts = logged.attack_metadata["attacker_reward_components"]
    assert logged.attacker_reward == pytest.approx(parts["total"])
    assert parts["induced_drop"] == pytest.approx(0.10)
    assert parts["damage_component"] == pytest.approx(1.0)
    assert parts["stealth_component"] == pytest.approx(0.375)

    assert logged.predicted_labels[0]["p_malicious"] == pytest.approx(0.25)
    assert logged.predicted_labels[0]["p_malicious_source"] == "calibrated"
    assert logged.predicted_labels[1]["p_malicious"] == pytest.approx(0.20)
    assert logged.predicted_labels[1]["p_malicious_source"] == "derived"
    assert logged.attack_metadata["poisoned_p_malicious"] == {0: 0.25}
    assert "parts(dmg=1.000 stealth=0.375" in caplog.text
    assert "p_mal={0:0.250}" in caplog.text
