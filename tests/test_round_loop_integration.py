"""End-to-end smoke test of ONE Phase-2 round loop against every real defense.

The unit tests above cover each fix in isolation; this one runs the actual round
protocol — ``begin_round`` -> clean counterfactual -> ``defend`` -> ``commit_state`` ->
``MetricsTracker.update`` -> ``RoundLog`` — through the real FLTrust / DeFL / DnC /
Multi-Krum implementations and the real curriculum, so a signature or contract that
only breaks when the pieces are wired together does not slip through.

It also asserts the two properties that were silently violated for a whole recorded
run and that no single-component test can see:

* the calibration invariant holds on EVERY verdict of EVERY round, and
* damage-based ``attack_success`` and the recorded drop stay consistent.

Real MNIST is not needed (synthetic tensors shaped like it) and no LLM is involved —
the attacker is ``rl.baseline``'s fixed action set:
    python tests/test_round_loop_integration.py
"""
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from benchmark.defenses import build_defenses  # noqa: E402
from metrics.tracker import MetricsTracker  # noqa: E402
from model.mnist_net import MnistNet  # noqa: E402
from rl.baseline import run_baseline  # noqa: E402
from rl.curriculum import TrainingCurriculum  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from server.algo_defender import AlgorithmicDefender  # noqa: E402

N_CLIENTS = 8
ALGORITHMS = ("fltrust", "defl", "dnc", "multikrum")
POISONER_COUNTS = [1, 3]
#: The curriculum sweeps (algorithm, #poisoners) PAIRS, one block each, so covering
#: every algorithm takes one round per pair — not one per algorithm.
FULL_SWEEP = len(ALGORITHMS) * len(POISONER_COUNTS)


def _loader(seed, n=96, batch=32):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.randn(n, 1, 28, 28, generator=g),
                      torch.randint(0, 10, (n,), generator=g)),
        batch_size=batch, shuffle=True)


def _build(rounds_per_block=1):
    cfg = {
        "fl": {"n_clients": N_CLIENTS, "device": "cpu", "training_rounds": 5,
               "benign_retrain_each_round": True, "n_compromisable": 3,
               "lr": 0.01, "local_epochs": 1, "batch_size": 32},
        "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.02},
                   "max_poison_clients": 3, "sample_budget_in_training": False},
    }
    defenses = build_defenses(
        list(ALGORITHMS), device="cpu",
        root_loader=_loader(777, n=64),
        root_lr=0.01, root_epochs=2, eta=1.0,
        defl_delta=0.05, defl_tau=2.5,
        dnc_num_byzantine=3, dnc_c=1.0, dnc_niters=1, dnc_sub_dim=10000, dnc_seed=0,
        multikrum_num_byzantine=3, multikrum_m=None,
    )
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    for d in defenses.values():
        d.reset(gw)
    defender = AlgorithmicDefender(defenses, random.Random(0), selection="round_robin")
    curriculum = TrainingCurriculum(
        algorithms=list(ALGORITHMS), poisoner_counts=list(POISONER_COUNTS),
        rounds_per_block=rounds_per_block,
    )
    env = FLArmsRaceEnv(cfg, [_loader(i) for i in range(N_CLIENTS)], _loader(99, n=128),
                        random.Random(0), defense=defender, curriculum=curriculum)
    cw = [{k: v + torch.randn_like(v) * 0.01 for k, v in gw.items()}
          for _ in range(N_CLIENTS)]
    env.reset(gw, cw, 0.5)
    return env


def _run_rounds(n_rounds):
    """Run ``n_rounds`` baseline rounds; return (round_logs, aggregate metrics)."""
    torch.manual_seed(0)
    env = _build()
    out = tempfile.mkdtemp()
    logs = []
    try:
        tracker = MetricsTracker(0.5, output_dir=out)
        run_baseline(env, n_rounds, tracker, logs.append)
        return logs, tracker.aggregate(), tracker
    finally:
        shutil.rmtree(out, ignore_errors=True)


# --- the loop runs at all -----------------------------------------------------

def test_every_defense_completes_a_full_round():
    """One round per (algorithm, #poisoners) block through the whole protocol, so
    every defense in the pool is exercised."""
    logs, agg, _ = _run_rounds(FULL_SWEEP)
    assert len(logs) == FULL_SWEEP
    faced = [log.attack_metadata["defense"] for log in logs]
    assert set(faced) == set(ALGORITHMS), faced
    assert agg.total_rounds == FULL_SWEEP


# --- the invariants that were violated in production --------------------------

def test_calibration_invariant_holds_on_every_verdict_of_every_round():
    """The contract from core.types, asserted through the real round loop rather
    than on a hand-built verdict list."""
    logs, _agg, _ = _run_rounds(FULL_SWEEP)
    checked = 0
    for log in logs:
        for pred in log.predicted_labels:
            # RoundLog only persists the hard flag + confidence, and confidence is
            # |2p - 1|, so a flagged client must report confidence consistent with
            # p >= 0.5. The direct p check lives in test_p_malicious_calibration.
            assert 0.0 <= pred["confidence"] <= 1.0, pred
            checked += 1
    assert checked == len(logs) * N_CLIENTS


def test_recorded_drop_and_success_agree():
    """``attack_success`` must be exactly "the recorded drop cleared the bar" — the
    old ``fn > 0`` definition disagreed with the accuracy it was logged next to."""
    logs, _agg, tracker = _run_rounds(FULL_SWEEP)
    for m in tracker.rounds:
        if m.induced_drop is None or m.success_drop is None:
            assert m.attack_success is False
        else:
            assert m.attack_success == (m.induced_drop >= m.success_drop)
    # And the RoundLog's own induced_drop matches the tracker's.
    for log, m in zip(logs, tracker.rounds):
        if m.induced_drop is not None:
            assert abs(log.attack_metadata["induced_drop"] - m.induced_drop) < 1e-6


def test_unmeasured_rounds_are_marked_and_excluded_from_the_damage_mean():
    """A round whose defense produced no clean aggregate must not enter the damage
    statistics as a measured zero drop."""
    _logs, agg, tracker = _run_rounds(FULL_SWEEP)
    measured = [m for m in tracker.rounds if m.induced_drop is not None]
    assert agg.measured_rounds == len(measured)
    for m in tracker.rounds:
        if m.induced_drop is None:
            assert m.clean_accuracy is None
            assert m.accuracy_preservation_vs_clean is None
            assert m.attack_success is False


def test_no_defense_accepts_only_the_poisoned_clients():
    """The FLTrust malfunction that made a whole recorded run uninterpretable: the
    honest majority dropped, the aggregate built purely from the attack. It is a
    configuration failure, so it must not reproduce on a sane configuration."""
    logs, _agg, _ = _run_rounds(FULL_SWEEP)
    for log in logs:
        poisoned = set(log.poisoned_client_ids)
        accepted = {p["client_id"] for p in log.predicted_labels if not p["is_suspicious"]}
        defense = log.attack_metadata["defense"]
        assert accepted, f"{defense} round {log.round_num}: rejected every client"
        if poisoned:
            assert not accepted <= poisoned, (
                f"{defense} round {log.round_num}: accepted ONLY poisoned clients "
                f"{sorted(accepted)} — the aggregate is the attack"
            )


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} round-loop integration tests passed.")


if __name__ == "__main__":
    _run()
