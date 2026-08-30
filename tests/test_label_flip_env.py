"""End-to-end tests for the label-flip attack inside the FL environment.

Covers the wiring in ``rl/env.py``: that poisoned clients really train on flipped
labels, that the clean counterfactual is the SAME clients on their real labels,
that ground truth matches what shipped, and that the ladder advances exactly once
per committed round.

Synthetic tensors shaped like MNIST — no download, no GPU, no LLM:
    python tests/test_label_flip_env.py
"""
import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                              # noqa: E402
from torch.utils.data import DataLoader, TensorDataset    # noqa: E402

from core.types import DetectionVerdict                   # noqa: E402
from model.mnist_net import MnistNet                      # noqa: E402
from rl.env import FLArmsRaceEnv                          # noqa: E402

N_CLIENTS = 6
SAMPLES = 40


def _loaders(n_clients=N_CLIENTS, samples=SAMPLES, seed=0):
    """Tiny per-client loaders shaped like MNIST (1x28x28 images, labels 0..9)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for cid in range(n_clients):
        x = torch.randn(samples, 1, 28, 28, generator=g)
        y = torch.randint(0, 10, (samples,), generator=g)
        out.append(DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False))
    return out


def _test_loader(seed=99):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.randn(32, 1, 28, 28, generator=g),
                                    torch.randint(0, 10, (32,), generator=g)),
                      batch_size=16)


def _cfg(poison_ids=(0,), **schedule):
    return {
        "fl": {"n_clients": N_CLIENTS, "device": "cpu", "training_rounds": 2,
               "lr": 0.05, "local_epochs": 1, "poison_seed": 0, "batch_size": 8,
               "freeze_global_in_phase2": True},
        "attack": {
            "type": "label_flip",
            "poison_client_ids": list(poison_ids),
            "schedule": dict({"start_fraction": 1.0, "step_fraction": 0.1,
                              "floor_fraction": 0.5, "caught_rule": "all"}, **schedule),
            "goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.1},
        },
    }


def _global_weights(seed=1234):
    """A FIXED starting model, so two envs built in one process start identically
    (``MnistNet()`` draws a fresh random init every call)."""
    torch.manual_seed(seed)
    return {k: v.clone() for k, v in MnistNet().state_dict().items()}


def _env(poison_ids=(0,), **schedule):
    gw = _global_weights()
    env = FLArmsRaceEnv(_cfg(poison_ids, **schedule), _loaders(), _test_loader(),
                        random.Random(0))
    env.reset(gw, [copy.deepcopy(gw) for _ in range(N_CLIENTS)], 0.5)
    return env


def _verdicts(env, flagged):
    return [DetectionVerdict(cid, cid in set(flagged), 1.0, "")
            for cid in range(env.n_clients)]


# ---------------------------------------------------------------------------

def test_only_the_configured_clients_are_poisoned():
    env = _env(poison_ids=(0, 2))
    ctx = env.begin_round()
    assert ctx.poisoned_ids == [0, 2]
    assert sorted(ctx.flip_plan) == [0, 2]
    updates = env.build_updates()
    assert [u.client_id for u in updates] == list(range(N_CLIENTS))
    assert [u.metadata["poisoned"] for u in updates] == [True, False, True, False, False, False]


def test_first_round_flips_every_label_the_client_holds():
    """The spec's starting point: 100% of the client's round data."""
    env = _env()
    ctx = env.begin_round()
    assert ctx.flip_fraction == 1.0
    assert ctx.flip_plan[0] == SAMPLES
    meta = env.poisoned_updates[0].metadata
    assert meta["n_flipped"] == SAMPLES and meta["n_local_samples"] == SAMPLES
    assert meta["attack"] == "label_flip" and meta["flip_fraction"] == 1.0


def test_poisoned_update_differs_from_the_clean_counterfactual():
    """The poisoned and honest updates come from the SAME client, the SAME data
    and the SAME starting model — so any difference between them is the labels,
    which is exactly what makes the counterfactual exact."""
    env = _env()
    env.begin_round()
    poisoned = env.poisoned_updates[0].weights
    honest = env.honest_updates[0].weights
    assert any(not torch.equal(poisoned[k], honest[k]) for k in honest), \
        "flipping every label must change the update"
    # ...and the honest cohort really is untouched by the attack.
    clean = env.build_updates(include_poison=False)
    assert all(torch.equal(clean[0].weights[k], honest[k]) for k in honest)
    assert all(v is False for v in [u.metadata["poisoned"] for u in clean])


def test_ground_truth_excludes_a_client_that_flipped_nothing():
    """A ladder level that rounds to zero flips sends an HONEST update. Calling it
    poison would corrupt the defender's reward and the ladder's own feedback."""
    # 2 samples at a 0.2 fraction rounds to 0 flips.
    env = FLArmsRaceEnv(_cfg((0,), start_fraction=0.2, floor_fraction=0.2),
                        _loaders(samples=2), _test_loader(), random.Random(0))
    gw = _global_weights()
    env.reset(gw, [copy.deepcopy(gw) for _ in range(N_CLIENTS)], 0.5)
    ctx = env.begin_round()
    assert ctx.flip_plan == {0: 0}
    assert ctx.poisoned_ids == [], "no flipped labels means no poisoned client"
    assert all(u.metadata["poisoned"] is False for u in env.build_updates())


def test_ladder_advances_once_per_committed_round_not_per_rollout():
    """The GRPO loop scores G rollouts against one cohort. If each of them moved
    the ladder, the attack schedule would depend on rl.G — a sampling
    hyperparameter — instead of on whether the defense caught anything."""
    env = _env()
    env.begin_round()
    assert env.attacker.fraction == 1.0
    rec = env.record_detection(_verdicts(env, flagged=[0]))
    assert rec["event"] == "step_down" and env.attacker.fraction == 0.9
    # Any further call in the SAME round is ignored.
    for _ in range(4):
        assert env.record_detection(_verdicts(env, flagged=[0])) == {}
    assert env.attacker.fraction == 0.9


def test_caught_and_missed_rounds_drive_the_saw_tooth_end_to_end():
    """The behaviour the whole change exists for, through the real env."""
    env = _env()
    sent = []
    # Caught every round: walk down to the floor and reset.
    for _ in range(7):
        ctx = env.begin_round()
        sent.append(ctx.flip_fraction)
        env.record_detection(_verdicts(env, flagged=[0]))
    assert sent == [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 1.0]

    # Now miss it: the level holds, unchanged, for as long as it keeps working.
    env2 = _env()
    held = []
    for _ in range(4):
        ctx = env2.begin_round()
        held.append(ctx.flip_fraction)
        env2.record_detection(_verdicts(env2, flagged=[]))
    assert held == [1.0, 1.0, 1.0, 1.0]


def test_multi_client_round_steps_only_when_every_insider_is_caught():
    env = _env(poison_ids=(0, 1))
    env.begin_round()
    # One insider survives -> the round succeeded -> hold.
    env.record_detection(_verdicts(env, flagged=[0]))
    assert env.attacker.fraction == 1.0
    env.begin_round()
    env.record_detection(_verdicts(env, flagged=[0, 1]))
    assert env.attacker.fraction == 0.9


def test_flip_count_scales_with_the_level():
    env = _env()
    counts = []
    for _ in range(4):
        ctx = env.begin_round()
        counts.append(ctx.flip_plan[0])
        env.record_detection(_verdicts(env, flagged=[0]))
    assert counts == [SAMPLES, round(0.9 * SAMPLES), round(0.8 * SAMPLES),
                      round(0.7 * SAMPLES)]


def test_poisoned_round_is_reproducible_from_the_same_seed():
    """A resumed run must reproduce the interrupted one's poison exactly — the
    same examples mislabelled AND the same SGD order over them."""
    a, b = _env(), _env()
    a.begin_round()
    b.begin_round()
    for k in a.poisoned_updates[0].weights:
        assert torch.equal(a.poisoned_updates[0].weights[k],
                           b.poisoned_updates[0].weights[k]), \
            "same seed + same round must produce the same poisoned update"


def test_frozen_anchor_survives_a_committed_round():
    env = _env()
    anchor = copy.deepcopy(env.global_weights)
    ctx = env.begin_round()
    updates = env.build_updates()
    env.commit(updates, _verdicts(env, flagged=[]))
    env.record_detection(_verdicts(env, flagged=[]))
    assert all(torch.equal(env.global_weights[k], anchor[k]) for k in anchor), \
        "a frozen simulated round must not move the global model"
    assert ctx.clean_measured is True and ctx.defense_sane is True


def test_benign_retrain_is_forced_on():
    """Label flipping IS local training on bad labels, so it cannot be replayed
    from frozen weights."""
    cfg = _cfg()
    cfg["fl"]["benign_retrain_each_round"] = False
    env = FLArmsRaceEnv(cfg, _loaders(), _test_loader(), random.Random(0))
    assert env.benign_retrain is True


def test_poisoned_client_reuses_its_honest_loader_next_round():
    """The flip set is per round; a poisoned loader must not leak into the next
    round's plan."""
    env = _env()
    env.begin_round()
    twin = env._flip_clients[0]
    assert len(twin.data_loader.dataset) == SAMPLES
    honest_labels = [int(twin.data_loader.dataset[i][1]) for i in range(SAMPLES)]
    env.record_detection(_verdicts(env, flagged=[0]))
    env.begin_round()
    assert [int(twin.data_loader.dataset[i][1]) for i in range(SAMPLES)] == honest_labels


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} label-flip env tests passed.")
