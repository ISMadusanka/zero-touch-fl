"""Tests for Phase 2's SIMULATED rounds on the frozen Phase-1 global.

The contract under test (rl.env.FLArmsRaceEnv with fl.freeze_global_in_phase2):
every defender-learning round starts from the SAME Phase-1 anchor, the clients
train on NEW local data, the round's aggregate is SCORED and then discarded, and
nothing the round produces leaks into the next one.

Plus the per-round data refresh (data.round_sampler.RoundDataSampler) that makes
consecutive frozen rounds differ at all.

Synthetic tensors shaped like MNIST — no download, no GPU:
    python tests/test_frozen_rounds.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from core.types import DetectionVerdict  # noqa: E402
from data.round_sampler import RoundDataSampler, build_round_data_sampler  # noqa: E402
from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402

N_CLIENTS = 4
TRAINING_ROUNDS = 45


def _loader(seed: int, n: int = 128):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.randn(n, 1, 28, 28, generator=g),
                      torch.randint(0, 10, (n,), generator=g)),
        batch_size=32, shuffle=True)


def _cfg(frozen=True, benign_retrain=True):
    return {
        "fl": {"n_clients": N_CLIENTS, "device": "cpu",
               "benign_retrain_each_round": benign_retrain,
               "freeze_global_in_phase2": frozen,
               "training_rounds": TRAINING_ROUNDS, "poison_seed": 0,
               "lr": 0.05, "local_epochs": 1, "batch_size": 32},
        # One insider, so the 4-client federation keeps an honest majority. The
        # tests that need a blatantly poisoned cohort build it themselves — see
        # _wrecked_updates — rather than relying on the (deliberately subtle) flip.
        "attack": {"type": "label_flip", "poison_client_ids": [0],
                   "goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.10}},
    }


def _env(frozen=True, benign_retrain=True, round_data=None):
    loaders = [_loader(i) for i in range(N_CLIENTS)]
    env = FLArmsRaceEnv(_cfg(frozen, benign_retrain), loaders, _loader(99, n=256),
                        random.Random(0), round_data=round_data)
    torch.manual_seed(1234)
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    cw = [{k: v + torch.randn_like(v) * 0.01 for k, v in gw.items()}
          for _ in range(N_CLIENTS)]
    env.reset(gw, cw, 0.5)
    return env


def _same(a, b) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def _wreck(sd):
    return {k: v * -50.0 for k, v in sd.items()}


def _wrecked_updates(env, ids=(0, 1)):
    """This round's cohort with ``ids`` replaced by a blatantly poisoned update.

    Built here rather than through the env because the real attack is a LABEL FLIP,
    a genuine SGD trajectory that barely moves the aggregate on these synthetic
    random-label loaders. These tests are about the ANCHOR's bookkeeping, so they
    need a round whose damage is unmistakable.
    """
    from core.types import ModelUpdate
    out = []
    for u in env.build_updates(include_poison=False):
        w = _wreck(u.weights) if u.client_id in set(ids) else u.weights
        out.append(ModelUpdate(u.client_id, w, dict(u.metadata or {})))
    return out


def _clean_verdicts(updates):
    return [DetectionVerdict(u.client_id, False, 0.0, "test") for u in updates]


# ---------------------------------------------------------------------------
# The anchor never moves
# ---------------------------------------------------------------------------

def test_committing_a_round_leaves_the_global_at_the_phase1_anchor():
    env = _env()
    anchor = env.global_weights
    for _ in range(3):
        env.begin_round()
        updates = _wrecked_updates(env, ids=(0,))
        env.commit(updates, _clean_verdicts(updates))
        assert _same(env.global_weights, anchor), \
            "a committed simulated round moved the global off the Phase-1 anchor"


def test_commit_returns_the_rounds_post_attack_accuracy():
    """Frozen or not, commit must report what the round's aggregate scored — that
    number is the attacker's reward signal."""
    env = _env()
    env.begin_round()
    updates = _wrecked_updates(env, ids=(0,))
    verdicts = _clean_verdicts(updates)
    expected = env.evaluate_updates(updates, verdicts)
    assert abs(env.commit(updates, verdicts) - expected) < 1e-12
    assert abs(env.last_round_accuracy - expected) < 1e-12


def test_current_accuracy_stays_the_anchors_accuracy():
    """``current_accuracy`` is what begin_round hands the clients and what the
    attacker's prompt calls the current global — in frozen mode that is always the
    anchor, however much damage a round did."""
    env = _env()
    env.begin_round()
    updates = _wrecked_updates(env)
    post = env.commit(updates, _clean_verdicts(updates))
    assert env.current_accuracy == env.baseline_accuracy
    assert post != env.current_accuracy or post == env.baseline_accuracy


def test_damage_does_not_carry_into_the_next_round():
    """Two identical rounds must produce identical clean counterfactuals: a
    catastrophic round leaves nothing behind for the next one to inherit."""
    env = _env(round_data=None)
    env.begin_round()
    clean_first = env.clean_reference_accuracy()
    updates = _wrecked_updates(env)
    env.commit(updates, _clean_verdicts(updates))

    env.begin_round()
    clean_second = env.clean_reference_accuracy()
    # Same anchor, same data (no sampler), so the next round's honest baseline is
    # unaffected by the wrecking round before it.
    assert abs(clean_first - clean_second) < 0.05


def test_unfrozen_mode_still_advances_the_global():
    env = _env(frozen=False)
    anchor = env.global_weights
    env.begin_round()
    updates = env.build_updates(include_poison=False)
    env.commit(updates, _clean_verdicts(updates))
    assert not _same(env.global_weights, anchor), \
        "freeze_global_in_phase2: false must keep the continuing-federation behaviour"


def test_benign_fl_interlude_is_skipped_when_frozen():
    env = _env()
    anchor = env.global_weights
    before = env.round_index
    assert env.run_benign_fl_round() is None
    assert env.round_index == before, "the skipped interlude still consumed a round number"
    assert _same(env.global_weights, anchor)


def test_benign_retrain_is_forced_on():
    """The label-flip poison IS local training on mislabelled data, so there is no
    frozen-replay mode to fall back to."""
    assert _env(benign_retrain=False).benign_retrain
    assert _env(frozen=False, benign_retrain=False).benign_retrain


def test_restore_reanchors_a_drifted_checkpoint():
    """A checkpoint written before the flag was flipped holds a global that DID
    drift; resuming into a frozen run must re-assert the Phase-1 anchor."""
    env = _env()
    anchor = env.global_weights
    drifted = {k: v + 1.0 for k, v in anchor.items()}
    env.restore_fl_state({"global_weights": drifted,
                          "client_weights": env.client_weights,
                          "current_accuracy": 0.1, "round_index": 7})
    assert _same(env.global_weights, anchor)
    assert env.current_accuracy == env.baseline_accuracy
    assert env.round_index == 7          # the round counter still resumes


# ---------------------------------------------------------------------------
# Clients train on new data every round
# ---------------------------------------------------------------------------

def test_rotate_gives_disjoint_slices_then_recuts():
    sampler = RoundDataSampler([_loader(0, n=100)], fraction=0.25, mode="rotate",
                               batch_size=32, seed=0)
    slices = [set(sampler.indices_for_round(r, 0)) for r in range(4)]
    assert all(len(s) == 25 for s in slices)
    assert len(set().union(*slices)) == 100, "one cycle must cover the whole shard"
    # The shard is re-shuffled per cycle, so round 4 is not a repeat of round 0.
    assert set(sampler.indices_for_round(4, 0)) != slices[0]


def test_resample_draws_a_fresh_subset_each_round():
    sampler = RoundDataSampler([_loader(0, n=100)], fraction=0.25, mode="resample",
                               batch_size=32, seed=0)
    draws = [tuple(sorted(sampler.indices_for_round(r, 0))) for r in range(5)]
    assert all(len(set(d)) == 25 for d in draws), "draws must be without replacement"
    assert len(set(draws)) == 5, "every round should get a different draw"


def test_sampling_is_a_pure_function_of_the_round_index():
    """Resume-safety: a restarted run must reproduce the data the interrupted one
    would have used."""
    a = RoundDataSampler([_loader(0, n=100)], fraction=0.25, seed=3)
    b = RoundDataSampler([_loader(0, n=100)], fraction=0.25, seed=3)
    assert a.indices_for_round(11, 0) == b.indices_for_round(11, 0)


def test_slices_stay_inside_the_clients_own_shard():
    """A slice must never pull another client's examples — that would erase the
    non-IID skew the FLTrust partition builds."""
    loaders = [_loader(i, n=64) for i in range(3)]
    sampler = RoundDataSampler(loaders, fraction=0.5, seed=0)
    for cid, loader in enumerate(loaders):
        shard = set(range(len(loader.dataset)))
        assert set(sampler.indices_for_round(2, cid)) <= shard


def test_batches_per_round_tracks_the_round_slice():
    """FLTrust's root fine-tuning is iteration-matched against this, so it must count
    the ROUND's batches, not the full shard's."""
    sampler = RoundDataSampler([_loader(0, n=100)], fraction=0.25, batch_size=10, seed=0)
    assert sampler.samples_per_round == 25
    assert sampler.batches_per_round == 3        # ceil(25 / 10)


def test_each_round_trains_clients_on_different_data():
    loaders = [_loader(i, n=128) for i in range(N_CLIENTS)]
    sampler = RoundDataSampler(loaders, fraction=0.25, batch_size=16, seed=0)
    env = FLArmsRaceEnv(_cfg(), loaders, _loader(99, n=256), random.Random(0),
                        round_data=sampler)
    gw = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    env.reset(gw, [gw for _ in range(N_CLIENTS)], 0.5)

    env.begin_round()
    first = [set(c.data_loader.dataset.indices) for c in env._clients]
    honest_first = {k: v.clone() for k, v in env.honest_updates[0].weights.items()}
    env.begin_round()
    second = [set(c.data_loader.dataset.indices) for c in env._clients]

    assert all(a != b for a, b in zip(first, second)), \
        "clients replayed the same examples on consecutive rounds"
    assert not _same(env.honest_updates[0].weights, honest_first), \
        "new data must produce a different honest update"


def test_refresh_off_returns_no_sampler():
    cfg = _cfg()
    cfg["fl"]["client_data_refresh"] = "none"
    assert build_round_data_sampler(cfg, [_loader(0)], seed=0) is None


def test_unknown_refresh_mode_is_rejected():
    try:
        RoundDataSampler([_loader(0)], mode="sometimes")
    except ValueError as e:
        assert "client_data_refresh" in str(e)
    else:
        raise AssertionError("an unknown refresh mode must not be silently accepted")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
