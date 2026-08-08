"""Tests for the SINGLE-INSIDER targeted setup: poison only client 0, and attack
the class client 0's own non-IID shard is dominated by (derived at runtime).

Covers
  * data.mnist_loader.client_label_counts — shard histograms through DataLoader /
    Subset / nested Subset / bare dataset
  * data.target_label.resolve_client_target_label — the derivation itself, the three
    config fields it pins, its no-ops and its errors
  * the consequences: the env exposes a pool of exactly [0] with budget 1, every
    round's goal carries the derived label, and the attacker can only land on client 0

Synthetic data + tiny tensors — no MNIST download, no GPU, no LLM:
    python tests/test_client_target_label.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset, TensorDataset  # noqa: E402

from agents.attacker_agent import AttackerAgent  # noqa: E402
from data.mnist_loader import client_label_counts, partition_noniid_fltrust  # noqa: E402
from data.target_label import (  # noqa: E402
    CLIENT_KEY, dominant_label, resolve_client_target_label,
)
from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402

N_CLIENTS = 6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeDS:
    """Minimal dataset stand-in exposing `.targets` (what the histogram reads)."""

    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)


def _balanced_targets(per_class: int, n_classes: int = 10):
    return [c for _ in range(per_class) for c in range(n_classes)]


def _cfg(*, from_client=0, label=2, n_compromisable=1, max_poison=1,
         goal_type="targeted_label", n_classes=10):
    """A config shaped like configs/targeted.yaml's single-insider setup."""
    return {
        "fl": {"n_clients": N_CLIENTS, "n_compromisable": n_compromisable,
               "device": "cpu", "lr": 0.01, "local_epochs": 1, "training_rounds": 5,
               "benign_retrain_each_round": False},
        "data": {"n_classes": n_classes},
        "attack": {
            "goal": {"type": goal_type, "label": label,
                     "target_class_drop": 0.8, "max_collateral": 0.05},
            CLIENT_KEY: from_client,
            "max_poison_clients": max_poison,
            "sample_budget_in_training": False,
            "sample_target_in_training": True,          # must be turned OFF by the resolver
            "target_labels": [0, 1, 2, 3, 4, 5],        # must be narrowed by the resolver
        },
    }


def _skewed_loaders(dominant_per_client, n_classes=10, n_per_label=4):
    """One loader per client; client c's shard is dominated by dominant_per_client[c].

    Real ``DataLoader(Subset(TensorDataset))`` objects, i.e. exactly the shape
    ``get_data_loaders`` returns, so the histogram is exercised on the production
    container rather than a stand-in.
    """
    loaders = []
    for cid, dom in enumerate(dominant_per_client):
        labels = [dom] * (n_per_label * n_classes)      # a clear majority ...
        labels += [c for c in range(n_classes) if c != dom]   # ... plus one of each other
        y = torch.tensor(labels)
        x = torch.zeros(len(y), 1, 28, 28)
        ds = TensorDataset(x, y)
        loaders.append(DataLoader(Subset(ds, list(range(len(y)))), batch_size=8))
    return loaders


def _test_loader(n_per_class: int = 4, n_classes: int = 10):
    g = torch.Generator().manual_seed(0)
    y = torch.arange(n_classes).repeat(n_per_class)
    x = torch.randn(len(y), 1, 28, 28, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


def _env(cfg, loaders):
    env = FLArmsRaceEnv(cfg, loaders, _test_loader(), random.Random(0))
    net = MnistNet()
    gw = {k: v.clone() for k, v in net.state_dict().items()}
    cw = [{k: v.clone() + 0.01 * i for k, v in gw.items()} for i in range(N_CLIENTS)]
    env.reset(gw, cw, 0.5)
    return env


# ---------------------------------------------------------------------------
# client_label_counts
# ---------------------------------------------------------------------------

def test_counts_through_dataloader_and_subset():
    loaders = _skewed_loaders([7] + [0] * (N_CLIENTS - 1))
    counts = client_label_counts(loaders[0], 10)
    assert counts[7] == 40                       # 4 * 10 of the dominant label
    assert all(counts[c] == 1 for c in range(10) if c != 7)
    assert sum(counts) == 49
    # The same numbers straight off the Subset (no DataLoader in the way).
    assert client_label_counts(loaders[0].dataset, 10) == counts


def test_counts_on_bare_dataset_and_nested_subsets():
    targets = _balanced_targets(3)                # 3 of each label 0..9
    base = FakeDS(targets)
    assert client_label_counts(base, 10) == [3] * 10

    # Subset picking every sample labelled 4, then a Subset of THAT taking 2 of them:
    # the index maps must compose, not stack.
    fours = [i for i, t in enumerate(targets) if t == 4]
    inner = Subset(base, fours)
    outer = Subset(inner, [0, 2])
    assert client_label_counts(inner, 10)[4] == 3 and sum(client_label_counts(inner, 10)) == 3
    assert client_label_counts(outer, 10)[4] == 2 and sum(client_label_counts(outer, 10)) == 2


def test_counts_match_the_real_noniid_partition():
    """On the shipped partition, client 0's own class is the one its group owns."""
    targets = _balanced_targets(200)
    ds = FakeDS(targets)
    shards = partition_noniid_fltrust(ds, n_clients=20, n_classes=10, bias_q=0.5, seed=0)
    counts = client_label_counts(Subset(ds, shards[0]), 10)
    assert sum(counts) == len(shards[0])
    # Round-robin groups put client 0 in group 0, and bias_q=0.5 >> IID's 0.1.
    assert dominant_label(counts) == 0, counts
    assert counts[0] > 0.4 * sum(counts), counts


def test_dominant_label_breaks_ties_on_the_lowest_class():
    assert dominant_label([5, 5, 1]) == 0
    assert dominant_label([0, 9, 9]) == 1
    assert dominant_label([0, 0, 3]) == 2


# ---------------------------------------------------------------------------
# resolve_client_target_label
# ---------------------------------------------------------------------------

def test_derives_client_zero_label_and_pins_the_run():
    cfg = _cfg(label=2)                                  # configured label is 2 ...
    loaders = _skewed_loaders([7, 1, 1, 1, 1, 1])        # ... but client 0 holds 7s
    info = resolve_client_target_label(cfg, loaders)

    assert info["client_id"] == 0 and info["label"] == 7
    assert info["n_samples"] == 49 and info["counts"][7] == 40
    assert abs(info["share"] - 40 / 49) < 1e-9
    assert info["n_classes_held"] == 10

    # All three fields pinned: the reward/prompt label, the sampling pool, and the
    # per-round redraw that would otherwise hand out a different label every round.
    assert cfg["attack"]["goal"]["label"] == 7
    assert cfg["attack"]["target_labels"] == [7]
    assert cfg["attack"]["sample_target_in_training"] is False
    # Everything else about the goal survives.
    assert cfg["attack"]["goal"]["target_class_drop"] == 0.8
    assert cfg["attack"]["goal"]["max_collateral"] == 0.05


def test_mutates_the_goal_in_place_so_existing_references_see_it():
    """main.py hands `attack.goal` to the attacker agent BEFORE the data is
    partitioned; the derivation must reach that same dict."""
    cfg = _cfg()
    held_by_agent = cfg["attack"]["goal"]
    resolve_client_target_label(cfg, _skewed_loaders([5, 0, 0, 0, 0, 0]))
    assert held_by_agent["label"] == 5


def test_off_switch_and_non_targeted_goal_change_nothing():
    loaders = _skewed_loaders([7, 1, 1, 1, 1, 1])

    off = _cfg(from_client=None, label=2)
    assert resolve_client_target_label(off, loaders) is None
    assert off["attack"]["goal"]["label"] == 2
    assert off["attack"]["target_labels"] == [0, 1, 2, 3, 4, 5]
    assert off["attack"]["sample_target_in_training"] is True

    untargeted = _cfg(goal_type="untargeted_degrade")
    assert resolve_client_target_label(untargeted, loaders) is None
    assert untargeted["attack"]["sample_target_in_training"] is True

    # No attack block at all (the untargeted config path).
    assert resolve_client_target_label({"fl": {}, "data": {}}, loaders) is None


def test_bad_client_id_is_a_loud_error():
    loaders = _skewed_loaders([7] + [0] * (N_CLIENTS - 1))
    for bad in (N_CLIENTS, -1, 99):
        try:
            resolve_client_target_label(_cfg(from_client=bad), loaders)
        except ValueError as e:
            assert str(bad) in str(e)
        else:
            raise AssertionError(f"client id {bad} should not be accepted")

    try:
        resolve_client_target_label(_cfg(from_client="zero"), loaders)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-integer client id should not be accepted")


def test_uncompromisable_client_warns_but_still_derives():
    """Deriving from a client the attacker cannot touch is a mistake worth shouting
    about, not a reason to refuse to run."""
    cfg = _cfg(from_client=3, n_compromisable=1)
    info = resolve_client_target_label(cfg, _skewed_loaders([0, 0, 0, 6, 0, 0]))
    assert info["label"] == 6 and cfg["attack"]["goal"]["label"] == 6


# ---------------------------------------------------------------------------
# what the env and the attacker then see
# ---------------------------------------------------------------------------

def test_pool_is_exactly_client_zero_with_budget_one():
    cfg = _cfg()
    loaders = _skewed_loaders([7, 1, 1, 1, 1, 1])
    resolve_client_target_label(cfg, loaders)
    env = _env(cfg, loaders)
    assert env.n_compromisable == 1 and env.budget_cap == 1
    for _ in range(4):
        ctx = env.begin_round()
        assert ctx.pool_ids == [0], ctx.pool_ids          # client 0 and nobody else
        assert list(ctx.pool_benign) == [0]
        assert ctx.budget == 1                            # one poisoned client per round
        assert ctx.goal["label"] == 7                     # the derived label, every round


def test_budget_sampling_cannot_widen_a_cap_of_one():
    """Belt and braces: even with sample_budget_in_training left ON, a cap of 1 can
    only ever draw 1 — so an old config cannot silently poison more clients."""
    cfg = _cfg()
    cfg["attack"]["sample_budget_in_training"] = True
    loaders = _skewed_loaders([7, 1, 1, 1, 1, 1])
    resolve_client_target_label(cfg, loaders)
    env = _env(cfg, loaders)
    assert all(env.begin_round().budget == 1 for _ in range(6))


def test_prompt_offers_only_client_zero_and_the_derived_row():
    cfg = _cfg()
    resolve_client_target_label(cfg, _skewed_loaders([7, 1, 1, 1, 1, 1]))
    goal = cfg["attack"]["goal"]

    agent = AttackerAgent({"attack_goal": goal, "n_clients": 20, "n_classes": 10})
    assert agent.targeted
    sd = MnistNet().state_dict()
    payload = json.loads(agent.build_user_prompt(
        1, 0.9, {0: {k: v.clone() for k, v in sd.items()}}, sd, 1, goal=goal))

    assert payload["controllable_client_ids"] == [0]
    assert payload["max_poison_clients"] == 1
    assert payload["attack_goal"]["label"] == 7
    assert payload["output_layer"]["row_for_target_label"] == 7
    # One poisoner out of 20: the dilution table offers k=1 only, and zeroing the
    # aggregated row needs factor 1 - 20/1 = -19.
    assert list(payload["federation"]["row_zero_factor"]) == ["1"]
    assert abs(payload["federation"]["row_zero_factor"]["1"] - (-19.0)) < 1e-9


def test_attacker_cannot_land_on_a_client_outside_the_pool():
    agent = AttackerAgent({"attack_goal": {"type": "targeted_label", "label": 7},
                           "n_clients": 20, "n_classes": 10})
    sd = MnistNet().state_dict()
    pool = {0: {k: v.clone() for k, v in sd.items()}}
    plan = ('{"clients":[{"id":%d,"operations":[{"op":"scale","target":"net.4.weight",'
            '"rows":[7],"factor":-19.0}]}]}')

    poisoned, chosen, n_malformed = agent.select_and_apply(plan % 3, pool, budget=1)
    assert chosen == [] and poisoned == {} and n_malformed == 1   # client 3 is unreachable

    poisoned, chosen, n_malformed = agent.select_and_apply(plan % 0, pool, budget=1)
    assert chosen == [0] and n_malformed == 0
    # Only row 7 moved — the surgical edit the targeted prompt asks for.
    row = poisoned[0]["net.4.weight"]
    assert torch.allclose(row[7], sd["net.4.weight"][7] * -19.0)
    for c in range(10):
        if c != 7:
            assert torch.allclose(row[c], sd["net.4.weight"][c])


# ---------------------------------------------------------------------------

def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} single-insider target-label tests passed.")


if __name__ == "__main__":
    _run()
