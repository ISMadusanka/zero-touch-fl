"""Tests for switching the FL dataset (MNIST <-> CIFAR-10).

Covers the contract the ``--dataset`` flag has to keep:

  * the registry resolves names + common misspellings, and rejects unknown ones;
  * ``build_model`` returns an architecture that accepts THAT dataset's inputs,
    and ``FedServer``/``FLArmsRaceEnv`` pick it up from the config;
  * per-dataset config overrides deep-merge and WIN over the shared blocks;
  * FL state (global model, client weights, baseline, live FL state, RL progress)
    is isolated per dataset, so a CIFAR-10 run can never load MNIST weights;
  * the LoRA adapter paths are NOT per dataset — the same LLM keeps fine-tuning
    from its last checkpoint whichever dataset a run selects;
  * pre-multi-dataset checkpoints (flat under ``checkpoints/``) still load.

No download and no GPU: models run on tiny random tensors, storage runs in a temp
dir.  Run with:  python tests/test_dataset_switch.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from core import run_config  # noqa: E402
from data import datasets as ds  # noqa: E402
from model import build_model, count_parameters  # noqa: E402
import storage.checkpoint as ckpt  # noqa: E402


# --------------------------------------------------------------------- registry
def test_registry_has_both_datasets():
    assert set(ds.DATASET_NAMES) == {"mnist", "cifar10"}
    assert ds.resolve("mnist").input_shape == (1, 28, 28)
    assert ds.resolve("cifar10").input_shape == (3, 32, 32)
    # Distinct download directories: two datasets must never share a cache.
    assert ds.resolve("mnist").data_dir != ds.resolve("cifar10").data_dir


def test_canonical_accepts_aliases_and_case():
    for spelling in ("cifar10", "CIFAR10", "cifar-10", "cifar_10", "ciffar10", "CIFFAR-10"):
        assert ds.canonical(spelling) == "cifar10", spelling
    assert ds.canonical("MNIST") == "mnist"
    assert ds.canonical(None) == ds.DEFAULT_DATASET


def test_unknown_dataset_names_the_valid_choices():
    try:
        ds.canonical("imagenet")
    except ValueError as e:
        assert "mnist" in str(e) and "cifar10" in str(e)
    else:
        raise AssertionError("unknown dataset should raise")


# ------------------------------------------------------------------ model factory
def test_each_dataset_gets_a_model_that_accepts_its_inputs():
    for name in ds.DATASET_NAMES:
        spec = ds.resolve(name)
        net = build_model(name)
        out = net(torch.randn(2, *spec.input_shape))
        assert out.shape == (2, spec.n_classes), (name, out.shape)
        assert count_parameters(net) > 0


def test_models_are_distinct_architectures():
    mnist_keys = set(build_model("mnist").state_dict())
    cifar_keys = set(build_model("cifar10").state_dict())
    assert mnist_keys != cifar_keys
    # A CIFAR-10 batch must NOT fit the MNIST model — this mismatch is exactly what
    # per-dataset checkpoint isolation exists to prevent.
    try:
        build_model("mnist")(torch.randn(2, 3, 32, 32))
    except RuntimeError:
        pass
    else:
        raise AssertionError("MNIST model should reject 3x32x32 inputs")


def test_cifar_model_has_no_batchnorm_buffers():
    """Every tensor in the state_dict must be a learnable parameter.

    The attack DSL scales/flips/clamps whatever is in the state_dict and FedAvg
    averages it; BatchNorm running stats (and the int64 num_batches_tracked
    counter) have no meaningful behaviour under either.
    """
    net = build_model("cifar10")
    param_names = {n for n, _ in net.named_parameters()}
    assert set(net.state_dict()) == param_names
    assert all(t.is_floating_point() for t in net.state_dict().values())


# ------------------------------------------------------- config resolution
def _cfg():
    return {
        "fl": {"training_rounds": 45, "lr": 0.002, "local_epochs": 3, "device": "cpu"},
        "data": {"dataset": "mnist", "iid": False, "data_dir": None},
        "attack": {"target_choices": [0.05, 0.30]},
        "datasets": {
            "cifar10": {"fl": {"training_rounds": 80, "lr": 0.01},
                        "attack": {"target_choices": [0.05, 0.2]}},
        },
    }


def test_cli_dataset_overrides_the_config_default():
    cfg, name = run_config.apply_dataset(_cfg(), "cifar10")
    assert name == "cifar10"
    assert cfg["data"]["dataset"] == "cifar10"
    # ...and with no CLI value the config's own default is used.
    _cfg2, name2 = run_config.apply_dataset(_cfg(), None)
    assert name2 == "mnist"


def test_per_dataset_overrides_win_and_leave_other_keys_alone():
    cfg, _ = run_config.apply_dataset(_cfg(), "cifar10")
    assert cfg["fl"]["training_rounds"] == 80      # overridden
    assert cfg["fl"]["lr"] == 0.01                 # overridden
    assert cfg["fl"]["local_epochs"] == 3          # untouched shared value
    assert cfg["attack"]["target_choices"] == [0.05, 0.2]
    # The un-selected dataset's config is unaffected.
    mnist_cfg, _ = run_config.apply_dataset(_cfg(), "mnist")
    assert mnist_cfg["fl"]["training_rounds"] == 45
    assert mnist_cfg["fl"]["lr"] == 0.002


def test_apply_dataset_does_not_mutate_the_input():
    original = _cfg()
    run_config.apply_dataset(original, "cifar10")
    assert original["fl"]["training_rounds"] == 45
    assert original["data"]["data_dir"] is None


def test_data_dir_defaults_to_the_dataset_directory():
    for name in ds.DATASET_NAMES:
        cfg, _ = run_config.apply_dataset(_cfg(), name)
        assert cfg["data"]["data_dir"] == ds.resolve(name).data_dir


def test_data_dir_pointing_at_another_dataset_is_rejected():
    """A leftover ``data_dir: ./data/mnist_raw`` must not silently make a CIFAR-10
    run download into the MNIST cache — it works, which is what makes it dangerous."""
    cfg = _cfg()
    cfg["data"]["data_dir"] = ds.resolve("mnist").data_dir
    run_config.apply_dataset(cfg, "mnist")          # same dataset: fine
    try:
        run_config.apply_dataset(cfg, "cifar10")
    except ValueError as e:
        assert "cifar10" in str(e) and "mnist" in str(e)
    else:
        raise AssertionError("cross-dataset data_dir should raise")


def test_unknown_override_section_is_rejected():
    cfg = _cfg()
    cfg["datasets"]["cifar10"]["flt"] = {"lr": 1.0}      # typo for "fl"
    try:
        run_config.apply_dataset(cfg, "cifar10")
    except ValueError as e:
        assert "flt" in str(e)
    else:
        raise AssertionError("an unknown override section should raise, not be ignored")


# ------------------------------------------------------------- artifact layout
def test_paths_are_per_dataset_but_adapters_are_not():
    mnist, cifar = run_config.run_paths("mnist"), run_config.run_paths("cifar10")
    for key in ("checkpoint_dir", "log_dir", "round_log", "metrics_dir",
                "benchmark_dir", "system_log"):
        assert mnist[key] != cifar[key], key
    assert "mnist" in mnist["checkpoint_dir"] and "cifar10" in cifar["checkpoint_dir"]

    # The LLM adapter is the ONE artifact that must be shared: that is what makes
    # fine-tuning continual across a dataset switch.
    cfg = {"rl": {"adapter_paths": {"attacker": "checkpoints/attacker_adapter",
                                    "defender": "checkpoints/defender_adapter"}}}
    paths = run_config.adapter_paths(cfg)
    assert paths["attacker"] == "checkpoints/attacker_adapter"
    assert "mnist" not in paths["attacker"] and "cifar10" not in paths["attacker"]
    # Defaults (no rl.adapter_paths configured) are dataset-free too.
    assert run_config.adapter_paths({})["attacker"] == os.path.join(
        run_config.CHECKPOINT_ROOT, "attacker_adapter")


# --------------------------------------------------------------------- storage
def _weights(dataset):
    return {k: v.clone() for k, v in build_model(dataset).state_dict().items()}


def test_checkpoints_are_isolated_per_dataset():
    original = ckpt.CHECKPOINT_DIR
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        try:
            mw, cw = _weights("mnist"), _weights("cifar10")
            ckpt.save_state(mw, [mw], 0.91, dataset="mnist")
            ckpt.save_state(cw, [cw], 0.55, dataset="cifar10")

            gm, clients, acc = ckpt.load_state(dataset="mnist")
            assert set(gm) == set(mw) and acc == 0.91 and len(clients) == 1
            gc, _clients, acc_c = ckpt.load_state(dataset="cifar10")
            assert set(gc) == set(cw) and acc_c == 0.55
            # The two must not share a single tensor key set...
            assert set(gm) != set(gc)
            # ...and each lives in its own directory.
            assert os.path.isfile(os.path.join(td, "mnist", "global_model.pt"))
            assert os.path.isfile(os.path.join(td, "cifar10", "global_model.pt"))
        finally:
            ckpt.CHECKPOINT_DIR = original


def test_progress_and_fl_state_are_isolated_per_dataset():
    original = ckpt.CHECKPOINT_DIR
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        try:
            ckpt.save_progress(120, round_index=165, controller={"learner": "attacker"},
                               dataset="mnist")
            # A CIFAR-10 run has done nothing yet: it must start from zero rather
            # than inherit the MNIST round counters (which index MNIST's FL state).
            assert ckpt.load_progress(dataset="cifar10")["rounds_done"] == 0
            assert ckpt.load_progress(dataset="mnist")["rounds_done"] == 120
            assert ckpt.load_progress(dataset="mnist")["round_index"] == 165

            ckpt.save_fl_state({"current_accuracy": 0.42}, dataset="cifar10")
            assert ckpt.load_fl_state(dataset="mnist") is None
            assert ckpt.load_fl_state(dataset="cifar10")["current_accuracy"] == 0.42
        finally:
            ckpt.CHECKPOINT_DIR = original


def test_state_exists_is_per_dataset():
    original = ckpt.CHECKPOINT_DIR
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        try:
            mw = _weights("mnist")
            ckpt.save_state(mw, [mw], 0.9, dataset="mnist")
            assert ckpt.state_exists(dataset="mnist")
            assert not ckpt.state_exists(dataset="cifar10")
        finally:
            ckpt.CHECKPOINT_DIR = original


def test_legacy_flat_checkpoint_still_loads_for_mnist():
    """A checkpoint tree written before multi-dataset support is MNIST by
    construction; an MNIST run reads it in place instead of re-running Phase 1."""
    original = ckpt.CHECKPOINT_DIR
    with tempfile.TemporaryDirectory() as td:
        ckpt.CHECKPOINT_DIR = td
        try:
            mw = _weights("mnist")
            ckpt.save_state(mw, [mw], 0.88, dataset=None)      # legacy flat layout
            assert os.path.isfile(os.path.join(td, "global_model.pt"))

            assert ckpt.state_exists(dataset="mnist")
            _gm, _cw, acc = ckpt.load_state(dataset="mnist")
            assert acc == 0.88
            # ...but a CIFAR-10 run must NOT pick up those MNIST weights.
            assert not ckpt.state_exists(dataset="cifar10")
            assert ckpt.load_state(dataset="cifar10") is None

            # Once the dataset-scoped copy exists it wins over the legacy file.
            ckpt.save_state(mw, [mw], 0.93, dataset="mnist")
            assert ckpt.load_state(dataset="mnist")[2] == 0.93
        finally:
            ckpt.CHECKPOINT_DIR = original


# ------------------------------------------------------------------ integration
def test_fed_server_builds_the_dataset_model():
    from server.fed_server import FedServer

    for name in ds.DATASET_NAMES:
        server = FedServer(device="cpu", dataset=name)
        assert server.dataset == name
        assert set(server.get_global_weights()) == set(build_model(name).state_dict())


def test_env_reads_the_dataset_from_the_config():
    import random

    from rl.env import FLArmsRaceEnv

    cfg = {
        "fl": {"n_clients": 3, "device": "cpu", "benign_retrain_each_round": False,
               "training_rounds": 1, "n_compromisable": 2, "lr": 0.01,
               "local_epochs": 1, "simulation_rounds": 1},
        "data": {"dataset": "cifar10"},
        "attack": {"max_poison_clients": 1},
    }
    env = FLArmsRaceEnv(cfg, None, None, random.Random(0))
    assert env.dataset == "cifar10"
    assert set(env.global_weights) == set(build_model("cifar10").state_dict())


def test_attacker_prompt_carries_the_dataset():
    """One shared adapter trains on both datasets, so the regime has to be in the
    observation — otherwise the policy sees an unlabelled mixture of two tasks."""
    import json

    from agents.attacker_agent import AttackerAgent

    agent = AttackerAgent()
    gw = _weights("cifar10")
    pool = {0: gw, 1: {k: v + 0.01 for k, v in gw.items()}}

    payload = json.loads(agent.build_user_prompt(1, 0.5, pool, gw, budget=1,
                                                 dataset="cifar10"))
    assert payload["dataset"] == "cifar10"
    # Omitted when not supplied, so older callers keep their exact prompt.
    assert "dataset" not in json.loads(agent.build_user_prompt(1, 0.5, pool, gw, budget=1))
    assert "`dataset` names the learning task" in agent.system_prompt()


def test_defender_prompt_carries_the_dataset():
    import json

    from agents.defender_agent import DefenderAgent

    agent = DefenderAgent()
    feats = {0: {"layers": {}, "whole": {}}}
    assert json.loads(agent.build_user_prompt(feats, dataset="mnist"))["dataset"] == "mnist"
    assert "dataset" not in json.loads(agent.build_user_prompt(feats))


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} dataset-switch tests passed.")


if __name__ == "__main__":
    _run()
