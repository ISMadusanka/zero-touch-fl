"""Tests for TARGETED label poisoning: per-class eval, row-level operators, the
targeted reward, the win-gate, and per-round label sampling.

Synthetic tensors shaped like MNIST — no download, no GPU, no LLM:
    python tests/test_targeted.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from agents.attack_ops import apply_plan, output_layer_keys  # noqa: E402
from agents.attacker_agent import AttackerAgent  # noqa: E402
from core.types import ClassEval, DetectionVerdict  # noqa: E402
from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from rl.rewards import (  # noqa: E402
    attacker_reward, goal_drop, goal_label, goal_target, targeted_terms,
)
from rl.switch import SwitchConfig, attacker_succeeded  # noqa: E402

N_CLIENTS = 6
GOAL = {"type": "targeted_label", "label": 2,
        "target_class_drop": 0.8, "max_collateral": 0.05}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _evenly_labelled_loader(n_per_class: int = 8, n_classes: int = 10):
    """A test set with exactly ``n_per_class`` samples of every label."""
    g = torch.Generator().manual_seed(0)
    y = torch.arange(n_classes).repeat(n_per_class)
    x = torch.randn(len(y), 1, 28, 28, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


def _client_loader(seed: int, n: int = 64):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.randn(n, 1, 28, 28, generator=g),
                      torch.randint(0, 10, (n,), generator=g)),
        batch_size=32, shuffle=True)


def _cfg(goal=None, sample_target=False, target_labels=None):
    return {
        "fl": {"n_clients": N_CLIENTS, "n_compromisable": 3, "device": "cpu",
               "lr": 0.01, "local_epochs": 1, "training_rounds": 5,
               "benign_retrain_each_round": False},
        "data": {"n_classes": 10},
        "attack": {"goal": goal if goal is not None else dict(GOAL),
                   "max_poison_clients": 3, "sample_budget_in_training": False,
                   "sample_target_in_training": sample_target,
                   "target_labels": target_labels or [0, 1, 2, 3, 4, 5]},
    }


def _env(goal=None, sample_target=False, target_labels=None, seed=0):
    cfg = _cfg(goal, sample_target, target_labels)
    loaders = [_client_loader(i) for i in range(N_CLIENTS)]
    env = FLArmsRaceEnv(cfg, loaders, _evenly_labelled_loader(), random.Random(seed))
    net = MnistNet()
    gw = {k: v.clone() for k, v in net.state_dict().items()}
    cw = [{k: v.clone() + 0.01 * i for k, v in gw.items()} for i in range(N_CLIENTS)]
    env.reset(gw, cw, 0.5)
    return env


def _eval(per_class):
    return ClassEval(overall=sum(per_class) / len(per_class),
                     per_class=list(per_class), support=[100] * len(per_class))


def _evaded(ids):
    return [DetectionVerdict(c, False, 0.9, "") for c in ids]


# ---------------------------------------------------------------------------
# per-class evaluation
# ---------------------------------------------------------------------------

def test_per_class_eval_matches_overall_accuracy():
    """The per-class breakdown and plain accuracy come from the same pass, so the
    support-weighted mean of the recalls must reproduce ``evaluate`` exactly."""
    env = _env()
    loader = _evenly_labelled_loader()
    acc = env.server.evaluate(loader)
    ev = env.server.evaluate_per_class(loader, 10)
    assert abs(ev.overall - acc) < 1e-12
    total = sum(ev.support)
    weighted = sum(r * s for r, s in zip(ev.per_class, ev.support)) / total
    assert abs(weighted - acc) < 1e-9
    assert len(ev.per_class) == 10 and sum(ev.support) == total


def test_class_eval_others_mean_excludes_the_target():
    ev = _eval([1.0] * 10)
    assert ev.recall(2) == 1.0
    ev2 = _eval([1.0, 1.0, 0.0] + [1.0] * 7)
    assert ev2.recall(2) == 0.0
    assert abs(ev2.others_mean(2) - 1.0) < 1e-12   # the zeroed class is excluded


def test_env_exposes_a_per_class_clean_counterfactual():
    env = _env()
    ctx = env.begin_round()
    assert ctx.clean_eval is not None
    assert abs(ctx.clean_eval.overall - ctx.clean_accuracy) < 1e-12
    assert len(ctx.clean_eval.per_class) == 10


# ---------------------------------------------------------------------------
# row-level operators (the surgical part)
# ---------------------------------------------------------------------------

def test_output_layer_keys_finds_the_classifier_head():
    sd = MnistNet().state_dict()
    head = output_layer_keys(sd, 10)
    assert head["weight_key"] == "net.4.weight"      # shape [10, 16] -> row c = class c
    assert head["bias_key"] == "net.4.bias"
    assert head["layer"] == "net.4"
    assert head["n_rows"] == 10


def test_rows_confines_an_operator_to_one_class():
    """The whole point: scaling with rows=[2] must change row 2 of the head — and
    its bias entry — while every other weight in the model stays byte-identical."""
    sd = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    plan = [{"op": "scale", "target": "net.4", "rows": [2], "factor": -3.0}]
    out, n_invalid = apply_plan(sd, plan)
    assert n_invalid == 0
    assert torch.allclose(out["net.4.weight"][2], sd["net.4.weight"][2] * -3.0)
    assert torch.allclose(out["net.4.bias"][2:3], sd["net.4.bias"][2:3] * -3.0)
    for c in range(10):
        if c == 2:
            continue
        assert torch.equal(out["net.4.weight"][c], sd["net.4.weight"][c])
        assert torch.equal(out["net.4.bias"][c:c + 1], sd["net.4.bias"][c:c + 1])
    assert torch.equal(out["net.2.weight"], sd["net.2.weight"])   # shared layer untouched
    assert torch.equal(out["net.2.bias"], sd["net.2.bias"])


def test_rows_without_the_param_still_hits_the_whole_tensor():
    """Back-compat: an op with no `rows` behaves exactly as before."""
    sd = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    out, n_invalid = apply_plan(sd, [{"op": "scale", "target": "net.4.weight", "factor": 2.0}])
    assert n_invalid == 0
    assert torch.allclose(out["net.4.weight"], sd["net.4.weight"] * 2.0)


def test_out_of_range_rows_are_skipped_as_invalid():
    """`rows` that address nothing must NOT silently fall back to the whole tensor
    — that would turn a mis-specified targeted attack into an untargeted one."""
    sd = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    out, n_invalid = apply_plan(sd, [{"op": "scale", "target": "net.4.weight",
                                      "rows": [99], "factor": -5.0}])
    assert n_invalid == 1
    assert torch.equal(out["net.4.weight"], sd["net.4.weight"])


def test_negative_row_index_resolves_from_the_end():
    sd = {k: v.clone() for k, v in MnistNet().state_dict().items()}
    out, _ = apply_plan(sd, [{"op": "scale", "target": "net.4.weight",
                              "rows": [-1], "factor": 0.0}])
    assert torch.count_nonzero(out["net.4.weight"][9]) == 0
    assert torch.equal(out["net.4.weight"][0], sd["net.4.weight"][0])


def test_row_targeted_attack_actually_suppresses_that_logit():
    """End-to-end sanity: driving the aggregated row negative must make the model
    stop predicting that class, with other classes' logits unchanged."""
    net = MnistNet()
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    x = torch.randn(32, 1, 28, 28, generator=torch.Generator().manual_seed(3))
    before = net(x)
    out, _ = apply_plan(sd, [
        {"op": "add_constant", "target": "net.4.bias", "rows": [2], "value": -50.0}])
    net.load_state_dict(out)
    after = net(x)
    assert after.argmax(dim=1).eq(2).sum().item() == 0        # class 2 never predicted
    keep = [c for c in range(10) if c != 2]
    assert torch.allclose(before[:, keep], after[:, keep])    # other logits untouched


# ---------------------------------------------------------------------------
# the targeted reward
# ---------------------------------------------------------------------------

def test_goal_target_reads_target_class_drop():
    assert abs(goal_target(GOAL) - 0.8) < 1e-12
    assert goal_label(GOAL) == 2
    assert goal_label({"type": "untargeted_degrade"}) is None


def test_targeted_terms_measures_the_target_class_only():
    clean = _eval([0.9] * 10)
    post = _eval([0.9, 0.9, 0.1] + [0.9] * 7)
    t = targeted_terms(GOAL, clean, post)
    assert t["label"] == 2
    assert abs(t["target_drop"] - 0.8) < 1e-9
    assert abs(t["collateral"]) < 1e-12
    # requested 0.8 but the class only had 0.9 to lose -> min(0.8, 0.9) = 0.8
    assert abs(t["effective_target"] - 0.8) < 1e-9


def test_effective_target_is_clamped_to_a_weak_class_clean_recall():
    """A class the honest model only gets 40% right cannot lose 80 points of
    recall; without the clamp that label would be unwinnable and rewards would be
    incomparable between labels in one run."""
    clean = _eval([0.9, 0.9, 0.4] + [0.9] * 7)
    post = _eval([0.9, 0.9, 0.0] + [0.9] * 7)
    t = targeted_terms(GOAL, clean, post)
    assert abs(t["effective_target"] - 0.4) < 1e-9
    assert abs(t["target_drop"] - 0.4) < 1e-9          # destroyed => goal met


def test_collateral_counts_only_losses():
    """A class that improves must not offset one that was destroyed."""
    clean = _eval([0.9] * 10)
    post = _eval([1.0, 0.9, 0.1, 0.5] + [0.9] * 6)     # class 0 up, class 3 down
    t = targeted_terms(GOAL, clean, post)
    assert t["collateral"] > 0.0
    assert abs(t["collateral"] - (0.4 / 9)) < 1e-9     # only class 3's 0.4 loss counts


def test_surgical_attack_beats_indiscriminate_destruction():
    """The core requirement. Both wreck class 2; only the surgical one is targeted,
    and it must score far higher or the policy just relearns the untargeted attack."""
    clean = _eval([0.9] * 10)
    surgical = _eval([0.9, 0.9, 0.0] + [0.9] * 7)
    nuke = _eval([0.05] * 10)
    kw = dict(goal=GOAL, poisoned_ids=[0], verdicts=_evaded([0]), n_malformed=0,
              pool_size=5, clean_eval=clean)
    r_surgical = attacker_reward(clean.overall, surgical.overall,
                                 post_eval=surgical, **kw)
    r_nuke = attacker_reward(clean.overall, nuke.overall, post_eval=nuke, **kw)
    assert r_surgical > r_nuke + 1.0, (r_surgical, r_nuke)


def test_doing_nothing_scores_worse_than_a_surgical_hit():
    clean = _eval([0.9] * 10)
    surgical = _eval([0.9, 0.9, 0.0] + [0.9] * 7)
    r_hit = attacker_reward(clean.overall, surgical.overall, GOAL, [0], _evaded([0]), 0,
                            pool_size=5, clean_eval=clean, post_eval=surgical)
    r_none = attacker_reward(clean.overall, clean.overall, GOAL, [], _evaded([]), 1,
                             pool_size=5, clean_eval=clean, post_eval=clean)
    assert r_hit > r_none


def test_eta_zero_removes_the_collateral_pressure():
    """Sanity on the knob: with eta=0 an indiscriminate attack is no longer
    punished, which is exactly the failure mode eta exists to prevent."""
    clean = _eval([0.9] * 10)
    nuke = _eval([0.05] * 10)
    kw = dict(goal=GOAL, poisoned_ids=[0], verdicts=_evaded([0]), n_malformed=0,
              pool_size=5, clean_eval=clean, post_eval=nuke)
    assert (attacker_reward(clean.overall, nuke.overall, eta=0.0, **kw)
            > attacker_reward(clean.overall, nuke.overall, eta=1.0, **kw))


def test_untargeted_reward_is_unchanged_by_the_new_arguments():
    """Regression guard for the OTHER experiment: an untargeted goal must ignore
    the per-class evals entirely."""
    goal = {"type": "untargeted_degrade", "target_accuracy_drop": 0.2}
    clean, post = _eval([0.9] * 10), _eval([0.7] * 10)
    a = attacker_reward(0.9, 0.7, goal, [0], _evaded([0]), 0, pool_size=5)
    b = attacker_reward(0.9, 0.7, goal, [0], _evaded([0]), 0, pool_size=5,
                        clean_eval=clean, post_eval=post)
    assert a == b


def test_goal_drop_switches_on_the_goal_type():
    clean = _eval([0.9] * 10)
    post = _eval([0.9, 0.9, 0.1] + [0.9] * 7)
    assert abs(goal_drop(GOAL, clean.overall, post.overall, clean, post) - 0.8) < 1e-9
    untargeted = {"type": "untargeted_degrade", "target_accuracy_drop": 0.2}
    assert abs(goal_drop(untargeted, 0.9, 0.7, clean, post) - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# the win-gate
# ---------------------------------------------------------------------------

def test_win_gate_requires_low_collateral():
    cfg = SwitchConfig(win_fraction=0.6, attacker_min_evaded=1.0)
    clean = _eval([0.9] * 10)
    surgical = targeted_terms(GOAL, clean, _eval([0.9, 0.9, 0.0] + [0.9] * 7))
    nuke = targeted_terms(GOAL, clean, _eval([0.05] * 10))
    assert attacker_succeeded(surgical["target_drop"], _evaded([0]), [0], cfg, GOAL, surgical)
    # The nuke destroys class 2 just as thoroughly but wrecks everything else too.
    assert nuke["target_drop"] >= 0.6 * nuke["effective_target"]
    assert not attacker_succeeded(nuke["target_drop"], _evaded([0]), [0], cfg, GOAL, nuke)


def test_win_gate_still_requires_evasion():
    cfg = SwitchConfig(win_fraction=0.6, attacker_min_evaded=1.0)
    clean = _eval([0.9] * 10)
    t = targeted_terms(GOAL, clean, _eval([0.9, 0.9, 0.0] + [0.9] * 7))
    caught = [DetectionVerdict(0, True, 0.9, "")]
    assert not attacker_succeeded(t["target_drop"], caught, [0], cfg, GOAL, t)


# ---------------------------------------------------------------------------
# per-round label sampling (generalization across classes)
# ---------------------------------------------------------------------------

def test_training_samples_the_label_each_round():
    env = _env(sample_target=True, target_labels=[0, 1, 2, 3, 4, 5])
    seen = set()
    for _ in range(60):
        ctx = env.begin_round()
        assert ctx.goal["type"] == "targeted_label"
        assert ctx.goal["label"] in {0, 1, 2, 3, 4, 5}
        # the rest of the goal must survive the swap
        assert ctx.goal["target_class_drop"] == 0.8
        assert ctx.goal["max_collateral"] == 0.05
        seen.add(ctx.goal["label"])
    assert len(seen) > 1, f"label never varied: {seen}"


def test_eval_pins_a_single_label():
    env = _env(goal={"type": "targeted_label", "label": 4,
                     "target_class_drop": 0.8, "max_collateral": 0.05},
               sample_target=False)
    for _ in range(5):
        assert env.begin_round().goal["label"] == 4


# ---------------------------------------------------------------------------
# the attacker's prompt
# ---------------------------------------------------------------------------

def test_targeted_agent_uses_the_targeted_prompt_and_observation():
    agent = AttackerAgent({"attack_goal": dict(GOAL), "n_clients": 20, "n_classes": 10})
    assert agent.targeted
    assert "YOUR OBJECTIVE IS TARGETED" in agent.system_prompt()
    assert '"rows"' in agent.system_prompt() or "rows" in agent.system_prompt()

    sd = MnistNet().state_dict()
    benign = {0: {k: v.clone() for k, v in sd.items()}}
    import json
    payload = json.loads(agent.build_user_prompt(1, 0.78, benign, sd, 2, goal=dict(GOAL)))
    assert payload["output_layer"]["weight"] == "net.4.weight"
    assert payload["output_layer"]["bias"] == "net.4.bias"
    assert payload["output_layer"]["row_for_target_label"] == 2
    # Dilution hint: poisoning k of 20 clients zeroes the aggregated row at 1 - 20/k.
    assert payload["federation"]["n_clients"] == 20
    assert abs(payload["federation"]["row_zero_factor"]["1"] - (-19.0)) < 1e-9
    assert abs(payload["federation"]["row_zero_factor"]["2"] - (-9.0)) < 1e-9


def test_untargeted_agent_keeps_the_original_prompt():
    agent = AttackerAgent({"attack_goal": {"type": "untargeted_degrade",
                                           "target_accuracy_drop": 0.2},
                           "n_clients": 20})
    assert not agent.targeted
    assert "YOUR OBJECTIVE IS TARGETED" not in agent.system_prompt()
    import json
    sd = MnistNet().state_dict()
    payload = json.loads(agent.build_user_prompt(
        1, 0.78, {0: {k: v.clone() for k, v in sd.items()}}, sd, 2))
    assert "output_layer" not in payload and "federation" not in payload


# ---------------------------------------------------------------------------

def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} targeted-poisoning tests passed.")


if __name__ == "__main__":
    _run()
