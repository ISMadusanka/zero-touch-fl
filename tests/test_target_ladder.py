"""Locks in the budget-conditioned target ladder (rl/rewards.py::target_for_budget)
and proves the reward path, the schedule's win gate, and the attacker prompt
cannot disagree about a round's target.

Torch-only (needs torch for ``FLArmsRaceEnv``/``MnistNet``, no MNIST download and
no GPU):  python tests/test_target_ladder.py
"""

import copy
import json
import os
import random
import sys
import yaml
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from agents.attacker_agent import AttackerAgent  # noqa: E402
from model.mnist_net import MnistNet  # noqa: E402
from rl.env import FLArmsRaceEnv  # noqa: E402
from rl.rewards import DEFAULT_TARGET_LADDER, goal_target, target_for_budget  # noqa: E402
from rl.switch import SwitchConfig, attacker_succeeded  # noqa: E402


@dataclass
class V:
    """Minimal DetectionVerdict stand-in (see tests/test_switch.py)."""
    client_id: int
    is_suspicious: bool


def _loader(seed, n=64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)


def _merge_cfg(base: dict, overrides: dict) -> dict:
    """One level of merging below ``fl``/``attack`` — a second-level key given
    in ``overrides`` (e.g. ``attack["goal"]``) REPLACES the base key wholesale
    rather than merging into it, mirroring D-04's "present config replaces the
    default wholesale, no per-rung merge" so a test can swap the ladder, the
    goal, or the budget cap without leftover base fields bleeding through."""
    result = copy.deepcopy(base)
    for section, section_overrides in overrides.items():
        if isinstance(section_overrides, dict) and isinstance(result.get(section), dict):
            result[section] = {**result[section], **copy.deepcopy(section_overrides)}
        else:
            result[section] = copy.deepcopy(section_overrides)
    return result


def _cfg(**overrides) -> dict:
    """Minimal env config: 4-client pool, budget fixed at the cap (no sampling)
    so a test can pin the round's poison budget deterministically. Overrides are
    merged one level deep so a test can swap the ladder, the goal, or the
    budget cap without repeating the whole dict."""
    base = {
        "fl": {"n_clients": 4, "device": "cpu", "benign_retrain_each_round": False,
               "training_rounds": 20, "n_compromisable": 3, "lr": 0.05, "local_epochs": 1},
        "attack": {"goal": {"type": "untargeted_degrade", "target_accuracy_drop": 0.20},
                   "max_poison_clients": 3, "sample_budget_in_training": False},
    }
    return _merge_cfg(base, overrides)


def _make_env(cfg: dict) -> FLArmsRaceEnv:
    n_clients = cfg["fl"]["n_clients"]
    loaders = [_loader(i) for i in range(n_clients)]
    env = FLArmsRaceEnv(cfg, loaders, _loader(999, 128), random.Random(0))
    net = MnistNet()
    gw = {k: v.clone() for k, v in net.state_dict().items()}
    cw = [{k: v.clone() + 0.01 * (i + 1) for k, v in gw.items()} for i in range(n_clients)]
    env.reset(gw, cw, baseline_accuracy=0.10)
    return env


# ---------------------------------------------------------------------------
# End-to-end: budget 3 resolves 0.06 through every real consumer
# ---------------------------------------------------------------------------

def test_budget_three_resolves_one_target_end_to_end():
    """Budget 3, no explicit ladder: begin_round() -> goal_target -> the win gate
    -> the attacker prompt all resolve 0.06. Every consumer is invoked for real
    (prohibition P-03) — none of this compares target_for_budget to itself."""
    cfg = _cfg()
    env = _make_env(cfg)
    ctx = env.begin_round()

    assert ctx.budget == 3
    assert ctx.goal["target_accuracy_drop"] == 0.06
    assert goal_target(ctx.goal) == 0.06

    prompt = AttackerAgent().build_user_prompt(
        ctx.round_num, ctx.global_accuracy, ctx.pool_benign, env.global_weights,
        ctx.budget, goal=ctx.goal)
    recovered = json.loads(prompt)["attack_goal"]["target_accuracy_drop"]
    assert recovered == 0.06

    switch_cfg = SwitchConfig(win_fraction=0.6)
    # Bar = 0.6 * 0.06 = 0.036: a drop of 0.036 with an evading poisoner wins;
    # one representable step below does not.
    assert attacker_succeeded(0.036, [V(0, False)], [0], switch_cfg, ctx.goal)
    assert not attacker_succeeded(0.035, [V(0, False)], [0], switch_cfg, ctx.goal)


# ---------------------------------------------------------------------------
# Every rung, both ends of the ladder
# ---------------------------------------------------------------------------

def test_every_rung_agrees_across_reward_win_gate_and_prompt():
    """Every rung 1..5 agrees, under exact ``==``, across the reward path
    (goal_target), the schedule's win gate (attacker_succeeded), and the
    attacker prompt — invoked for real at both ends of the ladder (budgets 1
    and 5 are inside this loop), not only mid-ladder. Never compare
    ``target_for_budget`` to itself (prohibition P-03)."""
    cfg = _cfg(fl={"n_clients": 6, "n_compromisable": 5},
               attack={"max_poison_clients": 5})
    env = _make_env(cfg)
    ctx = env.begin_round()
    switch_cfg = SwitchConfig(win_fraction=0.6)
    agent = AttackerAgent()

    for budget in range(1, 6):
        env.round_budget = budget
        goal = env._round_goal()
        expected = DEFAULT_TARGET_LADDER[budget]

        assert goal["target_accuracy_drop"] == expected
        assert goal_target(goal) == expected

        prompt = agent.build_user_prompt(
            ctx.round_num, ctx.global_accuracy, ctx.pool_benign, env.global_weights,
            budget, goal=goal)
        recovered = json.loads(prompt)["attack_goal"]["target_accuracy_drop"]
        assert recovered == expected

        # Win gate flips exactly at the proportional bar: win at the bar, loss
        # one representable step below (computed by subtraction, not scaling).
        bar = 0.6 * expected
        assert attacker_succeeded(bar, [V(0, False)], [0], switch_cfg, goal)
        assert not attacker_succeeded(bar - 1e-9, [V(0, False)], [0], switch_cfg, goal)


def test_ladder_rungs_are_pairwise_distinct():
    """Two budgets sharing a target would collapse two per-(defense x budget)
    result cells into one — precisely what D-03's startup validation rejects
    clamping to avoid, and exactly what Phase 2's normalizer and Phase 3's
    graded ASR are both keyed on."""
    values = [DEFAULT_TARGET_LADDER[b] for b in sorted(DEFAULT_TARGET_LADDER)]
    assert len(set(DEFAULT_TARGET_LADDER.values())) == len(DEFAULT_TARGET_LADDER)
    assert all(b > a for a, b in zip(values, values[1:]))


def test_off_ladder_budget_raises_at_construction():
    """An env whose budget_cap exceeds the ladder's coverage raises RuntimeError
    at construction naming every missing budget; a budget_cap exactly equal to
    the highest covered rung constructs successfully (inclusive boundary)."""
    cfg = _cfg(fl={"n_clients": 6, "n_compromisable": 5},
               attack={"max_poison_clients": 5,
                       "target_ladder": {1: 0.02, 2: 0.04, 3: 0.06}})
    try:
        _make_env(cfg)
        assert False, "expected RuntimeError for off-ladder budget_cap"
    except RuntimeError as e:
        assert "4" in str(e) and "5" in str(e)

    cfg_ok = _cfg(fl={"n_clients": 6, "n_compromisable": 5},
                  attack={"max_poison_clients": 3,
                          "target_ladder": {1: 0.02, 2: 0.04, 3: 0.06}})
    _make_env(cfg_ok)   # budget_cap == highest covered rung -> constructs fine


def test_off_ladder_budget_raises_in_the_pure_function():
    """target_for_budget raises RuntimeError for budget 0, a negative budget, and
    (highest rung + 1); an explicitly empty ladder is a config error, not an
    absent one — only None falls back to DEFAULT_TARGET_LADDER (D-04). Budget 1
    and the highest rung return normally."""
    for bad in (0, 6, -1):
        try:
            target_for_budget(bad)
            assert False, f"expected RuntimeError for budget {bad}"
        except RuntimeError:
            pass
    try:
        target_for_budget(3, {})
        assert False, "expected RuntimeError for an explicitly empty ladder"
    except RuntimeError:
        pass
    assert target_for_budget(1) == 0.02
    assert target_for_budget(5) == 0.12


def test_non_untargeted_goal_is_returned_unchanged():
    """A slow_degrade goal is returned unchanged by _round_goal() for every
    budget — the ladder maps budget to target_accuracy_drop, which goal_target
    does not read for slow_degrade; converting the goal type silently would
    change what the run measures."""
    cfg = _cfg(fl={"n_clients": 6, "n_compromisable": 5},
               attack={"max_poison_clients": 5,
                       "goal": {"type": "slow_degrade", "per_round_drop": 0.02}})
    env = _make_env(cfg)
    env.begin_round()
    for budget in range(1, 4):
        env.round_budget = budget
        goal = env._round_goal()
        assert goal == {"type": "slow_degrade", "per_round_drop": 0.02}
        assert "target_accuracy_drop" not in goal
        assert goal_target(goal) == 0.02


def test_round_goal_never_returns_the_fixed_config_target():
    """The GA-1 guard: after GOAL-03, training/--dry-run/--baseline all read
    _round_goal()'s output, so a sentinel target_accuracy_drop that is NOT a
    ladder rung (0.20, the configured fallback) must never leak through as the
    resolved target for any budget — this is what keeps the fixed fallback from
    silently reappearing as the training target."""
    cfg = _cfg(fl={"n_clients": 6, "n_compromisable": 5},
               attack={"max_poison_clients": 5,
                       "goal": {"type": "untargeted_degrade",
                                "target_accuracy_drop": 0.20}})
    env = _make_env(cfg)
    env.begin_round()
    for budget in range(1, 6):
        env.round_budget = budget
        goal = env._round_goal()
        assert goal["target_accuracy_drop"] != 0.20
        assert goal["target_accuracy_drop"] == DEFAULT_TARGET_LADDER[budget]


# ---------------------------------------------------------------------------
# Config drives the ladder (GOAL-02): no code edit required to re-tune it
# ---------------------------------------------------------------------------

def test_config_ladder_overrides_the_default_with_no_code_edit():
    """A config-supplied ``attack.target_ladder`` replaces
    ``DEFAULT_TARGET_LADDER`` wholesale (D-04) with no code edit: budgets 1 and
    2 resolve the config's non-default values (0.33/0.44, deliberately not on
    the default ladder so a fallback would be visible), and rung 3 -- present
    in ``DEFAULT_TARGET_LADDER`` but absent from this config -- raises rather
    than silently falling back, proving the default did not survive as a
    per-rung merge."""
    cfg = _cfg(fl={"n_clients": 4, "n_compromisable": 2},
               attack={"max_poison_clients": 2,
                       "target_ladder": {1: 0.33, 2: 0.44}})
    env = _make_env(cfg)

    env.round_budget = 1
    assert env._round_goal()["target_accuracy_drop"] == 0.33
    env.round_budget = 2
    assert env._round_goal()["target_accuracy_drop"] == 0.44

    try:
        target_for_budget(3, env.target_ladder)
        assert False, "expected RuntimeError: rung 3 must not survive from DEFAULT_TARGET_LADDER"
    except RuntimeError:
        pass


def test_quoted_string_ladder_keys_are_accepted():
    """A hand-edited or quoted-YAML ladder (``{'1': 0.33, '2': 0.44}``) -- the
    shape a quoted mapping key produces -- normalizes to integer keys and float
    values identically to the int-keyed form (D-01). A quoted key that
    silently missed would fall back to the default ladder and produce
    plausible-looking targets nothing would flag."""
    cfg = _cfg(fl={"n_clients": 4, "n_compromisable": 2},
               attack={"max_poison_clients": 2,
                       "target_ladder": {"1": 0.33, "2": 0.44}})
    env = _make_env(cfg)

    assert env.target_ladder == {1: 0.33, 2: 0.44}
    env.round_budget = 1
    assert env._round_goal()["target_accuracy_drop"] == 0.33
    env.round_budget = 2
    assert env._round_goal()["target_accuracy_drop"] == 0.44


def test_shipped_config_declares_a_complete_ladder():
    """The real ``configs/base.yaml`` -- not a test fixture -- declares a rung
    for every budget in ``[1, attack.max_poison_clients]``, with strictly
    increasing values. Deliberately does NOT assert the five literal numbers:
    Plan 01's ``target_for_budget`` behavior check pins those exactly once,
    and GOAL-06 may raise the bottom rung in Plan 03. Asserting the same five
    numbers here too would turn a sanctioned rung raise into an unrelated
    test edit in a second place."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "configs", "base.yaml")
    attack = yaml.safe_load(open(config_path))["attack"]
    ladder = {int(k): float(v) for k, v in attack["target_ladder"].items()}
    cap = int(attack["max_poison_clients"])

    assert set(ladder) >= set(range(1, cap + 1))
    values_in_order = [ladder[b] for b in range(1, cap + 1)]
    assert all(v > 0.0 for v in values_in_order)
    assert all(b > a for a, b in zip(values_in_order, values_in_order[1:]))


# ---------------------------------------------------------------------------
# Retirement is self-enforcing (GOAL-04): a tree-wide scan, not a one-time grep
# ---------------------------------------------------------------------------

_RETIRED_TOKENS = ("target_choices", "sample_target_in_training")
_SCAN_SKIP_DIRS = {".git", "__pycache__", ".planning", ".claude", "logs", "data",
                   "checkpoints", "results", ".venv", "node_modules"}
_SCAN_EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".txt", ".json"}


def test_retired_target_sampling_keys_have_no_reader_left():
    """Automated form of ROADMAP Success Criterion 2's tree-wide search: after
    this phase, neither retired token (``target_choices``,
    ``sample_target_in_training``) may appear anywhere in tracked source,
    config or documentation -- except inside this scanning test itself, which
    necessarily contains both tokens in its own search strings.

    ``.planning/`` is excluded because, per 01-PATTERNS.md's Deletion Sweep,
    those files are requirement and discussion documents describing this very
    retirement, not readers of the retired keys -- flagging them would fail
    the suite on the phase's own planning artifacts forever.

    The point of this test is that a reintroduced key fails the suite
    permanently, not only on the day someone remembers to grep for it."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    this_file = os.path.abspath(__file__)

    hits = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _SCAN_SKIP_DIRS]
        for fname in filenames:
            if os.path.splitext(fname)[1] not in _SCAN_EXTENSIONS:
                continue
            fpath = os.path.join(dirpath, fname)
            if os.path.abspath(fpath) == this_file:
                continue
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        for token in _RETIRED_TOKENS:
                            if token in line:
                                hits.append(f"{fpath}:{lineno}: {token}")
            except OSError:
                continue

    assert hits == [], "retired token(s) reintroduced:\n" + "\n".join(hits)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} target-ladder tests passed.")


if __name__ == "__main__":
    _run()
