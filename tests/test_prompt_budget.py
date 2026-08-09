"""Tests for the attacker prompt's CONTEXT-FILL budget.

``rl.max_context_fill`` caps how much of the model's context one attacker call
may occupy, counting ``rl.max_new_tokens`` reserved for the completion. The cap
is enforced by compacting the observation (``agents/attacker_agent.py::
_COMPACTION``), never by truncating it and never by dropping a client. These
tests cover ``agents/prompt_budget.py``, the compact encoding in
``agents/attack_ops.py``, and the ladder that ties them together.

Run on any box with torch:  python tests/test_prompt_budget.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import agents.attacker_agent as attacker_agent_mod  # noqa: E402
from agents.attack_ops import (  # noqa: E402
    LAYER_STAT_KEYS, WHOLE_STAT_KEYS, delta_details, delta_rows, delta_stats,
    layer_shapes,
)
from agents.attacker_agent import AttackerAgent  # noqa: E402
from agents.prompt_budget import PromptBudget, build_prompt_budget, estimate_tokens  # noqa: E402
from model.mnist_net import MnistNet  # noqa: E402

RL_CFG = {"max_seq_len": 16384, "max_new_tokens": 1536, "max_context_fill": 0.5}


def _global():
    torch.manual_seed(0)
    return MnistNet().state_dict()


def _client(g, seed):
    torch.manual_seed(seed)
    return {k: v + torch.randn_like(v) * 0.01 for k, v in g.items()}


def _pool(g, n):
    return {i: _client(g, i + 1) for i in range(n)}


# --- PromptBudget ---------------------------------------------------------

def test_budget_arithmetic():
    b = PromptBudget(context_window=16384, max_fill=0.5, reserved_output=1536)
    assert b.call_limit == 8192
    assert b.prompt_limit == 8192 - 1536
    assert b.fits(6000) and not b.fits(7000)
    assert abs(b.fill(6656) - 0.5) < 1e-9


def test_no_context_window_never_binds():
    b = PromptBudget(context_window=0)
    assert not b.active and b.fits(10 ** 9) and b.fill(10 ** 9) == 0.0


def test_reserved_output_cannot_eat_the_whole_budget():
    """A config with no room left for a prompt reports rather than silently passing."""
    b = PromptBudget(context_window=1024, max_fill=0.5, reserved_output=2048)
    assert b.prompt_limit == 1                    # floored, not negative
    assert not b.fits(2)


def test_real_tokenizer_is_preferred_and_failure_falls_back():
    b = PromptBudget(context_window=8192, token_counter=lambda s, u: 123)
    assert b.count("sys", "user") == 123 and b.exact()

    def boom(s, u):
        raise RuntimeError("no tokenizer")

    b2 = PromptBudget(context_window=8192, token_counter=boom)
    n = b2.count("sys", "user")
    assert n == estimate_tokens("sys", "user") and not b2.exact()


def test_build_from_rl_config():
    b = build_prompt_budget(RL_CFG)
    assert b.context_window == 16384 and b.reserved_output == 1536 and b.max_fill == 0.5


# --- compact observation encoding ----------------------------------------

def test_delta_rows_carry_the_same_numbers_as_delta_details():
    g = _global()
    c = _client(g, 7)
    verbose = delta_details(c, g, precision=10)
    rows = delta_rows(c, g, sig=10)
    for layer, names in ((k, LAYER_STAT_KEYS) for k in verbose["layers"]):
        for i, name in enumerate(names):
            assert abs(rows[layer][i] - verbose["layers"][layer][name]) < 1e-9, (layer, name)
    for i, name in enumerate(WHOLE_STAT_KEYS):
        assert abs(rows["whole"][i] - verbose["whole"][name]) < 1e-9, name


def test_compact_encoding_is_much_smaller():
    g = _global()
    pool = _pool(g, 10)
    verbose = json.dumps({str(c): delta_details(sd, g) for c, sd in pool.items()})
    compact = json.dumps({str(c): delta_rows(sd, g) for c, sd in pool.items()},
                         separators=(",", ":"))
    assert len(compact) < 0.45 * len(verbose)


def test_layer_shapes_are_sent_once():
    g = _global()
    assert layer_shapes(g) == {"net.2.weight": [16, 49], "net.2.bias": [16],
                               "net.4.weight": [10, 16], "net.4.bias": [10]}
    assert "shape" not in json.dumps(delta_rows(_client(g, 1), g))


def test_significant_figures_keep_small_magnitudes():
    """3 decimals would flatten rms_delta (~1e-2..1e-3); 3 sig figs does not."""
    g = _global()
    rows = delta_rows(_client(g, 3), g, sig=3)
    rms = rows["net.2.weight"][LAYER_STAT_KEYS.index("rms_delta")]
    assert rms > 0 and len(f"{rms:g}".lstrip("-0.").rstrip("0")) <= 3


# --- the compaction ladder ------------------------------------------------

def _agent(pool_size, **rl):
    cfg = dict(RL_CFG)
    cfg.update(rl)
    a = AttackerAgent({"fixed_poison_set": True, "rl": cfg})
    # A deterministic stand-in for the real tokenizer: ~1 token per 3 characters,
    # so a level's size is a pure function of its rendered length.
    a.bind_tokenizer(lambda s, u: (len(s) + len(u)) // 3)
    return a


def test_shipped_settings_fit_the_cap_at_full_detail():
    g = _global()
    a = _agent(10)
    a.build_user_prompt(1, 0.9, _pool(g, 10), g, budget=10)
    st = a.last_prompt_stats
    assert st["fits"] and st["fill"] < 0.5
    assert st["level"] == 0, st          # no observation detail sacrificed


def test_a_tight_window_compacts_instead_of_truncating():
    g = _global()
    a = _agent(10, max_seq_len=4096, max_new_tokens=1024)
    text = a.build_user_prompt(1, 0.9, _pool(g, 10), g, budget=10)
    st = a.last_prompt_stats
    assert st["level"] > 0               # the ladder was climbed
    payload = json.loads(text)           # ...and the JSON is still complete
    assert list(payload["client_update_stats"]) == [str(i) for i in range(10)]
    assert payload["poison_client_ids"] == list(range(10))


def test_no_client_is_ever_dropped_even_when_nothing_fits():
    g = _global()
    a = _agent(10, max_seq_len=512, max_new_tokens=256)
    text = a.build_user_prompt(1, 0.9, _pool(g, 10), g, budget=10)
    st = a.last_prompt_stats
    assert st["level"] == len(attacker_agent_mod._COMPACTION) - 1
    assert not st["fits"]                # reported, not hidden
    assert len(json.loads(text)["client_update_stats"]) == 10


def test_level_zero_matches_the_configured_precision():
    g = _global()
    a = _agent(2)
    text = a.build_user_prompt(1, 0.9, _pool(g, 2), g, budget=2)
    payload = json.loads(text)
    assert payload["stats_key"] == list(LAYER_STAT_KEYS)
    assert payload["whole_key"] == list(WHOLE_STAT_KEYS)
    expected = delta_rows(_pool(g, 2)[0], g, sig=a.detail_precision)
    assert payload["client_update_stats"]["0"] == expected


def test_larger_pools_cost_proportionally_more():
    g = _global()
    a = _agent(20)
    sizes = []
    for n in (1, 5, 10):
        a.build_user_prompt(1, 0.9, _pool(g, n), g, budget=n)
        sizes.append(a.last_prompt_stats["prompt_tokens"])
    assert sizes[0] < sizes[1] < sizes[2]


def test_no_budget_configured_still_builds_a_prompt():
    """An older config with no rl: block must not break the agent."""
    g = _global()
    a = AttackerAgent({"fixed_poison_set": True})
    text = a.build_user_prompt(1, 0.9, _pool(g, 10), g, budget=10)
    assert json.loads(text)["poison_client_ids"] == list(range(10))
    assert a.last_prompt_stats["level"] == 0     # inert budget => never compacts


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} prompt-budget tests passed.")


if __name__ == "__main__":
    _run()
