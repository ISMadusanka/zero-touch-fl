"""Tests for the DEFENDER prompt: its compact encoding and its context-fill band.

The defender gets ONE LLM call per round and must label EVERY client in it, so the
prompt is budgeted on two axes at once (``agents/prompt_budget.py``):

    prompt_tokens                             <= defender_max_prompt_fill  * max_seq_len
    prompt_tokens + defender_max_new_tokens   <= defender_max_context_fill * max_seq_len

Over the band the observation is COMPACTED (``agents/defender_agent.py::
_COMPACTION``), never truncated and never with a client dropped — a missing client
is an automatic wrong (default-benign) verdict. Below the band the fill is reported
as under-filled, because the richest rung of the ladder is then cheaper than the
config allows.

Run on any box with torch:  python tests/test_defender_prompt.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import yaml  # noqa: E402

import agents.defender_agent as defender_mod  # noqa: E402
from agents.attack_ops import sig_round  # noqa: E402
from agents.defender_agent import (  # noqa: E402
    CORE_LAYER_STAT_KEYS, LAYER_STAT_KEYS, WHOLE_STAT_KEYS, DefenderAgent,
)
from agents.prompt_budget import PromptBudget, build_prompt_budget  # noqa: E402
from core.types import ModelUpdate  # noqa: E402
from detector.features import compute_client_features  # noqa: E402
from model.mnist_net import MnistNet  # noqa: E402

RL_CFG = {"max_seq_len": 16384, "max_new_tokens": 1536, "max_context_fill": 0.5,
          "defender_max_new_tokens": 1024, "defender_max_context_fill": 0.60,
          "defender_max_prompt_fill": 0.30, "defender_min_prompt_fill": 0.20}


def _global():
    torch.manual_seed(0)
    return MnistNet().state_dict()


def _features(n_clients=20, n_poisoned=10, scale=6.0):
    """Per-client features for a cohort where the first ``n_poisoned`` are outliers."""
    g = _global()

    def weights(seed, mult):
        torch.manual_seed(seed)
        return {k: v + torch.randn_like(v) * 0.01 * mult for k, v in g.items()}

    updates = [
        ModelUpdate(client_id=i,
                    weights=weights(i + 1, scale if i < n_poisoned else 1.0),
                    metadata={})
        for i in range(n_clients)
    ]
    return compute_client_features(updates, g)


def _agent(**rl):
    cfg = dict(RL_CFG)
    cfg.update(rl)
    a = DefenderAgent({"rl": cfg})
    # A deterministic stand-in for the real tokenizer: ~1 token per 3 characters,
    # so a level's size is a pure function of its rendered length.
    a.bind_tokenizer(lambda s, u: (len(s) + len(u)) // 3)
    return a


def _level_sizes(agent, features):
    """Prompt tokens each compaction level would cost for ``features``."""
    return [agent.budget.count(agent.system_prompt(), agent._render(features, step))
            for step in defender_mod._COMPACTION]


# --- the prompt-only cap --------------------------------------------------

def test_prompt_cap_binds_before_the_total_cap():
    """``max_prompt_fill`` must be able to bite while the total cap still has room."""
    b = PromptBudget(context_window=16384, max_fill=0.60, reserved_output=1024,
                     max_prompt_fill=0.30)
    assert b.call_limit == 9830                      # 60% of the window
    assert b.prompt_cap == 4915                      # 30% of the window
    # The total cap alone would allow 9830 - 1024 = 8806 prompt tokens; the
    # prompt-only cap is what actually binds. This is the whole reason the two are
    # separate: a small reserved_output must not let the observation grow to fill
    # the entire call budget.
    assert b.prompt_limit == 4915
    assert b.fits(4915) and not b.fits(4916)


def test_total_cap_still_binds_when_it_is_the_tighter_one():
    b = PromptBudget(context_window=16384, max_fill=0.30, reserved_output=4000,
                     max_prompt_fill=0.90)
    assert b.prompt_limit == int(16384 * 0.30) - 4000
    assert b.prompt_limit < b.prompt_cap


def test_no_prompt_cap_reproduces_the_attacker_arithmetic():
    """``max_prompt_fill=None`` must leave the pre-existing behaviour untouched."""
    b = PromptBudget(context_window=16384, max_fill=0.5, reserved_output=1536)
    assert b.prompt_cap == 0 and b.prompt_limit == 8192 - 1536


def test_floor_is_reporting_only():
    b = PromptBudget(context_window=16384, max_fill=0.60, reserved_output=1024,
                     max_prompt_fill=0.30, min_prompt_fill=0.20)
    assert b.prompt_floor == 3276
    assert b.underfilled(3000) and not b.underfilled(3276)
    assert b.fits(3000)                  # under-filled is still perfectly valid
    assert abs(b.prompt_fill(3276) - 0.2) < 1e-3


def test_builder_reads_the_defender_overrides_and_falls_back():
    b = build_prompt_budget(RL_CFG, prefix="defender_")
    assert b.context_window == 16384          # max_seq_len is never prefixed
    assert b.reserved_output == 1024           # defender_max_new_tokens
    assert b.max_fill == 0.60                  # defender_max_context_fill
    assert b.max_prompt_fill == 0.30 and b.min_prompt_fill == 0.20
    # The attacker's own budget must be unaffected by the defender keys.
    a = build_prompt_budget(RL_CFG)
    assert a.reserved_output == 1536 and a.max_fill == 0.50 and a.max_prompt_fill is None
    # A config with no defender_* keys falls back to the shared ones.
    f = build_prompt_budget({"max_seq_len": 8192, "max_new_tokens": 256}, prefix="defender_")
    assert f.reserved_output == 256 and f.max_prompt_fill is None


# --- the compact encoding -------------------------------------------------

def test_rows_carry_the_same_numbers_as_the_feature_dict():
    """The positional encoding is a re-LAYOUT, not a re-computation: every number in
    a row is the feature dict's own value at the configured significant figures."""
    feats = _features(n_clients=4, n_poisoned=2)
    a = _agent()
    sig = a.detail_precision
    payload = json.loads(a.build_user_prompt(feats))
    for cid, f in feats.items():
        row = payload["clients"][str(cid)]
        for layer, stats in f["layers"].items():
            for i, name in enumerate(LAYER_STAT_KEYS):
                assert row[layer][i] == sig_round(stats[name], sig), (cid, layer, name)
        for i, name in enumerate(WHOLE_STAT_KEYS):
            assert row["whole"][i] == sig_round(f["whole"][name], sig), (cid, name)


def test_compact_encoding_is_much_smaller_than_the_named_key_form():
    feats = _features()
    verbose = json.dumps({"client_ids": list(feats),
                          "features": {str(c): f for c, f in feats.items()}})
    compact = _agent().build_user_prompt(feats)
    # Under half, despite the compact form ALSO carrying the cohort + ranking blocks
    # the verbose form never had.
    assert len(compact) < 0.6 * len(verbose), (len(compact), len(verbose))


def test_layer_names_and_legends_are_sent_once():
    feats = _features(n_clients=3, n_poisoned=1)
    payload = json.loads(_agent().build_user_prompt(feats))
    assert payload["layers"] == ["net.2", "net.4"]
    assert payload["layer_key"] == list(LAYER_STAT_KEYS)
    assert payload["whole_key"] == list(WHOLE_STAT_KEYS)
    # ...and NOT repeated inside any client's row.
    assert "rel_norm" not in json.dumps(payload["clients"])


def test_cohort_mirrors_a_client_row_and_reports_median_and_mad():
    feats = _features(n_clients=6, n_poisoned=3)
    payload = json.loads(_agent().build_user_prompt(feats))
    cohort, row = payload["cohort"], payload["clients"]["0"]
    assert set(cohort) == set(row)                     # same keys as one client row
    for key, ref in cohort.items():
        assert len(ref) == 2, key                      # [medians, mads]
        assert len(ref[0]) == len(ref[1]) == len(row[key]), key
    # The whole-model median really is the across-client median.
    values = sorted(f["whole"]["rel_norm"] for f in feats.values())
    expected = 0.5 * (values[2] + values[3])           # 6 clients -> mean of middle two
    assert abs(cohort["whole"][0][WHOLE_STAT_KEYS.index("rel_norm")] - expected) < 1e-3


def test_ranked_sorts_each_statistic_in_the_suspicious_direction():
    feats = _features(n_clients=8, n_poisoned=4)
    payload = json.loads(_agent().build_user_prompt(feats))
    ranked = payload["ranked"]
    assert set(ranked) == {k for k, _ in defender_mod._RANK_KEYS}
    for key, descending in defender_mod._RANK_KEYS:
        order = ranked[key]
        assert sorted(order) == sorted(feats)          # every client, no duplicates
        vals = [feats[cid]["whole"][key] for cid in order]
        assert vals == sorted(vals, reverse=descending), key


def test_outlier_clients_surface_at_the_top_of_the_rankings():
    """The oversized cohort half must lead the rel_norm ranking — otherwise the
    shortlist the prompt tells the model to walk is pointing the wrong way."""
    feats = _features(n_clients=20, n_poisoned=10, scale=6.0)
    payload = json.loads(_agent().build_user_prompt(feats))
    assert set(payload["ranked"]["rel_norm"][:10]) == set(range(10))


# --- the compaction ladder ------------------------------------------------

def test_shipped_settings_fit_the_cap_at_full_detail():
    a = _agent()
    a.build_user_prompt(_features())
    st = a.last_prompt_stats
    assert st["level"] == 0, st            # no observation detail sacrificed
    assert st["fits"]
    assert st["prompt_fill"] <= 0.30       # inside the prompt cap
    assert st["fill"] <= 0.60              # ...and the whole call inside its cap


def test_a_tight_window_compacts_instead_of_truncating():
    a = _agent(max_seq_len=4096, defender_max_new_tokens=512)
    text = a.build_user_prompt(_features())
    st = a.last_prompt_stats
    assert st["level"] > 0                 # the ladder was climbed
    payload = json.loads(text)             # ...and the JSON is still complete
    assert list(payload["clients"]) == [str(i) for i in range(20)]


def test_every_rung_of_the_ladder_is_strictly_cheaper_than_the_last():
    """A compaction level that does not actually save tokens cannot rescue a prompt,
    so the ladder would silently stall on it instead of reaching one that fits."""
    feats = _features()
    a = _agent()
    sizes = _level_sizes(a, feats)
    assert sizes == sorted(sizes, reverse=True), sizes
    assert len(set(sizes)) == len(sizes), sizes


def test_compaction_drops_detail_in_the_documented_order():
    """Walk the ladder one rung at a time and check WHAT each rung gave up.

    The cap is set to each level's own measured size rather than to a hand-picked
    context window, so this pins the DEGRADATION ORDER — precision, then the
    least-informative per-layer statistic, then the rankings, then per-layer detail
    entirely — without breaking every time the prompt's wording changes length.
    """
    feats = _features()
    sizes = _level_sizes(_agent(), feats)

    for level in range(1, len(defender_mod._COMPACTION)):
        a = _agent(defender_max_prompt_fill=(sizes[level] + 0.5) / RL_CFG["max_seq_len"])
        payload = json.loads(a.build_user_prompt(feats))
        st = a.last_prompt_stats
        assert st["level"] == level, (level, st)
        assert st["fits"]
        # Invariant at EVERY rung: every client, and the cohort reference that makes
        # their numbers interpretable, survive.
        assert len(payload["clients"]) == 20
        assert "whole" in payload["cohort"]
        assert all("whole" in row for row in payload["clients"].values())

        if level < 2:                                   # only precision was lost
            assert payload["layer_key"] == list(LAYER_STAT_KEYS)
        else:                                           # l2_norm goes next
            assert payload.get("layer_key", []) == list(CORE_LAYER_STAT_KEYS) or (
                "layer_key" not in payload)
        if level >= 3:
            assert "ranked" not in payload              # rankings before raw data
        else:
            assert "ranked" in payload
        if level >= 4:
            assert "layers" not in payload              # last resort: whole model only
            assert all(len(row) == 1 for row in payload["clients"].values())


def test_no_client_is_ever_dropped_even_when_nothing_fits():
    a = _agent(max_seq_len=512, defender_max_new_tokens=128)
    text = a.build_user_prompt(_features())
    st = a.last_prompt_stats
    assert st["level"] == len(defender_mod._COMPACTION) - 1
    assert not st["fits"]                  # reported, not hidden
    payload = json.loads(text)
    assert len(payload["clients"]) == 20
    assert "layers" not in payload         # last rung: whole-model statistics only
    assert all("whole" in row and len(row) == 1 for row in payload["clients"].values())


def test_under_filled_prompts_are_flagged_but_emitted_unchanged():
    """A floor is a report, never a rejection or a reason to pad."""
    a = _agent(defender_min_prompt_fill=0.90)
    text = a.build_user_prompt(_features())
    st = a.last_prompt_stats
    assert st["underfilled"] and st["fits"] and st["level"] == 0
    assert len(json.loads(text)["clients"]) == 20


def test_no_rl_block_still_builds_a_complete_prompt():
    """An older config with no rl: block must not break the agent."""
    a = DefenderAgent({})
    payload = json.loads(a.build_user_prompt(_features(n_clients=5, n_poisoned=2)))
    assert len(payload["clients"]) == 5
    assert a.last_prompt_stats["level"] == 0     # inert budget => never compacts


def test_empty_cohort_does_not_raise():
    a = _agent()
    payload = json.loads(a.build_user_prompt({}))
    assert payload["clients"] == {} and payload["n_clients"] == 0


# --- the prompt still round-trips through the parser ----------------------

def test_every_client_in_the_prompt_can_be_answered_and_parsed():
    feats = _features(n_clients=6, n_poisoned=3)
    a = _agent()
    payload = json.loads(a.build_user_prompt(feats))
    ids = payload["client_ids"]
    reply = json.dumps({"clients": [
        {"client_id": c, "is_suspicious": c < 3, "confidence": 0.9} for c in ids
    ]})
    verdicts = a.parse(reply, ids)
    assert [v.client_id for v in verdicts] == ids
    assert [v.is_suspicious for v in verdicts] == [True, True, True, False, False, False]


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} defender-prompt tests passed.")


if __name__ == "__main__":
    _run()
