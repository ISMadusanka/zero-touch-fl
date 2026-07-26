"""Tests for the attacker's client-SELECTION parsing + application.

Covers agents/attack_ops.extract_selection and AttackerAgent.select_and_apply:
pool filtering, dedup, budget truncation, distinct per-client plans, and the
benign fallback on garbage. Needs torch (apply_plan operates on tensors).

Run on any box with torch:  python tests/test_attacker_select.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from agents.attack_ops import extract_selection  # noqa: E402
from agents.attacker_agent import AttackerAgent  # noqa: E402


def _sd(scale=1.0):
    return {
        "net.2.weight": torch.ones(3, 3) * scale,
        "net.2.bias": torch.ones(3) * scale,
        "net.4.weight": torch.ones(2, 3) * scale,
        "net.4.bias": torch.ones(2) * scale,
    }


def _pool(n=5):
    # A distinct benign reference per client (so we can tell them apart).
    return {cid: _sd(scale=cid + 1) for cid in range(n)}


def test_extract_selection_shapes():
    per = extract_selection(
        '{"clients":[{"id":0,"operations":[{"op":"scale","factor":2}]},'
        '{"id":3,"operations":[{"op":"sign_flip"}]}]}')
    assert [e["id"] for e in per["per_client"]] == [0, 3]
    assert per["shared_ops"] is None

    shared = extract_selection('{"clients":[0,3],"operations":[{"op":"scale","factor":2}]}')
    assert shared["shared_ids"] == [0, 3] and shared["shared_ops"]

    ops_only = extract_selection('{"operations":[{"op":"scale","factor":2}]}')
    assert ops_only["per_client"] == [] and ops_only["shared_ids"] is None and ops_only["shared_ops"]

    assert extract_selection("not json at all") is None


def test_per_client_distinct_plans_applied():
    agent = AttackerAgent()
    pool = _pool()
    text = ('{"clients":[{"id":0,"operations":[{"op":"scale","factor":2.0}]},'
            '{"id":3,"operations":[{"op":"scale","factor":0.0}]}]}')
    poisoned, chosen, n_malformed = agent.select_and_apply(text, pool, budget=5)
    assert chosen == [0, 3] and n_malformed == 0
    # client 0 doubled, client 3 zeroed -> distinct, coordinated per-client plans.
    assert torch.allclose(poisoned[0]["net.2.weight"], pool[0]["net.2.weight"] * 2.0)
    assert torch.allclose(poisoned[3]["net.2.weight"], torch.zeros_like(pool[3]["net.2.weight"]))


def test_budget_truncation():
    agent = AttackerAgent()
    pool = _pool()
    text = ('{"clients":[{"id":0,"operations":[{"op":"scale","factor":2}]},'
            '{"id":1,"operations":[{"op":"scale","factor":2}]},'
            '{"id":2,"operations":[{"op":"scale","factor":2}]},'
            '{"id":4,"operations":[{"op":"scale","factor":2}]}]}')
    _poisoned, chosen, _ = agent.select_and_apply(text, pool, budget=2)
    assert chosen == [0, 1]                     # first `budget` valid picks kept


def test_pool_filtering_and_dedup():
    agent = AttackerAgent()
    pool = _pool()
    text = ('{"clients":[{"id":99,"operations":[{"op":"scale","factor":2}]},'
            '{"id":1,"operations":[{"op":"scale","factor":2}]},'
            '{"id":1,"operations":[{"op":"scale","factor":3}]}]}')
    _poisoned, chosen, _ = agent.select_and_apply(text, pool, budget=5)
    assert chosen == [1]                        # 99 dropped (not in pool), 1 deduped


def test_budget_clamped_to_pool():
    agent = AttackerAgent()
    pool = _pool(5)
    text = '{"operations":[{"op":"scale","factor":2}]}'   # shared plan, auto-select
    _poisoned, chosen, _ = agent.select_and_apply(text, pool, budget=99)
    assert chosen == [0, 1, 2, 3, 4]            # clamped to the 5-client pool


def test_garbage_poisons_nobody():
    """Unparseable output must NOT register a ground-truth poisoned client.

    It used to fall back to 'client 0, benign weights, marked poisoned', which
    made the ASR metric report a 100% success rate for an attack that sent honest
    weights, and penalized the defender for missing an undetectable client.
    """
    agent = AttackerAgent()
    pool = _pool()
    poisoned, chosen, n_malformed = agent.select_and_apply("total garbage", pool, budget=3)
    assert poisoned == {} and chosen == [] and n_malformed == 1


def test_empty_plan_client_counts_malformed_and_is_not_poisoned():
    agent = AttackerAgent()
    pool = _pool()
    poisoned, chosen, n_malformed = agent.select_and_apply(
        '{"clients":[{"id":2,"operations":[]}]}', pool, budget=3)
    assert poisoned == {} and chosen == [] and n_malformed == 1


def test_noop_plan_is_malformed_not_poison():
    """A plan that parses but changes nothing (identity scale, or ops that are all
    skipped as invalid) sends byte-identical benign weights -> not poison."""
    agent = AttackerAgent()
    for text in (
        '{"clients":[{"id":1,"operations":[{"op":"scale","target":"all","factor":1.0}]}]}',
        '{"clients":[{"id":1,"operations":[{"op":"backdoor","target":"all"}]}]}',
        '{"clients":[{"id":1,"operations":[{"op":"scale","target":"no.such.layer","factor":9}]}]}',
    ):
        poisoned, chosen, n_malformed = agent.select_and_apply(text, _pool(), budget=3)
        assert poisoned == {} and chosen == [] and n_malformed == 1, text


def test_partial_noop_keeps_only_the_effective_client():
    """One real plan + one no-op -> exactly one poisoned client, one wasted."""
    agent = AttackerAgent()
    pool = _pool()
    text = ('{"clients":[{"id":0,"operations":[{"op":"scale","target":"all","factor":3.0}]},'
            '{"id":1,"operations":[{"op":"scale","target":"all","factor":1.0}]}]}')
    poisoned, chosen, n_malformed = agent.select_and_apply(text, pool, budget=2)
    assert chosen == [0] and list(poisoned) == [0] and n_malformed == 1
    assert torch.allclose(poisoned[0]["net.2.weight"], pool[0]["net.2.weight"] * 3.0)


def test_invalid_ops_alongside_a_real_one_still_poison():
    """Skipped ops are counted but don't waste the client if the net effect is real."""
    agent = AttackerAgent()
    pool = _pool()
    text = ('{"clients":[{"id":4,"operations":[{"op":"nonsense"},'
            '{"op":"scale","target":"all","factor":2.0}]}]}')
    poisoned, chosen, n_malformed = agent.select_and_apply(text, pool, budget=1)
    assert chosen == [4] and n_malformed == 0
    assert torch.allclose(poisoned[4]["net.2.weight"], pool[4]["net.2.weight"] * 2.0)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} attacker-selection tests passed.")


if __name__ == "__main__":
    _run()
