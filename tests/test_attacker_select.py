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


def test_garbage_falls_back_to_one_benign_client():
    agent = AttackerAgent()
    pool = _pool()
    poisoned, chosen, n_malformed = agent.select_and_apply("total garbage", pool, budget=3)
    assert chosen == [0] and n_malformed == 1
    assert torch.allclose(poisoned[0]["net.2.weight"], pool[0]["net.2.weight"])  # unchanged


def test_empty_plan_client_counts_malformed():
    agent = AttackerAgent()
    pool = _pool()
    poisoned, chosen, n_malformed = agent.select_and_apply(
        '{"clients":[{"id":2,"operations":[]}]}', pool, budget=3)
    assert chosen == [2] and n_malformed == 1
    assert torch.allclose(poisoned[2]["net.2.weight"], pool[2]["net.2.weight"])


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} attacker-selection tests passed.")


if __name__ == "__main__":
    _run()
