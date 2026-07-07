"""Torch-free tests for DnC's selection logic (removal count, lowest-k keep,
intersection + fallback, subsampling). Runs anywhere:  python tests/test_dnc_logic.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.defenses.dnc import (   # noqa: E402
    num_to_remove, keep_lowest, finalize_keep, subsample_indices,
)


def test_num_to_remove_default_and_rounding():
    assert num_to_remove(1.0, 1, 5) == 1                 # paper iid default c=1
    assert num_to_remove(1.0, 3, 5) == 3
    assert num_to_remove((5 - 1) / 5, 1, 5) == 1         # non-iid c=(n-1)/n, rounds to 1
    assert num_to_remove(0.0, 4, 5) == 0                 # c=0 -> remove nobody


def test_num_to_remove_clamps_to_keep_one():
    assert num_to_remove(1.0, 10, 5) == 4                # can't remove >= n
    assert num_to_remove(1.0, 1, 1) == 0                 # single client -> keep it


def test_keep_lowest_picks_smallest_scores():
    assert keep_lowest([5.0, 1.0, 3.0, 2.0, 4.0], 3) == {1, 3, 2}
    assert keep_lowest([1.0, 1.0, 2.0], 2) == {0, 1}     # ties broken by index
    assert keep_lowest([1.0, 2.0], 0) == set()


def test_keep_lowest_treats_nan_as_worst_outlier():
    # A NaN score must be dropped (treated as +inf), not silently kept. Without the
    # NaN guard, naive (score, i) sorting would leave index 0 in the kept set.
    nan = float("nan")
    assert keep_lowest([nan, 1.0, 2.0, 3.0, 4.0], 4) == {1, 2, 3, 4}
    assert keep_lowest([1.0, nan, 2.0], 2) == {0, 2}     # the NaN client is dropped


def test_finalize_keep_intersection():
    assert finalize_keep([{0, 1, 2}, {1, 2, 3}], [0.0] * 4, 2) == {1, 2}


def test_finalize_keep_falls_back_when_intersection_empty():
    # disjoint per-iteration sets -> intersection empty -> lowest-mean-score keep.
    out = finalize_keep([{0, 1}, {2, 3}], [0.1, 0.2, 0.3, 0.4], 2)
    assert out == {0, 1}


def test_finalize_keep_empty_input_keeps_all():
    assert finalize_keep([], [0.0, 0.0, 0.0], 2) == {0, 1, 2}


def test_subsample_uses_all_dims_when_b_ge_d():
    assert subsample_indices(5, 10, random.Random(0)) == [0, 1, 2, 3, 4]


def test_subsample_is_sorted_distinct_in_range():
    r = subsample_indices(100, 8, random.Random(0))
    assert len(r) == 8 and len(set(r)) == 8
    assert r == sorted(r) and all(0 <= x < 100 for x in r)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} DnC logic tests passed.")


if __name__ == "__main__":
    _run()
