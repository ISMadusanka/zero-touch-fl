"""Torch-free tests for Multi-Krum's selection logic (neighbour count, selection
count, score formula, lowest-m selection). Runs anywhere:
    python tests/test_multikrum_logic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.defenses.multikrum import (   # noqa: E402
    k_closest_count, num_selected, krum_scores, select_lowest,
)


def test_k_closest_count_is_n_minus_f_minus_2_clamped():
    assert k_closest_count(5, 1) == 2          # paper n-f-2 for the default regime
    assert k_closest_count(5, 0) == 3
    assert k_closest_count(5, 3) == 1          # n-f-2 = 0 -> clamped up to 1
    assert k_closest_count(1, 0) == 0          # single client -> no neighbours


def test_num_selected_defaults_to_n_minus_f_and_clamps():
    assert num_selected(5, 1, None) == 4       # default m = n - f
    assert num_selected(5, 1, 2) == 2          # explicit override (paper's n-f-2 bound)
    assert num_selected(5, 1, 100) == 5        # clamp to n
    assert num_selected(5, 1, 0) == 1          # keep at least one


def _ring_dist():
    # clients 0-3 mutually close (dist 1), client 4 far (dist 100) from all.
    return [
        [0.0, 1.0, 1.0, 1.0, 100.0],
        [1.0, 0.0, 1.0, 1.0, 100.0],
        [1.0, 1.0, 0.0, 1.0, 100.0],
        [1.0, 1.0, 1.0, 0.0, 100.0],
        [100.0, 100.0, 100.0, 100.0, 0.0],
    ]


def test_krum_scores_sum_of_k_closest():
    scores = krum_scores(_ring_dist(), k=2)
    # benign: two closest are benign (1+1=2); outlier: two closest are 100+100=200.
    assert scores == [2.0, 2.0, 2.0, 2.0, 200.0]


def test_select_lowest_keeps_central_drops_outlier():
    scores = krum_scores(_ring_dist(), k=2)
    assert select_lowest(scores, 4) == {0, 1, 2, 3}            # outlier 4 dropped
    assert select_lowest(scores, 1) == {0}                     # Krum: single most-central


def test_select_lowest_nan_is_worst_and_clamps():
    nan = float("nan")
    assert select_lowest([nan, 1.0, 2.0, 3.0], 3) == {1, 2, 3}  # NaN never selected
    assert select_lowest([1.0, 2.0, 3.0], 10) == {0, 1, 2}      # m clamps to n


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} Multi-Krum logic tests passed.")


if __name__ == "__main__":
    _run()
