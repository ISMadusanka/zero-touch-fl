"""Torch-free tests for DeFL's pure logic (layer grouping, CLP rule, MOUD-Vote,
Beta model). These exercise everything except the tensor norms / aggregation, so
they run anywhere:  python tests/test_defl_logic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.defenses.defl import (   # noqa: E402
    group_layers, aggregate_fgnv, is_clp, per_layer_votes, per_layer_zscores,
    moud_vote, BetaTracker,
)


def test_group_layers_groups_weight_and_bias():
    keys = ["net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"]
    groups = group_layers(keys)
    assert list(groups.keys()) == ["net.2", "net.4"]          # order preserved
    assert groups["net.2"] == ["net.2.weight", "net.2.bias"]
    assert groups["net.4"] == ["net.4.weight", "net.4.bias"]


def test_group_layers_unknown_suffix_is_own_layer():
    groups = group_layers(["embedding", "fc.weight", "fc.bias"])
    assert list(groups.keys()) == ["embedding", "fc"]
    assert groups["embedding"] == ["embedding"]


def test_aggregate_fgnv_uniform_and_weighted():
    m = [[1.0, 2.0], [3.0, 4.0]]
    assert aggregate_fgnv(m, [1.0, 1.0]) == [2.0, 3.0]          # plain mean
    assert aggregate_fgnv(m, [1.0, 3.0]) == [2.5, 3.5]          # data-weighted


def test_is_clp_first_round_and_degenerate_base():
    assert is_clp(5.0, None, 0.05) is True                      # no history -> CLP
    assert is_clp(5.0, 0.0, 0.05) is True                       # zero base -> CLP


def test_is_clp_rise_and_decline():
    assert is_clp(1.05, 1.0, 0.05) is True                      # +5% == threshold
    assert is_clp(1.049, 1.0, 0.05) is False                    # +4.9% < threshold
    assert is_clp(0.5, 1.0, 0.05) is False                      # FGNV declining -> not CLP


def test_per_layer_votes_flags_outlier_both_layers():
    # 4 tight benign + 1 large outlier on both layers.
    m = [[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.0], [10.0, 10.0]]
    votes = per_layer_votes(m, tau=2.5)
    assert votes == [0, 0, 0, 0, 2]


def test_per_layer_votes_zero_spread_flags_any_deviation():
    # identical benign (MAD=0) -> only the differing client is an outlier.
    m = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [5.0, 1.0]]
    votes = per_layer_votes(m, tau=2.5)
    assert votes == [0, 0, 0, 0, 1]                             # outlier on layer 0 only


def test_moud_vote_adaptive_threshold_lowers_to_one():
    # outlier shows up on ONLY one layer -> thr=L finds nobody, drops to thr=1.
    m = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [10.0, 1.0]]
    flagged, votes, thr = moud_vote(m, tau=2.5)
    assert votes[4] == 1 and max(votes[:4]) == 0
    assert flagged == [False, False, False, False, True]
    # THE threshold that made ``votes/L`` an inverted p_malicious: one vote out of two
    # is a rejection here, so reporting 1/2 = 0.5 said "coin flip" about a client the
    # defense had just caught. Consumers need this number to place the boundary.
    assert thr == 1


def test_moud_vote_both_layers_uses_strict_threshold():
    m = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [10.0, 10.0]]
    flagged, _votes, thr = moud_vote(m, tau=2.5)
    assert flagged == [False, False, False, False, True]
    assert thr == 2                                            # L, not lowered


def test_moud_vote_clean_round_flags_nobody():
    m = [[1.0, 1.0]] * 5                                        # no outliers anywhere
    flagged, votes, thr = moud_vote(m, tau=2.5)
    assert votes == [0, 0, 0, 0, 0]
    assert flagged == [False] * 5                              # not forced to flag
    # Reported boundary is still the lowest the loop reaches, and every client is
    # strictly below it, so a calibrated p_malicious puts them all under 0.5.
    assert thr == 1


def test_per_layer_zscores_matches_the_vote_test():
    """The vote count is a threshold ON these z-scores, so they must agree."""
    m = [[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.0], [10.0, 10.0]]
    z = per_layer_zscores(m)
    votes = per_layer_votes(m, tau=2.5)
    assert votes == [sum(1 for zij in row if zij > 2.5) for row in z]
    # The outlier's magnitude is what survives thresholding: with L=2 the vote count
    # alone takes three values, far too coarse to be a reward gradient on its own.
    assert z[4][0] > 2.5 and z[4][1] > 2.5
    assert max(z[0]) < 2.5


def test_per_layer_zscores_zero_spread_is_infinite_not_large():
    """MAD=0 makes the z-score undefined; a deviating client must not be finite-ranked."""
    m = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [5.0, 1.0]]
    z = per_layer_zscores(m)
    assert z[4][0] == float("inf") and z[4][1] == 0.0
    assert all(zij == 0.0 for row in z[:4] for zij in row)


def test_beta_tracker_updates_and_prob():
    bt = BetaTracker()
    assert bt.prob(0) == 0.5                                    # init alpha=beta=1
    bt.update(0, benign=True)                                   # alpha 2, beta 1
    assert abs(bt.prob(0) - 2 / 3) < 1e-12
    bt.update(0, benign=False)                                  # alpha 2, beta 2
    assert abs(bt.prob(0) - 0.5) < 1e-12
    # repeated malicious votes drive prob down (soft down-weighting post-CLP).
    for _ in range(8):
        bt.update(1, benign=False)
    assert bt.prob(1) < 0.15


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} DeFL logic tests passed.")


if __name__ == "__main__":
    _run()
