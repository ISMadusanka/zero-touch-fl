"""Torch-free unit tests for ``benchmark.noise_probe.summarize_noise`` — the
statistics and verdict logic behind the GOAL-06 noise pre-flight gate.

Needs neither torch nor MNIST (only ``summarize_noise`` is imported, which is a
pure function over floats):

    python tests/test_noise_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.noise_probe import summarize_noise  # noqa: E402


def test_zero_variance_samples_clear_any_positive_rung():
    r = summarize_noise([0.80, 0.80, 0.80], rung=0.02)
    assert r["sd"] == 0.0
    assert r["threshold"] == 0.0
    assert r["clears"] is True
    assert r["required_rung"] == 0.0


def test_clears_uses_an_inclusive_comparison_at_exactly_the_margin():
    """A two-element list ``[m - d, m + d]`` has ``pstdev == d`` exactly (up to
    floating-point noise far below the comparison's tolerance), which makes
    this the boundary case: ``sd`` is exactly ``rung / sigma_margin``, and the
    inclusive ``>=`` comparison must record it as clearing."""
    rung, sigma_margin = 0.02, 3.0
    d = rung / sigma_margin
    r = summarize_noise([0.8 - d, 0.8 + d], rung=rung, sigma_margin=sigma_margin)
    assert abs(r["sd"] - d) < 1e-12
    assert r["clears"] is True


def test_a_defense_over_the_margin_fails_and_names_the_required_rung():
    r = summarize_noise([0.78, 0.82], rung=0.02, sigma_margin=3.0)
    assert r["clears"] is False
    # sd == 0.02, threshold == 0.06 -> required rung is 0.06 (already a whole cent).
    assert abs(r["required_rung"] - 0.06) < 1e-9


def test_fewer_than_two_samples_raises_instead_of_certifying():
    """A silent sd == 0.0 here would certify the rung from an aborted run."""
    for bad in ([], [0.8]):
        try:
            summarize_noise(bad, rung=0.02)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError as e:
            assert str(len(bad)) in str(e)


def test_standard_deviation_is_computed_on_unrounded_values():
    """Samples differing only in the 8th decimal place must still yield a
    strictly positive sd — proving no pre-rounding happened before pstdev."""
    r = summarize_noise([0.80000000, 0.80000001], rung=0.02)
    assert r["sd"] > 0.0


def test_identical_inputs_produce_identical_verdicts():
    samples = [0.781, 0.799, 0.812, 0.788]
    r1 = summarize_noise(samples, rung=0.02, sigma_margin=3.0)
    r2 = summarize_noise(list(samples), rung=0.02, sigma_margin=3.0)
    assert r1["sd"] == r2["sd"]
    assert r1["threshold"] == r2["threshold"]
    assert r1["clears"] == r2["clears"]


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} noise-probe tests passed.")


if __name__ == "__main__":
    _run()
