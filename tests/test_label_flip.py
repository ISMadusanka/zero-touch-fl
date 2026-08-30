"""Tests for the label-flipping attack: the flip itself and the adaptive ladder.

Covers ``data/label_flip.py`` and ``agents/label_flip_attacker.py``. The ladder
tests are torch-free; the dataset tests need torch only for the DataLoader.

    python tests/test_label_flip.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.label_flip_attacker import (   # noqa: E402
    FlipLadder, LabelFlipAttacker, build_attacker,
)
from data.label_flip import (              # noqa: E402
    choose_flip_positions, flip_label,
)


class _V:
    """Minimal DetectionVerdict stand-in (client_id + is_suspicious is all the
    ladder reads)."""

    def __init__(self, cid, suspicious):
        self.client_id = cid
        self.is_suspicious = suspicious


# ---------------------------------------------------------------------------
# The flip rule
# ---------------------------------------------------------------------------

def test_symmetric_flip_is_involutive_and_fixed_point_free():
    for y in range(10):
        assert flip_label(y) == 9 - y
        assert flip_label(flip_label(y)) == y      # involutive
        assert flip_label(y) != y                  # never a no-op on 10 classes
    assert flip_label(0) == 9 and flip_label(9) == 0
    assert flip_label(4) == 5 and flip_label(5) == 4


def test_flip_preserves_the_label_type():
    """A tensor label must come back as a tensor. Datasets disagree about what a
    label is (MNIST yields int, TensorDataset yields a 0-dim tensor), and
    default_collate refuses to stack a batch mixing the two — which is every
    partially-poisoned batch."""
    import torch
    assert isinstance(flip_label(3), int)
    t = flip_label(torch.tensor(3))
    assert isinstance(t, torch.Tensor) and t.dtype == torch.int64 and int(t) == 6


def test_flip_positions_are_deterministic_and_exactly_sized():
    a = choose_flip_positions(1000, 300, seed=7)
    b = choose_flip_positions(1000, 300, seed=7)
    c = choose_flip_positions(1000, 300, seed=8)
    assert len(a) == 300 and a == b, "same seed must select the same examples"
    assert a != c, "a different seed must select a different subset"
    assert all(0 <= i < 1000 for i in a)


def test_flip_positions_clamp_and_degenerate_cases():
    assert choose_flip_positions(50, 0, seed=1) == frozenset()
    assert choose_flip_positions(50, 999, seed=1) == frozenset(range(50))  # clamped
    assert choose_flip_positions(50, 50, seed=1) == frozenset(range(50))
    assert choose_flip_positions(0, 10, seed=1) == frozenset()


# ---------------------------------------------------------------------------
# The ladder — the behaviour the whole design rests on
# ---------------------------------------------------------------------------

def test_ladder_levels_are_exact_on_the_step_grid():
    """1.0/0.9/.../0.5 with no binary-representation drift.

    Repeated float subtraction from 1.0 lands on 0.7999999999999999 and
    0.5000000000000001, which would make "am I at the floor?" — the test that
    decides when the cycle resets — depend on rounding.
    """
    lad = FlipLadder(start=1.0, step=0.1, floor=0.5)
    assert lad.levels() == [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    assert lad.n_levels == 6 and lad.floor == 0.5


def test_caught_steps_down_missed_holds():
    """The core rule: back off when caught, repeat the level when not."""
    lad = FlipLadder()
    assert lad.fraction == 1.0
    assert lad.advance(caught=True) == "step_down"
    assert lad.fraction == 0.9
    assert lad.advance(caught=True) == "step_down"
    assert lad.fraction == 0.8
    # Not caught at 0.8 -> keep sending 0.8, for as long as it keeps working.
    for _ in range(5):
        assert lad.advance(caught=False) == "hold"
        assert lad.fraction == 0.8
    # ...until it is caught again.
    assert lad.advance(caught=True) == "step_down"
    assert lad.fraction == 0.7


def test_floor_resets_to_the_top_instead_of_descending():
    """Caught AT the floor restarts the cycle — it must not park there forever."""
    lad = FlipLadder(start=1.0, step=0.1, floor=0.5)
    for _ in range(5):                 # 1.0 -> 0.5
        lad.advance(caught=True)
    assert lad.fraction == 0.5 and lad.at_floor and lad.cycle == 0
    assert lad.advance(caught=True) == "reset"
    assert lad.fraction == 1.0 and lad.cycle == 1 and not lad.at_floor


def test_ladder_never_descends_below_the_floor():
    lad = FlipLadder(start=1.0, step=0.1, floor=0.5)
    seen = []
    for _ in range(40):
        seen.append(lad.fraction)
        lad.advance(caught=True)
    assert min(seen) == 0.5, "the floor is the lowest level ever sent"
    assert max(seen) == 1.0


def test_always_caught_produces_the_full_sawtooth():
    """The user-visible schedule: 1.0, .9, .8, .7, .6, .5, then 1.0 again."""
    lad = FlipLadder()
    seq = []
    for _ in range(13):
        seq.append(lad.fraction)
        lad.advance(caught=True)
    assert seq == [1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
                   1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 1.0]
    assert lad.cycle == 2


def test_floor_is_snapped_onto_the_step_grid():
    """A floor the steps cannot land on is moved UP to the nearest reachable
    level, never overshot — otherwise the ladder would descend past the
    configured floor."""
    lad = FlipLadder(start=1.0, step=0.1, floor=0.55)
    assert lad.floor == 0.6 and 0.55 not in lad.levels()
    assert min(lad.levels()) >= 0.55


def test_ladder_rejects_incoherent_grids():
    for kwargs in ({"start": 0.0}, {"start": 1.5}, {"step": 0.0}, {"step": -0.1},
                   {"floor": -0.1}, {"start": 0.5, "floor": 0.9}):
        try:
            FlipLadder(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"FlipLadder{kwargs} should have been rejected")


def test_ladder_state_round_trips():
    lad = FlipLadder()
    for _ in range(7):
        lad.advance(caught=True)
    saved = lad.state_dict()
    restored = FlipLadder()
    restored.load_state_dict(saved)
    assert restored.fraction == lad.fraction
    assert restored.cycle == lad.cycle and restored.level == lad.level


# ---------------------------------------------------------------------------
# The attacker: client set, plan, and the "caught" rule
# ---------------------------------------------------------------------------

def test_default_poison_set_is_client_zero():
    assert LabelFlipAttacker({}, n_clients=20).poison_client_ids == [0]
    assert LabelFlipAttacker({"poison_client_ids": 3}, n_clients=20).poison_client_ids == [3]


def test_poison_ids_are_validated_not_clamped():
    """An out-of-range id is DROPPED. Clamping would silently turn [0, 25] into
    two attacks on whatever id the clamp lands on, which is not what was asked."""
    a = LabelFlipAttacker({"poison_client_ids": [0, 25, 1, 1, -2]}, n_clients=20)
    assert a.poison_client_ids == [0, 1]
    try:
        LabelFlipAttacker({"poison_client_ids": [99]}, n_clients=20)
    except ValueError:
        pass
    else:
        raise AssertionError("an entirely invalid poison set must be rejected")


def test_plan_applies_the_fraction_to_each_clients_own_sample_count():
    """Non-IID shards are unequal, so the ladder is a PROPORTION, not a count."""
    a = LabelFlipAttacker({"poison_client_ids": [0, 1]}, n_clients=4)
    assert a.plan({0: 1000, 1: 500}) == {0: 1000, 1: 500}      # 100%
    a.ladder.advance(caught=True)                               # -> 90%
    assert a.plan({0: 1000, 1: 500}) == {0: 900, 1: 450}
    # The 1000-sample walk from the spec.
    counts = {0: 1000, 1: 1000}
    a2 = LabelFlipAttacker({"poison_client_ids": [0, 1]}, n_clients=4)
    walk = []
    for _ in range(6):
        walk.append(a2.plan(counts)[0])
        a2.ladder.advance(caught=True)
    assert walk == [1000, 900, 800, 700, 600, 500]


def test_plan_clamps_to_what_the_client_actually_holds():
    a = LabelFlipAttacker({}, n_clients=4)
    assert a.plan({0: 3})[0] == 3
    assert a.plan({0: 0})[0] == 0        # a client with no round data flips nothing
    assert a.plan({})[0] == 0            # missing count -> no flips, no crash


def test_caught_rule_all_is_the_default_and_needs_every_client_flagged():
    a = LabelFlipAttacker({"poison_client_ids": [0, 1, 2]}, n_clients=10)
    assert a.caught_rule == "all"
    ids = [0, 1, 2]
    assert a.caught([_V(0, True), _V(1, True), _V(2, True)], ids) is True
    # One surviving insider is a SUCCESSFUL round for the attack -> hold.
    assert a.caught([_V(0, True), _V(1, True), _V(2, False)], ids) is False
    assert a.caught([_V(0, False), _V(1, False), _V(2, False)], ids) is False


def test_caught_rule_any_and_majority():
    ids = [0, 1, 2]
    v = [_V(0, True), _V(1, False), _V(2, False)]
    assert LabelFlipAttacker({"poison_client_ids": ids, "schedule": {"caught_rule": "any"}},
                             n_clients=10).caught(v, ids) is True
    assert LabelFlipAttacker({"poison_client_ids": ids, "schedule": {"caught_rule": "majority"}},
                             n_clients=10).caught(v, ids) is False
    v2 = [_V(0, True), _V(1, True), _V(2, False)]
    assert LabelFlipAttacker({"poison_client_ids": ids, "schedule": {"caught_rule": "majority"}},
                             n_clients=10).caught(v2, ids) is True


def test_caught_uses_ground_truth_not_the_configured_set():
    """A client whose level rounded to zero flips sent an HONEST update. Holding
    the defense responsible for not flagging it would step the ladder on a
    detection that could not have happened."""
    a = LabelFlipAttacker({"poison_client_ids": [0, 1]}, n_clients=10)
    # Only client 0 actually shipped flipped labels this round.
    assert a.caught([_V(0, True), _V(1, False)], poisoned_ids=[0]) is True
    # No poison at all -> nothing to catch.
    assert a.caught([_V(0, True), _V(1, True)], poisoned_ids=[]) is False


def test_record_round_reports_the_transition():
    a = LabelFlipAttacker({}, n_clients=10)
    rec = a.record_round([_V(0, True)], [0])
    assert rec["flip_fraction"] == 1.0        # the level the round was SENT at
    assert rec["caught"] is True
    assert rec["event"] == "step_down"
    assert rec["next_flip_fraction"] == 0.9   # ...and where it moved to
    rec2 = a.record_round([_V(0, False)], [0])
    assert rec2["caught"] is False and rec2["event"] == "hold"
    assert rec2["flip_fraction"] == 0.9 and rec2["next_flip_fraction"] == 0.9


def test_flip_seed_is_stable_and_varies_by_round_and_client():
    a = LabelFlipAttacker({"poison_client_ids": [0, 1]}, n_clients=4, seed=3)
    b = LabelFlipAttacker({"poison_client_ids": [0, 1]}, n_clients=4, seed=3)
    assert a.flip_seed(5, 0) == b.flip_seed(5, 0), "must survive a restart"
    assert a.flip_seed(5, 0) != a.flip_seed(6, 0), "new round -> new examples"
    assert a.flip_seed(5, 0) != a.flip_seed(5, 1), "different clients differ"
    assert LabelFlipAttacker({}, n_clients=4, seed=4).flip_seed(5, 0) != a.flip_seed(5, 0)


def test_attacker_state_round_trips_through_a_checkpoint():
    a = LabelFlipAttacker({"poison_client_ids": [0, 1]}, n_clients=4)
    for _ in range(8):
        a.record_round([_V(0, True), _V(1, True)], [0, 1])
    saved = a.state_dict()
    b = LabelFlipAttacker({"poison_client_ids": [0, 1]}, n_clients=4)
    b.load_state_dict(saved)
    assert b.fraction == a.fraction and b.ladder.cycle == a.ladder.cycle


def test_build_attacker_rejects_any_other_attack_type():
    """Label flipping is the only implemented attack — an unrecognised type is a
    config error, not a request for a different attack."""
    assert build_attacker({"attack": {"type": "label_flip"}}, n_clients=4)
    for bad in ("model_replacement", "sign_flip", "scale", "gaussian_noise"):
        try:
            build_attacker({"attack": {"type": bad}}, n_clients=4)
        except ValueError:
            continue
        raise AssertionError(f"attack.type={bad!r} should have been rejected")


# ---------------------------------------------------------------------------
# The dataset wrapper (needs torch)
# ---------------------------------------------------------------------------

def test_flipped_loader_flips_exactly_n_labels_and_nothing_else():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from data.label_flip import build_flipped_loader

    x = torch.arange(100).float().view(100, 1)
    y = torch.arange(100) % 10
    honest = DataLoader(TensorDataset(x, y), batch_size=10, shuffle=False)

    poisoned = build_flipped_loader(honest, 30, seed=0)
    assert len(poisoned.dataset) == 100, "the client trains on the SAME examples"

    n_flipped = 0
    for i in range(100):
        xi, yi = poisoned.dataset[i]
        assert torch.equal(xi, x[i]), "inputs are untouched — labels only"
        if int(yi) != int(y[i]):
            assert int(yi) == 9 - int(y[i])
            n_flipped += 1
    assert n_flipped == 30

    # 0 flips must produce a byte-identical honest dataset.
    clean = build_flipped_loader(honest, 0, seed=0)
    assert all(int(clean.dataset[i][1]) == int(y[i]) for i in range(100))


def test_flipped_loader_preserves_batch_size():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from data.label_flip import build_flipped_loader

    honest = DataLoader(TensorDataset(torch.zeros(64, 1), torch.zeros(64, dtype=torch.long)),
                        batch_size=16, shuffle=True)
    assert build_flipped_loader(honest, 10, seed=0).batch_size == 16


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} label-flip tests passed.")
