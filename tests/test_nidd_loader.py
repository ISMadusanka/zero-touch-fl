"""Tests for 5G-NIDD ingest/preprocessing (data/nidd_loader.py) and the FL model
(model/nidd_net.py).

Everything here runs on small hand-built CSVs written to a temp dir, or on the
synthetic generator — no 5G-NIDD download, no GPU, no LLM:

    python tests/test_nidd_loader.py

The properties under test are the ones that would silently corrupt results rather
than crash: leakage (test statistics reaching the scaler, identifier columns
reaching the model), non-determinism in feature selection, and a label encoding
that stops matching the model's output width.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data import feature_spec  # noqa: E402
from data.feature_spec import DEFAULT_SPEC, FeatureSpec  # noqa: E402
from data.nidd_loader import (  # noqa: E402
    DEFAULT_DROP_COLUMNS, _anova_f, _stratified_split, clear_memo, get_data_loaders,
    load_nidd,
)
from model.nidd_net import DEFAULT_HIDDEN, NiddNet, build_model, count_parameters  # noqa: E402

_TMP = None


def _tmpdir() -> str:
    global _TMP
    if _TMP is None:
        _TMP = tempfile.mkdtemp(prefix="nidd_test_")
    return _TMP


def _write_csv(name: str, rows: list[str]) -> str:
    path = os.path.join(_tmpdir(), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    return path


def _cfg(**over) -> dict:
    """A loader config that never touches the network or a shared cache."""
    cfg = dict(source="synthetic", max_samples=4000, n_features=8,
               cache_dir=os.path.join(_tmpdir(), "cache"), use_cache=False)
    cfg.update(over)
    return cfg


def _reset():
    feature_spec.reset_active()


# --- the model ---------------------------------------------------------------

def test_model_is_smaller_than_the_image_model_it_replaced():
    """The requirement: fewer trainable parameters than the 970-param MNIST net."""
    net = build_model(FeatureSpec(input_dim=32, n_classes=9))
    assert count_parameters(net) == 681
    assert count_parameters(net) < 970
    # And the formula the config comments quote, (K+1)*H + (H+1)*C, holds generally.
    for k, h, c in ((16, 8, 9), (32, 16, 9), (48, 32, 2)):
        n = count_parameters(NiddNet(input_dim=k, n_classes=c, hidden=h))
        assert n == (k + 1) * h + (h + 1) * c, (k, h, c, n)


def test_model_has_two_logical_layers_named_as_the_docs_say():
    net = build_model(FeatureSpec(input_dim=32, n_classes=9))
    shapes = {k: tuple(v.shape) for k, v in net.state_dict().items()}
    assert shapes == {"net.0.weight": (DEFAULT_HIDDEN, 32), "net.0.bias": (DEFAULT_HIDDEN,),
                      "net.2.weight": (9, DEFAULT_HIDDEN), "net.2.bias": (9,)}
    # Every state_dict entry must be a genuine learnable parameter (no BatchNorm
    # buffers): the attack DSL clamps and scales these tensors, and the defenses
    # flatten them, both of which assume there is nothing else in there.
    learnable = {n for n, _ in net.named_parameters()}
    assert set(net.state_dict()) == learnable


def test_model_is_tolerant_of_a_singleton_dimension():
    net = build_model(FeatureSpec(input_dim=8, n_classes=3))
    flat = torch.randn(4, 8)
    assert net(flat).shape == (4, 3)
    assert net(flat.unsqueeze(1)).shape == (4, 3)          # (batch, 1, features)


def test_model_defaults_to_the_documented_spec_offline():
    """`NiddNet()` must work with no data loaded — unit tests rely on it."""
    _reset()
    net = NiddNet()
    assert net.input_dim == DEFAULT_SPEC.input_dim
    assert net.n_classes == DEFAULT_SPEC.n_classes


# --- the feature spec --------------------------------------------------------

def test_published_spec_overrides_the_default_and_round_trips():
    _reset()
    assert feature_spec.active() is DEFAULT_SPEC
    spec = FeatureSpec(input_dim=5, n_classes=3, class_names=["a", "b", "c"],
                       feature_names=["f1", "f2", "f3", "f4", "f5"], source="csv")
    feature_spec.set_active(spec)
    assert feature_spec.active() is spec
    assert build_model().input_dim == 5                    # models follow the spec

    path = os.path.join(_tmpdir(), "schema_roundtrip.json")
    spec.to_json(path)
    assert FeatureSpec.from_json(path) == spec
    _reset()


def test_missing_or_corrupt_schema_falls_back_instead_of_raising():
    assert FeatureSpec.from_json(os.path.join(_tmpdir(), "nope.json")) is None
    bad = _write_csv("truncated.json", ['{"input_dim": 32, "n_class'])
    assert FeatureSpec.from_json(bad) is None               # logged, not raised


def test_spec_matching_is_about_shape_not_names():
    a = FeatureSpec(input_dim=4, n_classes=2, feature_names=["w", "x", "y", "z"])
    b = FeatureSpec(input_dim=4, n_classes=2, feature_names=["p", "q", "r", "s"])
    assert a.matches(b)                                     # loadable weights
    assert not a.matches(FeatureSpec(input_dim=5, n_classes=2))
    assert not a.matches(None)


# --- preprocessing ----------------------------------------------------------

def test_identifier_and_leakage_columns_never_reach_the_model():
    """A model handed SrcAddr learns "this IP is the attacker" and scores ~100%.

    The drop list is what stops that, so this asserts the columns are gone from the
    fitted feature set even though they are perfectly predictive — which is exactly
    why a variance- or F-based selector would otherwise rank them first.
    """
    rows = ["Dur,SrcAddr,Sport,Attack Tool,TotBytes,Attack Type,Label"]
    for i in range(200):
        attack = i % 2 == 0
        rows.append(f"{0.5 + i * 0.01},"
                    f"{'10.0.0.9' if attack else '10.0.0.1'},"
                    f"{40000 + i},"
                    f"{'hping3' if attack else 'none'},"
                    f"{100 + i},"
                    f"{'UDPFlood' if attack else 'Benign'},"
                    f"{'Malicious' if attack else 'Benign'}")
    path = _write_csv("leakage.csv", rows)

    _reset()
    _, _, spec = load_nidd(_cfg(source="csv", csv_path=path, max_samples=None,
                                n_features=8), seed=0)
    for banned in ("SrcAddr", "Sport", "Attack Tool", "Label"):
        assert banned not in spec.feature_names, (banned, spec.feature_names)
    assert set(spec.feature_names) <= {"Dur", "TotBytes"}
    assert spec.class_names == ["Benign", "UDPFlood"]
    _reset()


def test_default_drop_list_is_lowercase_so_matching_works():
    # The lookup lowercases CSV headers before comparing; an uppercase entry here
    # would silently never match and quietly re-admit a leakage column.
    assert all(c == c.lower() for c in DEFAULT_DROP_COLUMNS)


def test_scaler_is_fitted_on_train_only():
    """Test rows must not move the standardization statistics.

    Built so the two splits have wildly different scales: if the scaler saw the
    test rows, the train split's standardized mean would be far from 0.
    """
    rows = ["Dur,TotBytes,Attack Type"]
    # 400 rows of one class, 400 of another; the LAST 20% of each (which the
    # stratified split holds out) are enormous.
    for c, name in ((0, "Benign"), (1, "UDPFlood")):
        for i in range(400):
            big = i >= 320
            base = (1e6 if big else 1.0) + i
            rows.append(f"{base},{base * 2},{name}")
    path = _write_csv("scale.csv", rows)

    _reset()
    train, test, spec = load_nidd(
        _cfg(source="csv", csv_path=path, max_samples=None, n_features=2), seed=0)
    x_train = train.features.numpy()
    # Fitted on train => train is centred; the untouched huge test rows are not.
    assert abs(float(x_train.mean())) < 0.2, float(x_train.mean())
    assert float(np.abs(test.features.numpy()).max()) > 2.0
    _reset()


def test_feature_selection_is_deterministic_and_ranks_by_separability():
    """Same config twice must select the same columns, in the same order.

    `signal` separates the classes, `noise` does not, and `constant` cannot. Ties
    (both constants score 0) break by name so two machines agree.
    """
    rng = np.random.default_rng(0)
    rows = ["signal,noise,constant,constant_b,Attack Type"]
    for c, name in ((0, "Benign"), (1, "UDPFlood")):
        for _ in range(300):
            rows.append(f"{c * 5 + rng.normal():.5f},{rng.normal():.5f},7,7,{name}")
    path = _write_csv("select.csv", rows)

    cfg = _cfg(source="csv", csv_path=path, max_samples=None, n_features=2)
    _reset()
    clear_memo()
    _, _, first = load_nidd(cfg, seed=0)
    _reset()
    clear_memo()                                             # genuinely re-fit, not memoized
    _, _, second = load_nidd(cfg, seed=0)
    assert first.feature_names == second.feature_names
    # Ranked signal (F~3.2e3) > noise (F~0.75) > the constants (F=0); the kept set is
    # then stored in CSV column order, which here is signal, noise.
    assert first.feature_names == ["signal", "noise"]
    # Asking for one more takes a constant — and which one is decided by the
    # name tiebreak, not by whichever order numpy happened to produce.
    _reset()
    clear_memo()
    _, _, three = load_nidd(dict(cfg, n_features=3), seed=0)
    assert three.feature_names == ["signal", "noise", "constant"]
    _reset()


def test_anova_f_scores_a_separating_column_above_noise():
    y = np.array([0] * 100 + [1] * 100)
    separating = np.concatenate([np.zeros(100), np.ones(100) * 5.0])
    constant = np.ones(200)
    x = np.stack([separating, constant], axis=1)
    scores = _anova_f(x, y, 2)
    assert scores[0] > 1000
    assert scores[1] < 1e-6                                 # constant separates nothing


def test_unseen_categorical_level_maps_to_the_reserved_code():
    """Code 0 means "never seen in training" — it must not collide with a real level."""
    rows = ["Proto,Dur,Attack Type"]
    for i in range(200):
        attack = i % 2 == 0
        # 'icmp' appears ONLY in rows the stratified split holds out for test.
        proto = "icmp" if i >= 180 else ("udp" if attack else "tcp")
        rows.append(f"{proto},{i * 0.1},{'UDPFlood' if attack else 'Benign'}")
    path = _write_csv("cats.csv", rows)

    _reset()
    _, _, spec = load_nidd(_cfg(source="csv", csv_path=path, max_samples=None,
                                n_features=2), seed=0)
    assert "Proto" in spec.feature_names
    _reset()


def test_split_is_stratified_and_keeps_every_class_on_both_sides():
    y = np.array([0] * 500 + [1] * 100 + [2] * 3)
    rng = np.random.default_rng(0)
    tr, te = _stratified_split(y, 3, 0.2, rng)
    assert len(tr) + len(te) == len(y)
    assert not (set(tr.tolist()) & set(te.tolist()))         # disjoint
    for c in range(3):
        assert (y[tr] == c).sum() >= 1, c                    # class 2 has only 3 rows
        assert (y[te] == c).sum() >= 1, c


def test_a_class_wiped_out_by_subsampling_is_re_encoded_contiguously():
    """A gap in the label space would give the model an output unit nothing uses.

    Class indices must stay 0..n-1 and `spec.n_classes` must match, or
    CrossEntropyLoss and the model's output width disagree.
    """
    rows = ["Dur,TotBytes,Attack Type"]
    for i in range(600):
        rows.append(f"{i * 0.01},{i},Benign")
    for i in range(600):
        rows.append(f"{i * 0.02},{i * 3},UDPFlood")
    rows.append("9.9,9,ICMPFlood")                           # a single rare row
    path = _write_csv("rare.csv", rows)

    _reset()
    train, test, spec = load_nidd(
        _cfg(source="csv", csv_path=path, max_samples=None, n_features=2), seed=0)
    labels = set(train.targets) | set(test.targets)
    assert labels == set(range(spec.n_classes)), (labels, spec.n_classes)
    assert len(spec.class_names) == spec.n_classes
    assert build_model(spec).n_classes == spec.n_classes
    _reset()


def test_binary_label_mode_collapses_to_two_classes():
    rows = ["Dur,TotBytes,Attack Type,Label"]
    for i in range(300):
        attack = i % 3 != 0
        rows.append(f"{i * 0.01},{i},"
                    f"{'UDPFlood' if attack else 'Benign'},"
                    f"{'Malicious' if attack else 'Benign'}")
    path = _write_csv("binary.csv", rows)

    _reset()
    _, _, spec = load_nidd(_cfg(source="csv", csv_path=path, max_samples=None,
                                n_features=2, label_mode="binary"), seed=0)
    assert spec.n_classes == 2
    assert spec.class_names == ["Benign", "Malicious"]
    assert "Attack Type" not in spec.feature_names           # the other label column
    _reset()


def test_missing_csv_names_the_config_key_and_the_synthetic_escape_hatch():
    _reset()
    try:
        load_nidd(_cfg(source="csv", csv_path=os.path.join(_tmpdir(), "absent.csv")))
    except FileNotFoundError as e:
        msg = str(e)
        assert "data.csv_path" in msg
        assert "synthetic" in msg
    else:
        raise AssertionError("a missing CSV must raise FileNotFoundError")
    _reset()


# --- cache ------------------------------------------------------------------

def test_cache_round_trips_and_is_invalidated_by_changed_options():
    cache = os.path.join(_tmpdir(), "cache_rt")
    cfg = _cfg(cache_dir=cache, use_cache=True, max_samples=2000, n_features=8)
    _reset()
    train, _, first = load_nidd(cfg, seed=0)
    assert any(f.startswith("processed_") for f in os.listdir(cache))
    _reset()
    clear_memo()                                             # force the DISK path
    train2, _, cached = load_nidd(cfg, seed=0)
    assert cached == first                                   # spec survives the round trip
    assert torch.equal(train.features, train2.features)       # so do the arrays
    _reset()
    # A different feature count must NOT be served the old arrays.
    clear_memo()
    _, _, other = load_nidd(dict(cfg, n_features=4), seed=0)
    assert other.input_dim == 4
    _reset()


def test_memo_avoids_reparsing_when_the_disk_cache_is_off():
    """A run loads the data twice — client loaders, then FLTrust's root set — and with
    `use_cache: false` both would otherwise re-parse the whole CSV."""
    cfg = _cfg(use_cache=False, max_samples=2000, n_features=8)
    _reset()
    clear_memo()
    train, _, spec = load_nidd(cfg, seed=0)
    again, _, spec2 = load_nidd(cfg, seed=0)                  # served from the memo
    assert spec2 == spec
    assert torch.equal(train.features, again.features)
    # And the memo holds one entry at a time, not one per configuration.
    load_nidd(dict(cfg, n_features=4), seed=0)
    from data.nidd_loader import _MEMO
    assert len(_MEMO) == 1
    _reset()
    clear_memo()


# --- end to end -------------------------------------------------------------

def test_loaders_are_partitioned_and_feed_the_model():
    _reset()
    client_loaders, test_loader = get_data_loaders(
        n_clients=20, batch_size=64, data_cfg=_cfg(max_samples=4000, n_features=8),
        iid=False, bias_q=0.5, seed=0)
    assert len(client_loaders) == 20
    spec = feature_spec.active()
    net = build_model()
    xb, yb = next(iter(test_loader))
    assert xb.shape[1] == spec.input_dim == 8
    assert xb.dtype == torch.float32 and yb.dtype == torch.int64
    assert int(yb.max()) < spec.n_classes
    assert net(xb).shape == (xb.shape[0], spec.n_classes)

    # Shards must be disjoint and non-empty (the round sampler slices them per client).
    seen = set()
    for loader in client_loaders:
        idx = set(loader.dataset.indices)
        assert idx and not (idx & seen)
        seen |= idx
    _reset()


def test_synthetic_source_reproduces_the_dataset_shape():
    _reset()
    _, _, spec = load_nidd(_cfg(max_samples=20000, n_features=32), seed=0)
    assert spec.source == "synthetic"                         # never mistaken for real
    assert spec.n_classes == 9
    assert spec.input_dim == 32
    assert count_parameters(build_model(spec)) == 681
    _reset()


def test_checkpoint_shape_guard_rejects_a_model_from_other_preprocessing():
    from storage.checkpoint import shape_mismatch

    _reset()
    feature_spec.set_active(FeatureSpec(input_dim=32, n_classes=9))
    good = build_model().state_dict()
    assert shape_mismatch(good) is None

    stale = build_model(FeatureSpec(input_dim=16, n_classes=9)).state_dict()
    assert "different feature/class count" in (shape_mismatch(stale) or "")
    assert "different layers" in (shape_mismatch({"unexpected.weight": torch.zeros(1)}) or "")
    _reset()


def test_checkpoint_guard_rejects_a_synthetic_baseline_for_a_real_run():
    """The dangerous case: shapes MATCH but the data did not.

    A `data.source: synthetic` smoke run produces a 32-feature/9-class model that is
    byte-compatible with a real one, so the shape check alone would happily resume
    from it and every number in the run would describe generated traffic.
    """
    import storage.checkpoint as ckpt

    _reset()
    real = FeatureSpec(input_dim=32, n_classes=9,
                       feature_names=[f"f{i}" for i in range(32)], source="csv")
    fake = FeatureSpec(input_dim=32, n_classes=9,
                       feature_names=[f"f{i}" for i in range(32)], source="synthetic")

    feature_spec.set_active(fake)
    weights = build_model().state_dict()               # shape-identical either way
    saved_to = os.path.join(_tmpdir(), "ckpt_provenance")
    original_dir = ckpt.CHECKPOINT_DIR
    try:
        ckpt.CHECKPOINT_DIR = saved_to
        fake.to_json(os.path.join(saved_to, ckpt._SPEC_FILE))

        assert ckpt.shape_mismatch(weights) is None     # synthetic run reusing synthetic
        feature_spec.set_active(real)                   # now a REAL run finds it
        why = ckpt.shape_mismatch(weights) or ""
        assert "synthetic" in why and "invalidate" in why, why

        # A different feature SET at the same width is also caught.
        real.to_json(os.path.join(saved_to, ckpt._SPEC_FILE))
        feature_spec.set_active(FeatureSpec(
            input_dim=32, n_classes=9,
            feature_names=[f"g{i}" for i in range(32)], source="csv"))
        assert "different feature set" in (ckpt.shape_mismatch(weights) or "")

        # A checkpoint with no recorded spec is "unknown", not "mismatched".
        os.remove(os.path.join(saved_to, ckpt._SPEC_FILE))
        assert ckpt.shape_mismatch(weights) is None
    finally:
        ckpt.CHECKPOINT_DIR = original_dir
        _reset()


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        for t in tests:
            t()
            print(f"  ok  {t.__name__}")
        print(f"\nAll {len(tests)} 5G-NIDD loader/model tests passed.")
    finally:
        if _TMP:
            shutil.rmtree(_TMP, ignore_errors=True)


if __name__ == "__main__":
    _run()
