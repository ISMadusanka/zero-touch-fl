"""5G-NIDD loading, preprocessing and partitioning across clients.

5G-NIDD (Samarakoon et al., 2022) is 1,215,890 labelled network flows captured on
the University of Oulu 5G Test Network: benign traffic plus eight attacks — five
DoS (UDPFlood, HTTPFlood, SlowrateDoS, SYNFlood, ICMPFlood) and three port scans
(TCPConnectScan, SYNScan, UDPScan) — described by ~52 Argus flow features. Nine
classes counting benign.

It is **tabular**, which is the whole reason the model changed shape: there is no
image grid to convolve or average-pool, so the FL model is a plain fully-connected
network over a feature vector (``model/nidd_net.py``).

Pipeline
--------
``get_data_loaders`` is the only entry point the system calls. It:

 1. reads the CSV (or generates a synthetic stand-in — see ``source``);
 2. drops identifier / leakage columns (below) and encodes the label;
 3. subsamples to ``max_samples`` so a Phase-2 round costs about what it cost on
    MNIST — 1.2M flows over 20 clients would be ~60k examples per client per
    round, which is a different experiment, not a faster one;
 4. splits train/test **stratified** and seeded;
 5. FITS preprocessing ON THE TRAIN SPLIT ONLY — categorical vocabularies,
    missing-value medians, standardization statistics, and the top-K feature
    selector — then transforms both splits;
 6. caches the result and publishes a :class:`~data.feature_spec.FeatureSpec` so
    ``model.build_model`` knows the input width.

Steps 4-then-5 are in that order deliberately: fitting the scaler or the feature
selector on all rows and splitting afterwards leaks test-set statistics into the
model, which on an intrusion-detection dataset inflates accuracy enough to matter
(the reward here IS test accuracy, so a leak would inflate every number the arms
race produces).

Leakage columns
---------------
``SrcAddr``/``DstAddr``/``Sport``/``Dport`` are dropped by default and this is not
cosmetic. In 5G-NIDD the attack traffic originates from a handful of testbed
hosts, so a model given the source address learns "this IP is the attacker" and
reports ~100% accuracy without learning anything about traffic. The same goes for
``Attack Tool`` (a direct restatement of the label) and the timestamps. Override
with ``data.drop_columns`` if you want them back.

Schema tolerance
----------------
5G-NIDD ships as several CSVs (per-attack and combined) whose exact column
spelling varies between releases and mirrors, so nothing here hardcodes a column
list. The label column is auto-detected from candidates, the drop list is applied
only where it matches, and every remaining column is classified numeric or
categorical by trying to parse it. The one thing pinned down is the model's input
width: ``data.n_features`` selects exactly K columns, so the parameter count is a
config constant rather than a property of whichever CSV was handed over.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from data.feature_spec import SCHEMA_FILENAME, FeatureSpec, set_active

logger = logging.getLogger(__name__)

#: Columns removed before anything else, when present (case-insensitive). Two
#: groups: restatements of the label, and identifiers that make the task trivial
#: for the wrong reason. See "Leakage columns" above.
DEFAULT_DROP_COLUMNS = (
    # --- label restatements ---
    "attack tool", "attack_tool", "attacktool",
    # --- host / port / flow identifiers ---
    "srcaddr", "dstaddr", "src_addr", "dst_addr", "saddr", "daddr",
    "sport", "dport", "src_port", "dst_port",
    "svid", "dvid", "srcid", "flgs", "state",
    # --- timestamps and row counters ---
    "starttime", "lasttime", "stime", "ltime", "seq", "unnamed: 0", "index",
)

#: Label column candidates, most specific first. ``multiclass`` wants the attack
#: type; ``binary`` wants the benign/malicious flag.
_LABEL_CANDIDATES = {
    "multiclass": ("attack type", "attack_type", "attacktype", "label", "class"),
    "binary": ("label", "binary label", "binary_label", "attack", "class"),
}

#: The nine canonical class names, used by the synthetic generator and as the
#: documented ordering. Real runs take their names from the CSV.
CLASS_NAMES = ("Benign", "HTTPFlood", "ICMPFlood", "SYNFlood", "SYNScan",
               "SlowrateDoS", "TCPConnectScan", "UDPFlood", "UDPScan")

#: Published class mix of 5G-NIDD, as a fraction of all 1,215,890 flows. Benign
#: is 39.3%; the eight attacks are the paper's within-attack percentages scaled
#: by the 60.7% attack share. Used ONLY by the synthetic generator, so a
#: no-CSV smoke run has the same brutal class imbalance a real run has.
_SYNTHETIC_CLASS_MIX = {
    "Benign": 0.3930, "UDPFlood": 0.3757, "HTTPFlood": 0.1153,
    "SlowrateDoS": 0.0601, "TCPConnectScan": 0.0164, "SYNScan": 0.0164,
    "UDPScan": 0.0127, "SYNFlood": 0.0079, "ICMPFlood": 0.0009,
}

_EPS = 1e-8

#: Standardized features are clipped to +-this many standard deviations. Flow
#: statistics are heavy-tailed (byte counts, rates), and a single 300-sigma outlier
#: in one column otherwise dominates the first layer's gradients and destabilizes
#: local training on whichever client happens to hold that flow.
_CLIP = 10.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TabularDataset(Dataset):
    """Preprocessed flows in memory: float32 features, int64 labels.

    Exposes ``.targets`` (a plain int list) because that is the interface
    :func:`partition_noniid_fltrust` and the rest of the system already expect
    from a torchvision dataset — keeping it means the partitioner, the per-round
    sampler and every ``Subset``-unwrapping helper work unchanged.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32)
        self.labels = torch.as_tensor(np.ascontiguousarray(labels), dtype=torch.long)
        #: int labels for every sample, in row order (torchvision-compatible).
        self.targets = self.labels.tolist()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def _read_csv(csv_path: str):
    """Read one CSV, or concatenate every ``*.csv`` in a directory.

    5G-NIDD is distributed both as ``Combined.csv`` and as per-attack files; a
    directory of the latter is concatenated so either layout works. Columns are
    unioned across files, so a per-attack CSV missing a column contributes NaN
    there rather than dropping the file.
    """
    import pandas as pd

    if os.path.isdir(csv_path):
        files = sorted(
            os.path.join(csv_path, f) for f in os.listdir(csv_path)
            if f.lower().endswith(".csv")
        )
        if not files:
            raise FileNotFoundError(f"no .csv files under {csv_path!r}")
        logger.info(f"5G-NIDD: reading {len(files)} CSV file(s) from {csv_path}")
        frames = [pd.read_csv(f, low_memory=False) for f in files]
        return pd.concat(frames, ignore_index=True, sort=False)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"5G-NIDD CSV not found at {csv_path!r}.\n"
            f"  The dataset is not redistributable, so it is not vendored here. Get it from\n"
            f"    https://ieee-dataport.org/documents/"
            f"5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless\n"
            f"  and point `data.csv_path` in configs/base.yaml at the CSV (or at a directory\n"
            f"  of per-attack CSVs).\n"
            f"  To smoke-test the pipeline without the real data, set `data.source: synthetic`\n"
            f"  — that generates 5G-NIDD-shaped traffic and is NOT a substitute for results."
        )
    logger.info(f"5G-NIDD: reading {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


def _synthetic_frame(n_rows: int, seed: int, n_features: int = 48):
    """5G-NIDD-shaped traffic for offline smoke tests. **Not real data.**

    Generates ``n_features`` columns — mostly numeric flow statistics plus a few
    categorical ones — with per-class mean offsets so the classes are learnable
    but not trivially separable, and the published class imbalance so the rare
    classes are as rare as they really are. Named so it is obvious in any log or
    schema dump that the run was synthetic.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    names = list(CLASS_NAMES)
    probs = np.array([_SYNTHETIC_CLASS_MIX[n] for n in names], dtype=np.float64)
    probs = probs / probs.sum()
    y = rng.choice(len(names), size=n_rows, p=probs)

    n_num = max(1, n_features - 3)
    # Each class gets a random centre; features are that centre plus noise, with a
    # deliberately low signal-to-noise ratio on most columns so feature selection
    # has something to do.
    centres = rng.normal(0.0, 1.0, size=(len(names), n_num))
    salience = rng.random(n_num) ** 2          # most columns weakly informative
    x = centres[y] * salience + rng.normal(0.0, 1.0, size=(n_rows, n_num))
    # Flow features are heavy-tailed and non-negative (byte/packet counts, rates).
    x[:, : n_num // 2] = np.exp(x[:, : n_num // 2])

    frame = {f"Feat{i}": x[:, i].astype(np.float32) for i in range(n_num)}
    # A few categorical columns, mirroring Proto / Cause / sDSb.
    frame["Proto"] = np.array(["tcp", "udp", "icmp"])[rng.integers(0, 3, n_rows)]
    frame["Cause"] = np.array(["Start", "Status"])[rng.integers(0, 2, n_rows)]
    frame["sDSb"] = np.array(["cs0", "af11", "ef"])[rng.integers(0, 3, n_rows)]
    frame["Attack Type"] = np.array(names)[y]
    frame["Label"] = np.where(y == 0, "Benign", "Malicious")
    return pd.DataFrame(frame)


def _find_label_column(columns, label_mode: str, explicit: str | None):
    """Resolve the label column name (case/whitespace tolerant)."""
    lookup = {str(c).strip().lower(): c for c in columns}
    if explicit:
        key = str(explicit).strip().lower()
        if key not in lookup:
            raise KeyError(
                f"configured data.label_column {explicit!r} is not in the CSV; "
                f"columns are: {', '.join(map(str, columns))}"
            )
        return lookup[key]
    for cand in _LABEL_CANDIDATES.get(label_mode, _LABEL_CANDIDATES["multiclass"]):
        if cand in lookup:
            return lookup[cand]
    raise KeyError(
        f"could not find a label column for label_mode={label_mode!r}; tried "
        f"{_LABEL_CANDIDATES.get(label_mode)}. Set data.label_column explicitly. "
        f"Columns are: {', '.join(map(str, columns))}"
    )


# ---------------------------------------------------------------------------
# Preprocessing (fit on train only)
# ---------------------------------------------------------------------------

def _anova_f(x: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """One-way ANOVA F statistic per column — the feature-selection criterion.

    F = (between-class variance / (k-1)) / (within-class variance / (N-k)). High
    F means the column's mean differs a lot between attack classes relative to
    the spread inside them, which is exactly "this feature separates the classes".

    Chosen over mutual information because it is closed-form, needs no binning
    choices, and is deterministic — the selected feature set must be identical on
    every machine or two runs train different models under the same config.
    Constant columns score 0 and sort last.
    """
    n_samples, n_cols = x.shape
    grand_mean = x.mean(axis=0)
    between = np.zeros(n_cols, dtype=np.float64)
    within = np.zeros(n_cols, dtype=np.float64)
    present = 0
    for c in range(n_classes):
        mask = (y == c)
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        present += 1
        xc = x[mask]
        mean_c = xc.mean(axis=0)
        between += n_c * (mean_c - grand_mean) ** 2
        within += ((xc - mean_c) ** 2).sum(axis=0)
    df_between = max(1, present - 1)
    df_within = max(1, n_samples - present)
    return (between / df_between) / (within / df_within + _EPS)


class _Preprocessor:
    """Fitted-on-train column encoders, imputer, scaler and top-K selector.

    Kept as one object because the four steps share fitted state and must be
    applied to the test split — and later to the FLTrust root sample — in exactly
    the same way.
    """

    def __init__(self, n_features: int):
        self.n_features = int(n_features)
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.vocab: dict[str, dict[str, int]] = {}
        self.medians: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.selected: list[int] = []
        self.feature_names: list[str] = []

    # -- column typing -------------------------------------------------
    def _classify_columns(self, frame) -> None:
        """Split columns into numeric and categorical.

        A column is numeric when at least 90% of its non-null values parse as
        numbers. The threshold (rather than "all values parse") tolerates the
        sentinel strings Argus exports scatter through otherwise-numeric columns.
        """
        import pandas as pd

        for col in frame.columns:
            series = frame[col]
            parsed = pd.to_numeric(series, errors="coerce")
            non_null = series.notna().sum()
            ok = parsed.notna().sum()
            if non_null == 0 or ok >= 0.9 * non_null:
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

    def _encode(self, frame) -> np.ndarray:
        """Frame -> float matrix, in ``numeric_cols + categorical_cols`` order."""
        import pandas as pd

        blocks = []
        if self.numeric_cols:
            num = frame[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            blocks.append(num.to_numpy(dtype=np.float64, copy=False))
        for col in self.categorical_cols:
            codes = frame[col].astype("string").fillna("<NA>").map(
                self.vocab[col]).fillna(0.0)
            blocks.append(codes.to_numpy(dtype=np.float64, copy=False).reshape(-1, 1))
        x = np.hstack(blocks) if blocks else np.zeros((len(frame), 0))
        return np.where(np.isfinite(x), x, np.nan)

    # -- fit / transform -----------------------------------------------
    def fit(self, frame, y: np.ndarray, n_classes: int) -> None:
        self._classify_columns(frame)
        # Categorical vocabulary: rank by frequency so code 1 is the most common
        # level. Unseen levels at transform time map to 0, a reserved code that
        # therefore means "not present in training".
        for col in self.categorical_cols:
            counts = frame[col].astype("string").fillna("<NA>").value_counts()
            self.vocab[col] = {str(level): float(i + 1)
                               for i, level in enumerate(counts.index)}

        x = self._encode(frame)
        self.medians = np.nanmedian(x, axis=0) if x.size else np.zeros(0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        x = self._impute(x)

        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        # Floor the scale so constant columns become exact zeros instead of
        # dividing by ~0 and exploding into the model's first layer.
        self.std = np.where(std > 1e-6, std, 1.0)
        x = (x - self.mean) / self.std
        # Clip HERE too, matching transform(): the selector must rank the features the
        # model will actually see. Flow columns are heavy-tailed, and an unclipped
        # 300-sigma outlier inflates a column's between-class variance enough to win a
        # top-K slot on the strength of a handful of rows that get flattened away.
        np.clip(x, -_CLIP, _CLIP, out=x)

        scores = _anova_f(x, y, n_classes)
        all_names = list(self.numeric_cols) + list(self.categorical_cols)
        k = min(self.n_features, x.shape[1])
        if k < self.n_features:
            logger.warning(
                f"data.n_features={self.n_features} but only {x.shape[1]} usable "
                f"column(s) survived the drop list — using all {k}"
            )
        # Sort by score desc, then name asc: ties (constant columns all score 0)
        # must break the same way on every machine or the model differs by run.
        order = sorted(range(len(all_names)),
                       key=lambda i: (-float(scores[i]), str(all_names[i])))
        self.selected = sorted(order[:k])
        self.feature_names = [all_names[i] for i in self.selected]
        kept = ", ".join(f"{all_names[i]}({scores[i]:.3g})" for i in order[:8])
        logger.info(f"5G-NIDD: selected {k}/{len(all_names)} features by ANOVA F; "
                    f"strongest: {kept}")

    def _impute(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        idx = np.where(np.isnan(x))
        if idx[0].size:
            x = x.copy()
            x[idx] = np.take(self.medians, idx[1])
        return x

    def transform(self, frame) -> np.ndarray:
        x = self._impute(self._encode(frame))
        x = (x - self.mean) / self.std
        np.clip(x, -_CLIP, _CLIP, out=x)          # see _CLIP
        return x[:, self.selected].astype(np.float32)


# ---------------------------------------------------------------------------
# Split / subsample helpers
# ---------------------------------------------------------------------------

def _stratified_indices(y: np.ndarray, n_classes: int, rng: np.random.Generator,
                        max_samples: int | None, balance: str) -> np.ndarray:
    """Row indices to keep, preserving (or equalizing) the class mix.

    ``natural`` keeps 5G-NIDD's real, severely imbalanced mix — ICMPFlood is
    0.09% of flows — which is what makes accuracy on this dataset mean what the
    literature means by it. ``balanced`` caps every class at the same count,
    which trades faithfulness for every class actually being learnable.
    """
    by_class = [np.flatnonzero(y == c) for c in range(n_classes)]
    for idx in by_class:
        rng.shuffle(idx)

    if balance == "balanced":
        per = (max_samples // max(1, sum(1 for i in by_class if len(i)))
               if max_samples else min((len(i) for i in by_class if len(i)), default=0))
        keep = [idx[:per] for idx in by_class]
    elif max_samples is not None and max_samples < len(y):
        frac = max_samples / len(y)
        # At least one row per non-empty class: proportional rounding would erase
        # ICMPFlood entirely at small max_samples, silently reducing n_classes.
        keep = [idx[:max(1, int(round(frac * len(idx))))] for idx in by_class if len(idx)]
    else:
        keep = by_class

    out = np.concatenate([k for k in keep if len(k)]) if any(len(k) for k in keep) \
        else np.zeros(0, dtype=np.int64)
    rng.shuffle(out)
    return out


def _stratified_split(y: np.ndarray, n_classes: int, test_fraction: float,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Seeded stratified train/test split over row positions."""
    train, test = [], []
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        if not len(idx):
            continue
        rng.shuffle(idx)
        n_test = int(round(test_fraction * len(idx)))
        # Never let a class vanish from either side while it has >= 2 rows.
        n_test = min(max(n_test, 1), len(idx) - 1) if len(idx) > 1 else 0
        test.append(idx[:n_test])
        train.append(idx[n_test:])
    tr = np.concatenate(train) if train else np.zeros(0, dtype=np.int64)
    te = np.concatenate(test) if test else np.zeros(0, dtype=np.int64)
    rng.shuffle(tr)
    rng.shuffle(te)
    return tr, te


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(options: dict) -> str:
    """Fingerprint of everything that changes the processed arrays.

    The CSV is fingerprinted by (path, size, mtime) rather than by hashing its
    contents — a 1 GB hash on every startup would cost more than the parse it is
    meant to skip, and size+mtime catches every realistic edit.
    """
    payload = dict(options)
    path = payload.get("csv_path")
    if path and os.path.isdir(path):
        # Stat every CSV, not the directory: a directory's own mtime does not change
        # when a file inside it is rewritten, so fingerprinting the dir alone would
        # keep serving a stale cache after the data was replaced.
        payload["_csv_stat"] = sorted(
            [f, int(os.stat(os.path.join(path, f)).st_size),
             int(os.stat(os.path.join(path, f)).st_mtime)]
            for f in os.listdir(path) if f.lower().endswith(".csv")
        )
    elif path and os.path.exists(path):
        st = os.stat(path)
        payload["_csv_stat"] = [int(st.st_size), int(st.st_mtime)]
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_cache(cache_dir: str, key: str):
    npz = os.path.join(cache_dir, f"processed_{key}.npz")
    spec = FeatureSpec.from_json(os.path.join(cache_dir, SCHEMA_FILENAME))
    if not os.path.exists(npz) or spec is None:
        return None
    try:
        blob = np.load(npz, allow_pickle=False)
        data = (blob["x_train"], blob["y_train"], blob["x_test"], blob["y_test"])
    except (OSError, ValueError, KeyError) as e:
        logger.warning(f"5G-NIDD: ignoring unreadable cache {npz}: {e}")
        return None
    logger.info(f"5G-NIDD: loaded preprocessed cache {npz} "
                f"({len(data[1])} train / {len(data[3])} test)")
    return data, spec


def _save_cache(cache_dir: str, key: str, arrays, spec: FeatureSpec) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    x_train, y_train, x_test, y_test = arrays
    np.savez_compressed(os.path.join(cache_dir, f"processed_{key}.npz"),
                        x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    spec.to_json(os.path.join(cache_dir, SCHEMA_FILENAME))


#: In-process memo, keyed by the same fingerprint as the disk cache. A run calls
#: ``load_nidd`` at least twice — once via ``get_data_loaders`` and once via
#: ``build_root_loader`` for FLTrust — and with ``data.use_cache: false`` both would
#: otherwise re-parse the full CSV. Holds one entry: the arrays are hundreds of MB,
#: and nothing needs two preprocessings live at once.
_MEMO: dict[str, tuple] = {}


def clear_memo() -> None:
    """Drop the in-process preprocessing memo (frees the arrays; used by tests that
    need to exercise the on-disk cache path rather than be served from memory)."""
    _MEMO.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_nidd(data_cfg: dict | None = None, seed: int = 0):
    """Return ``(train_dataset, test_dataset, spec)`` for 5G-NIDD.

    Reads every knob from the ``data:`` block of ``configs/base.yaml``; see that
    file for what each one does. Results are cached under ``data.cache_dir``,
    keyed by the options plus the CSV's size/mtime, so the ~1M-row parse happens
    once rather than on every run, benchmark and resume.
    """
    cfg = dict(data_cfg or {})
    cache_dir = str(cfg.get("cache_dir", "./data/5gnidd_processed"))
    source = str(cfg.get("source", "csv")).strip().lower()
    csv_path = str(cfg.get("csv_path", "./data/5gnidd_raw/Combined.csv"))
    label_mode = str(cfg.get("label_mode", "multiclass")).strip().lower()
    n_features = int(cfg.get("n_features", 32))
    max_samples = cfg.get("max_samples", 100000)
    max_samples = int(max_samples) if max_samples else None
    test_fraction = float(cfg.get("test_fraction", 0.2))
    balance = str(cfg.get("class_balance", "natural")).strip().lower()
    extra_drop = tuple(str(c).strip().lower() for c in (cfg.get("drop_columns") or ()))
    label_column = cfg.get("label_column")

    options = dict(source=source, csv_path=csv_path, label_mode=label_mode,
                   n_features=n_features, max_samples=max_samples,
                   test_fraction=test_fraction, balance=balance,
                   drop=sorted(extra_drop), label_column=label_column, seed=int(seed))
    key = _cache_key(options)

    cached = _MEMO.get(key)
    if cached is None and cfg.get("use_cache", True):
        cached = _load_cache(cache_dir, key)
    if cached is not None:
        (x_train, y_train, x_test, y_test), spec = cached
    else:
        (x_train, y_train, x_test, y_test), spec = _build(
            source=source, csv_path=csv_path, label_mode=label_mode,
            label_column=label_column, n_features=n_features,
            max_samples=max_samples, test_fraction=test_fraction,
            balance=balance, extra_drop=extra_drop, seed=int(seed),
        )
        if cfg.get("use_cache", True):
            _save_cache(cache_dir, key, (x_train, y_train, x_test, y_test), spec)
    _MEMO.clear()
    _MEMO[key] = ((x_train, y_train, x_test, y_test), spec)

    set_active(spec)
    logger.info(f"5G-NIDD ready — {spec.describe()}; "
                f"{len(y_train)} train / {len(y_test)} test flows")
    return TabularDataset(x_train, y_train), TabularDataset(x_test, y_test), spec


def _build(*, source, csv_path, label_mode, label_column, n_features,
           max_samples, test_fraction, balance, extra_drop, seed):
    """Do the actual read + fit + transform (cache miss path)."""
    rng = np.random.default_rng(seed)

    if source == "synthetic":
        logger.warning(
            "5G-NIDD: data.source=synthetic — training on GENERATED traffic, not the "
            "real dataset. Use this only to smoke-test the pipeline; no accuracy, "
            "attack-success or defense number from such a run means anything."
        )
        frame = _synthetic_frame(int(max_samples or 50000), seed)
    elif source == "csv":
        frame = _read_csv(csv_path)
    else:
        raise ValueError(f"unknown data.source {source!r} — expected 'csv' or 'synthetic'")

    frame.columns = [str(c).strip() for c in frame.columns]
    label_col = _find_label_column(frame.columns, label_mode, label_column)

    # --- labels ---
    raw_labels = frame[label_col].astype("string").fillna("Unknown").str.strip()
    if label_mode == "binary":
        benign = raw_labels.str.lower().isin(("benign", "normal", "0", "false"))
        raw_labels = np.where(benign, "Benign", "Malicious")
        class_names = ["Benign", "Malicious"]
    else:
        class_names = sorted(set(raw_labels.tolist()))
    encoding = {name: i for i, name in enumerate(class_names)}
    y_all = np.array([encoding[str(v)] for v in raw_labels], dtype=np.int64)
    n_classes = len(class_names)

    # --- drop label + identifier / leakage columns ---
    drop = set(DEFAULT_DROP_COLUMNS) | set(extra_drop)
    keep_cols = [c for c in frame.columns
                 if c != label_col and str(c).strip().lower() not in drop]
    dropped = [c for c in frame.columns if c != label_col and c not in keep_cols]
    if dropped:
        logger.info(f"5G-NIDD: dropped {len(dropped)} identifier/leakage column(s): "
                    f"{', '.join(map(str, dropped))}")
    # The *other* label column ("Label" when training multiclass, and vice versa)
    # is a restatement of the target and must go, whatever it is called.
    for other in set(_LABEL_CANDIDATES["multiclass"]) | set(_LABEL_CANDIDATES["binary"]):
        keep_cols = [c for c in keep_cols if str(c).strip().lower() != other]
    features = frame[keep_cols]

    # --- subsample, then split, THEN fit (no test-set leakage) ---
    keep = _stratified_indices(y_all, n_classes, rng, max_samples, balance)
    features, y_all = features.iloc[keep].reset_index(drop=True), y_all[keep]
    counts = {class_names[c]: int((y_all == c).sum()) for c in range(n_classes)}
    logger.info(f"5G-NIDD: class counts after subsampling ({balance}): {counts}")
    # A class can be wiped out by an aggressive max_samples; re-encode so the
    # label space stays contiguous (a gap would make the model allocate an output
    # unit no example ever uses, and shift every class index after the gap).
    present = sorted({int(c) for c in np.unique(y_all)})
    if len(present) < n_classes:
        logger.warning(f"5G-NIDD: {n_classes - len(present)} class(es) have no rows "
                       f"after subsampling — re-encoding to {len(present)} classes")
        remap = {old: new for new, old in enumerate(present)}
        y_all = np.array([remap[int(v)] for v in y_all], dtype=np.int64)
        class_names = [class_names[old] for old in present]
        n_classes = len(class_names)

    tr_idx, te_idx = _stratified_split(y_all, n_classes, test_fraction, rng)
    train_frame, y_train = features.iloc[tr_idx], y_all[tr_idx]
    test_frame, y_test = features.iloc[te_idx], y_all[te_idx]

    pre = _Preprocessor(n_features)
    pre.fit(train_frame, y_train, n_classes)          # TRAIN ONLY
    x_train = pre.transform(train_frame)
    x_test = pre.transform(test_frame)

    spec = FeatureSpec(
        input_dim=int(x_train.shape[1]), n_classes=n_classes,
        class_names=list(class_names), feature_names=list(pre.feature_names),
        source=source, dataset="5gnidd",
    )
    return (x_train, y_train, x_test, y_test), spec


def build_root_loader(root_size: int = 100, batch_size: int = 64,
                      data_cfg: dict | None = None, seed: int = 0):
    """A small CLEAN root dataset held by the server, for FLTrust.

    FLTrust (Cao et al., NDSS 2021) bootstraps trust from a handful of clean
    examples the server collects itself; we sample ``root_size`` of them from the
    5G-NIDD training split with a fixed seed so runs are reproducible. Used by
    both the benchmark panel and the arms race's algorithmic defender.

    The sample is drawn uniformly, as in the paper — NOT class-balanced. On a
    dataset this imbalanced that means the root set is mostly benign and
    UDPFlood traffic, which is a real property of FLTrust's threat model (the
    server's clean data is just clean, not curated) and affects the trust
    direction g0 it derives.
    """
    train_dataset, _, _ = load_nidd(data_cfg, seed=seed)
    g = torch.Generator().manual_seed(int(seed))
    n = len(train_dataset)
    size = max(1, min(int(root_size), n))
    idx = torch.randperm(n, generator=g)[:size].tolist()
    # The same seeded generator drives the shuffle, so FLTrust's root fine-tuning
    # (and therefore its trusted reference direction g0) is reproducible across runs
    # instead of drawing from the ambient global RNG. Within a round g0 is computed
    # once and cached — see benchmark.defenses.fltrust.FLTrust._root_update.
    return DataLoader(Subset(train_dataset, idx),
                      batch_size=max(1, min(int(batch_size), size)),
                      shuffle=True, generator=g)


def partition_iid(dataset, n_clients: int):
    """Split dataset into n_clients equal IID shards."""
    indices = torch.randperm(len(dataset)).tolist()
    shard_size = len(dataset) // n_clients
    return [indices[i * shard_size: (i + 1) * shard_size] for i in range(n_clients)]


def _dataset_targets(dataset) -> list:
    """Integer labels for every sample, robust across dataset implementations."""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:  # exotic dataset: fall back to (slow) per-item access
        return [int(dataset[i][1]) for i in range(len(dataset))]
    if isinstance(targets, torch.Tensor):
        return targets.tolist()
    return [int(t) for t in targets]


def partition_noniid_fltrust(dataset, n_clients: int, n_classes: int = 9,
                             bias_q: float = 0.5, seed: int = 0):
    """Non-IID partition following FLTrust (Cao et al., NDSS 2021).

    Clients are split into ``M = n_classes`` groups. A training example with
    label ``l`` is assigned to group ``l`` with probability ``bias_q`` and to any
    OTHER group with probability ``(1 - bias_q) / (M - 1)``. Within a group the
    assigned examples are split evenly across that group's clients.

    ``bias_q = 1/M`` reproduces an IID split; a larger ``bias_q`` gives a stronger
    non-IID skew (each group is dominated by its own class). The paper's default
    is ``0.5``. For 5G-NIDD's nine classes with 20 clients this yields 9 groups of
    2-3 clients each.

    Note that on 5G-NIDD the groups are NOT equal in size the way they were on
    MNIST, whose ten classes are near-uniform. Here the classes are severely
    imbalanced (Benign ~39% of flows, UDPFlood ~38%, ICMPFlood ~0.09%), so the
    groups that attract the two dominant classes hold far more examples. At the
    shipped settings that is a measured 3.6x spread across 20 clients (2510 /
    2946 / 9078 examples at min / median / max) — much less than the 400x the raw
    class ratio suggests, because at ``bias_q=0.5`` half of every class is spread
    uniformly over the other groups, which floors the rare-class groups. Raising
    ``bias_q`` toward 1.0 removes that floor and the spread grows sharply.

    This is why ``RoundDataSampler`` computes its per-round slice size per client
    rather than once for the federation.

    Returns a list of ``n_clients`` index lists (shards), ordered by client id;
    the shards are disjoint and cover every sample.
    """
    rng = random.Random(seed)
    M = max(1, int(n_classes))

    # --- Split clients into M groups, as evenly as possible (20/9 -> 2-3 each). ---
    # Round-robin keeps groups balanced; earlier groups get the extra client when
    # n_clients is not divisible by M.
    groups: list[list[int]] = [[] for _ in range(M)]
    for cid in range(n_clients):
        groups[cid % M].append(cid)
    nonempty = [g for g in range(M) if groups[g]]  # guards n_clients < M

    # --- Route each sample to a group by the bias rule ---
    targets = _dataset_targets(dataset)
    group_indices: list[list[int]] = [[] for _ in range(M)]
    for idx, lbl in enumerate(targets):
        l = int(lbl) % M
        if rng.random() < bias_q:
            g = l
        else:  # pick uniformly among the M-1 groups other than l
            g = rng.randrange(M - 1) if M > 1 else 0
            if g >= l:
                g += 1
        if not groups[g]:  # chosen group has no clients (only when n_clients < M)
            g = l if groups[l] else rng.choice(nonempty)
        group_indices[g].append(idx)

    # --- Within each group, split its indices evenly across the group's clients ---
    shards: list[list[int]] = [[] for _ in range(n_clients)]
    for g in range(M):
        members = groups[g]
        if not members:
            continue
        idxs = group_indices[g]
        rng.shuffle(idxs)
        k = len(members)
        base, rem = divmod(len(idxs), k)
        start = 0
        for j, cid in enumerate(members):
            count = base + (1 if j < rem else 0)
            shards[cid] = idxs[start:start + count]
            start += count

    return shards


def get_data_loaders(n_clients: int, batch_size: int, data_cfg: dict | None = None,
                     iid: bool = True, bias_q: float = 0.5, seed: int = 0):
    """Return per-client train loaders and a global test loader.

    Args:
        n_clients: Number of federated clients.
        batch_size: Training batch size.
        data_cfg: The ``data:`` block of the config (paths, feature count,
            subsampling, label mode). See ``configs/base.yaml``.
        iid: If True, use IID partitioning. If False, use the FLTrust non-IID
             partition (``partition_noniid_fltrust``).
        bias_q: FLTrust bias probability (only used when ``iid=False``). ``1/M``
             is IID; larger is more non-IID; paper default ``0.5``.
        seed: RNG seed for the partition and the preprocessing split.

    The class count driving the non-IID grouping comes from the DATA, not from a
    constant: a binary run (``data.label_mode: binary``) forms 2 groups, a
    multiclass run 9.
    """
    train_dataset, test_dataset, spec = load_nidd(data_cfg, seed=seed)

    if iid:
        shards = partition_iid(train_dataset, n_clients)
    else:
        shards = partition_noniid_fltrust(
            train_dataset, n_clients, n_classes=spec.n_classes,
            bias_q=bias_q, seed=seed,
        )

    sizes = [len(s) for s in shards]
    logger.info(f"5G-NIDD: {n_clients} client shard(s), "
                f"{'IID' if iid else f'non-IID q={bias_q}'} — "
                f"min {min(sizes) if sizes else 0} / median "
                f"{int(np.median(sizes)) if sizes else 0} / max "
                f"{max(sizes) if sizes else 0} examples")

    client_loaders = [
        DataLoader(Subset(train_dataset, shard), batch_size=batch_size, shuffle=True)
        for shard in shards
    ]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return client_loaders, test_loader
