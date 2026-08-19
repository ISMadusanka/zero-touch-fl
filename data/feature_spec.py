"""The contract between 5G-NIDD preprocessing and the FL model.

MNIST needed no such contract: every component could hardcode "1x28x28 in, 10
classes out" because those numbers are properties of the *dataset*. 5G-NIDD's
are properties of the *preprocessing* — how many flow features survived the drop
list, how many were kept by the feature selector, how many attack classes the CSV
actually contained — so they are only known after ``data.nidd_loader`` has run.

That is a problem, because the components that build the model do not load the
data:

    server.FedServer(device=...)          rl/env.py, benchmark/{phase1,harness}.py,
                                          benchmark/defenses/fltrust.py
    model.build_model()                   tests

This module is how they agree. ``data.nidd_loader.get_data_loaders`` fits the
preprocessing, writes the resulting :class:`FeatureSpec` to ``schema.json`` beside
the processed-data cache, and installs it as the *active* spec; everything that
builds a model calls :func:`active` and gets the same shape. A run that never
loads data (unit tests) falls back to :data:`DEFAULT_SPEC`, so ``NiddNet()`` and
``FedServer()`` still work offline.

**This module deliberately imports nothing heavy** (no torch, no pandas), so it
can be imported from ``model/`` without dragging the data stack in.

The spec is also what makes a stale checkpoint detectable: Phase-1 weights are
only loadable by a model of the same shape, so ``storage.checkpoint`` compares
the saved ``input_dim``/``n_classes`` against the active spec instead of letting
``load_state_dict`` raise deep inside a resume. See :func:`FeatureSpec.matches`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

#: Filename written into the processed-data cache directory.
SCHEMA_FILENAME = "schema.json"


@dataclass(frozen=True)
class FeatureSpec:
    """Shape of one preprocessed 5G-NIDD example, and what the classes mean.

    ``input_dim`` is the width of the feature vector a client trains on — the
    number of columns that survived the drop list and the top-K selector, NOT the
    number of columns in the CSV. ``n_classes`` is however many distinct labels
    the ingested rows actually carried, which can be fewer than the dataset's
    nine if the run subsampled hard or was pointed at a per-attack CSV.
    """

    input_dim: int
    n_classes: int
    #: Class index -> label string, e.g. ``["Benign", "HTTPFlood", ...]``. Index
    #: order is the label encoding the loader applied (sorted, so it is stable
    #: across runs and machines).
    class_names: list[str] = field(default_factory=list)
    #: The selected feature columns, in model-input order. Kept so a run can be
    #: audited ("which 32 flow features is this model actually using?") and so a
    #: cache built under a different drop list is recognisably different.
    feature_names: list[str] = field(default_factory=list)
    #: Where the rows came from: ``"kaggle"`` (downloaded mirror, the default) or
    #: ``"csv"`` (a local file) for real 5G-NIDD, ``"synthetic"`` for the generated
    #: stand-in. Carried through to logs and checkpoints so a synthetic run is
    #: never mistaken for a real one, and so a checkpoint records which of the two
    #: real routes produced it.
    source: str = "csv"
    dataset: str = "5gnidd"

    # ------------------------------------------------------------------
    def matches(self, other: "FeatureSpec | None") -> bool:
        """True when a model built for ``other`` can load weights built for us.

        Only the tensor shapes matter — feature *names* can differ (a re-fit that
        selected a different but equally-sized feature set still produces loadable
        weights, even though the model then means something different). Names are
        compared by the caller when it wants to warn about that.
        """
        return (other is not None
                and int(other.input_dim) == int(self.input_dim)
                and int(other.n_classes) == int(self.n_classes))

    def describe(self) -> str:
        """One-line summary for logs."""
        # Only NON-real sources get a tag. Marking the two real routes as well
        # would put brackets on every ordinary run and train the eye to skip
        # them, which is precisely the signal `[SYNTHETIC]` needs to keep.
        tag = "" if self.source in ("csv", "kaggle") else f" [{self.source.upper()}]"
        return (f"{self.dataset}{tag}: {self.input_dim} flow features -> "
                f"{self.n_classes} classes ({', '.join(self.class_names) or 'unnamed'})")

    # ------------------------------------------------------------------
    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @staticmethod
    def from_json(path: str) -> "FeatureSpec | None":
        """Load a spec, or ``None`` if it is missing or unreadable.

        Unreadable is treated the same as missing on purpose: a truncated
        ``schema.json`` (killed mid-write) should send the caller back to the
        default rather than abort a run with a JSON error.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (FileNotFoundError, NotADirectoryError):
            return None
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"ignoring unreadable feature schema at {path}: {e}")
            return None
        known = {f for f in FeatureSpec.__dataclass_fields__}
        return FeatureSpec(**{k: v for k, v in raw.items() if k in known})


#: Used when nothing has loaded data yet — unit tests, ``--help``, and any
#: ``NiddNet()`` / ``FedServer()`` built before ``get_data_loaders`` runs.
#:
#: The numbers are the shipped ``configs/base.yaml`` defaults (``data.n_features:
#: 32``) against 5G-NIDD's nine classes, so an offline-constructed model has the
#: same shape a real run produces and test fixtures stay meaningful. It is a
#: FALLBACK, never an override: a loaded schema always wins.
DEFAULT_SPEC = FeatureSpec(
    input_dim=32,
    n_classes=9,
    class_names=["Benign", "HTTPFlood", "ICMPFlood", "SYNFlood", "SYNScan",
                 "SlowrateDoS", "TCPConnectScan", "UDPFlood", "UDPScan"],
    feature_names=[],
    source="default",
)

_active: FeatureSpec | None = None


def set_active(spec: FeatureSpec) -> FeatureSpec:
    """Install ``spec`` as the shape every subsequently-built model uses.

    Called by ``data.nidd_loader.get_data_loaders``. Re-installing a *different*
    shape mid-process is logged loudly: models built before the change do not
    match ones built after, which in practice means a stale checkpoint or two
    datasets in one process.
    """
    global _active
    if _active is not None and not _active.matches(spec):
        logger.warning(
            f"active feature spec CHANGED: {_active.describe()} -> {spec.describe()}; "
            f"models built earlier in this process have the old shape"
        )
    _active = spec
    return spec


def active(cache_dir: str | None = None) -> FeatureSpec:
    """The spec models should be built against.

    Resolution order: whatever :func:`set_active` installed, else ``schema.json``
    under ``cache_dir`` (so a benchmark process that builds a model before
    touching the data still agrees with the run that wrote the cache), else
    :data:`DEFAULT_SPEC`.
    """
    if _active is not None:
        return _active
    if cache_dir:
        spec = FeatureSpec.from_json(os.path.join(cache_dir, SCHEMA_FILENAME))
        if spec is not None:
            return spec
    return DEFAULT_SPEC


def reset_active() -> None:
    """Forget the installed spec (tests; lets each case start from the default)."""
    global _active
    _active = None
