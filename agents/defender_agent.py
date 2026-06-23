"""Defender LLM agent — classifies each client benign/malicious directly.

Like the attacker, this is a **pure prompt-builder + output-parser** (no LLM
call inside). It receives the per-client, per-layer statistical feature vectors
produced by `detector/features.py` and outputs one classification per client.

Crucially, the defender's *prompt* contains ONLY the feature vectors — never
the ground truth. The ground-truth poisoned set is used solely to compute the
verifiable reward that trains this policy (train-time signal, not an input),
preserving the production-realistic, oracle-free observation.
"""

import json
import logging

from agents.weight_codec import extract_json
from core.types import DetectionVerdict

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the defender in a federated-learning system trained on MNIST.

Every round each client submits a model update. A subset of clients are
malicious (model poisoning) and the rest are honest. You do NOT know which.
You are given statistical features of every client's weight delta (client minus
global), and you must decide, for EACH client, whether it is malicious.

Assume the HONEST clients are the MAJORITY, so the robust reference statistics
(medians) describe honest behaviour and outliers are suspicious.

Per-client features:
- `layers`: stats per logical layer (e.g. "net.2" = hidden, "net.4" = output):
   * l2_norm        : magnitude of this layer's update.
   * rel_norm       : l2_norm / median over clients. >> 1 means abnormally large.
   * cos_to_median  : cosine similarity to the median update of the others.
                      Near 1 = aligned with the crowd; low or negative = anomalous
                      direction (e.g. sign-flipping).
   * sign_agreement : fraction of coordinates whose sign matches the median sign.
                      Honest updates mostly agree (~>0.6); flipped/poisoned ones
                      drop well below 0.5.
- `whole`: model-wide stats:
   * l2_norm / rel_norm : as above, across all parameters.
   * cos_to_mean        : cosine to the mean update.
   * max_pairwise_cos   : highest similarity to any other single client (colluding
                          Sybils look unusually similar to each other).
   * dnc_score          : spectral outlier score; larger = stronger outlier.

A client is likely malicious when several signals agree: high rel_norm, low
cos_to_median / cos_to_mean, low sign_agreement, or high dnc_score. Flagging an
honest client (false positive) is costly, so require corroborating evidence.

OUTPUT FORMAT — respond with ONLY a single JSON object, no prose, no markdown:
{"clients": [
   {"client_id": <int>, "is_suspicious": <true|false>,
    "confidence": <float 0..1>, "reason": "<short>"},
   ...
]}
Include EXACTLY one entry for every client_id you were given."""


class DefenderAgent:
    """Pure defender policy: builds the prompt, parses per-client verdicts."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        # Confidence to assume when the model omits/garbles a client entry.
        self.default_confidence = float(config.get("default_confidence", 0.0))

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_prompt(self, features: dict[int, dict]) -> str:
        """Serialize per-client feature vectors into a user message.

        ``features`` is keyed by client_id (from ``compute_client_features``).
        """
        payload = {
            "client_ids": list(features.keys()),
            "features": {str(cid): feats for cid, feats in features.items()},
        }
        return json.dumps(payload)

    def parse(self, text, client_ids: list[int]) -> list[DetectionVerdict]:
        """Parse LLM output into one DetectionVerdict per requested client.

        Robust to missing/garbled entries: any client the model failed to label
        defaults to benign with ``default_confidence`` and reason "unparsed".
        Ordering follows ``client_ids``.
        """
        raw = extract_json(text)
        by_id: dict[int, dict] = {}

        entries = []
        if isinstance(raw, dict):
            if isinstance(raw.get("clients"), list):
                entries = raw["clients"]
            else:
                # Accept a plain {client_id: {...}} mapping too.
                for k, v in raw.items():
                    if isinstance(v, dict):
                        v = dict(v)
                        v.setdefault("client_id", k)
                        entries.append(v)
        elif isinstance(raw, list):
            entries = raw

        for e in entries:
            if not isinstance(e, dict):
                continue
            cid = _coerce_int(e.get("client_id"))
            if cid is None:
                continue
            by_id[cid] = e

        verdicts: list[DetectionVerdict] = []
        for cid in client_ids:
            e = by_id.get(cid)
            if e is None:
                verdicts.append(DetectionVerdict(cid, False, self.default_confidence, "unparsed"))
                continue
            verdicts.append(DetectionVerdict(
                client_id=cid,
                is_suspicious=bool(e.get("is_suspicious", False)),
                confidence=_coerce_float(e.get("confidence"), self.default_confidence),
                reason=str(e.get("reason", ""))[:200],
            ))
        return verdicts


def _coerce_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _coerce_float(x, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):  # NaN / Inf guard
        return default
    return max(0.0, min(1.0, v))
