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

from agents.json_utils import extract_json
from core.types import DetectionVerdict

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the defender in a federated-learning system. Each round every client submits a model update; some clients are malicious (model poisoning) and the rest honest, and you do NOT know which. Given statistical features of each client's weight delta (client - global), decide for EACH client whether it is malicious.

Honest clients are the MAJORITY, so median-based references describe honest behaviour and outliers are suspicious.

Features per client:
- `layers` -- per-layer stats:
  * l2_norm: this layer's update magnitude. rel_norm: l2_norm / median over clients (>> 1 = abnormally large).
  * cos_to_median: cosine to the coordinate-wise median update across clients (near 1 = normal; low/negative = anomalous direction, e.g. sign-flip).
  * sign_agreement: fraction of coordinates whose sign matches the median (honest ~> 0.6; flipped drops below 0.5).
- `whole` -- model-wide: l2_norm, rel_norm; cos_to_mean; max_pairwise_cos (colluding Sybils look alike -> high); dnc_score (spectral outlier, higher = stronger).

Flag a client malicious when SEVERAL signals agree: high rel_norm, low cos_to_median/cos_to_mean, low sign_agreement, or high dnc_score. False positives are costly -- require corroborating evidence.

OUTPUT FORMAT -- respond with ONLY a single JSON object, no prose, no markdown:
{"clients": [
   %CLIENT_SCHEMA%,
   ...
]}
Include EXACTLY one entry for every client_id you were given.%REASON_HINT%"""

# Per-client output schema, with and without the free-text "reason". The reason
# is a short natural-language explanation that costs generation tokens (one string
# per client — 20 clients = 20 strings per verdict, per rollout). It is purely
# informational (logged only; never used by the reward or metrics), so it can be
# turned off to save tokens via ``emit_reason`` in configs/defender_agent.yaml.
_SCHEMA_WITH_REASON = ('{"client_id": <int>, "is_suspicious": <true|false>, '
                       '"confidence": <float 0..1>, "reason": "<short>"}')
_SCHEMA_NO_REASON = ('{"client_id": <int>, "is_suspicious": <true|false>, '
                     '"confidence": <float 0..1>}')
_REASON_HINT_OFF = ('\nOutput ONLY those fields per client — do NOT add a '
                    '"reason" or any other keys.')


class DefenderAgent:
    """Pure defender policy: builds the prompt, parses per-client verdicts."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        # Confidence to assume when the model omits/garbles a client entry.
        self.default_confidence = float(config.get("default_confidence", 0.0))
        # Whether to ask the LLM for a short per-client "reason". Off by default to
        # save generation tokens; flip ``emit_reason: true`` in the config to restore
        # the explanations. The prompt is built once here so it stays consistent
        # across generation and (during training) the log-prob passes.
        self.emit_reason = bool(config.get("emit_reason", False))
        schema = _SCHEMA_WITH_REASON if self.emit_reason else _SCHEMA_NO_REASON
        hint = "" if self.emit_reason else _REASON_HINT_OFF
        self._system = (
            SYSTEM_PROMPT.replace("%CLIENT_SCHEMA%", schema).replace("%REASON_HINT%", hint)
        )

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        return self._system

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
                # When reasons are disabled, drop any stray field the model emits so
                # the verdict stays empty-reason regardless of model behaviour.
                reason=(str(e.get("reason", ""))[:200] if self.emit_reason else ""),
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
