"""Defender LLM agent — classifies each client benign/malicious directly.

Like the attacker, this is a **pure prompt-builder + output-parser** (no LLM
call inside). It receives the per-client, per-layer statistical feature vectors
produced by `detector/features.py` and outputs one classification per client.

Crucially, the defender's *prompt* contains ONLY the feature vectors — never
the ground truth. The ground-truth poisoned set is used solely to compute the
verifiable reward that trains this policy (train-time signal, not an input),
preserving the production-realistic, oracle-free observation.

CONTEXT BUDGET — the defender gets **one LLM call per round** and must label
EVERY client in it, so the whole verdict list is produced from a single pass over
a prompt that grows with the federation (20 clients x per-layer statistics). Two
things follow, and both are enforced here rather than left to chance:

* **The observation is encoded positionally** (``layer_key`` / ``whole_key``
  legends sent once, arrays of bare numbers per client), the same trick the
  attacker uses. The old named-key JSON repeated ~60 characters of statistic
  names per layer per client — at 20 clients that was the largest single term in
  the prompt, spent entirely on text the model already knows from the legend.
* **The prompt is held inside a band** (``rl.defender_min_prompt_fill`` ..
  ``rl.defender_max_prompt_fill``, 20-30% of the window) with the whole call —
  prompt plus reserved completion — under ``rl.defender_max_context_fill``
  (60%). A 3B instruct model's output quality falls off well before its context
  is full: instructions at the top start competing with data in the middle, and
  a defender that loses the output schema emits an unparseable verdict list,
  which scores as "flagged nothing". Above the band the payload is COMPACTED
  (see :data:`_COMPACTION`), never truncated and never with a client dropped.

The room the compact encoding frees is deliberately spent back on content that
makes a single call more decidable rather than banked:

* a ``cohort`` block giving each whole-model statistic's across-client median and
  MAD, so "is this client unusual?" is a subtraction instead of an eyeball over
  20 rows. ``rel_norm`` arrives cohort-relative already, but ``cos_to_mean``,
  ``dnc_score`` and ``max_pairwise_cos`` do not — without a reference scale the
  model has to invent one.
* a ``ranked`` block pre-sorting client ids by the statistics that carry the most
  signal, worst-looking first. This is presentation only — the same numbers are
  in ``clients`` — but sorting 20 rows in-context is exactly the kind of work a
  small model does unreliably, and getting it wrong costs a whole round's verdict.
"""

import json
import logging

from agents.attack_ops import extract_json, sig_round
from agents.prompt_budget import build_prompt_budget
from core.types import DetectionVerdict

logger = logging.getLogger(__name__)


#: Per-layer statistics, in the order the compact row encoding emits them.
#: Ordered most- to least-decisive, because the lossy compaction levels drop
#: from the END of this tuple.
LAYER_STAT_KEYS = ("rel_norm", "cos_to_median", "sign_agreement", "l2_norm")

#: Whole-model statistics, same convention.
WHOLE_STAT_KEYS = ("rel_norm", "cos_to_mean", "max_pairwise_cos", "dnc_score",
                   "l2_norm")

#: The per-layer statistics kept once the observation has to be compacted.
#: ``l2_norm`` is the absolute magnitude that ``rel_norm`` already expresses
#: relative to the cohort, so it is the first thing worth losing.
CORE_LAYER_STAT_KEYS = ("rel_norm", "cos_to_median", "sign_agreement")

#: Whole-model statistics the ``ranked`` block pre-sorts on, each with the
#: direction that looks suspicious. Kept short on purpose: one ordering per
#: statistic costs one integer per client, and four orderings is enough to make
#: every family of attack (oversized, misdirected, sign-flipped, cloned) surface
#: at the top of some list.
_RANK_KEYS = (
    ("rel_norm", True),           # descending: abnormally large updates first
    ("cos_to_mean", False),       # ascending: most misaligned directions first
    ("dnc_score", True),          # descending: strongest spectral outliers first
    ("max_pairwise_cos", True),   # descending: likeliest colluding clones first
)


SYSTEM_PROMPT = """You are the defender in a federated-learning system. Each round every client submits a model update; some clients are malicious (model poisoning) and the rest honest, and you do NOT know which. Given statistical features of each client's weight delta (client - global), decide for EACH client whether it is malicious.

Honest clients are the MAJORITY, so median-based references describe honest behaviour and outliers are suspicious.

READING THE OBSERVATION -- it is packed positionally to stay short:
- `layer_key` names the statistics inside every per-layer array, in that order; `whole_key` names them inside every "whole" array.
- `layers` lists the model's layer names once (they are the same for every client).
- `clients` maps each client_id to {<layer name>: [...], "whole": [...]}.
- `cohort` has the SAME keys as one client's row, but each maps to [medians, mads]: the across-client median and median absolute deviation of the statistic at that position. It tells you what "honest" looks like this round. Judge a client by how far it sits outside its own reference: deviation = |value - median| / (mad + 1e-6), read position by position against the same legend. Around 3 or more is a strong outlier.
- `ranked` pre-sorts client_ids by the most informative whole-model statistics, most suspicious first. It is a shortlist, not a verdict -- the top of a list is where to look, and `cohort` is what decides.

WHAT THE STATISTICS MEAN
- rel_norm: this update's L2 norm divided by the cohort median norm. Much greater than 1 = abnormally large.
- cos_to_median (per layer) / cos_to_mean (whole model): alignment with the cohort's central update. Near 1 = normal; low or negative = anomalous direction, e.g. a sign flip.
- sign_agreement: fraction of coordinates whose sign matches the cohort median sign. Honest is typically above 0.6; a flipped update drops below 0.5.
- max_pairwise_cos: highest cosine to any OTHER client. Colluding Sybils submit near-identical updates, so unusually high values implicate a GROUP -- flag them together or not at all.
- dnc_score: spectral outlier score (squared projection on the top singular direction of the centred updates). Higher = stronger outlier.
- l2_norm: raw magnitude, for scale.

HOW TO DECIDE
1. Read `cohort` first to establish the honest scale for this round.
2. Walk `ranked` from the top and check each candidate against `cohort` and its per-layer numbers.
3. Flag a client when SEVERAL INDEPENDENT signals agree: high rel_norm, low cos_to_median/cos_to_mean, low sign_agreement, high dnc_score, or a high max_pairwise_cos shared with other suspects. One mildly-off statistic is not enough.
4. False positives are costly -- a flagged honest client is dropped from the average and its data is lost -- so require corroborating evidence. Equally, do not flag zero clients merely because no single signal is extreme; poisoning that is spread across many clients shows up as a consistent cluster rather than one spike.
5. Set `confidence` to how strongly the evidence agrees: ~0.5 borderline, ~1.0 several signals far outside the cohort.

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


#: The observation's compaction ladder, tried in order until the prompt fits the
#: context budget. Level 0 is the richest rendering; every later level trades
#: detail for room, least-informative first. Dropping a CLIENT is deliberately
#: not an option — the defender must return a verdict for every client, so an
#: observation missing one guarantees a wrong (default-benign) label.
_COMPACTION = (
    dict(sig=None, layer_keys=LAYER_STAT_KEYS, include_layers=True, ranked=True,
         label="full per-layer stats + rankings"),
    dict(sig=3, layer_keys=LAYER_STAT_KEYS, include_layers=True, ranked=True,
         label="3 significant figures"),
    dict(sig=3, layer_keys=CORE_LAYER_STAT_KEYS, include_layers=True, ranked=True,
         label="core per-layer stats"),
    dict(sig=3, layer_keys=CORE_LAYER_STAT_KEYS, include_layers=True, ranked=False,
         label="core per-layer stats, no rankings"),
    dict(sig=3, layer_keys=(), include_layers=False, ranked=False,
         label="whole-model stats only"),
)


def _median(values) -> float:
    """Median of a sequence (0.0 when empty).

    Hand-rolled rather than ``numpy``/``statistics`` so this module stays a pure
    prompt builder: the cohort block is computed from the already-extracted
    feature numbers, and the agent remains importable and testable without the
    numeric stack the feature extractor needs.
    """
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if not n:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def _mad(values) -> float:
    """Median absolute deviation — the robust spread the cohort block reports."""
    xs = [float(v) for v in values]
    if not xs:
        return 0.0
    med = _median(xs)
    return _median([abs(x - med) for x in xs])


class DefenderAgent:
    """Pure defender policy: builds the prompt, parses per-client verdicts."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        # Confidence to assume when the model omits/garbles a client entry.
        self.default_confidence = float(config.get("default_confidence", 0.0))
        # Significant figures at the richest compaction level.
        self.detail_precision = int(config.get("detail_precision", 4))
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
        # How much of the LLM's context one defender call may fill. Built from the
        # ``rl:`` block main.py / run_benchmark.py injects, reading the
        # ``defender_*`` overrides first; absent an ``rl`` block the budget is
        # inert and only reports.
        self.budget = build_prompt_budget(config.get("rl"), label="defender prompt",
                                          prefix="defender_")
        #: Fill report for the most recent ``build_user_prompt`` (for logging/tests).
        self.last_prompt_stats: dict = {}
        self._logged_level: int | None = None

    # ------------------------------------------------------------------
    def bind_tokenizer(self, token_counter) -> None:
        """Use the real tokenizer for the context-fill check.

        ``token_counter(system, user) -> int`` — pass
        ``LLMPolicy.count_prompt_tokens`` so the count reflects the same
        chat-template rendering generation uses. Without this the budget falls
        back to a character heuristic (see :mod:`agents.prompt_budget`).
        """
        self.budget.bind(token_counter)

    def system_prompt(self) -> str:
        return self._system

    def build_user_prompt(self, features: dict[int, dict]) -> str:
        """Serialize per-client feature vectors into a user message.

        ``features`` is keyed by client_id (from ``compute_client_features``).

        The observation is emitted at the first compaction level (see
        :data:`_COMPACTION`) whose prompt fits ``self.budget``; the payload is
        made coarser rather than truncated, and no client is ever dropped from
        it. When the richest level lands *below* the budget's target band the
        fill is reported as under-filled — the observation could afford to say
        more — but it is still emitted unchanged.
        """
        if not features:
            self._report(0, self.budget.count(self._system, "{}"), 0)
            return json.dumps({"n_clients": 0, "client_ids": [], "clients": {}},
                              separators=(",", ":"))

        text = ""
        level = 0
        n_tokens = 0
        for level, step in enumerate(_COMPACTION):
            text = self._render(features, step)
            n_tokens = self.budget.count(self._system, text)
            if self.budget.fits(n_tokens) or level == len(_COMPACTION) - 1:
                break

        self._report(level, n_tokens, len(features))
        return text

    # ------------------------------------------------------------------
    def _render(self, features: dict[int, dict], step: dict) -> str:
        """One compaction level's user message."""
        sig = self.detail_precision if step["sig"] is None else step["sig"]
        layer_keys = tuple(step["layer_keys"])
        include_layers = bool(step["include_layers"]) and bool(layer_keys)
        client_ids = list(features.keys())

        payload: dict = {
            "n_clients": len(client_ids),
            "client_ids": client_ids,
        }
        if include_layers:
            # Layer names are identical for every client (FL shares one
            # architecture), so they are sent once here instead of per client.
            first = features[client_ids[0]].get("layers", {})
            payload["layers"] = list(first.keys())
            payload["layer_key"] = list(layer_keys)
        payload["whole_key"] = list(WHOLE_STAT_KEYS)
        payload["cohort"] = self._cohort(features, sig, layer_keys, include_layers)
        if step["ranked"]:
            payload["ranked"] = self._ranked(features)
        payload["clients"] = {
            str(cid): _rows(feats, sig, layer_keys, include_layers)
            for cid, feats in features.items()
        }
        # Compact separators: at 20 clients the default ", " / ": " padding is a
        # few hundred tokens of pure whitespace.
        return json.dumps(payload, separators=(",", ":"))

    def _cohort(self, features: dict[int, dict], sig: int,
                layer_keys: tuple, include_layers: bool) -> dict:
        """Across-client median + MAD for every statistic, per layer and whole-model.

        This is the reference scale the system prompt tells the model to judge
        against. Several statistics are ABSOLUTE — ``cos_to_mean``,
        ``max_pairwise_cos``, ``dnc_score``, every per-layer ``cos_to_median``
        and ``sign_agreement`` — so without a cohort reference the model has to
        infer "typical" from the rows themselves. That is the step a small model
        most often gets wrong, and it decides every verdict in the round.

        Shaped to MIRROR a client row (:func:`_rows`): the same ``<layer>`` /
        ``"whole"`` keys, each mapping to ``[medians, mads]`` under the same
        ``layer_key`` / ``whole_key`` legend. One layout to explain, and a
        client's numbers line up positionally with the reference they are read
        against.
        """
        clients = list(features.values())

        def rows(pick, keys):
            cols = [[float(pick(f).get(k, 0.0)) for f in clients] for k in keys]
            return [[sig_round(_median(c), sig) for c in cols],
                    [sig_round(_mad(c), sig) for c in cols]]

        cohort = {"whole": rows(lambda f: f.get("whole", {}), WHOLE_STAT_KEYS)}
        if include_layers:
            for layer in features[list(features)[0]].get("layers", {}):
                cohort[layer] = rows(
                    lambda f, ly=layer: f.get("layers", {}).get(ly, {}), layer_keys)
        return cohort

    def _ranked(self, features: dict[int, dict]) -> dict:
        """Client ids pre-sorted by each ranking statistic, most suspicious first.

        Presentation only — every number is already in ``clients`` — but it
        removes the one piece of work a small model does least reliably in a
        single pass: ordering 20 rows by a column it has to locate positionally.
        """
        out: dict[str, list[int]] = {}
        for key, descending in _RANK_KEYS:
            out[key] = sorted(
                features,
                key=lambda cid, k=key: float(features[cid].get("whole", {}).get(k, 0.0)),
                reverse=descending,
            )
        return out

    def _report(self, level: int, n_tokens: int, n_clients: int) -> None:
        """Record the fill, and log it when the compaction level changes."""
        step = _COMPACTION[level]
        fits = self.budget.fits(n_tokens)
        underfilled = self.budget.underfilled(n_tokens)
        self.last_prompt_stats = {
            "prompt_tokens": int(n_tokens),
            "fill": round(self.budget.fill(n_tokens), 4),
            "prompt_fill": round(self.budget.prompt_fill(n_tokens), 4),
            "level": level,
            "level_label": step["label"],
            "n_clients": int(n_clients),
            "fits": bool(fits),
            "underfilled": bool(underfilled),
            "exact": self.budget.exact(),
        }
        if level == self._logged_level:
            logger.debug(self.budget.describe(n_tokens))
            return
        self._logged_level = level
        detail = (f"{self.budget.describe(n_tokens)} — {n_clients} client(s), "
                  f"observation at level {level} ({step['label']})")
        if not fits:
            # Out of compaction levels: the prompt is over the cap but complete.
            # Nothing is truncated, so the round continues — but the whole point
            # of the cap is that output quality degrades before this point, and
            # a defender that garbles its verdict list flags nobody.
            logger.warning(
                f"{detail}. Every compaction level still exceeds the "
                f"{self.budget.max_prompt_fill or self.budget.max_fill:.0%} prompt cap "
                f"({self.budget.prompt_limit} prompt tokens). Raise rl.max_seq_len, "
                f"lower rl.defender_max_new_tokens, or raise "
                f"rl.defender_max_prompt_fill."
            )
        elif underfilled:
            # Not a problem to fix at runtime, but worth surfacing: the richest
            # rung of the ladder is cheaper than the band the config asked for,
            # so there is context available for a richer observation.
            logger.info(
                f"{detail}. Below the {self.budget.min_prompt_fill:.0%} target floor "
                f"({self.budget.prompt_floor} tokens) — there is room for a richer "
                f"observation at this federation size."
            )
        else:
            logger.info(detail)

    # ------------------------------------------------------------------
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


def _rows(feats: dict, sig: int, layer_keys: tuple, include_layers: bool) -> dict:
    """One client's features as compact positional rows.

    ``{<layer>: [<layer_key order>], "whole": [<whole_key order>]}`` — the
    statistic names live in the prompt's legend instead of being repeated per
    layer per client.
    """
    row: dict = {}
    if include_layers:
        for name, stats in feats.get("layers", {}).items():
            row[name] = [sig_round(stats.get(k, 0.0), sig) for k in layer_keys]
    whole = feats.get("whole", {})
    row["whole"] = [sig_round(whole.get(k, 0.0), sig) for k in WHOLE_STAT_KEYS]
    return row


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
