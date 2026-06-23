"""Serialize / parse raw model weights for the attacker LLM.

In this design the attacker LLM literally emits poisoned weights (the user's
chosen output representation). This module is the hardening layer around that
fragile contract:

  * ``dump_weights``  renders a client's benign weights into compact per-layer
    flat arrays for the prompt (reduced precision to save tokens).
  * ``parse_round``   turns the LLM's JSON output back into validated
    state_dicts — one per poisoned client.

ROBUSTNESS CONTRACT: parsing NEVER raises on bad model output. Any block that
is missing, mis-shaped, non-numeric, NaN/Inf, or out of range falls back to the
client's benign weights and is counted as *malformed*. The caller turns the
malformed count into a reward penalty, so a bad generation costs reward instead
of crashing training.
"""

import copy
import json
import logging
import re

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization (weights -> prompt)
# ---------------------------------------------------------------------------

def dump_weights(state_dict: dict, precision: int = 4) -> dict:
    """Render a state_dict as ``{key: [flat floats]}`` for the prompt."""
    out = {}
    for k, v in state_dict.items():
        out[k] = [round(float(x), precision) for x in v.flatten().tolist()]
    return out


def weight_schema(state_dict: dict) -> dict:
    """``{key: [shape...]}`` so the LLM knows the exact expected array lengths."""
    return {k: list(v.shape) for k, v in state_dict.items()}


def total_params(state_dict: dict) -> int:
    return int(sum(v.numel() for v in state_dict.values()))


# ---------------------------------------------------------------------------
# Parsing (LLM output -> validated state_dicts)
# ---------------------------------------------------------------------------

def extract_json(text) -> dict | None:
    """Best-effort JSON object extraction (mirrors llm_client._extract_json).

    Public helper — also used by the defender agent to parse its verdict list.
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # First-to-last brace: tolerates leading chain-of-thought / trailing prose.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _flatten_numeric(arr) -> list[float] | None:
    """Flatten an arbitrarily nested list into floats; None if any non-numeric.

    Accepts both flat (``[...]``) and nested (``[[...], [...]]``) arrays so the
    LLM may emit either a flat list or the natural 2-D weight matrix.
    """
    out: list[float] = []

    def _walk(x) -> bool:
        if isinstance(x, bool):
            return False                      # reject True/False sneaking in
        if isinstance(x, (int, float)):
            out.append(float(x))
            return True
        if isinstance(x, list):
            return all(_walk(item) for item in x)
        return False

    return out if _walk(arr) else None


def _parse_client_block(
    block, reference: dict, max_abs: float
) -> dict | None:
    """Parse one client's ``{key: array}`` block into a validated state_dict.

    Returns None on any unrecoverable problem (caller falls back to benign).
    """
    if not isinstance(block, dict):
        return None
    out = {}
    for key, ref_tensor in reference.items():
        if key not in block:
            return None
        flat = _flatten_numeric(block[key])
        if flat is None or len(flat) != ref_tensor.numel():
            return None
        t = torch.tensor(flat, dtype=torch.float32).reshape(ref_tensor.shape)
        # Scrub non-finite values, then clamp into a sane range.
        t = torch.nan_to_num(t, nan=0.0, posinf=max_abs, neginf=-max_abs)
        t = t.clamp(-max_abs, max_abs)
        out[key] = t.to(ref_tensor.dtype)
    return out


def parse_round(
    text, references: dict[int, dict], max_abs: float = 100.0
) -> tuple[dict[int, dict], int]:
    """Parse the attacker's per-round output for all poisoned clients.

    Expected output shape (keys are client ids, as int or str)::

        {"0": {"net.2.weight": [...], "net.2.bias": [...], ...},
         "3": {...}}

    Args:
        text:        Raw LLM output (str) or already-parsed dict.
        references:  ``{client_id: benign_state_dict}`` for each poisoned client;
                     used both for shape validation and as the fallback weights.
        max_abs:     Absolute clamp bound for emitted weights.

    Returns:
        ``(poisoned, n_malformed)`` where ``poisoned`` maps every requested
        client_id to a validated state_dict (benign fallback when malformed),
        and ``n_malformed`` counts clients that fell back.
    """
    raw = extract_json(text)
    poisoned: dict[int, dict] = {}
    n_malformed = 0

    for cid, ref in references.items():
        block = None
        if isinstance(raw, dict):
            block = raw.get(str(cid), raw.get(cid))
            # Single-client convenience: if only one client is poisoned and the
            # model returned the weights at the top level, accept that too.
            if block is None and len(references) == 1 and any(
                k in raw for k in ref
            ):
                block = raw
        parsed = _parse_client_block(block, ref, max_abs) if block is not None else None
        if parsed is None:
            poisoned[cid] = copy.deepcopy(ref)   # benign fallback
            n_malformed += 1
        else:
            poisoned[cid] = parsed

    if n_malformed:
        logger.warning(
            f"weight_codec: {n_malformed}/{len(references)} client block(s) "
            f"malformed — fell back to benign weights"
        )
    return poisoned, n_malformed
