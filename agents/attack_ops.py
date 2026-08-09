"""Attack-plan DSL — primitive weight operators the attacker LLM composes.

Instead of emitting raw poisoned weights, the attacker LLM observes compact
per-layer *statistics* of the benign weights and emits an **attack plan**: an
ordered list of primitive mathematical operations. A deterministic interpreter
(`apply_plan`) then applies that plan to the actual benign weights to produce
the poisoned weights sent to the server.

This keeps the LLM's job small and RL-trainable (tens of tokens, not thousands),
PyTorch does all the arithmetic exactly, and composing primitives lets the
attacker discover novel attacks (e.g. "flip the top 30% of the output layer,
then add noise to the hidden layer, then clip").

Operators (10; targets are "all", a layer group like "net.2", or a full key
like "net.2.weight"):
  scale, sign_flip, add_gaussian_noise, mask, clip, add_constant, permute,
  scale_neurons, blend_random, quantize
"""

import json
import logging
import re

import torch

logger = logging.getLogger(__name__)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# JSON extraction (shared with the defender agent)
# ---------------------------------------------------------------------------

def extract_json(text):
    """Best-effort JSON extraction from raw model output (str or already-parsed)."""
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    fence = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    brace = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Update details (the attacker's observation)
# ---------------------------------------------------------------------------

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two flat tensors (0 if either is degenerate)."""
    denom = float(a.norm()) * float(b.norm())
    if denom < _EPS:
        return 0.0
    return float(torch.dot(a, b) / denom)


#: Per-layer statistics, in the order the compact row encoding emits them.
LAYER_STAT_KEYS = ("rel_update", "rms_delta", "energy_frac", "sign_flip_frac",
                   "std_ratio", "absmean_ratio")
#: Whole-model statistics, in the order the compact row encoding emits them.
#: ``energy_frac`` is omitted (trivially 1.0) and ``cos_to_global`` added.
WHOLE_STAT_KEYS = ("rel_update", "rms_delta", "sign_flip_frac", "std_ratio",
                   "absmean_ratio", "cos_to_global")


def layer_shapes(global_sd: dict) -> dict:
    """``{param key: shape}`` for the model — the attacker's layer table.

    Shapes are the same for every client (FL shares one architecture), so the
    compact encoding sends this ONCE instead of repeating it inside each
    client's stats.
    """
    return {k: list(v.shape) for k, v in global_sd.items()}


def _sig(x: float, digits: int) -> float:
    """Round to ``digits`` SIGNIFICANT figures.

    Significant figures rather than decimal places because these statistics span
    orders of magnitude: ``rms_delta`` is ~1e-2 while ``energy_frac`` is ~1e-1
    and ``sign_flip_frac`` can be 0. Fixed decimals either waste characters on
    the large values or quantize the small ones to zero — and ``rms_delta`` is
    exactly what the attacker calibrates additive operators against.
    """
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        return 0.0
    return float(f"{v:.{max(1, int(digits))}g}")


def delta_stats(client_sd: dict, global_sd: dict) -> dict:
    """Raw (unrounded) update statistics — the shared core of both encodings.

    Returns ``{"layers": {key: {stat: float}}, "whole": {stat: float}}``. See
    :func:`delta_details` for what each statistic means. Callers that render the
    same client at several compaction levels compute this ONCE and pass it to
    :func:`format_rows`.
    """
    keys = list(global_sd.keys())
    g_flat = torch.cat([global_sd[k].flatten().float() for k in keys])
    c_flat = torch.cat([client_sd[k].flatten().float() for k in keys])
    d_norm_sq = float((c_flat - g_flat).pow(2).sum())

    def _stats(c: torch.Tensor, g: torch.Tensor) -> dict:
        d = c - g
        dl_norm = float(d.norm())
        count = max(1, int(d.numel()))
        return {
            "rel_update": dl_norm / (float(g.norm()) + _EPS),
            "rms_delta": dl_norm / (count ** 0.5),
            "energy_frac": (dl_norm ** 2) / (d_norm_sq + _EPS),
            "sign_flip_frac": float((torch.sign(c) != torch.sign(g)).float().mean()),
            "std_ratio": float(d.std(unbiased=False)) / (float(g.std(unbiased=False)) + _EPS),
            "absmean_ratio": float(d.abs().mean()) / (float(g.abs().mean()) + _EPS),
        }

    layers = {k: _stats(client_sd[k].flatten().float(), global_sd[k].flatten().float())
              for k in keys}
    whole = _stats(c_flat, g_flat)
    whole.pop("energy_frac", None)                    # trivially 1.0 for the whole model
    whole["cos_to_global"] = _cos(c_flat - g_flat, g_flat)
    return {"layers": layers, "whole": whole}


def format_rows(stats: dict, *, sig: int = 4,
                layer_keys: tuple = LAYER_STAT_KEYS,
                whole_keys: tuple = WHOLE_STAT_KEYS,
                include_layers: bool = True) -> dict:
    """Render :func:`delta_stats` output as compact positional rows.

    See :func:`delta_rows` for the encoding and why it exists.
    """
    row = {}
    if include_layers and layer_keys:
        for k, s in stats["layers"].items():
            row[k] = [_sig(s[name], sig) for name in layer_keys]
    row["whole"] = [_sig(stats["whole"][name], sig) for name in whole_keys]
    return row


def delta_rows(client_sd: dict, global_sd: dict, *, sig: int = 4,
               layer_keys: tuple = LAYER_STAT_KEYS,
               whole_keys: tuple = WHOLE_STAT_KEYS,
               include_layers: bool = True) -> dict:
    """COMPACT form of :func:`delta_details`: one ARRAY of numbers per layer.

    Same statistics, encoded positionally — ``{"net.2.weight": [rel_update,
    rms_delta, ...], ..., "whole": [...]}`` — with the stat names sent once in
    the prompt's legend and the shapes once in :func:`layer_shapes`.

    This exists purely for the CONTEXT BUDGET. The named-key encoding repeats
    ~90 characters of statistic names and a ``shape`` entry per layer per
    client, which at a 10-client pool was the single largest term in the
    attacker's prompt (~4.2k tokens of the ~5.5k total). Positionally the same
    numbers cost about a third of that, so the observation survives intact
    instead of having clients or statistics dropped from it. ``layer_keys`` /
    ``whole_keys`` / ``include_layers`` are the further, lossy steps the
    attacker agent escalates to only if the compact form still does not fit.
    """
    return format_rows(delta_stats(client_sd, global_sd), sig=sig,
                       layer_keys=layer_keys, whole_keys=whole_keys,
                       include_layers=include_layers)


def delta_details(client_sd: dict, global_sd: dict, precision: int = 4) -> dict:
    """Summarize a client's HONEST UPDATE ``Δ = client − global`` for the attacker.

    A partial-insider attacker sees only the global model (broadcast to every
    participant each round) and its OWN clients' benign weights — never the other
    clients' updates (defender-only), and never a stable pool baseline (the poison
    budget makes the controllable pool's size/membership vary, and those clients
    are the poison target, not a reference). So EVERY statistic here is normalized
    only against the global ``G`` or the client's own weights. That keeps them
    self-contained, dimensionless, and independent of the model architecture and
    of how many clients are poisoned this round:

      per layer (and whole model):
        rel_update      ‖Δ‖ / ‖G‖               update size relative to the model
        rms_delta       ‖Δ‖ / sqrt(count)       per-coordinate step size
        energy_frac     ‖Δ_layer‖² / ‖Δ‖²       share of the update in this layer
        sign_flip_frac  mean(sign c ≠ sign G)   fraction of weights flipped vs G
        std_ratio       std(Δ) / std(G)         spread of Δ vs the model's own
        absmean_ratio   mean|Δ| / mean|G|       typical |change| vs the model's own
      whole model also: cos_to_global = cos(Δ, G)

    ``client_sd`` and ``global_sd`` must share keys (same architecture).

    This is the VERBOSE, self-describing encoding (named statistics + a ``shape``
    per layer). :func:`delta_rows` is the positional encoding the attacker prompt
    now uses; both compute the same numbers via :func:`_delta_stats`.
    """
    def r(x):
        return round(float(x), precision)

    stats = delta_stats(client_sd, global_sd)
    layers = {}
    for k, s in stats["layers"].items():
        entry = {"shape": list(global_sd[k].shape)}   # context only, not a magnitude
        entry.update({name: r(v) for name, v in s.items()})
        layers[k] = entry
    return {"layers": layers, "whole": {name: r(v) for name, v in stats["whole"].items()}}


# ---------------------------------------------------------------------------
# Plan extraction + validation
# ---------------------------------------------------------------------------

def extract_plan(text):
    """Return a normalized ``{"operations": [...]}`` plan, or None if unusable."""
    raw = extract_json(text)
    if isinstance(raw, dict):
        ops = raw.get("operations")
        if isinstance(ops, list):
            return ops
        if "op" in raw:                       # a single bare operation
            return [raw]
        return None
    if isinstance(raw, list):
        return raw
    return None


def _coerce_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def extract_selection(text):
    """Parse the attacker's client-selection output into a normalized structure.

    Returns ``{"per_client": [{"id": int, "operations": [...]}, ...],
    "shared_ops": [...] | None, "shared_ids": [int, ...] | None}`` or ``None`` if
    nothing usable was found. Robust to markdown / surrounding prose (via
    ``extract_json``). Accepted shapes:

      * ``{"clients": [{"id": 0, "operations": [ops]}, ...]}``  (canonical: a
        DISTINCT plan per client — enables coordinated multi-client attacks)
      * ``{"clients": [0, 3], "operations": [ops]}``            (explicit ids +
        one shared plan)
      * ``{"operations": [ops]}`` / a bare ``[ops]`` list        (shared plan, no
        ids -> the caller auto-selects clients)
      * ``targets`` / ``ids`` / ``client_ids`` are accepted as aliases for a
        shared id list.
    """
    raw = extract_json(text)
    if raw is None:
        return None
    if isinstance(raw, list):                 # bare list -> a shared ops plan
        return {"per_client": [], "shared_ops": raw, "shared_ids": None}
    if not isinstance(raw, dict):
        return None

    per_client = []
    shared_ids = None
    clients = raw.get("clients")
    if isinstance(clients, list):
        int_ids = []
        for entry in clients:
            if isinstance(entry, dict):
                cid = _coerce_int(entry.get("id", entry.get("client_id")))
                if cid is None:
                    continue
                ops = entry.get("operations")
                if isinstance(ops, dict):
                    ops = [ops]
                elif not isinstance(ops, list):
                    ops = []
                per_client.append({"id": cid, "operations": ops})
            else:                              # a bare int in the clients list
                cid = _coerce_int(entry)
                if cid is not None:
                    int_ids.append(cid)
        if int_ids:
            shared_ids = int_ids

    shared_ops = None
    ops = raw.get("operations")
    if isinstance(ops, list):
        shared_ops = ops
    elif isinstance(ops, dict):
        shared_ops = [ops]
    elif "op" in raw:                          # a single bare operation at top level
        shared_ops = [raw]

    if shared_ids is None:
        for key in ("targets", "ids", "client_ids"):
            v = raw.get(key)
            if isinstance(v, list):
                ids = [c for c in (_coerce_int(x) for x in v) if c is not None]
                if ids:
                    shared_ids = ids
                    break

    if not per_client and shared_ops is None and shared_ids is None:
        return None
    return {"per_client": per_client, "shared_ops": shared_ops, "shared_ids": shared_ids}


# ---------------------------------------------------------------------------
# Operator helpers
# ---------------------------------------------------------------------------

def _f(op, key, default):
    try:
        v = float(op.get(key, default))
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _gen(op):
    g = torch.Generator()
    g.manual_seed(int(_f(op, "seed", 0)) & 0x7FFFFFFF)
    return g


def _topk_mask(t, fraction, largest=True):
    flat = t.flatten()
    n = flat.numel()
    k = int(round(_clamp(fraction, 0.0, 1.0) * n))
    mask = torch.zeros(n, dtype=torch.bool)
    if k > 0:
        idx = torch.topk(flat.abs(), min(k, n), largest=largest).indices
        mask[idx] = True
    return mask.view_as(t)


def _resolve_target(target, sd):
    # A list/tuple targets several layers at once (union of resolved keys) —
    # the LLM sometimes emits "target": ["net.2.weight", "net.4.weight"].
    if isinstance(target, (list, tuple)):
        keys = []
        for t in target:
            for k in _resolve_target(t, sd):
                if k not in keys:
                    keys.append(k)
        return keys
    if target is None or target in ("all", "*"):
        return list(sd.keys())
    if not isinstance(target, str):
        return []                       # unsupported target type -> invalid op
    if target in sd:
        return [target]
    prefix = target if target.endswith(".") else f"{target}."
    return [k for k in sd if k.startswith(prefix)]


# ---------------------------------------------------------------------------
# Operators: (tensor, op_dict) -> tensor
# ---------------------------------------------------------------------------

def _op_scale(t, op):
    return t * _f(op, "factor", 1.0)


def _op_sign_flip(t, op):
    m = _topk_mask(t, _f(op, "fraction", 1.0), largest=True)
    return torch.where(m, -t, t)


def _op_add_gaussian_noise(t, op):
    sigma = abs(_f(op, "sigma", 0.0))
    return t + torch.randn(t.shape, generator=_gen(op)) * sigma


def _op_mask(t, op):
    frac = _clamp(_f(op, "fraction", 0.0), 0.0, 1.0)
    mode = str(op.get("mode", "largest")).lower()
    if mode == "random":
        n = t.numel()
        k = int(round(frac * n))
        mask = torch.zeros(n, dtype=torch.bool)
        if k > 0:
            idx = torch.randperm(n, generator=_gen(op))[:k]
            mask[idx] = True
        mask = mask.view_as(t)
    else:
        mask = _topk_mask(t, frac, largest=(mode != "smallest"))
    return t.masked_fill(mask, 0.0)


def _op_clip(t, op):
    v = abs(_f(op, "value", 1.0))
    return t.clamp(-v, v)


def _op_add_constant(t, op):
    return t + _f(op, "value", 0.0)


def _op_permute(t, op):
    flat = t.flatten()
    perm = torch.randperm(flat.numel(), generator=_gen(op))
    return flat[perm].view_as(t)


def _op_scale_neurons(t, op):
    """Scale a fraction of the most important output units (rows)."""
    factor = _f(op, "factor", 1.0)
    frac = _clamp(_f(op, "fraction", 1.0), 0.0, 1.0)
    out = t.clone()
    n_units = t.shape[0]
    k = int(round(frac * n_units))
    if k <= 0:
        return out
    if t.dim() >= 2:
        row_norms = t.flatten(1).norm(dim=1)
    else:
        row_norms = t.abs()
    idx = torch.topk(row_norms, min(k, n_units), largest=True).indices
    out[idx] = out[idx] * factor
    return out


def _op_blend_random(t, op):
    alpha = _clamp(_f(op, "alpha", 0.0), 0.0, 1.0)
    scale = max(float(t.std(unbiased=False)), _EPS)
    rand = (torch.rand(t.shape, generator=_gen(op)) * 2.0 - 1.0) * scale
    return (1.0 - alpha) * t + alpha * rand


def _op_quantize(t, op):
    step = abs(_f(op, "step", 0.0))
    if step <= _EPS:
        return t
    return torch.round(t / step) * step


OP_FUNCS = {
    "scale": _op_scale,
    "sign_flip": _op_sign_flip,
    "add_gaussian_noise": _op_add_gaussian_noise,
    "mask": _op_mask,
    "clip": _op_clip,
    "add_constant": _op_add_constant,
    "permute": _op_permute,
    "scale_neurons": _op_scale_neurons,
    "blend_random": _op_blend_random,
    "quantize": _op_quantize,
}


# Human-readable operator reference for the attacker prompt.
OPERATOR_DOCS = """Operators (compose several for novel attacks). Each op is
{"op":<name>,"target":<target>, ...params}. `target` is "all", a layer name, a full
parameter key, or a list of these -- use the exact names shown in `client_update_stats`
(a layer groups its ".weight" and ".bias"):
- scale {"factor":float}: multiply weights (negative flips sign, |factor|>1 amplifies).
- sign_flip {"fraction":0..1}: negate the top-fraction largest-magnitude weights (1.0=all).
- add_gaussian_noise {"sigma":float>=0,"seed":int}: add zero-mean noise of std sigma.
- mask {"fraction":0..1,"mode":"largest"|"smallest"|"random","seed":int}: zero out a fraction of weights.
- clip {"value":float>=0}: clamp weights into [-value, value].
- add_constant {"value":float}: add a constant to every weight.
- permute {"seed":int}: randomly shuffle weights within the target (destroys structure).
- scale_neurons {"fraction":0..1,"factor":float}: scale the top-fraction most important output units (rows).
- blend_random {"alpha":0..1,"seed":int}: move toward random noise: (1-alpha)*w + alpha*noise.
- quantize {"step":float>0}: round weights to a coarse grid of size step."""


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

def apply_plan(benign: dict, plan, max_abs: float = 100.0) -> tuple[dict, int]:
    """Apply an attack plan to benign weights → poisoned weights.

    Robust: unknown ops / bad params / bad targets are skipped and counted in
    ``n_invalid``; the result is always a valid state_dict (NaN/Inf scrubbed,
    clamped to ±max_abs, cast back to the benign dtype).

    Returns ``(poisoned_state_dict, n_invalid_ops)``.
    """
    poisoned = {k: v.clone().float() for k, v in benign.items()}
    n_invalid = 0
    ops = plan if isinstance(plan, list) else []

    for op in ops:
        # Guard the WHOLE op: any malformed shape (bad target type, weird params,
        # op-fn failure) counts as invalid and is skipped — never crashes training.
        try:
            if not isinstance(op, dict):
                n_invalid += 1
                continue
            fn = OP_FUNCS.get(op.get("op"))
            keys = _resolve_target(op.get("target", "all"), poisoned)
            if fn is None or not keys:
                n_invalid += 1
                continue
            for k in keys:
                poisoned[k] = fn(poisoned[k], op)
        except Exception as e:
            logger.warning(f"attack_ops: skipping bad op {op!r}: {e}")
            n_invalid += 1

    out = {}
    for k, v in poisoned.items():
        t = torch.nan_to_num(v, nan=0.0, posinf=max_abs, neginf=-max_abs).clamp(-max_abs, max_abs)
        out[k] = t.to(benign[k].dtype)
    return out, n_invalid
