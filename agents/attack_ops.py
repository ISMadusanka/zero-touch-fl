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
# Layer details (the attacker's observation)
# ---------------------------------------------------------------------------

def layer_details(state_dict: dict, precision: int = 4) -> dict:
    """Per-layer summary statistics describing benign weights (no raw values)."""
    def r(x):
        return round(float(x), precision)

    out = {}
    for k, v in state_dict.items():
        t = v.flatten().float()
        out[k] = {
            "shape": list(v.shape),
            "count": int(t.numel()),
            "mean": r(t.mean()),
            "std": r(t.std(unbiased=False)),
            "min": r(t.min()),
            "max": r(t.max()),
            "l2_norm": r(t.norm()),
            "abs_mean": r(t.abs().mean()),
        }
    return out


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
    if target in (None, "all", "*"):
        return list(sd.keys())
    if target in sd:
        return [target]
    prefix = target if str(target).endswith(".") else f"{target}."
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
OPERATOR_DOCS = """Available primitive operators (compose several for novel attacks).
Each operation is {"op": <name>, "target": <"all" | layer-group e.g. "net.2" | full key e.g. "net.4.weight">, ...params}:
- scale            {"factor": float}                  multiply weights by factor (negative flips sign, |f|>1 amplifies).
- sign_flip        {"fraction": 0..1}                 negate the top-`fraction` largest-magnitude weights (1.0 = all).
- add_gaussian_noise {"sigma": float>=0, "seed": int} add zero-mean Gaussian noise of std `sigma`.
- mask             {"fraction": 0..1, "mode": "largest"|"smallest"|"random", "seed": int}  zero out a fraction of weights.
- clip             {"value": float>=0}                clamp weights into [-value, value].
- add_constant     {"value": float}                   add a constant to every weight.
- permute          {"seed": int}                      randomly shuffle weights within the target (destroys structure).
- scale_neurons    {"fraction": 0..1, "factor": float} scale the top-`fraction` most important output units (rows).
- blend_random     {"alpha": 0..1, "seed": int}       interpolate toward random noise: (1-alpha)*w + alpha*noise.
- quantize         {"step": float>0}                  round weights to a coarse grid of size `step`."""


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
        if not isinstance(op, dict):
            n_invalid += 1
            continue
        name = op.get("op")
        fn = OP_FUNCS.get(name)
        keys = _resolve_target(op.get("target", "all"), poisoned)
        if fn is None or not keys:
            n_invalid += 1
            continue
        for k in keys:
            try:
                poisoned[k] = fn(poisoned[k], op)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"attack_ops: op={name} failed on {k}: {e}")
                n_invalid += 1

    out = {}
    for k, v in poisoned.items():
        t = torch.nan_to_num(v, nan=0.0, posinf=max_abs, neginf=-max_abs).clamp(-max_abs, max_abs)
        out[k] = t.to(benign[k].dtype)
    return out, n_invalid
