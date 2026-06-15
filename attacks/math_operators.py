"""Mathematical operators for dynamic weight poisoning.

Each operator is a pure function:  (tensor, **params) → (tensor, metadata)

The LLM composes pipelines of these operators and decides which to apply
to which layers, giving it fine-grained creative control over the attack.

Operators
---------
scale       — Multiply weights by a scalar factor
shift       — Add a constant bias to weights
rotate      — Apply Givens rotation on flattened weight pairs
mask        — Zero-out a fraction of weights
permute     — Randomly shuffle weight indices
inject_noise— Add Gaussian noise
invert      — Negate weights with optional scaling
align       — Interpolate between local and global weights
clip        — Clamp weight magnitudes to a threshold
smooth      — Average weights toward their mean (reduce variance)
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

_OPERATORS: dict[str, callable] = {}


def _register_op(name: str):
    """Decorator to register a mathematical operator."""
    def wrapper(fn):
        _OPERATORS[name] = fn
        return fn
    return wrapper


def get_operator(name: str):
    """Look up a registered operator by name."""
    if name not in _OPERATORS:
        raise ValueError(
            f"Unknown operator: {name}. "
            f"Available: {list(_OPERATORS.keys())}"
        )
    return _OPERATORS[name]


def available_operators() -> list[str]:
    """Return names of all registered operators."""
    return list(_OPERATORS.keys())


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

@_register_op("scale")
def op_scale(w: torch.Tensor, **params) -> tuple[torch.Tensor, dict]:
    """Multiply weights by a scalar factor.

    w' = w * factor

    Args:
        w: Weight tensor.
        factor (float): Scaling multiplier. Default 2.0.
            >1 amplifies, <1 diminishes, negative inverts+scales.
    """
    factor = float(params.get("factor", 2.0))
    result = w * factor
    meta = {
        "op": "scale",
        "factor": factor,
        "norm_before": round(w.norm().item(), 6),
        "norm_after": round(result.norm().item(), 6),
    }
    logger.debug(f"  scale: factor={factor}, norm {meta['norm_before']:.6f} → {meta['norm_after']:.6f}")
    return result, meta


@_register_op("shift")
def op_shift(w: torch.Tensor, **params) -> tuple[torch.Tensor, dict]:
    """Add a constant bias to all weights.

    w' = w + delta

    Args:
        w: Weight tensor.
        delta (float): Constant to add. Default 0.1.
    """
    delta = float(params.get("delta", 0.1))
    result = w + delta
    meta = {
        "op": "shift",
        "delta": delta,
        "mean_before": round(w.mean().item(), 6),
        "mean_after": round(result.mean().item(), 6),
    }
    logger.debug(f"  shift: delta={delta}, mean {meta['mean_before']:.6f} → {meta['mean_after']:.6f}")
    return result, meta


@_register_op("rotate")
def op_rotate(w: torch.Tensor, **params) -> tuple[torch.Tensor, dict]:
    """Apply Givens rotation to adjacent weight pairs in flattened space.

    For each consecutive pair (w[2i], w[2i+1]), apply:
        w'[2i]   =  cos(θ)·w[2i] - sin(θ)·w[2i+1]
        w'[2i+1] =  sin(θ)·w[2i] + cos(θ)·w[2i+1]

    This preserves the L2 norm while changing the direction in parameter
    space — very stealthy against norm-based detection.

    Args:
        w: Weight tensor.
        angle (float): Rotation angle in radians. Default 0.1.
    """
    angle = float(params.get("angle", 0.1))
    flat = w.flatten().clone()
    n = flat.numel()

    # Pair-wise Givens rotation
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    pairs = n // 2
    if pairs > 0:
        even = flat[:2 * pairs].view(pairs, 2)
        x, y = even[:, 0].clone(), even[:, 1].clone()
        even[:, 0] = cos_a * x - sin_a * y
        even[:, 1] = sin_a * x + cos_a * y
        flat[:2 * pairs] = even.flatten()

    result = flat.reshape(w.shape)
    meta = {
        "op": "rotate",
        "angle_rad": angle,
        "angle_deg": round(math.degrees(angle), 2),
        "pairs_rotated": pairs,
        "norm_before": round(w.norm().item(), 6),
        "norm_after": round(result.norm().item(), 6),
    }
    logger.debug(f"  rotate: angle={angle:.4f} rad, {pairs} pairs, norm preserved: {meta['norm_before']:.6f} → {meta['norm_after']:.6f}")
    return result, meta


@_register_op("mask")
def op_mask(w: torch.Tensor, **params) -> tuple[torch.Tensor, dict]:
    """Zero-out a fraction of weights (structural dropout).

    w'[i] = 0 if randomly selected, else w[i]

    Args:
        w: Weight tensor.
        ratio (float): Fraction of weights to zero [0.0, 1.0]. Default 0.3.
        seed (int): Random seed for reproducibility. Default None.
    """
    ratio = float(params.get("ratio", 0.3))
    ratio = max(0.0, min(1.0, ratio))
    seed = params.get("seed", None)

    if seed is not None:
        gen = torch.Generator().manual_seed(int(seed))
    else:
        gen = None

    mask = torch.rand_like(w.float(), generator=gen) > ratio
    result = w * mask.to(w.dtype)

    n_masked = int((~mask).sum().item())
    meta = {
        "op": "mask",
        "ratio": ratio,
        "total_params": w.numel(),
        "masked_count": n_masked,
        "norm_before": round(w.norm().item(), 6),
        "norm_after": round(result.norm().item(), 6),
    }
    logger.debug(f"  mask: ratio={ratio}, masked {n_masked}/{w.numel()} weights")
    return result, meta


@_register_op("permute")
def op_permute(w: torch.Tensor, **params) -> tuple[torch.Tensor, dict]:
    """Randomly shuffle weight indices in flattened space.

    Disrupts learned feature associations while preserving the overall
    weight distribution — hard to detect via distribution-based methods.

    Args:
        w: Weight tensor.
        seed (int): Random seed for the permutation. Default 42.
    """
    seed = int(params.get("seed", 42))

    flat = w.flatten().clone()
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(flat.numel(), generator=gen)
    flat = flat[perm]
    result = flat.reshape(w.shape)

    meta = {
        "op": "permute",
        "seed": seed,
        "n_params": w.numel(),
        "norm_before": round(w.norm().item(), 6),
        "norm_after": round(result.norm().item(), 6),
    }
    logger.debug(f"  permute: seed={seed}, shuffled {w.numel()} weights (norm preserved)")
    return result, meta


@_register_op("inject_noise")
def op_inject_noise(w: torch.Tensor, **params) -> tuple[torch.Tensor, dict]:
    """Add Gaussian noise to weights.

    w' = w + N(0, σ²)

    Args:
        w: Weight tensor.
        sigma (float): Standard deviation of noise. Default 0.5.
    """
    sigma = float(params.get("sigma", 0.5))
    noise = torch.randn_like(w) * sigma
    result = w + noise

    meta = {
        "op": "inject_noise",
        "sigma": sigma,
        "noise_norm": round(noise.norm().item(), 6),
        "norm_before": round(w.norm().item(), 6),
        "norm_after": round(result.norm().item(), 6),
    }
    logger.debug(f"  inject_noise: sigma={sigma}, noise_norm={meta['noise_norm']:.6f}")
    return result, meta


@_register_op("invert")
def op_invert(w: torch.Tensor, **params) -> tuple[torch.Tensor, dict]:
    """Negate weights with optional scaling (generalized sign-flip).

    w' = -w * factor

    Args:
        w: Weight tensor.
        factor (float): Post-inversion scaling. Default 1.0.
            factor=1.0 is a pure sign-flip.
    """
    factor = float(params.get("factor", 1.0))
    result = -w * factor

    meta = {
        "op": "invert",
        "factor": factor,
        "norm_before": round(w.norm().item(), 6),
        "norm_after": round(result.norm().item(), 6),
    }
    logger.debug(f"  invert: factor={factor}, norm {meta['norm_before']:.6f} → {meta['norm_after']:.6f}")
    return result, meta


@_register_op("align")
def op_align(w: torch.Tensor, **params) -> tuple[torch.Tensor, dict]:
    """Interpolate between local weights and a reference (global weights).

    w' = w + α · (reference - w) = (1 - α)·w + α·reference

    α > 1 overshoots past the reference (amplifies divergence).
    α < 0 pushes weights *away* from the reference.

    Args:
        w: Weight tensor (local weights).
        reference: Reference tensor (global weights). Required.
        alpha (float): Interpolation factor. Default 0.5.
    """
    alpha = float(params.get("alpha", 0.5))
    reference = params.get("reference")

    if reference is None:
        logger.warning("  align: no reference tensor provided — returning unchanged")
        return w, {"op": "align", "error": "no reference provided"}

    result = w + alpha * (reference - w)
    meta = {
        "op": "align",
        "alpha": alpha,
        "norm_before": round(w.norm().item(), 6),
        "norm_after": round(result.norm().item(), 6),
        "distance_to_ref_before": round((w - reference).norm().item(), 6),
        "distance_to_ref_after": round((result - reference).norm().item(), 6),
    }
    logger.debug(
        f"  align: alpha={alpha}, dist_to_ref {meta['distance_to_ref_before']:.6f} → {meta['distance_to_ref_after']:.6f}"
    )
    return result, meta


@_register_op("clip")
def op_clip(w: torch.Tensor, **params) -> tuple[torch.Tensor, dict]:
    """Clamp weight magnitudes to a threshold.

    w' = clamp(w, -threshold, +threshold)

    Args:
        w: Weight tensor.
        threshold (float): Maximum absolute value. Default 0.5.
    """
    threshold = float(params.get("threshold", 0.5))
    threshold = max(0.001, threshold)  # prevent zero-clamping
    result = torch.clamp(w, -threshold, threshold)

    n_clipped = int(((w.abs() > threshold).sum()).item())
    meta = {
        "op": "clip",
        "threshold": threshold,
        "clipped_count": n_clipped,
        "total_params": w.numel(),
        "norm_before": round(w.norm().item(), 6),
        "norm_after": round(result.norm().item(), 6),
    }
    logger.debug(f"  clip: threshold={threshold}, clipped {n_clipped}/{w.numel()} weights")
    return result, meta


@_register_op("smooth")
def op_smooth(w: torch.Tensor, **params) -> tuple[torch.Tensor, dict]:
    """Average weights toward their mean (reduce variance).

    w' = (1 - β)·w + β·mean(w)

    β=1.0 collapses all weights to a single value (the mean).
    β=0.0 leaves weights unchanged.

    Args:
        w: Weight tensor.
        beta (float): Smoothing factor [0.0, 1.0]. Default 0.3.
    """
    beta = float(params.get("beta", 0.3))
    beta = max(0.0, min(1.0, beta))
    mean_val = w.mean()
    result = (1 - beta) * w + beta * mean_val

    meta = {
        "op": "smooth",
        "beta": beta,
        "weight_mean": round(mean_val.item(), 6),
        "std_before": round(w.std().item(), 6),
        "std_after": round(result.std().item(), 6),
    }
    logger.debug(f"  smooth: beta={beta}, std {meta['std_before']:.6f} → {meta['std_after']:.6f}")
    return result, meta
