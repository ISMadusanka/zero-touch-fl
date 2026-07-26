"""Per-client, per-layer statistical features for the defender LLM.

The math here is salvaged from the old ``AnomalyDetector._compute_features``,
but this module makes **no decisions** — it only turns the raw client updates
into compact, JSON-friendly statistical vectors. Classifying which clients are
malicious is now the defender LLM's job (and the verifiable reward, computed
elsewhere, supervises it via RL).

For each client we compute, relative to the current global model:
  * Per logical layer (grouped by state_dict prefix, e.g. ``net.2`` / ``net.4``):
      - ``l2_norm``        : ‖Δ_layer‖
      - ``rel_norm``       : ‖Δ_layer‖ / median(‖Δ_layer‖ over ALL clients)
      - ``cos_to_median``  : cosine(Δ_layer, coordinate-wise median of ALL
                             clients' Δ_layer)
      - ``sign_agreement`` : fraction of coordinates whose sign matches the
                             coordinate-wise median sign (catches sign-flip /
                             targeted attacks that preserve norm)

    The median references are taken over all clients INCLUDING the one being
    scored (not leave-one-out). With a benign majority the median is dominated by
    honest clients either way, so a single outlier still stands out; computing one
    shared reference per layer also keeps this O(n) instead of O(n²).
  * Whole model (all params flattened):
      - ``l2_norm`` / ``rel_norm``
      - ``cos_to_mean``      : cosine(Δ, mean Δ)                 (FLTrust-style)
      - ``max_pairwise_cos`` : max cosine to any other client    (FoolsGold)
      - ``dnc_score``        : squared projection on the top right singular
                               vector of the centered updates    (DnC / spectral)

All statistics assume a benign majority (median/MAD-based references), so the
poisoned fraction must stay below 0.5.
"""

import logging

import numpy as np
import torch

from core.types import ModelUpdate

logger = logging.getLogger(__name__)

_EPS = 1e-8


def median_absolute_deviation(values: np.ndarray) -> float:
    """Robust spread: median(|x_i - median(x)|)."""
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def layer_groups(state_dict_keys) -> dict[str, list[str]]:
    """Group state_dict keys into logical layers by their prefix.

    ``net.2.weight`` and ``net.2.bias`` both map to layer ``net.2``.
    Insertion order is preserved so the feature dict is deterministic.
    """
    groups: dict[str, list[str]] = {}
    for k in state_dict_keys:
        prefix = ".".join(k.split(".")[:-1]) or k
        groups.setdefault(prefix, []).append(k)
    return groups


def _flat_delta(weights: dict, global_weights: dict, keys) -> torch.Tensor:
    """Flattened weight delta (client − global) over the given keys."""
    return torch.cat([
        (weights[k] - global_weights[k]).flatten().float() for k in keys
    ])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item())


def compute_client_features(
    updates: list[ModelUpdate], global_weights: dict, precision: int = 5
) -> dict[int, dict]:
    """Compute per-client per-layer + whole-model features.

    Returns a dict keyed by ``client_id``:
        {client_id: {"layers": {layer: {...}}, "whole": {...}}}
    """
    client_ids = [u.client_id for u in updates]
    n = len(updates)
    keys = list(global_weights.keys())
    groups = layer_groups(keys)

    def r(x: float) -> float:
        return round(float(x), precision)

    # ---- Per-layer deltas: layer -> list[tensor] (one per client) ----
    per_layer_deltas: dict[str, list[torch.Tensor]] = {}
    for layer, layer_keys in groups.items():
        per_layer_deltas[layer] = [
            _flat_delta(u.weights, global_weights, layer_keys) for u in updates
        ]

    # ---- Whole-model deltas ----
    whole_deltas = [_flat_delta(u.weights, global_weights, keys) for u in updates]
    whole_stack = torch.stack(whole_deltas)                       # (n, D)
    mean_delta = whole_stack.mean(dim=0)
    whole_norms = np.array([float(d.norm().item()) for d in whole_deltas])
    median_whole_norm = float(np.median(whole_norms))

    # Pairwise cosine matrix (whole model) for FoolsGold-style max similarity.
    norms_t = whole_stack.norm(dim=1, keepdim=True).clamp(min=_EPS)
    normalized = whole_stack / norms_t
    pairwise_cos = (normalized @ normalized.T).cpu().numpy()

    # DnC spectral outlier score: project centered updates on top singular vec.
    centered = whole_stack - mean_delta
    try:
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        top_v = Vh[0]
        dnc_scores = [float((c @ top_v).item()) ** 2 for c in centered]
    except Exception as e:  # pragma: no cover - degenerate inputs
        logger.warning(f"features: SVD failed ({e}); dnc_scores=0")
        dnc_scores = [0.0] * n

    # ---- Per-layer reference stats (computed once per layer) ----
    layer_ref = {}
    for layer, deltas in per_layer_deltas.items():
        stacked = torch.stack(deltas)                             # (n, d_layer)
        norms = np.array([float(d.norm().item()) for d in deltas])
        coord_median = stacked.median(dim=0).values               # (d_layer,)
        layer_ref[layer] = {
            "norms": norms,
            "median_norm": float(np.median(norms)),
            "coord_median": coord_median,
            "median_sign": torch.sign(coord_median),
        }

    features: dict[int, dict] = {}
    for i, cid in enumerate(client_ids):
        layer_feats = {}
        for layer, deltas in per_layer_deltas.items():
            ref = layer_ref[layer]
            d_i = deltas[i]
            norm_i = float(d_i.norm().item())
            rel_norm = norm_i / (ref["median_norm"] + _EPS)
            cos_med = _cosine(d_i, ref["coord_median"])
            sign_agree = float(
                (torch.sign(d_i) == ref["median_sign"]).float().mean().item()
            )
            layer_feats[layer] = {
                "l2_norm": r(norm_i),
                "rel_norm": r(rel_norm),
                "cos_to_median": r(cos_med),
                "sign_agreement": r(sign_agree),
            }

        # FoolsGold: max cosine to any OTHER client.
        row = pairwise_cos[i].copy()
        row[i] = -2.0
        max_pairwise = float(np.max(row)) if n > 1 else 0.0

        features[cid] = {
            "layers": layer_feats,
            "whole": {
                "l2_norm": r(whole_norms[i]),
                "rel_norm": r(whole_norms[i] / (median_whole_norm + _EPS)),
                "cos_to_mean": r(_cosine(whole_deltas[i], mean_delta)),
                "max_pairwise_cos": r(max_pairwise),
                "dnc_score": r(dnc_scores[i]),
            },
        }

    return features
