"""Scaling model poisoning: amplifies the weight delta.

Supports selective scaling — when k is specified, only the k weights with
the largest gradient magnitudes are scaled, leaving the rest as honest updates.
"""

import logging

import torch
from core.interfaces import BaseAttack
from attacks.registry import register

logger = logging.getLogger(__name__)


@register("scaling")
class ScalingAttack(BaseAttack):

    def __init__(self):
        self.last_metadata: dict = {}

    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Scale the weight delta by `factor`.

        When k is specified, only the top-k weight deltas (ranked by gradient
        magnitude) are scaled. The rest remain honest.

        Args:
            weights:        The honest local weights this client would have sent.
            global_weights: The current global model weights.
            params:
                factor (float): Scaling multiplier for the delta. Defaults to 10.0.
                k (int|None): Number of weights to scale.
                              None = scale ALL weights (original behaviour).
                              Minimum 10 when specified.
        """
        factor = params.get("factor", 10.0)
        k = params.get("k", None)  # None → scale all weights

        # ── Compute honest gradients per layer ────────────────────────────
        gradients = {}
        for key in weights:
            gradients[key] = weights[key] - global_weights[key]

        total_params = sum(t.numel() for t in weights.values())

        # ── Case 1: scale ALL weights ─────────────────────────────────────
        if k is None:
            poisoned = {}
            for key in weights:
                poisoned[key] = global_weights[key] + gradients[key] * factor

            self.last_metadata = {
                "k": "all",
                "total_params": total_params,
                "affected_per_layer": {
                    key: weights[key].numel() for key in weights
                },
            }
            logger.info(f"Scaling (all): factor={factor}, scaled all {total_params} params")
            return poisoned

        # ── Case 2: strategic top-k selection ─────────────────────────────
        raw_k = int(k)
        k = max(10, min(raw_k, total_params))  # clamp to [10, total_params]
        if raw_k != k:
            logger.info(f"Scaling: k clamped from {raw_k} → {k} (range [10, {total_params}])")
        logger.info(f"Scaling (selective): factor={factor}, k={k}/{total_params} params")

        # Flatten all gradients into a single vector and track layer sizes
        layer_keys = list(weights.keys())
        layer_sizes = {key: gradients[key].numel() for key in layer_keys}
        flat = torch.cat([gradients[key].flatten() for key in layer_keys])

        # Select top-k indices by absolute gradient magnitude
        _, topk_flat_indices = torch.topk(flat.abs(), k)
        target_mask_flat = torch.zeros(flat.numel(), dtype=torch.bool)
        target_mask_flat[topk_flat_indices] = True

        # Reconstruct per-layer masks and apply selective scaling
        offset = 0
        poisoned = {}
        affected_per_layer = {}
        affected_indices_per_layer = {}

        for key in layer_keys:
            size = layer_sizes[key]
            layer_mask_flat = target_mask_flat[offset : offset + size]
            layer_mask = layer_mask_flat.reshape(weights[key].shape)

            # Record which indices in this layer were targeted
            layer_target_indices = layer_mask_flat.nonzero(as_tuple=True)[0].tolist()
            if layer_target_indices:
                affected_per_layer[key] = len(layer_target_indices)
                affected_indices_per_layer[key] = layer_target_indices

            # Scaled weights where mask is True, honest weights elsewhere
            scaled = global_weights[key] + gradients[key] * factor
            poisoned[key] = torch.where(layer_mask, scaled, weights[key])

            offset += size

        # ── Compute gradient magnitude stats for memory ───────────────────
        all_abs = flat.abs()
        targeted_mags = all_abs[topk_flat_indices]
        untargeted_mask = ~target_mask_flat
        untargeted_mags = all_abs[untargeted_mask]

        self.last_metadata = {
            "k": k,
            "total_params": total_params,
            "affected_per_layer": affected_per_layer,
            "affected_indices_per_layer": affected_indices_per_layer,
            "avg_targeted_grad_magnitude": round(targeted_mags.mean().item(), 6),
            "avg_untargeted_grad_magnitude": round(
                untargeted_mags.mean().item(), 6
            ) if untargeted_mags.numel() > 0 else 0.0,
        }

        # Log per-layer breakdown
        layer_summary = ", ".join(f"{lyr}={cnt}" for lyr, cnt in affected_per_layer.items())
        logger.info(f"  Scaled per layer: {layer_summary}")
        logger.info(
            f"  Grad magnitudes — targeted avg: {self.last_metadata['avg_targeted_grad_magnitude']:.6f}, "
            f"untargeted avg: {self.last_metadata['avg_untargeted_grad_magnitude']:.6f}"
        )

        return poisoned
