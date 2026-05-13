"""Noise injection model poisoning: adds Gaussian noise to weights.

Supports selective injection — when k is specified, only the k weights with
the largest gradient magnitudes are noised, leaving the rest as honest updates.
"""

import logging

import torch
from core.interfaces import BaseAttack
from attacks.registry import register

logger = logging.getLogger(__name__)


@register("noise_injection")
class NoiseInjectionAttack(BaseAttack):

    def __init__(self):
        self.last_metadata: dict = {}

    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Add Gaussian noise scaled by `scale` to client weights.

        When k is specified, noise is only injected into the top-k weights
        (ranked by gradient magnitude). The rest remain honest.

        Args:
            weights:        The honest local weights this client would have sent.
            global_weights: The current global model weights.
            params:
                scale (float): Noise standard deviation. Defaults to 1.0.
                k (int|None): Number of weights to inject noise into.
                              None = inject into ALL weights (original behaviour).
                              Minimum 10 when specified.
        """
        scale = params.get("scale", 1.0)
        k = params.get("k", None)  # None → noise all weights

        # ── Compute honest gradients (needed for top-k ranking) ───────────
        gradients = {}
        for key in weights:
            gradients[key] = weights[key] - global_weights[key]

        total_params = sum(t.numel() for t in weights.values())

        # ── Case 1: inject noise into ALL weights ─────────────────────────
        if k is None:
            poisoned = {}
            for key in weights:
                noise = torch.randn_like(weights[key]) * scale
                poisoned[key] = weights[key] + noise

            self.last_metadata = {
                "k": "all",
                "total_params": total_params,
                "affected_per_layer": {
                    key: weights[key].numel() for key in weights
                },
            }
            logger.info(f"Noise injection (all): scale={scale}, noised all {total_params} params")
            return poisoned

        # ── Case 2: strategic top-k selection ─────────────────────────────
        raw_k = int(k)
        k = max(10, min(raw_k, total_params))  # clamp to [10, total_params]
        if raw_k != k:
            logger.info(f"Noise injection: k clamped from {raw_k} → {k} (range [10, {total_params}])")
        logger.info(f"Noise injection (selective): scale={scale}, k={k}/{total_params} params")

        # Flatten all gradients into a single vector and track layer sizes
        layer_keys = list(weights.keys())
        layer_sizes = {key: gradients[key].numel() for key in layer_keys}
        flat = torch.cat([gradients[key].flatten() for key in layer_keys])

        # Select top-k indices by absolute gradient magnitude
        _, topk_flat_indices = torch.topk(flat.abs(), k)
        target_mask_flat = torch.zeros(flat.numel(), dtype=torch.bool)
        target_mask_flat[topk_flat_indices] = True

        # Reconstruct per-layer masks and apply selective noise
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

            # Noised weights where mask is True, honest weights elsewhere
            noise = torch.randn_like(weights[key]) * scale
            noised = weights[key] + noise
            poisoned[key] = torch.where(layer_mask, noised, weights[key])

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
        logger.info(f"  Noised per layer: {layer_summary}")
        logger.info(
            f"  Grad magnitudes — targeted avg: {self.last_metadata['avg_targeted_grad_magnitude']:.6f}, "
            f"untargeted avg: {self.last_metadata['avg_untargeted_grad_magnitude']:.6f}"
        )

        return poisoned
