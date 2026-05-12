"""Sign-flip model poisoning: negates and scales the honest gradient update.

Supports selective flipping — when k is specified, only the k weights with
the largest gradient magnitudes are flipped, making the attack stealthier.
"""

import logging

import torch
from core.interfaces import BaseAttack
from attacks.registry import register

logger = logging.getLogger(__name__)


@register("sign_flip")
class SignFlipAttack(BaseAttack):

    def __init__(self):
        self.last_metadata: dict = {}

    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Return g_mal = -c * g_honest for the top-k weights by gradient magnitude.

        When k is specified, only the k weights with the largest gradient
        magnitudes are flipped — the rest are left as honest updates.
        This makes the attack harder to detect via simple norm checks.

        Args:
            weights:        The honest local weights this client would have sent.
            global_weights: The current global model weights.
            params:
                c (float): Positive scaling factor. Typical values: 1, 2, 4.
                           Larger c = stronger attack but easier to detect.
                           Must be > 0. Defaults to 1.0.
                k (int|None): Number of weights to selectively flip.
                              None = flip ALL weights (original behaviour).
                              Minimum 10 when specified.
        """
        c = params.get("c", 1.0)
        k = params.get("k", None)  # None → flip all weights

        if c <= 0:
            raise ValueError(f"Scaling factor c must be > 0, got {c}")

        # ── Compute honest gradients per layer ────────────────────────────
        gradients = {}
        for key in weights:
            gradients[key] = weights[key] - global_weights[key]

        total_params = sum(t.numel() for t in weights.values())

        # ── Case 1: flip ALL weights (k not specified) ────────────────────
        if k is None:
            poisoned = {}
            for key in weights:
                poisoned[key] = global_weights[key] + (-c * gradients[key])

            self.last_metadata = {
                "k": "all",
                "total_params": total_params,
                "flipped_per_layer": {
                    key: weights[key].numel() for key in weights
                },
            }
            logger.info(f"Sign-flip (all): c={c}, flipped all {total_params} params")
            return poisoned

        # ── Case 2: strategic top-k selection ─────────────────────────────
        raw_k = int(k)
        k = max(10, min(raw_k, total_params))  # clamp to [10, total_params]
        if raw_k != k:
            logger.info(f"Sign-flip: k clamped from {raw_k} → {k} (range [10, {total_params}])")
        logger.info(f"Sign-flip (selective): c={c}, k={k}/{total_params} params")

        # Flatten all gradients into a single vector and track layer sizes
        layer_keys = list(weights.keys())
        layer_sizes = {key: gradients[key].numel() for key in layer_keys}
        flat = torch.cat([gradients[key].flatten() for key in layer_keys])

        # Select top-k indices by absolute gradient magnitude
        _, topk_flat_indices = torch.topk(flat.abs(), k)
        flip_mask_flat = torch.zeros(flat.numel(), dtype=torch.bool)
        flip_mask_flat[topk_flat_indices] = True

        # Reconstruct per-layer masks and apply selective flip
        offset = 0
        poisoned = {}
        flipped_per_layer = {}
        flipped_indices_per_layer = {}

        for key in layer_keys:
            size = layer_sizes[key]
            layer_mask_flat = flip_mask_flat[offset : offset + size]
            layer_mask = layer_mask_flat.reshape(weights[key].shape)

            # Record which indices in this layer were flipped
            layer_flip_indices = layer_mask_flat.nonzero(as_tuple=True)[0].tolist()
            if layer_flip_indices:
                flipped_per_layer[key] = len(layer_flip_indices)
                flipped_indices_per_layer[key] = layer_flip_indices

            # Flipped weights where mask is True, honest weights elsewhere
            flipped = global_weights[key] + (-c * gradients[key])
            poisoned[key] = torch.where(layer_mask, flipped, weights[key])

            offset += size

        # ── Compute gradient magnitude stats for memory ───────────────────
        all_abs = flat.abs()
        flipped_mags = all_abs[topk_flat_indices]
        unflipped_mask = ~flip_mask_flat
        unflipped_mags = all_abs[unflipped_mask]

        self.last_metadata = {
            "k": k,
            "total_params": total_params,
            "flipped_per_layer": flipped_per_layer,
            "flipped_indices_per_layer": flipped_indices_per_layer,
            "avg_flipped_grad_magnitude": round(flipped_mags.mean().item(), 6),
            "avg_unflipped_grad_magnitude": round(
                unflipped_mags.mean().item(), 6
            ) if unflipped_mags.numel() > 0 else 0.0,
        }

        # Log per-layer breakdown
        layer_summary = ", ".join(f"{lyr}={cnt}" for lyr, cnt in flipped_per_layer.items())
        logger.info(f"  Flipped per layer: {layer_summary}")
        logger.info(
            f"  Grad magnitudes — flipped avg: {self.last_metadata['avg_flipped_grad_magnitude']:.6f}, "
            f"unflipped avg: {self.last_metadata['avg_unflipped_grad_magnitude']:.6f}"
        )

        return poisoned
