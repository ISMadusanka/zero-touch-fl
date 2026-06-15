"""Dynamic mathematical attack — applies a sequence of mathematical operators.

The LLM agent decides which operators to apply on which layers and weights
when poisoning the client model.
"""

import logging
import torch
from core.interfaces import BaseAttack
from attacks.registry import register

logger = logging.getLogger(__name__)

@register("dynamic_math")
class DynamicMathAttack(BaseAttack):

    def __init__(self):
        self.last_metadata: dict = {}

    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Apply mathematical operations to craft the attack payload.

        Args:
            weights: The honest local weights.
            global_weights: The current global model weights.
            params: Must contain 'operations' (list of dicts).
        """
        operations = params.get("operations", [])
        
        # Start with the honest weights
        poisoned = {k: v.clone() for k, v in weights.items()}
        
        ops_applied = []

        for i, op in enumerate(operations):
            op_name = op.get("op")
            if not op_name:
                continue

            layers = op.get("layers", list(weights.keys()))
            if not layers:
                layers = list(weights.keys())

            layers_affected = []

            for layer in layers:
                if layer not in poisoned:
                    continue
                
                layers_affected.append(layer)
                w = poisoned[layer]
                g = global_weights[layer]
                
                try:
                    if op_name == "scale":
                        factor = op.get("factor", 1.0)
                        # scale the gradient (w - g)
                        w = g + factor * (w - g)
                    elif op_name == "shift":
                        value = op.get("value", 0.0)
                        w = w + value
                    elif op_name == "rotate":
                        shifts = op.get("shifts", 1)
                        w = torch.roll(w, shifts=int(shifts))
                    elif op_name == "mask":
                        fraction = op.get("fraction", 0.5)
                        # Zero out a random fraction of the weights
                        mask = torch.rand_like(w) > fraction
                        w = w * mask
                    elif op_name == "permute":
                        # Randomly shuffle weights
                        flat_w = w.flatten()
                        perm = torch.randperm(flat_w.numel())
                        w = flat_w[perm].reshape(w.shape)
                    elif op_name == "inject_noise":
                        std = op.get("std", 0.1)
                        w = w + torch.randn_like(w) * std
                    elif op_name == "invert":
                        # Sign-flip the honest update
                        w = g - (w - g)
                    elif op_name == "align":
                        # Interpolate towards global weights
                        alpha = op.get("alpha", 0.5)
                        w = (1 - alpha) * w + alpha * g
                    elif op_name == "clip":
                        # Clip values
                        min_val = op.get("min", -1.0)
                        max_val = op.get("max", 1.0)
                        w = torch.clamp(w, min=min_val, max=max_val)
                    elif op_name == "quantize":
                        # Round to nearest multiple of step
                        step = op.get("step", 0.1)
                        if step > 0:
                            w = torch.round(w / step) * step
                    else:
                        logger.warning(f"DynamicMathAttack: Unknown operator '{op_name}'")
                        
                    poisoned[layer] = w
                except Exception as e:
                    logger.error(f"DynamicMathAttack: Error applying '{op_name}' to layer '{layer}': {e}")
            
            if layers_affected:
                ops_applied.append({"op": op_name, "layers": layers_affected})

        logger.info(f"DynamicMathAttack applied {len(ops_applied)} operations")

        # Track metadata for memory
        self.last_metadata = {
            "operations_applied": ops_applied,
            "total_params": sum(t.numel() for t in weights.values()),
            "affected_per_layer": {layer: weights[layer].numel() for layer in set(l for op in ops_applied for l in op["layers"])}
        }

        return poisoned
