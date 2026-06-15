"""Operator pipeline engine — applies LLM-composed mathematical operator
pipelines to client weights for dynamic poisoning attacks.

The LLM outputs a structured pipeline specifying which operators to apply
to which layers, and this engine executes the pipeline, producing poisoned
weights and detailed metadata for memory/logging.

Registered as ``"math_ops"`` in the attack registry.
"""

import copy
import logging

from core.interfaces import BaseAttack
from attacks.registry import register
from attacks.math_operators import get_operator, available_operators

logger = logging.getLogger(__name__)

# Maximum operators the LLM can chain on a single layer
MAX_OPS_PER_LAYER = 5


@register("math_ops")
class MathOperatorAttack(BaseAttack):
    """Executes an LLM-composed pipeline of mathematical operators.

    The pipeline is a list of per-layer operation specs::

        [
            {
                "layer": "net.2.weight",           # or "*" for all layers
                "ops": [
                    {"op": "scale", "params": {"factor": 2.5}},
                    {"op": "rotate", "params": {"angle": 0.05}},
                ]
            },
            ...
        ]

    Operators are applied in order within each layer specification.
    Multiple specs can target the same layer (they compose sequentially).
    """

    def __init__(self):
        self.last_metadata: dict = {}

    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Apply the operator pipeline to client weights.

        Args:
            weights:        The honest local weights this client would have sent.
            global_weights: The current global model weights.
            params:
                operations (list[dict]): The pipeline specification from the LLM.
                    Each entry has "layer" (str) and "ops" (list[dict]).
        """
        operations = params.get("operations", [])
        if not operations:
            logger.warning("MathOperatorAttack: empty operations pipeline — returning honest weights")
            self.last_metadata = {"error": "empty pipeline"}
            return copy.deepcopy(weights)

        poisoned = copy.deepcopy(weights)
        layer_keys = list(weights.keys())
        total_params = sum(t.numel() for t in weights.values())

        # Track detailed metadata
        pipeline_metadata = []
        layers_affected = set()

        for spec_idx, spec in enumerate(operations):
            target_layer = spec.get("layer", "")
            ops_list = spec.get("ops", [])

            if not ops_list:
                logger.debug(f"Pipeline spec {spec_idx}: no ops for layer '{target_layer}' — skipping")
                continue

            # Enforce max ops per layer
            if len(ops_list) > MAX_OPS_PER_LAYER:
                logger.warning(
                    f"Pipeline spec {spec_idx}: {len(ops_list)} ops exceeds max "
                    f"{MAX_OPS_PER_LAYER} — truncating"
                )
                ops_list = ops_list[:MAX_OPS_PER_LAYER]

            # Resolve target layers ("*" = all layers)
            if target_layer == "*":
                resolved_layers = layer_keys
            else:
                # Support both exact match and partial match
                resolved_layers = [k for k in layer_keys if k == target_layer]
                if not resolved_layers:
                    # Try partial match (e.g., "weight" matches "net.2.weight")
                    resolved_layers = [k for k in layer_keys if target_layer in k]
                if not resolved_layers:
                    logger.warning(
                        f"Pipeline spec {spec_idx}: layer '{target_layer}' not found "
                        f"in model (available: {layer_keys}) — skipping"
                    )
                    continue

            for layer_name in resolved_layers:
                layers_affected.add(layer_name)
                layer_ops_meta = []

                for op_idx, op_spec in enumerate(ops_list):
                    op_name = op_spec.get("op", "")
                    op_params = op_spec.get("params", {})

                    try:
                        op_fn = get_operator(op_name)
                    except ValueError as e:
                        logger.warning(
                            f"Pipeline spec {spec_idx}, op {op_idx}: {e} — skipping"
                        )
                        continue

                    # Special handling for 'align' — inject global weights as reference
                    if op_name == "align" and "reference" not in op_params:
                        op_params["reference"] = global_weights[layer_name]

                    # Apply the operator
                    logger.info(
                        f"  Applying {op_name}({op_params}) to layer '{layer_name}'"
                    )
                    poisoned[layer_name], op_meta = op_fn(
                        poisoned[layer_name], **op_params
                    )
                    layer_ops_meta.append(op_meta)

                pipeline_metadata.append({
                    "layer": layer_name,
                    "ops_applied": layer_ops_meta,
                    "n_params": weights[layer_name].numel(),
                })

        # Summary metadata
        self.last_metadata = {
            "pipeline_type": "math_ops",
            "total_params": total_params,
            "n_specs": len(operations),
            "layers_affected": sorted(layers_affected),
            "n_layers_affected": len(layers_affected),
            "pipeline_detail": pipeline_metadata,
            "available_operators": available_operators(),
        }

        # Log summary
        ops_summary = []
        for pm in pipeline_metadata:
            op_names = [o["op"] for o in pm["ops_applied"]]
            ops_summary.append(f"{pm['layer']}: {' → '.join(op_names)}")
        logger.info(
            f"MathOperatorAttack: applied pipeline to {len(layers_affected)} layers "
            f"({total_params} total params)"
        )
        for line in ops_summary:
            logger.info(f"  {line}")

        return poisoned
