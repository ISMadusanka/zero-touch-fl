"""Scaling model poisoning: amplifies the weight delta."""

from core.interfaces import BaseAttack
from attacks.registry import register


@register("scaling")
class ScalingAttack(BaseAttack):

    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Scale the weight delta by `factor`."""
        factor = params.get("factor", 10.0)
        poisoned = {}
        for key in weights:
            delta = weights[key] - global_weights[key]
            poisoned[key] = global_weights[key] + delta * factor
        return poisoned
