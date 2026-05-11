"""Sign-flip model poisoning: negates all weight updates."""

import copy
from core.interfaces import BaseAttack
from attacks.registry import register


@register("sign_flip")
class SignFlipAttack(BaseAttack):

    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Flip and scale the sign of the weight delta (weights - global)."""
        factor = params.get("factor", 1.0)
        poisoned = {}
        for key in weights:
            delta = weights[key] - global_weights[key]
            poisoned[key] = global_weights[key] - (delta * factor*10)
        return poisoned
