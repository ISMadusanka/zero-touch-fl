"""Sign-flip model poisoning: negates all weight updates."""

import copy
from core.interfaces import BaseAttack
from attacks.registry import register


@register("sign_flip")
class SignFlipAttack(BaseAttack):

    def execute(self, weights: dict, global_weights: dict, **params) -> dict:
        """Flip the sign of the weight delta and optionally scale it."""
        multiplier = params.get("multiplier", 1.0)
        poisoned = {}
        for key in weights:
            delta = weights[key] - global_weights[key]
            # negate and scale: global - (multiplier * delta)
            poisoned[key] = global_weights[key] - (multiplier * delta)
        return poisoned
