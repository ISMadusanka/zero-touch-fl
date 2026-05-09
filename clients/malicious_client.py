"""Malicious client — applies model poisoning attacks to saved weights."""

import copy
from core.types import ModelUpdate
from attacks.registry import get_attack


class MaliciousClient:
    """Applies an attack to its saved (honest) weights before submitting."""

    def __init__(self, client_id: int):
        self.client_id = client_id

    def poison(
        self,
        saved_weights: dict,
        global_weights: dict,
        attack_name: str,
        attack_params: dict,
    ) -> ModelUpdate:
        """Apply the named attack to the saved weights."""
        attack = get_attack(attack_name)
        # Deep copy so saved weights stay pristine across rounds
        weights = copy.deepcopy(saved_weights)
        poisoned = attack.execute(weights, global_weights, **attack_params)
        return ModelUpdate(
            client_id=self.client_id,
            weights=poisoned,
            metadata={"attack": attack_name, "params": attack_params},
        )
