"""Malicious client — applies model poisoning attacks to saved weights."""

import copy
import logging
from core.types import ModelUpdate
from attacks.registry import get_attack

logger = logging.getLogger(__name__)


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

        # Capture attack-specific metadata (e.g. flipped indices from sign_flip)
        attack_metadata = getattr(attack, "last_metadata", {})
        if attack_metadata:
            layer_info = attack_metadata.get("flipped_per_layer", attack_metadata.get("affected_per_layer", {}))
            logger.info(
                f"  Client {self.client_id}: attack_metadata captured — "
                f"k={attack_metadata.get('k', 'N/A')}, "
                f"layers={list(layer_info.keys())}"
            )

        return ModelUpdate(
            client_id=self.client_id,
            weights=poisoned,
            metadata={
                "attack": attack_name,
                "params": attack_params,
                "attack_metadata": attack_metadata,
            },
        )
