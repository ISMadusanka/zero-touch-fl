"""Malicious client — applies model poisoning attacks to saved weights.

Supports both the new LLM-driven mathematical operator pipelines
(``math_ops``) and legacy predefined attacks for backward compatibility.
"""

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
        """Apply the named attack to the saved weights.

        For ``math_ops`` attacks, ``attack_params`` contains an
        ``operations`` key with the full operator pipeline specification
        from the LLM attacker agent.
        """
        attack = get_attack(attack_name)
        # Deep copy so saved weights stay pristine across rounds
        weights = copy.deepcopy(saved_weights)
        poisoned = attack.execute(weights, global_weights, **attack_params)

        # Capture attack-specific metadata
        attack_metadata = getattr(attack, "last_metadata", {})
        if attack_metadata:
            # Handle both old-style (flipped_per_layer) and new-style (layers_affected) metadata
            if "layers_affected" in attack_metadata:
                # New math_ops format
                logger.info(
                    f"  Client {self.client_id}: pipeline metadata captured — "
                    f"layers={attack_metadata.get('layers_affected', [])}, "
                    f"n_specs={attack_metadata.get('n_specs', 0)}"
                )
            else:
                # Legacy format
                layer_info = attack_metadata.get(
                    "flipped_per_layer",
                    attack_metadata.get("affected_per_layer", {}),
                )
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
