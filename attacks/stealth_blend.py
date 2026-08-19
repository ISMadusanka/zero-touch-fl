"""Stealth blending — evade cosine-similarity defenses (FLTrust, etc.).

After the LLM attacker generates poisoned weights via ``apply_plan``, this
module blends the poisoned delta with the original benign delta so the
resulting update stays directionally aligned with honest updates (and therefore
with FLTrust's server-side reference update ``g0``).

    Δ_benign  = w_benign  − w_global    (naturally aligned with g0)
    Δ_poison  = w_poisoned − w_global   (may point away from g0)

    Δ_final   = (1 − α) · Δ_benign  +  α · Δ_poison
    w_final   = w_global + Δ_final

With a small α (0.1–0.3), ``cos(Δ_final, g0) > 0`` because the benign
component dominates the direction, so FLTrust keeps the client instead of
zeroing its trust score.  The α · Δ_poison component still shifts the target
class's neurons; over many rounds these small shifts accumulate.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def stealth_blend(
    global_weights: dict,
    benign_weights: dict,
    poisoned_weights: dict,
    alpha: float = 0.2,
) -> dict:
    """Blend a poisoned update with the benign update for stealth.

    Args:
        global_weights: current global model state_dict.
        benign_weights: the honest client update (before poisoning).
        poisoned_weights: the LLM-generated poisoned state_dict.
        alpha: blend ratio.  0 = fully benign (no attack), 1 = fully poisoned
               (no stealth).  Values in [0.1, 0.3] are the practical sweet spot.

    Returns:
        A new state_dict whose delta from the global model is
        ``(1 − α) · Δ_benign + α · Δ_poison``.
    """
    blended: dict = {}
    for k in global_weights:
        g = global_weights[k].float()
        b = benign_weights[k].float()
        p = poisoned_weights[k].float()

        delta_benign = b - g
        delta_poison = p - g
        delta_final = (1.0 - alpha) * delta_benign + alpha * delta_poison

        blended[k] = (g + delta_final).to(global_weights[k].dtype)
    return blended
