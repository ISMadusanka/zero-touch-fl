"""Attack registry for the benchmark — published untargeted model-poisoning
baselines to compare head-to-head against the trained attacker LLM.

Attack classes import torch, so they are imported LAZILY inside
``build_attacker``; importing this package stays cheap and torch-free (so
``--help`` works without torch, mirroring ``benchmark/defenses``).

``"llm"`` is the sentinel for the trained attacker (the harness's default path);
``build_attacker("llm", ...)`` returns ``None`` so the caller uses the LLM
pipeline instead of a :class:`ScriptedAttacker`.
"""
from __future__ import annotations

#: Scripted (non-LLM) attackers, in report order. "llm" is handled by the caller.
AVAILABLE_ATTACKS = [
    "noise", "sign_flip", "scaling",        # trivial Byzantine baselines
    "lie", "ipm", "min_max", "min_sum", "fang",  # optimization / directed attacks
]


def build_attacker(
    name: str,
    *,
    seed: int = 0,
    n_clients: int | None = None,
    attack_scale: float | None = None,
    lie_z: float | None = None,
    ipm_eps: float | None = None,
    attack_dev: str = "sign",
    fang_lambda: float = 3.0,
):
    """Build a :class:`ScriptedAttacker`, or ``None`` for the ``"llm"`` sentinel.

    ``attack_scale`` overrides the primary magnitude knob of the trivial
    baselines (``sigma`` for noise, ``factor`` for sign_flip/scaling). ``lie_z``
    and ``ipm_eps`` pin those attacks' parameters (else auto-derived from
    ``n_clients`` and the per-round quota). ``attack_dev`` selects the Min-Max /
    Min-Sum perturbation direction (``sign`` | ``std`` | ``unit``).
    """
    if name == "llm":
        return None

    from benchmark.attacks.base import ScriptedAttacker
    from benchmark.attacks.simple import NoiseAttack, SignFlipAttack, ScalingAttack
    from benchmark.attacks.lie import LIEAttack
    from benchmark.attacks.ipm import IPMAttack
    from benchmark.attacks.min_max import MinMaxAttack, MinSumAttack
    from benchmark.attacks.fang import FangAttack

    if name == "noise":
        attack = NoiseAttack(sigma=attack_scale if attack_scale is not None else 1.0)
    elif name == "sign_flip":
        attack = SignFlipAttack(factor=attack_scale if attack_scale is not None else 1.0)
    elif name == "scaling":
        attack = ScalingAttack(factor=attack_scale if attack_scale is not None else 10.0)
    elif name == "lie":
        attack = LIEAttack(z=lie_z, n_clients=n_clients)
    elif name == "ipm":
        attack = IPMAttack(epsilon=ipm_eps, n_clients=n_clients)
    elif name == "min_max":
        attack = MinMaxAttack(direction=attack_dev)
    elif name == "min_sum":
        attack = MinSumAttack(direction=attack_dev)
    elif name == "fang":
        attack = FangAttack(lam=fang_lambda)
    else:
        raise ValueError(f"unknown attacker '{name}' "
                         f"(available: llm, {', '.join(AVAILABLE_ATTACKS)})")
    return ScriptedAttacker(attack, seed=seed)
