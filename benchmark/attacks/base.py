"""Attack interface for the benchmark — published model-poisoning baselines.

Mirrors ``benchmark/defenses/``: a registry of attacks with a uniform interface
so each can be run against the SAME defense panel as the trained attacker LLM,
for a head-to-head comparison (hold the defense fixed, vary only the attacker).

Threat model — identical to the LLM attacker's (a **partial insider**). Every
attack here sees ONLY the global model ``G`` and its own controllable clients'
benign weights; it never sees the honest majority's updates. So all benign
statistics (the update mean ``mu`` and coordinate-wise std ``sigma``) are
estimated over the **compromised clients alone**, exactly the information the LLM
gets in its prompt. The federation size ``n`` is treated as a known structural
constant (needed by LIE's ``z`` and IPM's auto scale), not as knowledge of other
clients' data.

An ``Attack`` CRAFTS malicious client weights directly (it does NOT go through
the operator DSL the LLM composes): given the controllable pool's benign
state_dicts, the global model, and an exact poison quota ``budget``, it returns
poisoned state_dicts for exactly ``budget`` chosen clients. The server sees each
as a full weight vector ``W``; FedAvg averages the accepted ones, and because
``W = G + Delta`` poisoning ``W`` is equivalent to injecting a crafted update
``Delta`` — which is how these attacks are defined in their papers.

All attacks are **untargeted** (goal: degrade global accuracy), matching the
project's ``untargeted_degrade`` goal.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger("benchmark.attacks")

_SAFE_ABS = 1e6   # numerical guard only (scrub inf/nan) — NOT an attack-strength clamp


class BenignStats:
    """Per-round benign-update statistics estimated over the compromised clients.

    Given ``pool_benign = {cid: benign_state_dict}`` and the global model, this
    holds the honest updates ``Delta_i = W_i - G`` for the controllable clients
    and their coordinate-wise mean/std — the only signal a partial insider has to
    craft an attack. It also flattens/unflattens between the per-key state_dict
    layout and a single vector (needed by the distance-based attacks).
    """

    def __init__(self, pool_benign: dict[int, dict], global_sd: dict):
        if not pool_benign:
            raise ValueError("BenignStats needs a non-empty controllable pool")
        self.cids: list[int] = list(pool_benign.keys())
        self.global_sd = global_sd
        self.keys: list[str] = list(global_sd.keys())
        self._ref = pool_benign[self.cids[0]]      # dtype/shape template

        # Per-key stacked honest updates: delta_by_key[k] has shape (f, *param_shape).
        self.delta_by_key: dict[str, torch.Tensor] = {
            k: torch.stack([pool_benign[c][k].float() - global_sd[k].float()
                            for c in self.cids])
            for k in self.keys
        }
        self.mean: dict[str, torch.Tensor] = {k: v.mean(0) for k, v in self.delta_by_key.items()}
        # Population std (unbiased=False): 0 when only one client is controlled.
        self.std: dict[str, torch.Tensor] = {
            k: v.std(0, unbiased=False) for k, v in self.delta_by_key.items()}

    @property
    def n_benign(self) -> int:
        return len(self.cids)

    # -- flat <-> per-key -------------------------------------------------
    def flat_deltas(self) -> torch.Tensor:
        """Honest updates as a dense matrix, shape ``(f, d)``."""
        return torch.stack([
            torch.cat([self.delta_by_key[k][i].reshape(-1) for k in self.keys])
            for i in range(self.n_benign)
        ])

    def flat_mean(self) -> torch.Tensor:
        return torch.cat([self.mean[k].reshape(-1) for k in self.keys])

    def flat_std(self) -> torch.Tensor:
        return torch.cat([self.std[k].reshape(-1) for k in self.keys])

    def unflatten_delta(self, vec: torch.Tensor) -> dict:
        """Turn a flat update vector back into a per-key delta state_dict."""
        out, i = {}, 0
        for k in self.keys:
            n = self.global_sd[k].numel()
            out[k] = vec[i:i + n].reshape(self.global_sd[k].shape)
            i += n
        return out

    # -- delta -> deliverable weights ------------------------------------
    def weights_from_flat_delta(self, vec: torch.Tensor) -> dict:
        """``G + delta`` as a deliverable state_dict (dtype-matched, scrubbed)."""
        return self.weights_from_delta(self.unflatten_delta(vec))

    def weights_from_delta(self, delta_by_key: dict) -> dict:
        """``G + delta`` as a deliverable state_dict (per-key delta input)."""
        out = {}
        for k in self.keys:
            w = self.global_sd[k].float() + delta_by_key[k].float()
            w = torch.nan_to_num(w, nan=0.0, posinf=_SAFE_ABS, neginf=-_SAFE_ABS)
            out[k] = w.to(self._ref[k].dtype)
        return out


def unit(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vec / (vec.norm() + eps)


class Attack(ABC):
    """One untargeted model-poisoning attack.

    ``craft`` returns ``{cid: poisoned_state_dict}`` for exactly ``min(budget,
    pool_size)`` chosen clients. Scripted attacks have no learned client
    selection, so they take the first ``budget`` clients of the pool (the same
    convention as ``rl/baseline.py``); in the benchmark the pool is already
    widened to exactly the quota, so this poisons clients ``0..budget-1``.
    """

    name: str = "attack"
    #: True when all malicious clients send the SAME crafted update (collusion,
    #: the canonical form for optimization attacks). False when each malicious
    #: client manipulates its own honest update independently (the trivial
    #: Byzantine baselines). Purely informational — used in logs.
    colludes: bool = True

    def choose_ids(self, pool_benign: dict[int, dict], budget: int) -> list[int]:
        ids = list(pool_benign.keys())
        budget = max(1, min(int(budget), len(ids)))
        return ids[:budget]

    @abstractmethod
    def craft(self, pool_benign: dict[int, dict], global_sd: dict, budget: int,
              gen: torch.Generator) -> dict[int, dict]:
        raise NotImplementedError


class ScriptedAttacker:
    """Adapts an :class:`Attack` (pure crafting) to the harness's ``act`` protocol.

    The benchmark harness treats the attacker as a black box that, each round,
    turns the round context into ``(poisoned_by_client, chosen_ids,
    n_malformed)``. A scripted attack always fills the exact quota and never
    emits a malformed plan, so ``n_malformed`` is always 0. The RNG is a single
    seeded ``torch.Generator`` so a run is reproducible.
    """

    is_scripted = True

    def __init__(self, attack: Attack, seed: int = 0):
        self.attack = attack
        self.name = attack.name
        self._gen = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)

    def act(self, env, ctx):
        poisoned = self.attack.craft(ctx.pool_benign, env.global_weights,
                                     ctx.budget, self._gen)
        chosen_ids = list(poisoned.keys())
        return poisoned, chosen_ids, 0
