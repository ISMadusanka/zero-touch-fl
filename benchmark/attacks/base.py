"""Attack interface for the benchmark + the flat-vector helpers every attack shares.

A benchmark ``Attack`` turns one round's HONEST client updates into the poisoned
updates a fixed set of compromised clients submits. Every attack in the panel is
handed the *same* round: the same honest updates, the same reference model, and —
critically — the **same poisoned client ids** (see ``benchmark/harness.py``), so
the only thing that varies between rows of the report is the attack itself.

Two conventions the implementations rely on:

* **Deltas, not weights.** Clients submit ABSOLUTE weights (``w_i``), but every
  published attack is written in terms of the local update / gradient
  ``Δ_i = w_i − g`` against the model the round started from. Attacks therefore
  work in flattened delta space and the harness converts back:
  ``w_mal = g + Δ_mal``. ``g`` is ``env.global_weights``, the model every client
  in this round actually trained from, so the deltas are the real local updates
  rather than a difference against some defense's drifted global.
* **Knowledge.** ``AttackContext.known_ids`` is the set of clients whose honest
  updates the adversary may look at. ``partial`` (the default) = exactly the
  compromised clients, which is the knowledge the trained LLM attacker gets;
  ``full`` = every client in the federation, the omniscient setting most of the
  papers state their attack in. The choice is global to a run so all baselines
  are compared under one assumption.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# state_dict <-> flat vector
# ---------------------------------------------------------------------------

def float_keys(state: dict) -> list:
    """Ordered state_dict keys that take part in the arithmetic.

    Only floating-point tensors: an integer buffer (a BatchNorm
    ``num_batches_tracked`` counter, say) has no meaningful "add 0.3 of a
    perturbation", and averaging one is already outside what these attacks
    describe. The current model (``model/nidd_net.py``) is deliberately BN-free so
    this is every key, but the guard keeps the attacks correct if that changes —
    non-float entries are passed through untouched by :func:`unflatten_like`.
    """
    return [k for k, v in state.items() if v.is_floating_point()]


def flatten(state: dict, keys: list):
    """Concatenate ``keys`` of one state_dict into a 1-D float32 tensor."""
    import torch
    return torch.cat([state[k].reshape(-1).to(torch.float32) for k in keys])


def unflatten_like(vec, reference: dict, keys: list) -> dict:
    """Inverse of :func:`flatten`: a NEW state_dict shaped like ``reference``.

    Keys outside ``keys`` are cloned from ``reference`` unchanged. Every tensor is
    freshly allocated — the honest updates are shared by every attack and every
    defense in the round, so an attack that wrote into them in place would corrupt
    the whole panel.
    """
    out, off = {}, 0
    for k, ref in reference.items():
        if k in keys:
            n = ref.numel()
            out[k] = vec[off:off + n].reshape(ref.shape).to(ref.dtype).clone()
            off += n
        else:
            out[k] = ref.clone()
    return out


def stack_deltas(honest: dict, global_weights: dict, keys: list, ids):
    """``len(ids) x d`` matrix of local updates ``Δ_i = w_i − g``, in ``ids`` order."""
    import torch
    g = flatten(global_weights, keys)
    return torch.stack([flatten(honest[int(cid)], keys) - g for cid in ids])


# ---------------------------------------------------------------------------
# Round context
# ---------------------------------------------------------------------------

@dataclass
class AttackContext:
    """Everything an attack may look at for one round.

    ``poisoned_ids`` is decided ONCE per round by the harness and shared by every
    attack in the panel — that is what makes the rows comparable, and it is why an
    attack must return exactly these ids and no others.
    """
    round_num: int
    global_weights: dict                    # g: the model every client trained from
    honest: dict                            # {cid: honest absolute state_dict}, ALL clients
    poisoned_ids: list                      # the exact clients to poison this round
    known_ids: list                         # clients whose honest updates the adversary sees
    pool_ids: list                          # the compromised (controllable) clients
    n_clients: int                          # federation size
    goal: dict = field(default_factory=dict)
    reference_accuracy: float = 0.0         # undefended accuracy, one round stale
    keys: list = field(default_factory=list)

    @property
    def n_malicious(self) -> int:
        return len(self.poisoned_ids)

    def known_deltas(self):
        """``len(known_ids) x d`` matrix of the honest updates the adversary sees."""
        return stack_deltas(self.honest, self.global_weights, self.keys, self.known_ids)

    def deltas_for(self, ids):
        return stack_deltas(self.honest, self.global_weights, self.keys, ids)

    def to_states(self, deltas_by_cid: dict) -> dict:
        """Flat malicious deltas -> the absolute state_dicts the clients submit."""
        g = flatten(self.global_weights, self.keys)
        return {int(cid): unflatten_like(g + d, self.global_weights, self.keys)
                for cid, d in deltas_by_cid.items()}


# ---------------------------------------------------------------------------
# Attack interface
# ---------------------------------------------------------------------------

class Attack(ABC):
    """One attack, evaluated against the whole defense panel."""

    #: registry name, also the report's row label
    name: str = "attack"
    #: short citation shown in the report legend
    citation: str = ""
    #: True for the trained LLM policy (the system under test), False for baselines
    is_llm: bool = False
    #: False for a control row that poisons nothing (``clean``). The harness then
    #: scores that row against an EMPTY ground-truth poisoned set, so the oracle
    #: flags nobody and every flag a real defense raises is a false positive.
    poisons: bool = True

    def reset(self) -> None:
        """Clear any cross-round state. Called once before the run."""
        return None

    @abstractmethod
    def craft(self, ctx: AttackContext) -> dict:
        """Return ``{client_id: poisoned_state_dict}`` for exactly ``ctx.poisoned_ids``.

        Implementations must not mutate anything reachable from ``ctx``.
        """
        raise NotImplementedError


class DeltaAttack(Attack):
    """Convenience base for the model-poisoning attacks: craft in flat delta space."""

    def craft(self, ctx: AttackContext) -> dict:
        return ctx.to_states(self.craft_deltas(ctx))

    @abstractmethod
    def craft_deltas(self, ctx: AttackContext) -> dict:
        """Return ``{client_id: flat malicious delta}`` for exactly ``ctx.poisoned_ids``."""
        raise NotImplementedError


def broadcast(ids, vec) -> dict:
    """Every compromised client submits the same malicious update.

    This is what LIE / IPM / Min-Max / Min-Sum / Fang-Krum all specify — the
    colluding clients agree on one vector — so it gets one shared helper rather
    than being re-derived in each module. Each client gets its own tensor so a
    later in-place write cannot alias across clients.
    """
    return {int(cid): vec.clone() for cid in ids}
