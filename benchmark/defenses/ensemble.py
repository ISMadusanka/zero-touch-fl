"""Ensemble defense — every classical robust-aggregation algorithm, voting together.

This is the defense used when the defender LLM is switched off
(``python main.py --env linux --freeze defender``), and it is also selectable as a
single panel entry in the benchmark (``--defenses fedavg,ensemble``).

Instead of picking one Byzantine-robust rule, it runs SEVERAL of them against the
SAME global model and the SAME client updates and combines their per-client
decisions by vote:

    reject client i  <=>  at least ``min_votes`` members flagged it

Members ship as FLTrust + Multi-Krum + DnC + DeFL, which is deliberately a
mixture of *different* detection principles, so an attack that is invisible to
one is usually loud to another:

  * ``fltrust``    — DIRECTION: cosine of the client's delta against a server-side
                     reference update trained on a small clean root set.
  * ``multikrum``  — DISTANCE: how far the update sits from the bulk of the others.
  * ``dnc``        — SPECTRAL: projection on the top principal direction of the
                     centred updates.
  * ``defl``       — PER-LAYER: robust median/MAD outlier vote over each layer's
                     gradient-norm statistic.

Vote rule (``vote``): ``"majority"`` (default, ``ceil(n/2)``), ``"any"`` (union —
maximum recall, worst false-positive rate), ``"all"`` (unanimous — minimum false
positives, easiest to slip past), or an explicit integer. Majority is the default
because two of the members (Multi-Krum, DnC) are *budgeted* filters that drop a
fixed number of clients EVERY round whether or not anyone is malicious: under
``"any"`` their standing quota alone would evict honest clients from the average
every round.

**What "defend" means here.** A member's own aggregation rule (FLTrust's
trust-weighted rescaling, DeFL's Beta weighting) is not what the ensemble
combines — only its ACCEPT/REJECT read-out is. The ensemble then aggregates the
survivors with plain FedAvg, exactly like the rest of this codebase does with the
defender LLM's verdicts. That keeps one aggregation rule across the whole system,
so the round's clean counterfactual, the attacker's reward and the win-gate all
keep measuring the same thing, and swapping the LLM defender for this one changes
*who decides*, not *how the server averages*.

Cross-round member state (DeFL's Beta counts and its previous total FGNV, DnC's
subsampling RNG) is snapshotted and restored around a NON-committing detection,
so scoring the ``G`` candidate attacks of a GRPO round cannot corrupt the defense
the committed round then faces.
"""
import copy
from contextlib import contextmanager

from core.types import DetectionVerdict

from benchmark.defenses.base import Defense, StepResult

# The classical (non-LLM, non-oracle) defenses combined by default.
DEFAULT_MEMBERS = ("fltrust", "multikrum", "dnc", "defl")

# Members that must never be part of the ensemble: ``oracle`` reads the
# ground-truth poisoned set (it would make the arms race meaningless),
# ``llm_defender`` is precisely the component ``--freeze defender`` removes, and
# ``ensemble`` would recurse.
FORBIDDEN_MEMBERS = ("oracle", "llm_defender", "ensemble")

# Mutable attributes a member carries ACROSS rounds. Restored after a scoring
# (non-committing) detection so a rollout cannot advance the defense's memory.
_MUTABLE_ATTRS = ("_beta", "_prev_total_fgnv", "_rng")


def resolve_min_votes(vote, n_members: int) -> int:
    """How many members must flag a client before the ensemble rejects it.

    ``"majority"`` -> ``ceil(n/2)``, ``"any"`` -> 1, ``"all"``/``"unanimous"`` -> n,
    an int (or int-like string) -> itself, clamped to ``[1, n]``.
    """
    n = max(1, int(n_members))
    if isinstance(vote, str):
        key = vote.strip().lower()
        if key in ("", "majority", "default"):
            return (n + 1) // 2
        if key == "any":
            return 1
        if key in ("all", "unanimous"):
            return n
        try:
            vote = int(key)
        except ValueError:
            raise ValueError(
                f"unknown vote rule {vote!r}; use majority | any | all | <int>"
            ) from None
    return max(1, min(int(vote), n))


def combine_votes(member_flags: dict, client_ids: list, min_votes: int) -> list:
    """Combine per-member flags into one verdict per client.

    ``member_flags`` maps a member name to the set of client ids IT flagged.
    A client is rejected when at least ``min_votes`` members flagged it.

    ``confidence`` is the agreement behind the decision, in [0, 1]: for a rejected
    client the fraction of members that flagged it, for an accepted one the
    fraction that did not. It is not decoration — the attacker's stealth reward
    reads it (``rl.rewards._soft_malicious_prob``), so a plan that fools three of
    four members scores strictly better than one that fools none, which is the
    continuous signal GRPO needs to make progress against a hard 0/1 defense.
    """
    n = max(1, len(member_flags))
    verdicts = []
    for cid in client_ids:
        voters = sorted(name for name, flagged in member_flags.items() if cid in flagged)
        votes = len(voters)
        rejected = votes >= min_votes
        confidence = votes / n if rejected else 1.0 - votes / n
        verdicts.append(DetectionVerdict(
            client_id=cid,
            is_suspicious=rejected,
            confidence=float(confidence),
            reason=f"{votes}/{n} votes" + (f": {','.join(voters)}" if voters else ""),
        ))
    return verdicts


@contextmanager
def _member_scratch(member: Defense, global_weights: dict, *, freeze: bool):
    """Run one member against ``global_weights`` as its current global model.

    ``freeze`` snapshots and restores the member's cross-round state so a
    non-committing (rollout-scoring) detection leaves the defense exactly as it
    was. The member's ``_global`` is always restored — the ensemble supplies the
    reference model on every call, so a member's own evolving global is scratch.
    """
    saved = ({a: copy.deepcopy(getattr(member, a)) for a in _MUTABLE_ATTRS
              if hasattr(member, a)} if freeze else {})
    previous_global = member._global
    member._global = global_weights
    try:
        yield
    finally:
        member._global = previous_global
        for attr, value in saved.items():
            setattr(member, attr, value)


class EnsembleDefense(Defense):
    """Several classical defenses deciding together (see the module docstring)."""

    name = "ensemble"

    def __init__(self, members: dict, vote="majority", device: str = "cpu"):
        super().__init__(device)
        members = dict(members)
        if not members:
            raise ValueError("EnsembleDefense needs at least one member defense")
        bad = [n for n in members if n in FORBIDDEN_MEMBERS]
        if bad:
            raise ValueError(f"ensemble members may not include {bad} "
                             f"(forbidden: {list(FORBIDDEN_MEMBERS)})")
        llm = [n for n, m in members.items() if getattr(m, "requires_llm", False)]
        if llm:
            raise ValueError(f"ensemble members must be non-LLM defenses, got {llm}")
        self.members = members
        self.vote = vote
        self.min_votes = resolve_min_votes(vote, len(members))
        # Lazy: keeps this module importable without torch (server.aggregation
        # imports torch), matching the other defense modules.
        from server.aggregation import FedAvgAggregator
        self._agg = FedAvgAggregator()

    def describe(self) -> str:
        return (f"ensemble[{','.join(self.members)}] "
                f"min_votes={self.min_votes}/{len(self.members)} (vote={self.vote})")

    def reset(self, init_global):
        super().reset(init_global)
        for member in self.members.values():
            member.reset(init_global)

    # ------------------------------------------------------------------
    def detect(self, updates, global_weights, *, advance_state: bool = False):
        """Per-client verdicts for ``updates`` measured against ``global_weights``.

        Returns ``(verdicts, info)``. ``advance_state=False`` (the default) makes
        this a pure function of its inputs — use it to score candidate attacks;
        pass ``True`` exactly once per round, for the committed updates, so the
        members' cross-round memory advances by one round like the FL state does.
        """
        client_ids = [u.client_id for u in updates]
        member_flags: dict[str, set] = {}
        for name, member in self.members.items():
            with _member_scratch(member, global_weights, freeze=not advance_state):
                result = member.step(updates, set())
            member_flags[name] = {v.client_id for v in result.verdicts if v.is_suspicious}

        verdicts = combine_votes(member_flags, client_ids, self.min_votes)
        info = {
            "min_votes": self.min_votes,
            "n_members": len(self.members),
            "per_member": {n: sorted(f) for n, f in member_flags.items()},
            "rejected": sorted(v.client_id for v in verdicts if v.is_suspicious),
        }
        return verdicts, info

    def step(self, updates, poisoned_ids) -> StepResult:
        verdicts, info = self.detect(updates, self._global, advance_state=True)
        new_global = self._agg.aggregate(updates, verdicts)
        if new_global is not None:
            self._global = new_global
        return StepResult(new_global, verdicts, info=info)
