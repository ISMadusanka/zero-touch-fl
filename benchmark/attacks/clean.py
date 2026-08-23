"""``clean`` — the no-attack control row.

Not an attack. The "compromised" clients submit their honest updates unchanged, so
each defense's row is the accuracy it reaches on this round with NOTHING poisoned.

It exists because ``acc_drop`` in every other row is measured against the clean
Phase-1 baseline, which is a fixed number from before the run — fine for RANKING
attacks against each other (they all share it), but it does not answer "how much
did this attack cost *this defense*, this round". That needs the per-defense
counterfactual, and this row is it: read any cell as ``clean_row - attack_row`` for
the same defense.

Because nothing is poisoned, this row's GROUND TRUTH is empty (``poisons = False``),
which the harness honours: the oracle flags nobody and becomes plain FedAvg over the
whole federation — the genuine no-attack aggregate — and every flag a real defense
raises is a false positive, so the row doubles as a clean-round false-alarm rate.
"""
from benchmark.attacks.base import Attack


class Clean(Attack):
    name = "clean"
    citation = "no-attack control (not an attack)"
    #: No client is poisoned, so this row is scored against an EMPTY poisoned set.
    poisons = False

    def craft(self, ctx) -> dict:
        # Cloned, not passed through: every other attack returns fresh tensors and a
        # defense that wrote in place would otherwise corrupt the honest updates the
        # rest of the panel is still going to use this round.
        return {int(cid): {k: v.clone() for k, v in ctx.honest[int(cid)].items()}
                for cid in ctx.poisoned_ids}
