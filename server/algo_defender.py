"""AlgorithmicDefender — the non-LLM server-side defense for the Phase-2 arms race.

The defender LLM is currently DISABLED (``defense.mode: algorithmic`` in
``configs/base.yaml``). In its place the server runs the published defense
algorithms already implemented for the benchmark — **FLTrust**, **DeFL**, **DnC**
and **Multi-Krum** — and uses **one of them per FL round**: normally the one the
training curriculum's current block pins (:meth:`AlgorithmicDefender.select`, see
``rl/curriculum.py``), or, without a curriculum, one drawn per
``defense.selection``. That one algorithm defends the whole round: the clean
counterfactual, every GRPO rollout
scored in that round, and the committed aggregate all go through it, so the
attacker's G candidate plans stay comparable and the round's reward means "how
well did this plan do *against this defense*".

Two things differ from the LLM defender, and they matter:

* **The algorithm produces the aggregate too.** The LLM defender only emitted
  per-client verdicts and ``FedAvgAggregator`` averaged whoever was not flagged.
  FLTrust (trust-weighted, norm-rescaled) and DeFL (Beta-weighted, CLP-gated) are
  fundamentally *re-weighting aggregators* — reducing them to a drop-list would
  throw away the defense. So :meth:`AlgorithmicDefender.run` returns BOTH the
  verdicts (for detection metrics / the attacker's evasion reward) and the new
  global model the defense computed, and the env commits that state directly.
* **Nothing here learns.** With the defender frozen, only the attacker trains;
  ``rl/schedule.py`` runs attacker-only phases.

Statefulness: DeFL carries Beta counts + S(t-1) across rounds and DnC carries a
subsampling RNG. Scoring a candidate rollout must not disturb that memory, so a
non-committing :meth:`run` snapshots and restores it (``Defense.state_snapshot`` /
``state_restore``). Because only one algorithm runs per round, a rotating
algorithm's memory advances only on the rounds it is actually selected — that is
inherent to rotation, and it is why the training curriculum gives each algorithm
a CONTIGUOUS block of rounds (and why ``round_robin`` selection exists as an
alternative to ``random`` when no curriculum is active).

The ground-truth-reading ``oracle`` and the ``llm_defender`` are deliberately not
selectable here (see :data:`ALGORITHMS`).
"""

import logging
import random
from dataclasses import dataclass, field

from core.types import DetectionVerdict

logger = logging.getLogger(__name__)


#: Defenses that may take part in the rotation. All four are genuine, published
#: algorithms that observe only what a real server sees.
ALGORITHMS = ("fltrust", "defl", "dnc", "multikrum")

#: Benchmark defenses that are NOT valid rotation members, with the reason.
_NOT_SELECTABLE = {
    "oracle": "reads the ground-truth poisoned set — an upper bound, not a defense",
    "llm_defender": "IS the defender LLM that this mode replaces",
    "fedavg": "is the no-defense baseline, not a defense",
}


@dataclass
class DefenseOutcome:
    """One algorithm's decision for one set of client updates."""
    algorithm: str
    verdicts: list[DetectionVerdict]
    new_global: dict | None            # the defense's aggregate; None = keep the current global
    info: dict = field(default_factory=dict)


class AlgorithmicDefender:
    """A pool of defense algorithms, one of which defends each round.

    ``defenses`` is an ordered ``{name: Defense}`` mapping (built by
    :func:`build_algorithmic_defender`). ``rng`` is a *dedicated*
    ``random.Random`` so drawing the algorithm never perturbs the env's
    poison/budget stream.
    """

    def __init__(self, defenses: dict, rng: random.Random, selection: str = "random"):
        if not defenses:
            raise ValueError("AlgorithmicDefender needs at least one defense algorithm")
        self._defenses = dict(defenses)
        self._names = list(self._defenses)
        self._rng = rng
        self._selection = str(selection).lower()
        if self._selection not in ("random", "round_robin"):
            raise ValueError(f"defense.selection must be random|round_robin, "
                             f"got {selection!r}")
        self._rr = -1
        self._current = self._names[0]

    # ------------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def current(self) -> str:
        """The algorithm defending the round in progress."""
        return self._current

    def describe(self) -> str:
        return f"{self._selection} over {self._names}"

    def choose(self) -> str:
        """Draw THIS round's algorithm. Call exactly once per FL round, before any
        candidate is scored, so the whole round is defended by the same algorithm."""
        if len(self._names) == 1:
            self._current = self._names[0]
        elif self._selection == "round_robin":
            self._rr = (self._rr + 1) % len(self._names)
            self._current = self._names[self._rr]
        else:
            self._current = self._rng.choice(self._names)
        return self._current

    def select(self, name: str) -> str:
        """Pin THIS round's algorithm explicitly, instead of drawing one.

        Used by the training curriculum (``rl/curriculum.py``), which sweeps the
        pool deterministically — one algorithm per block of rounds — so
        ``defense.selection`` does not apply. Setting ``_current`` here keeps
        :meth:`run` correct when it is called without an explicit ``algorithm``,
        and leaves the ``round_robin`` cursor and the draw RNG untouched so
        switching back mid-run is not skewed by the curriculum's rounds.
        """
        key = str(name).strip().lower()
        if key not in self._defenses:
            raise KeyError(f"unknown defense algorithm {name!r} (have {self._names})")
        self._current = key
        return key

    # ------------------------------------------------------------------
    def run(self, updates, global_weights: dict, *, commit: bool = False,
            algorithm: str | None = None) -> DefenseOutcome:
        """Defend one candidate round.

        ``updates`` are the client submissions (absolute weights, as everywhere in
        this codebase) and ``global_weights`` the CURRENT global model they are
        measured against. Returns the per-client verdicts plus the aggregate the
        defense produced (``None`` when it declined to update the global).

        ``commit=False`` — scoring a rollout or the clean counterfactual — rolls
        back any cross-round state the algorithm mutated, so repeated scoring
        calls within a round are independent and identically defended.
        ``commit=True`` lets that state advance, exactly once per committed round.
        """
        name = algorithm or self._current
        defense = self._defenses.get(name)
        if defense is None:
            raise KeyError(f"unknown defense algorithm {name!r} (have {self._names})")

        defense.sync_global(global_weights)
        snapshot = None if commit else defense.state_snapshot()
        try:
            # ``poisoned_ids`` is ground truth and is deliberately NOT passed: no
            # selectable algorithm may read it (see _NOT_SELECTABLE['oracle']).
            result = defense.step(updates, set())
        finally:
            if snapshot is not None:
                defense.state_restore(snapshot)

        info = dict(result.info or {})
        info["algorithm"] = name
        info["committed"] = bool(commit)
        return DefenseOutcome(name, result.verdicts, result.new_global, info)


# ---------------------------------------------------------------------------
# Construction from configs/base.yaml
# ---------------------------------------------------------------------------

def configured_algorithms(cfg: dict) -> list[str]:
    """The validated ``defense.algorithms`` list for this config (order preserved)."""
    dcfg = (cfg.get("defense") or {})
    raw = dcfg.get("algorithms") or list(ALGORITHMS)
    names, seen = [], set()
    for item in raw:
        name = str(item).strip().lower()
        if not name or name in seen:
            continue
        if name in _NOT_SELECTABLE:
            raise ValueError(f"defense.algorithms: '{name}' {_NOT_SELECTABLE[name]}; "
                             f"pick from {list(ALGORITHMS)}")
        if name not in ALGORITHMS:
            raise ValueError(f"defense.algorithms: unknown algorithm '{name}' "
                             f"(available: {list(ALGORITHMS)})")
        names.append(name)
        seen.add(name)
    if not names:
        raise ValueError("defense.algorithms is empty — list at least one of "
                         f"{list(ALGORITHMS)}, or set defense.mode: llm")
    return names


def defense_mode(cfg: dict) -> str:
    """``'algorithmic'`` (default) or ``'llm'`` — who defends in Phase 2."""
    mode = str(((cfg.get("defense") or {}).get("mode") or "algorithmic")).lower()
    if mode not in ("algorithmic", "llm"):
        raise ValueError(f"defense.mode must be algorithmic|llm, got {mode!r}")
    return mode


def resolve_root_epochs(configured, root_batches: int, client_iterations: int | None) -> int:
    """FLTrust's server-side local epochs R_l over the root set.

    ``configured is None`` means "match an honest client's local training", which is
    what the paper assumes and what this codebase got wrong. FLTrust rescales EVERY
    accepted client delta to ``||g0||`` (Eq. 3) and applies ``w <- w + eta*g``, so the
    global model moves by about ``||g0||`` per round: the server's reference update
    sets the whole system's learning rate. The paper specifies R_l in *iterations*,
    but this config expresses local training in *epochs*, and the root set is far
    smaller than a client shard — 100 examples at batch 64 is 2 iterations/epoch
    against a client's ~40. At ``root_epochs: 1`` the root update therefore took ~2
    SGD steps against a client's ~120, making ``||g0||`` roughly 60x too small.

    The consequence was not a slightly slow defense but a broken training signal:
    with the aggregate pinned to a near-zero magnitude, the clean counterfactual and
    every poisoned rollout landed at essentially the same accuracy, so the damage
    term was ~0 for all G candidates and the whole group was degenerate. Roughly a
    quarter of all rounds (FLTrust's share of the rotation) produced no gradient, and
    the configured ``target_accuracy_drop`` of 0.05-0.30 was not reachable in
    principle.

    Matching ITERATIONS instead of epochs fixes the scale while leaving the algorithm
    and the "server holds only a few clean examples" premise untouched. Pass an
    explicit integer to override.
    """
    if configured is not None:
        return max(1, int(configured))
    if not client_iterations or root_batches <= 0:
        return 1
    return max(1, round(int(client_iterations) / int(root_batches)))


def build_algorithmic_defender(cfg: dict, *, root_loader=None,
                               root_loader_factory=None,
                               seed: int | None = None,
                               client_iterations: int | None = None) -> AlgorithmicDefender | None:
    """Build the round-rotating defender described by ``cfg`` (the whole base config).

    Returns ``None`` when ``defense.mode`` is ``llm`` — the caller then keeps the
    defender-LLM path. ``root_loader`` (or ``root_loader_factory``, called only if
    needed) supplies FLTrust's clean root dataset.

    ``client_iterations`` is how many SGD steps an honest client takes per round
    (``local_epochs * batches_per_client``). It is used only to size FLTrust's root
    fine-tuning when ``defense.fltrust.root_epochs`` is null — see
    :func:`resolve_root_epochs`.
    """
    if defense_mode(cfg) != "algorithmic":
        return None

    dcfg = (cfg.get("defense") or {})
    fl = cfg.get("fl", {})
    attack = cfg.get("attack", {})
    names = configured_algorithms(cfg)

    if "fltrust" in names and root_loader is None:
        if root_loader_factory is None:
            raise ValueError("defense.algorithms includes 'fltrust' but no clean root "
                             "dataset was supplied (root_loader/root_loader_factory)")
        root_loader = root_loader_factory()

    # DnC / Multi-Krum need an ASSUMED upper bound on the number of malicious
    # clients (a hyperparameter, never per-round truth). Default it to the attack
    # budget the config actually grants, clamped to keep an honest majority.
    n_clients = int(fl.get("n_clients", 1))
    default_byz = max(1, min(int(attack.get("max_poison_clients", 1)),
                             max(1, (n_clients - 1) // 2)))
    assumed_byz = dcfg.get("assumed_byzantine")
    assumed_byz = default_byz if assumed_byz is None else max(1, int(assumed_byz))

    ft = dcfg.get("fltrust") or {}
    dl = dcfg.get("defl") or {}
    dn = dcfg.get("dnc") or {}
    mk = dcfg.get("multikrum") or {}

    seed = int(fl.get("poison_seed", 0)) if seed is None else int(seed)
    dnc_seed = seed if dn.get("seed") is None else int(dn["seed"])

    # R_l: null (the default) sizes the root fine-tuning to match an honest client's
    # ITERATION count, so FLTrust's ||g0|| — which sets the whole system's step size —
    # is not ~60x too small. See resolve_root_epochs.
    root_epochs = resolve_root_epochs(
        ft.get("root_epochs"),
        root_batches=(len(root_loader) if root_loader is not None else 0),
        client_iterations=client_iterations,
    )
    if "fltrust" in names:
        logger.info(
            f"FLTrust root fine-tuning: root_epochs={root_epochs} over "
            f"{len(root_loader) if root_loader is not None else 0} batch(es) "
            f"= ~{root_epochs * (len(root_loader) if root_loader is not None else 0)} "
            f"SGD iterations (honest client: ~{client_iterations or 'unknown'})"
        )

    from benchmark.defenses import build_defenses
    defenses = build_defenses(
        names,
        device=fl.get("device", "cpu"),
        root_loader=root_loader,
        root_lr=float(ft.get("root_lr") or fl.get("lr", 0.002)),
        root_epochs=root_epochs,
        eta=float(ft.get("eta", 1.0)),
        defl_delta=float(dl.get("delta", 0.05)),
        defl_tau=float(dl.get("tau", 2.5)),
        dnc_num_byzantine=assumed_byz,
        dnc_c=float(dn.get("c", 1.0)),
        dnc_niters=int(dn.get("niters", 1)),
        dnc_sub_dim=int(dn.get("sub_dim", 10000)),
        dnc_seed=dnc_seed,
        multikrum_num_byzantine=assumed_byz,
        multikrum_m=mk.get("m"),
    )

    selection = str(dcfg.get("selection", "random")).lower()
    # A dedicated stream: the algorithm draw must not shift the env's poison /
    # budget / target sampling, so the two stay independently reproducible.
    rng_seed = dcfg.get("seed")
    rng = random.Random(seed if rng_seed is None else int(rng_seed))

    defender = AlgorithmicDefender(defenses, rng, selection=selection)
    logger.info(
        f"Defender LLM DISABLED — algorithmic defense active: {defender.describe()} "
        f"(assumed_byzantine={assumed_byz}"
        + (f", fltrust_root_batches={len(root_loader)}" if root_loader is not None else "")
        + ")"
    )
    return defender
