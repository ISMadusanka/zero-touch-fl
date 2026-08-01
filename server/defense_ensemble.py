"""Algorithmic defense panel — the defender side with the defender LLM OFF.

Used by ``python main.py --freeze defender``. The defender LLM is deactivated and
the server instead judges each round with a robust-FL algorithm drawn from:

    FLTrust (NDSS'21) · Multi-Krum (NeurIPS'17) · DnC (NDSS'21) · DeFL (AAAI'23)

Each algorithm reports its own per-client accept/reject decision (the same
``DetectionVerdict`` the LLM defender emits, so nothing downstream changes).

Two modes (``defense.mode``)
---------------------------
* ``single`` (**default**) — ONE algorithm judges the round; ``defense.selection``
  decides which (``rotate`` round-robin, ``random``, or ``fixed`` = the first
  configured). The choice is made once per round in :meth:`begin_round` and held
  for the clean-counterfactual evaluation, all ``G`` rollout scorings, and the
  commit, so every accuracy in a round is measured under the same defense.

* ``union`` — the legacy behaviour: every algorithm judges the round and the
  REJECTIONS ARE UNIONED, so a client is dropped if ANY algorithm flags it.

**Why ``single`` is the default.** Unioning four aggregators is far more
trigger-happy than any of them alone. Multi-Krum and DnC each drop a fixed number
of clients every round *by construction*, DeFL always flags at least one, and
FLTrust zeroes the trust of any client whose update points away from its root
update — which, under a strongly non-IID split, is most of the honest federation.
Measured on this repo's own Phase-1 checkpoint (20 clients, ``noniid_bias: 0.5``),
a CLEAN round with no attacker at all produced:

    fltrust 12/20 · multikrum 1/20 · dnc 1/20 · defl 2/20  ->  UNION 14/20

i.e. FedAvg ran on 6 of 20 honest clients and the clean accuracy fell 0.782 ->
0.757 before any attack existed. Two things follow, and both break training:

1. There is no gap left to attack through. An attack must evade all four at once
   while overcoming 20x FedAvg dilution; no plan in the operator DSL can do both,
   so the attacker's damage term is unreachable and the policy converges to the
   only reachable gradient — evade everything, achieve nothing.
2. The damage term stops measuring the attack. Because the union's membership
   churns from round to round, the accuracy swing caused by *which honest clients
   got dropped* (sd ~0.012) dwarfs the swing caused by the attack itself
   (~0.0003). GRPO is then climbing defense noise.

Judging with one algorithm at a time keeps every defense in the curriculum while
leaving a real, learnable frontier: a clean round scores 0.782/0.774/0.774/0.783
under fltrust/multikrum/dnc/defl respectively — within 0.01 of the undefended
baseline instead of 0.025 below it.

Other details that make the panel usable as a training-time opponent
--------------------------------------------------------------------
* **Confidence.** The per-algorithm "confidence" fields are not comparable
  (FLTrust reports ``1 - trust`` in [0,1]; Multi-Krum/DnC report raw outlier scores
  that are unbounded; DeFL reports a vote fraction), and the attacker's stealth
  reward reads confidence as a probability. So the panel reports the **fraction of
  ACTIVE algorithms that flagged the client**: in ``union`` mode, flagged by 1 of 4
  → ``0.25``, by all 4 → ``1.0``; in ``single`` mode there is one active algorithm,
  so a flag is ``1.0`` and a clear is ``is_suspicious=False, confidence=1.0``.

* **Rollout scoring must not advance history.** GRPO scores ``G`` candidate attacks
  per round before committing one. DeFL's critical-learning-period test compares
  against the previous round's FGNV and its Beta trust counts accumulate, so a
  scoring pass would corrupt them ``G`` times per round. ``verdicts(..., commit=False)``
  snapshots and restores each defense's ``state_dict`` around the call; only
  ``commit=True`` lets history advance.

Unlike the benchmark — where each defense evolves its OWN global model so the
panel can be compared head to head — here the active algorithm judges the SAME
shared global model that the arms race is training against, and the surviving
clients are FedAvg-ed into it. The algorithms are used purely as detectors; their
internal aggregates are discarded.
"""

import logging

from core.types import DetectionVerdict

logger = logging.getLogger(__name__)

# The purely algorithmic defenses. Deliberately excludes ``oracle`` (reads ground
# truth), ``llm_defender`` (that IS the agent we are switching off) and ``fedavg``
# (no defense at all — it would flag nothing and only dilute the consensus).
ALGORITHMIC = ("fltrust", "multikrum", "dnc", "defl")


class DefenseEnsemble:
    """Judges rounds with the configured robust-FL algorithms.

    ``mode="single"`` (default) uses ONE algorithm per round, chosen by
    ``selection``; ``mode="union"`` runs them all and unions their rejections. See
    the module docstring for why single is the default.
    """

    MODES = ("single", "union")
    SELECTIONS = ("rotate", "random", "fixed")

    # Even one aggregator drops clients on a clean round (Multi-Krum and DnC drop a
    # fixed number by construction, DeFL always flags at least one), which is
    # absorbed by measuring the clean counterfactual under the SAME defense (see
    # FLArmsRaceEnv.clean_reference_accuracy). But if the panel starts rejecting most
    # of the federation there is nothing left to average, so warn.
    _OVERFLAG_WARN_EVERY = 50
    # A persistently failing algorithm must stay visible, but a full traceback on
    # every one of millions of rounds would bury logs/system.log.
    _ERROR_TRACEBACK_EVERY = 500

    def __init__(self, defenses: dict, *, mode: str = "single",
                 selection: str = "rotate", rng=None):
        if not defenses:
            raise ValueError("DefenseEnsemble needs at least one defense")
        if mode not in self.MODES:
            raise ValueError(f"defense.mode must be one of {self.MODES}, got {mode!r}")
        if selection not in self.SELECTIONS:
            raise ValueError(
                f"defense.selection must be one of {self.SELECTIONS}, got {selection!r}")
        self.defenses = dict(defenses)          # ordered {name: Defense}
        self.mode = mode
        self.selection = selection
        self._rng = rng
        self._overflag_rounds = 0
        self._error_counts: dict[str, int] = {}
        # Index of the algorithm judging the CURRENT round (``single`` mode only).
        # Advanced by begin_round(), never mid-round: the clean counterfactual, the
        # G rollout scorings and the commit must all see the same defense or `drop`
        # compares accuracies measured under different filters.
        self._active_index = 0
        self._round_started = False

    @property
    def names(self) -> list[str]:
        return list(self.defenses)

    @property
    def active_names(self) -> list[str]:
        """The algorithms judging the current round (all of them in ``union``)."""
        if self.mode == "union":
            return self.names
        return [self.names[self._active_index % len(self.defenses)]]

    def __len__(self) -> int:
        return len(self.defenses)

    def describe(self) -> str:
        if self.mode == "union":
            return "+".join(self.names)
        return f"one-of[{','.join(self.names)}] ({self.selection})"

    # ------------------------------------------------------------------
    def begin_round(self) -> list[str]:
        """Pick the algorithm(s) that will judge this round; returns their names.

        Called once per FL round by ``FLArmsRaceEnv.begin_round`` BEFORE the clean
        counterfactual is computed. In ``union`` mode this is a no-op. Safe to call
        when no defense is attached (the env guards that).
        """
        if self.mode == "union":
            return self.names
        n = len(self.defenses)
        if not self._round_started:
            self._round_started = True          # first round uses index 0 as-is
        elif self.selection == "rotate":
            self._active_index = (self._active_index + 1) % n
        elif self.selection == "random" and self._rng is not None:
            self._active_index = self._rng.randrange(n)
        return self.active_names

    # ------------------------------------------------------------------
    def verdicts(self, updates, global_weights, *, commit: bool = False):
        """Judge one round of ``updates`` against ``global_weights``.

        Returns ``(verdicts, info)`` — one ``DetectionVerdict`` per update (ordered
        like ``updates``) plus a per-algorithm breakdown for the logs. With
        ``commit=False`` (scoring a GRPO rollout) any cross-round state the
        algorithms keep is rolled back afterwards.

        Only the ACTIVE algorithms run (all of them in ``union`` mode, this round's
        single pick otherwise — see :meth:`begin_round`).
        """
        active = self.active_names
        snapshot = (None if commit
                    else {n: self.defenses[n].state_dict() for n in active})

        hits: dict[int, list[str]] = {u.client_id: [] for u in updates}
        per_defense: dict[str, list[int]] = {}
        errors: dict[str, str] = {}
        try:
            for name in active:
                defense = self.defenses[name]
                # Re-point at the SHARED global; do not reset (DeFL's history must
                # survive). The algorithm's own aggregate is ignored — we only want
                # its verdicts.
                defense.set_global_weights(global_weights)
                try:
                    # The ground-truth poisoned set is never handed to a defense:
                    # these are real detectors, not the benchmark's oracle.
                    result = defense.step(updates, set())
                except Exception as exc:
                    # One algorithm blowing up (e.g. an SVD that fails to converge)
                    # must not kill a multi-day training run, and it must not silently
                    # weaken the panel either: the failure is logged AND carried in
                    # `info` so it shows up in logs/round_data/rounds.jsonl.
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    per_defense[name] = []
                    count = self._error_counts.get(name, 0) + 1
                    self._error_counts[name] = count
                    tail = ("the other algorithms still decide" if len(active) > 1
                            else "it is the ONLY defense this round, so the round is undefended")
                    msg = (f"Defense '{name}' failed this round (#{count}) — it flags "
                           f"nobody and {tail}")
                    if count % self._ERROR_TRACEBACK_EVERY == 1:
                        logger.exception(msg)
                    else:
                        logger.warning(f"{msg}: {errors[name]}")
                    continue
                flags = sorted(v.client_id for v in result.verdicts if v.is_suspicious)
                per_defense[name] = flags
                for cid in flags:
                    if cid in hits:
                        hits[cid].append(name)
        finally:
            if snapshot is not None:
                for name, state in snapshot.items():
                    self.defenses[name].load_state_dict(state)

        # Confidence is the fraction of the algorithms that ACTUALLY RAN which
        # flagged the client — so in single mode a flag is 1.0, and a defense that
        # errored out does not silently dilute the score toward "probably benign".
        n = max(1, len(active))
        out = []
        for u in updates:
            names = hits.get(u.client_id, [])
            if names:
                out.append(DetectionVerdict(
                    u.client_id, True, len(names) / n,
                    f"flagged by {','.join(names)} ({len(names)}/{n})"))
            else:
                out.append(DetectionVerdict(
                    u.client_id, False, 1.0,
                    f"cleared by {'all ' if n > 1 else ''}{','.join(active)}"))
        flagged = sorted(cid for cid, names in hits.items() if names)
        if commit and len(flagged) > len(updates) / 2:
            self._overflag_rounds += 1
            if self._overflag_rounds % self._OVERFLAG_WARN_EVERY == 1:
                hint = ("Trim `defense.algorithms` or lower `defense.assumed_malicious`"
                        if self.mode == "union" else
                        "Lower `defense.assumed_malicious`, or drop this algorithm from "
                        "`defense.algorithms`")
                logger.warning(
                    f"Defense [{'+'.join(active)}] rejected {len(flagged)}/{len(updates)} "
                    f"clients ({per_defense}) — over-flagging, so FedAvg is running on a "
                    f"small minority (all-flagged rounds leave the global unchanged and "
                    f"the attacker can never score a drop). {hint} if this persists. "
                    f"[{self._overflag_rounds} such round(s) so far]"
                )
        info = {
            # NOT "mode": the round log nests this under `attack_metadata.defense`
            # alongside {"mode": "algorithmic" | "llm_defender"}, which says WHO
            # judged the round. Reusing the key would silently overwrite that and
            # break every consumer that branches on it.
            "panel_mode": self.mode,
            "algorithms": list(active),      # what actually judged THIS round
            "configured": self.names,
            "per_defense_flags": per_defense,
            "flagged": flagged,
        }
        if errors:
            info["errors"] = errors
        return out, info


# ---------------------------------------------------------------------------
# Construction from the config's ``defense:`` block
# ---------------------------------------------------------------------------

def build_ensemble(config: dict, *, device: str | None = None,
                   seed: int | None = None, rng=None) -> DefenseEnsemble:
    """Build the defense panel described by ``config['defense']``.

    Defaults (see ``configs/base.yaml``) configure all four algorithms and judge
    each round with ONE of them (``mode: single``, ``selection: rotate``); set
    ``mode: union`` for the legacy union-of-rejections behaviour. The assumed
    adversary budget Multi-Krum (``f``) and DnC (``m``) need is a HYPERPARAMETER,
    not per-round truth: it defaults to ``attack.max_poison_clients``, clamped to
    keep a benign majority. FLTrust additionally needs a small clean root dataset,
    built here from the MNIST training set with the run's seed.

    ``rng`` is the run's ``random.Random``; only ``selection: random`` uses it, and
    passing it keeps the choice reproducible under the run seed.
    """
    from benchmark.defenses import build_defenses
    from data.mnist_loader import build_root_loader

    fl = config.get("fl", {})
    attack = config.get("attack", {})
    dcfg = dict(config.get("defense") or {})

    names = [str(n).strip() for n in dcfg.get("algorithms", list(ALGORITHMIC)) if str(n).strip()]
    unknown = [n for n in names if n not in ALGORITHMIC]
    if unknown:
        raise ValueError(
            f"defense.algorithms contains non-algorithmic entries {unknown}; "
            f"available: {list(ALGORITHMIC)}"
        )
    if not names:
        raise ValueError("defense.algorithms is empty — nothing would defend the aggregate")

    mode = str(dcfg.get("mode", "single")).strip().lower()
    selection = str(dcfg.get("selection", "rotate")).strip().lower()

    device = device or fl.get("device", "cpu")
    seed = int(fl.get("poison_seed", 0)) if seed is None else int(seed)
    n_clients = int(fl.get("n_clients", 1))

    # Assumed #malicious f/m: the attacker's largest possible budget, capped so the
    # defense still assumes an honest majority.
    assumed = dcfg.get("assumed_malicious")
    if assumed is None:
        assumed = attack.get("max_poison_clients", fl.get("n_compromisable", 1))
    assumed = max(1, min(int(assumed), max(1, (n_clients - 1) // 2)))

    root_loader = None
    if "fltrust" in names:
        root_loader = build_root_loader(
            data_dir=config.get("data", {}).get("data_dir", "./data/mnist_raw"),
            root_size=int(dcfg.get("root_size", 100)),
            batch_size=int(fl.get("batch_size", 64)),
            seed=seed,
        )

    defenses = build_defenses(
        names,
        device=device,
        root_loader=root_loader,
        root_lr=float(dcfg.get("root_lr") or fl.get("lr", 0.002)),
        root_epochs=int(dcfg.get("root_epochs", 1)),
        eta=float(dcfg.get("eta", 1.0)),
        defl_delta=float(dcfg.get("defl_delta", 0.05)),
        defl_tau=float(dcfg.get("defl_tau", 2.5)),
        dnc_num_byzantine=assumed,
        dnc_c=float(dcfg.get("dnc_c", 1.0)),
        dnc_niters=int(dcfg.get("dnc_niters", 1)),
        dnc_sub_dim=int(dcfg.get("dnc_sub_dim", 10000)),
        dnc_seed=seed,
        multikrum_num_byzantine=assumed,
        multikrum_m=dcfg.get("multikrum_m"),
    )
    ensemble = DefenseEnsemble(defenses, mode=mode, selection=selection, rng=rng)
    rule = ("a client is dropped from FedAvg if ANY of them flags it"
            if mode == "union" else
            "ONE algorithm judges each round and its rejections are dropped from FedAvg")
    logger.info(
        f"Algorithmic defense panel: {ensemble.describe()} "
        f"(assumed_malicious={assumed} of {n_clients} clients) — {rule}; "
        f"the defender LLM is OFF"
    )
    return ensemble
