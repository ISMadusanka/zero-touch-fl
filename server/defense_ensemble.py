"""Algorithmic defense ensemble — the defender side with the defender LLM OFF.

Used by ``python main.py --freeze defender``. The defender LLM is deactivated and
the server instead runs EVERY implemented robust-FL algorithm over the same round
of client updates:

    FLTrust (NDSS'21) · Multi-Krum (NeurIPS'17) · DnC (NDSS'21) · DeFL (AAAI'23)

Each algorithm reports its own per-client accept/reject decision (the same
``DetectionVerdict`` the LLM defender emits, so nothing downstream changes). The
ensemble takes the **union of the rejections**: a client is flagged if ANY single
algorithm flags it. Flagged clients are then dropped by ``FedAvgAggregator``
exactly as flagged clients always were — so a poisoner only lands its attack when
it slips past *all* the algorithms at once, and one detection is enough to make
the round's attack fail.

Two details make the ensemble usable as a training-time opponent:

* **Confidence = consensus.** The per-algorithm "confidence" fields are not
  comparable (FLTrust reports ``1 - trust`` in [0,1]; Multi-Krum/DnC report raw
  outlier scores that are unbounded; DeFL reports a vote fraction), and the
  attacker's stealth reward reads confidence as a probability. So the ensemble
  reports the **fraction of algorithms that flagged the client** instead: flagged
  by 1 of 4 → ``confidence=0.25``, flagged by all 4 → ``1.0``, flagged by none →
  ``is_suspicious=False, confidence=1.0`` (all algorithms confidently agree it is
  benign). That is a well-defined [0,1] signal that decreases smoothly as the
  attacker evades more algorithms, which is what gives GRPO a gradient to climb
  instead of a single binary caught/not-caught bit.

* **Rollout scoring must not advance history.** GRPO scores ``G`` candidate attacks
  per round before committing one. DeFL's critical-learning-period test compares
  against the previous round's FGNV and its Beta trust counts accumulate, so a
  scoring pass would corrupt them ``G`` times per round. ``verdicts(..., commit=False)``
  snapshots and restores each defense's ``state_dict`` around the call; only
  ``commit=True`` lets history advance.

Unlike the benchmark — where each defense evolves its OWN global model so the
panel can be compared head to head — here every algorithm judges the SAME shared
global model that the arms race is training against, and the surviving clients are
FedAvg-ed into it. The algorithms are used purely as detectors; their internal
aggregates are discarded.
"""

import logging

from core.types import DetectionVerdict

logger = logging.getLogger(__name__)

# The purely algorithmic defenses. Deliberately excludes ``oracle`` (reads ground
# truth), ``llm_defender`` (that IS the agent we are switching off) and ``fedavg``
# (no defense at all — it would flag nothing and only dilute the consensus).
ALGORITHMIC = ("fltrust", "multikrum", "dnc", "defl")


class DefenseEnsemble:
    """Runs several robust-FL algorithms as detectors and unions their rejections."""

    # Unioning several aggregators is deliberately trigger-happy: Multi-Krum and DnC
    # drop a fixed number of clients EVERY round by construction and DeFL always
    # flags at least one, so some honest clients are dropped even on a clean round.
    # That is absorbed by measuring the clean counterfactual under the same defense
    # (see FLArmsRaceEnv.clean_reference_accuracy) — but if the panel starts
    # rejecting most of the federation there is nothing left to average, so warn.
    _OVERFLAG_WARN_EVERY = 50
    # A persistently failing algorithm must stay visible, but a full traceback on
    # every one of millions of rounds would bury logs/system.log.
    _ERROR_TRACEBACK_EVERY = 500

    def __init__(self, defenses: dict):
        if not defenses:
            raise ValueError("DefenseEnsemble needs at least one defense")
        self.defenses = dict(defenses)          # ordered {name: Defense}
        self._overflag_rounds = 0
        self._error_counts: dict[str, int] = {}

    @property
    def names(self) -> list[str]:
        return list(self.defenses)

    def __len__(self) -> int:
        return len(self.defenses)

    def describe(self) -> str:
        return "+".join(self.names)

    # ------------------------------------------------------------------
    def verdicts(self, updates, global_weights, *, commit: bool = False):
        """Judge one round of ``updates`` against ``global_weights``.

        Returns ``(verdicts, info)`` — one ``DetectionVerdict`` per update (ordered
        like ``updates``) plus a per-algorithm breakdown for the logs. With
        ``commit=False`` (scoring a GRPO rollout) any cross-round state the
        algorithms keep is rolled back afterwards.
        """
        snapshot = (None if commit
                    else {n: d.state_dict() for n, d in self.defenses.items()})

        hits: dict[int, list[str]] = {u.client_id: [] for u in updates}
        per_defense: dict[str, list[int]] = {}
        errors: dict[str, str] = {}
        try:
            for name, defense in self.defenses.items():
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
                    # weaken the panel either: the round proceeds with the remaining
                    # algorithms and the failure is logged AND carried in `info` so it
                    # shows up in logs/round_data/rounds.jsonl.
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    per_defense[name] = []
                    count = self._error_counts.get(name, 0) + 1
                    self._error_counts[name] = count
                    msg = (f"Defense '{name}' failed this round (#{count}) — it flags "
                           f"nobody and the other algorithms still decide")
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

        n = len(self.defenses)
        out = []
        for u in updates:
            names = hits.get(u.client_id, [])
            if names:
                out.append(DetectionVerdict(
                    u.client_id, True, len(names) / n,
                    f"flagged by {','.join(names)} ({len(names)}/{n})"))
            else:
                out.append(DetectionVerdict(
                    u.client_id, False, 1.0, f"cleared by all {n}"))
        flagged = sorted(cid for cid, names in hits.items() if names)
        if commit and len(flagged) > len(updates) / 2:
            self._overflag_rounds += 1
            if self._overflag_rounds % self._OVERFLAG_WARN_EVERY == 1:
                logger.warning(
                    f"Defense ensemble rejected {len(flagged)}/{len(updates)} clients "
                    f"({per_defense}) — the union is over-flagging, so FedAvg is running "
                    f"on a small minority (all-flagged rounds leave the global unchanged "
                    f"and the attacker can never score a drop). Trim `defense.algorithms` "
                    f"or lower `defense.assumed_malicious` in the config if this persists. "
                    f"[{self._overflag_rounds} such round(s) so far]"
                )
        info = {
            "algorithms": self.names,
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
                   seed: int | None = None) -> DefenseEnsemble:
    """Build the ensemble described by ``config['defense']``.

    Defaults (see ``configs/base.yaml``) run all four algorithms. The assumed
    adversary budget Multi-Krum (``f``) and DnC (``m``) need is a HYPERPARAMETER,
    not per-round truth: it defaults to ``attack.max_poison_clients``, clamped to
    keep a benign majority. FLTrust additionally needs a small clean root dataset,
    built here from the MNIST training set with the run's seed.
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
    ensemble = DefenseEnsemble(defenses)
    logger.info(
        f"Algorithmic defense ensemble: {ensemble.describe()} "
        f"(assumed_malicious={assumed} of {n_clients} clients) — a client is dropped "
        f"from FedAvg if ANY of them flags it; the defender LLM is OFF"
    )
    return ensemble
