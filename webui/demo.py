"""The built-in demo version, and the benchmark result it replays.

Why this exists
---------------
The panel's rule is that it *watches* real runs (see ``webui/README.md``). This
module is the one deliberate exception, and it is fenced off rather than woven in:
a single version id — :data:`DEMO_ID` — is a **fixture**, and a benchmark aimed at
it runs :mod:`webui.demo_bench` instead of ``benchmark.run_benchmark``. Every real
version keeps taking the real path, and nothing here can be reached by one:
:func:`is_demo` gates on an id the on-disk store cannot produce, because
``versions._safe_id`` only accepts ``v<digits>[-slug]`` and the store's own
``_next_index`` starts at 1.

What it replays
---------------
:data:`REFERENCE` is a hand-authored attack × defense result for the ``llm`` row,
quoted at :data:`REFERENCE_ROUNDS` rounds. It is a fixture, not a measurement, and
its numbers are not all derivable from one another the way
``benchmark.metrics.DefenseMetrics`` derives them from confusion counts — the
precision and F1 quoted for a row do not follow from its TPR and FPR, and neither
does its evasion rate. So the demo does **not** try to reproduce the table by
feeding synthetic rounds through the real metrics: it emits the table as the
authoritative ``summary`` event and generates a per-round stream that is merely
*plausible and consistent with it* (the accuracy trajectory ends at
``final_accuracy`` and averages ``mean_accuracy``; detection converges on
``detection_rate``/``fpr``). The live theatre therefore reads like the run that
produced the table, and the final table is exactly the fixture.

Away from ``REFERENCE_ROUNDS`` the fixture is perturbed rather than reused
verbatim, deterministically in the round count and the seed, and by more when the
run is shorter — fewer rounds really are noisier. ``mean_acc_drop`` is always
recomputed from :data:`BASELINE_ACCURACY` so the arithmetic the report checks
still holds.
"""
import hashlib

#: The one version id that is a fixture. Deliberately outside the range the store
#: generates (``_next_index`` starts at 1), so no real snapshot can collide.
DEMO_ID = "v000"

#: The round count :data:`REFERENCE` is quoted at. Ask for exactly this many and
#: the fixture is replayed verbatim.
REFERENCE_ROUNDS = 250

#: Phase-1 clean accuracy every row's drop is measured against. Chosen to agree
#: with the fixture: every row's ``mean_acc + acc_drop`` comes to this number.
BASELINE_ACCURACY = 0.897

#: The fixture: one row per defense, for the ``llm`` attack.
#:
#: ``evasion`` is the benchmark's ``attack_success_rate`` (rounds a poisoned client
#: slipped past detection) and ``goal`` is its weighted ``goal_success_rate``.
REFERENCE = {
    "fedavg":     {"detection_rate": 0.000, "fpr": 0.000, "precision": 0.00,
                   "f1": 0.00, "final_accuracy": 0.281, "mean_accuracy": 0.451,
                   "evasion": 1.000, "goal": 0.945},
    "oracle":     {"detection_rate": 1.000, "fpr": 0.000, "precision": 1.00,
                   "f1": 1.00, "final_accuracy": 0.896, "mean_accuracy": 0.895,
                   "evasion": 0.000, "goal": 0.012},
    "fltrust":    {"detection_rate": 0.210, "fpr": 0.150, "precision": 0.35,
                   "f1": 0.28, "final_accuracy": 0.612, "mean_accuracy": 0.690,
                   "evasion": 0.900, "goal": 0.580},
    "defl":       {"detection_rate": 0.101, "fpr": 0.040, "precision": 0.55,
                   "f1": 0.34, "final_accuracy": 0.430, "mean_accuracy": 0.618,
                   "evasion": 0.880, "goal": 0.725},
    "dnc":        {"detection_rate": 0.136, "fpr": 0.080, "precision": 0.48,
                   "f1": 0.30, "final_accuracy": 0.402, "mean_accuracy": 0.585,
                   "evasion": 0.900, "goal": 0.780},
    "multikrum":  {"detection_rate": 0.164, "fpr": 0.050, "precision": 0.62,
                   "f1": 0.45, "final_accuracy": 0.520, "mean_accuracy": 0.665,
                   "evasion": 0.720, "goal": 0.550},
    # Not in the quoted table -- the fixture was taken from a run whose panel had
    # no defender column. Included so a defender-targeted demo run has a row too,
    # placed between Multi-Krum and Oracle: a trained detector, not a clairvoyant.
    "llm_defender": {"detection_rate": 0.640, "fpr": 0.030, "precision": 0.88,
                     "f1": 0.74, "final_accuracy": 0.812, "mean_accuracy": 0.838,
                     "evasion": 0.240, "goal": 0.190},
}

#: How each OTHER attack row compares to `llm` on the damage it gets through.
#: The trained policy is the strongest row -- it is the system under test, and the
#: published baselines are the control that says the legs were comparable.
#: Scales the drop; detection is scaled inversely (a blunter attack is easier to
#: catch), and `clean` poisons nothing at all.
ATTACK_STRENGTH = {
    "llm": 1.00, "min_max": 0.82, "min_sum": 0.78, "fang": 0.71, "fang_krum": 0.68,
    "lie": 0.64, "ipm": 0.57, "scaling": 0.52, "mimic": 0.44, "sign_flip": 0.38,
    "noise": 0.31, "label_flip": 0.27, "clean": 0.0,
}

#: How the attack's reach scales with HOW MANY clients it poisons, as
#: ``(poisoners, strength)`` anchors interpolated linearly in between.
#:
#: The fixture was taken with 10 poisoned clients, so that is where strength is
#: 1.0 and the table is reproduced verbatim; more than 10 does not read as
#: stronger, because the quoted row is already what a half-poisoned federation
#: looks like. Below it the attack has fewer updates to hide among and fewer to
#: push with, so it lands less damage AND is easier to spot -- which is the pair
#: of movements this models. The anchors are the bands asked for: 10+, 6-10, 3-6,
#: 1-3, made continuous inside each band so 9 poisoners is not indistinguishable
#: from 6.
POISONER_ANCHORS = ((1, 0.20), (3, 0.45), (6, 0.72), (10, 1.00))

#: Roles the fixture claims to hold, so it appears on both benchmark axes.
DEMO_ROLES = ("attacker", "defender")


def poisoner_strength(n_poison) -> float:
    """How much of the quoted attack ``n_poison`` poisoned clients deliver, in (0, 1].

    1.0 at and above the fixture's 10, falling through the bands below it. Returns
    1.0 for ``None`` so a caller that does not care about the count gets the
    fixture unchanged.
    """
    if n_poison is None:
        return 1.0
    n = max(1, int(n_poison))
    lo_n, lo_s = POISONER_ANCHORS[0]
    if n <= lo_n:
        return lo_s
    for hi_n, hi_s in POISONER_ANCHORS[1:]:
        if n <= hi_n:
            span = hi_n - lo_n
            return lo_s + (hi_s - lo_s) * ((n - lo_n) / span if span else 1.0)
        lo_n, lo_s = hi_n, hi_s
    return lo_s              # at or past the last anchor: the fixture as quoted


def _record() -> dict:
    """The fixture's version record, in the shape ``versions.listing`` returns."""
    return {
        "id": DEMO_ID,
        "index": 0,
        "label": DEMO_ID,
        "notes": "",
        "created": "2026-07-10T21:32:16",
        "adapters": {role: f"{role}_adapter" for role in DEMO_ROLES},
        "roles": sorted(DEMO_ROLES),
        "base_model": "unsloth/Qwen2.5-7B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "rounds_done": 1001450,
        "round_index": 1001450,
        "size_bytes": None,
        # What the versions table and the pickers read. `available` is true so the
        # row can be selected like any other; `demo` is what routes the run to
        # webui.demo_bench and what the page uses to keep rename/delete local.
        "available": {role: True for role in DEMO_ROLES},
        "demo": True,
        "training": {
            "rounds": REFERENCE_ROUNDS,
            "mean_attacker_reward": 0.8,
            "mean_induced_drop": 0.315,
            "mean_defender_reward": 0.41,
            "learners_seen": ["attacker"],
            "defenses_seen": sorted(k for k in REFERENCE if k not in
                                    ("fedavg", "oracle", "llm_defender")),
        },
    }


#: Merged into the version listing by the server. A list, so a second fixture is
#: a one-line addition.
LISTING = [_record()]


def is_demo(vid) -> bool:
    """True for the fixture's id (and its aliases), false for every real version."""
    return str(vid or "").strip().lower() == DEMO_ID


def get(vid: str) -> dict:
    if not is_demo(vid):
        raise KeyError(vid)
    return _record()


# ---------------------------------------------------------------------------
# Deriving a run's numbers from the fixture
# ---------------------------------------------------------------------------

def _rng(*parts) -> "object":
    """A tiny deterministic generator seeded by ``parts``.

    ``random.Random`` would do, but seeding it from a tuple of mixed types is
    version-sensitive; hashing a joined string keeps a given (rounds, seed) pair
    producing the same table on every machine and Python build.
    """
    import random
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _deviation(rounds: int) -> float:
    """How far from the fixture a run of ``rounds`` may drift, as a fraction.

    Zero at :data:`REFERENCE_ROUNDS` — that is the promise. Away from it the
    spread grows with how much shorter the run is, because a shorter run really is
    noisier, and saturates so a very long run stays close rather than wandering.
    """
    if rounds == REFERENCE_ROUNDS:
        return 0.0
    ratio = REFERENCE_ROUNDS / max(1, rounds)
    return min(0.15, 0.04 * (ratio ** 0.5 if ratio > 1 else 1 / ratio ** 0.5))


def _jitter(rng, value: float, spread: float, lo: float = 0.0,
            hi: float = 1.0) -> float:
    """``value`` nudged by up to ``spread`` of itself, clamped into [lo, hi].

    Exact 0.0 and 1.0 are left alone: they are the structural facts of a row
    (FedAvg detects nothing, Oracle reads the ground truth), not measurements that
    could have come out differently.
    """
    if spread <= 0.0 or value in (0.0, 1.0):
        return value
    return max(lo, min(hi, value * (1.0 + rng.uniform(-spread, spread))))


def defense_row(defense: str, rounds: int, seed: int = 0) -> dict:
    """The fixture's row for ``defense``, adjusted for a run of ``rounds``."""
    ref = REFERENCE.get(defense)
    if ref is None:
        # A defense the fixture does not quote: sit it between FLTrust and DnC
        # rather than dropping the column, so an unusual panel still fills in.
        ref = REFERENCE["fltrust"]
    spread = _deviation(rounds)
    rng = _rng(defense, rounds, seed)
    row = {
        "detection_rate": _jitter(rng, ref["detection_rate"], spread),
        "fpr": _jitter(rng, ref["fpr"], spread),
        "precision": _jitter(rng, ref["precision"], spread),
        "f1": _jitter(rng, ref["f1"], spread),
        "evasion": _jitter(rng, ref["evasion"], spread),
        "goal": _jitter(rng, ref["goal"], spread),
    }
    # Accuracy is perturbed through the DROP, not the level. Nudging the level
    # would move a row by the same absolute amount however well it did, which is
    # not how these rows differ: Oracle rejects every poisoned update, so its drop
    # is ~0.002 and stays hundredths of a point away from clean whatever the round
    # count, while FedAvg's 0.446 has room to move. Perturbing the level sent
    # Oracle's mean accuracy to 0.827 on a short run, which reads as a broken
    # oracle rather than a shorter measurement.
    for key in ("mean_accuracy", "final_accuracy"):
        drop = _jitter(rng, BASELINE_ACCURACY - ref[key], spread,
                       0.0, BASELINE_ACCURACY - 0.02)
        row[key] = BASELINE_ACCURACY - drop
    # The report and the panel both check this identity, so it is derived, never
    # jittered: a drop that does not equal baseline - mean would read as a bug.
    row["mean_acc_drop"] = BASELINE_ACCURACY - row["mean_accuracy"]
    return row


def attack_row(attack: str, defense: str, rounds: int, seed: int = 0,
               n_poison=None) -> dict:
    """One matrix cell: ``defense``'s row scaled to how hard this attack pushes.

    Two things set that. **Which attack** it is -- the fixture describes the
    ``llm`` row, and every other attack is that row moved toward the clean
    baseline by its :data:`ATTACK_STRENGTH`. And **how many clients it poisons**
    -- see :func:`poisoner_strength`. They multiply, because they are the same
    kind of weakening: less damage through the defense, and more of the attack
    caught, because a blunter or smaller attack is easier to see.

    At the fixture's own point -- the ``llm`` row with 10 poisoners, or with the
    count left unspecified -- nothing is scaled and the quoted row is returned.
    """
    row = defense_row(defense, rounds, seed)
    strength = ATTACK_STRENGTH.get(attack, 0.6) * poisoner_strength(n_poison)
    if strength >= 1.0 - 1e-9:
        return row

    rng = _rng(attack, defense, rounds, seed, n_poison)
    scaled = dict(row)
    if strength <= 0.0:
        # `clean` poisons nothing: no attack to detect, no damage, and any flag a
        # defense raises on it is a false alarm. Its accuracy IS this defense's
        # clean accuracy, which is what makes it the control row.
        scaled.update({"detection_rate": 0.0, "precision": 0.0, "f1": 0.0,
                       "evasion": 0.0, "goal": 0.0})
        # Perturb the DROP, not the level -- the same trap as in defense_row. A
        # control row that poisons nothing sits a few thousandths under clean;
        # nudging its accuracy by a percentage of 0.893 moved it by a tenth of a
        # point, which made the row that attacks nothing look worse than the ones
        # that do.
        for key in ("mean_accuracy", "final_accuracy"):
            scaled[key] = BASELINE_ACCURACY - _jitter(rng, 0.004, 0.4, 0.0, 0.02)
        scaled["mean_acc_drop"] = BASELINE_ACCURACY - scaled["mean_accuracy"]
        return scaled

    # Damage scales with strength; accuracy is what is left of the baseline.
    # The jitter here is deliberately SMALLER than the step between two adjacent
    # poisoner counts. At 9 poisoners the attack is scaled to 93%, a 7% move; a
    # +/-10% wobble on top of that let the ramp run backwards, so stepping the
    # quota down could show the attack doing better -- the opposite of what the
    # control is there to demonstrate.
    for key in ("mean_accuracy", "final_accuracy"):
        drop = (BASELINE_ACCURACY - row[key]) * strength
        scaled[key] = max(0.02, BASELINE_ACCURACY - drop * rng.uniform(0.985, 1.015))
    scaled["mean_acc_drop"] = BASELINE_ACCURACY - scaled["mean_accuracy"]
    # A blunter attack is easier to catch, so detection moves the other way -- but
    # never past what the ground-truth-reading oracle achieves.
    # Detection is the headline movement -- a smaller or blunter attack is easier
    # to see -- so it gets the full lift. Precision and F1 move the same way but
    # more slowly: they start high enough that the same multiplier would pin them
    # against 1.0 and report a flawless detector on a row that is still missing
    # two thirds of the attack.
    lift = 1.0 + (1.0 - strength) * 1.4
    soft = 1.0 + (1.0 - strength) * 0.7
    scaled["detection_rate"] = min(1.0, _jitter(rng, row["detection_rate"] * lift, 0.02))
    # The false-positive rate is about how twitchy the defense is on honest
    # clients, which the attack's size does not drive, so it only wobbles.
    scaled["fpr"] = _jitter(rng, row["fpr"], 0.10)
    # These may approach 1 but must not REACH it while honest clients are still
    # being flagged: precision 1.00 next to a 5% false-positive rate says nothing
    # honest was ever flagged and that something was, in the same row.
    ceiling = 1.0 if scaled["fpr"] <= 0.0 else 0.95
    for key in ("precision", "f1"):
        scaled[key] = min(ceiling, _jitter(rng, row[key] * soft, 0.03))
    scaled["evasion"] = min(1.0, _jitter(rng, row["evasion"] * (0.6 + 0.4 * strength), 0.02))
    scaled["goal"] = min(1.0, _jitter(rng, row["goal"] * strength, 0.02))
    if defense == "oracle":
        # The oracle reads the ground truth whatever the attack is.
        scaled.update({"detection_rate": 1.0, "precision": 1.0, "f1": 1.0,
                       "fpr": 0.0, "evasion": 0.0})
    if defense == "fedavg":
        # No defense: nothing is ever flagged, so everything gets through.
        scaled.update({"detection_rate": 0.0, "precision": 0.0, "f1": 0.0,
                       "fpr": 0.0, "evasion": 1.0})
    return scaled
