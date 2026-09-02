"""The fine-tuned-model version store.

Training writes its adapters in place -- ``checkpoints/attacker_adapter`` and,
when the defender LLM is the side learning, ``checkpoints/defender_adapter``
(every ``rl.save_every`` rounds and at every phase boundary; see
``rl/schedule.py::_save_adapters``, which writes ONLY the trainable side). That is
the right thing for resuming a run and the wrong thing for evaluating one: by the
time a benchmark finishes, the adapter it was told to load has already moved on,
and there is no way to ask "how did the policy at round 400 compare to the policy
at round 900?".

A *version* fixes that: it is a copy of the adapter directories, taken at a moment
the user chose, next to the resume state and a metadata record of what training
looked like when it was taken. Nothing here modifies the live checkpoints -- a
snapshot is a read of ``checkpoints/`` and a write into ``checkpoints/versions/``.

The benchmark already accepts ``--attacker-adapter`` / ``--defender-adapter``, so a
version needs no benchmark change at all: :func:`adapter_path` hands back the
directory and the CLI is pointed at it.

Layout::

    checkpoints/versions/
        v001-after-first-cycle/
            version.json          <- the record below
            attacker_adapter/     <- copied verbatim
            defender_adapter/     <- copied when it exists
            rl_progress.json      <- the resume state at snapshot time

A version is self-describing: ``version.json`` records the base model the adapter
is dimensioned for, so a version taken against a different ``rl.model`` can be
refused at benchmark time rather than failing deep inside PEFT -- and ``roles``
says which sides it can be benchmarked as, since a one-sided training run
snapshots one adapter.
"""
import datetime
import json
import os
import re
import shutil

VERSIONS_DIR = os.path.join("checkpoints", "versions")
LIVE_ADAPTERS = {
    "attacker": os.path.join("checkpoints", "attacker_adapter"),
    "defender": os.path.join("checkpoints", "defender_adapter"),
}
PROGRESS_FILE = os.path.join("checkpoints", "rl_progress.json")
ROUND_LOG = os.path.join("logs", "round_data", "rounds.jsonl")

#: How much of the round log to read back when summarising a snapshot. The log is
#: append-only across every run ever made, so the whole file is not worth reading
#: to describe the last few hundred rounds.
_TAIL_BYTES = 4 * 1024 * 1024
#: Rounds averaged into a version's "how was it training" record.
_SUMMARY_WINDOW = 50

_SLUG = re.compile(r"[^a-z0-9]+")


class VersionError(RuntimeError):
    pass


def _slug(text: str, limit: int = 40) -> str:
    s = _SLUG.sub("-", (text or "").strip().lower()).strip("-")
    return s[:limit].strip("-")


def adapter_exists(path: str) -> bool:
    """True if ``path`` looks like a saved LoRA adapter directory."""
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _read_json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Summarising the training run a snapshot was taken from
# ---------------------------------------------------------------------------

def _tail_rounds(path: str = ROUND_LOG, limit: int = _SUMMARY_WINDOW) -> list:
    """The last ``limit`` parsable records of the round log, oldest first."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return []
    try:
        with open(path, "rb") as f:
            if size > _TAIL_BYTES:
                f.seek(size - _TAIL_BYTES)
                f.readline()          # discard the partial line we landed in
            raw = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = []
    for line in raw.split("\n")[-(limit * 3):]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out[-limit:]


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 6) if xs else None


def _detection_rates(row: dict) -> dict:
    """TPR/FPR for one round, from the verdicts and the ground-truth poisoned set.

    This is the defender's score sheet, and the round log carries the two halves
    of it rather than the rates themselves (``predicted_labels`` and
    ``poisoned_client_ids``), so it is derived the same way the live panel derives
    it. ``None`` where the round has no such clients -- a round with nothing
    poisoned has no TPR, and averaging a 0 in would read as a missed attack.
    """
    poisoned = set(row.get("poisoned_client_ids") or [])
    verdicts = row.get("predicted_labels") or []
    tp = fn = fp = tn = 0
    for v in verdicts:
        bad = v.get("client_id") in poisoned
        hit = bool(v.get("is_suspicious"))
        if bad and hit:
            tp += 1
        elif bad:
            fn += 1
        elif hit:
            fp += 1
        else:
            tn += 1
    return {"tpr": tp / (tp + fn) if (tp + fn) else None,
            "fpr": fp / (fp + tn) if (fp + tn) else None}


def training_summary(rounds=None) -> dict:
    """Describe the recent training rounds -- the "was this a good moment to snapshot"
    record stored with a version.

    Damage is averaged over MEASURED rounds only. A round with
    ``clean_measured: false`` has no counterfactual, so its ``induced_drop`` is not a
    measurement of anything (``rl/schedule.py`` does not even apply a gradient on
    those); averaging it in would quietly pull the number toward zero.
    """
    rows = _tail_rounds() if rounds is None else rounds
    if not rows:
        return {"rounds": 0}
    meta = [r.get("attack_metadata") or {} for r in rows]
    measured = [m for m in meta
                if m.get("clean_measured", True) and m.get("defense_sane", True)]
    terms = [m.get("reward_terms") or {} for m in meta]
    train = [m.get("train") or {} for m in meta]
    wins = [bool(m.get("learner_success")) for m in meta]
    detect = [_detection_rates(r) for r in rows]
    return {
        "rounds": len(rows),
        "first_round": rows[0].get("round_num"),
        "last_round": rows[-1].get("round_num"),
        "measured_rounds": len(measured),
        "mean_attacker_reward": _mean([r.get("attacker_reward") for r in rows]),
        "mean_induced_drop": _mean([m.get("induced_drop") for m in measured]),
        "max_induced_drop": (max((m.get("induced_drop") or 0.0) for m in measured)
                             if measured else None),
        "win_rate": round(sum(wins) / len(wins), 4) if wins else None,
        "mean_damage_term": _mean([t.get("damage") for t in terms]),
        "mean_stealth_term": _mean([t.get("stealth") for t in terms]),
        "mean_potency": _mean([m.get("attack_potency") for m in meta]),
        "mean_grpo_loss": _mean([t.get("loss") for t in train]),
        "mean_reward_spread": _mean([t.get("reward_spread") for t in train]),
        "final_accuracy": rows[-1].get("test_accuracy"),
        "baseline_accuracy": rows[-1].get("baseline_accuracy"),
        "defenses_seen": sorted({m.get("defense") for m in meta if m.get("defense")}),
        "goal": rows[-1].get("attack_goal"),
        # --- the defender side -------------------------------------------------
        # A defender-only run (``--learn defender`` under ``defense.mode: llm``)
        # produces rounds whose attacker numbers are frozen and whose reward,
        # detection rate and win gate all belong to the DEFENDER. Summarising only
        # the attacker's columns described such a snapshot as a flat line and said
        # nothing about the policy that was actually trained.
        "mean_defender_reward": _mean([r.get("defender_reward") for r in rows]),
        "mean_tpr": _mean([d["tpr"] for d in detect if d["tpr"] is not None]),
        "mean_fpr": _mean([d["fpr"] for d in detect if d["fpr"] is not None]),
        # Which side(s) the rounds in this window were training -- "attacker",
        # "defender", both (an alternating arms race) or "none" (dry-run/baseline).
        "learners_seen": sorted({str(r.get("learning_agent")) for r in rows
                                 if r.get("learning_agent")}),
    }


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

def _next_index() -> int:
    top = 0
    for name in os.listdir(VERSIONS_DIR) if os.path.isdir(VERSIONS_DIR) else []:
        m = re.match(r"^v(\d+)", name)
        if m:
            top = max(top, int(m.group(1)))
    return top + 1


def create(label: str = "", notes: str = "", roles=("attacker", "defender"),
           base_model: str = "", extra: dict | None = None) -> dict:
    """Snapshot the live adapters into a new version. Returns its record.

    Whichever of ``roles`` is on disk is copied, so an attacker-only run, a
    defender-only run (``--learn defender``) and a two-sided one each snapshot
    exactly what they trained. :attr:`roles` on the record says which.

    Raises :class:`VersionError` when NONE of them is there -- the ordinary "you
    have not trained yet" case, worth a clear message rather than an empty
    directory that fails later inside the benchmark.
    """
    # At least ONE of the requested roles has to be on disk. Requiring the
    # ATTACKER specifically is wrong once ``--learn defender`` is a UI choice: a
    # defender-only run writes ``checkpoints/defender_adapter`` and deliberately
    # never touches the attacker's (see ``rl/schedule.py::_save_adapters``), so
    # the one adapter that run produced could not be snapshotted at all.
    present = [role for role in roles
               if LIVE_ADAPTERS.get(role) and adapter_exists(LIVE_ADAPTERS[role])]
    if not present:
        looked = ", ".join(LIVE_ADAPTERS[r] for r in roles if r in LIVE_ADAPTERS)
        raise VersionError(
            f"no trained adapter to snapshot -- looked in {looked}. Run training "
            f"first (an adapter is written every rl.save_every rounds), and note "
            f"that only the side named by --learn is written to disk.")

    index = _next_index()
    slug = _slug(label)
    vid = f"v{index:03d}" + (f"-{slug}" if slug else "")
    dest = os.path.join(VERSIONS_DIR, vid)
    if os.path.exists(dest):
        raise VersionError(f"version directory already exists: {dest}")
    os.makedirs(dest, exist_ok=False)

    copied = {}
    try:
        for role in roles:
            live = LIVE_ADAPTERS.get(role)
            if not live or not adapter_exists(live):
                continue
            shutil.copytree(live, os.path.join(dest, f"{role}_adapter"))
            copied[role] = f"{role}_adapter"
        if os.path.isfile(PROGRESS_FILE):
            shutil.copy2(PROGRESS_FILE, os.path.join(dest, "rl_progress.json"))
    except OSError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise VersionError(f"could not copy the adapters: {exc}") from exc

    progress = _read_json(os.path.join(dest, "rl_progress.json"), {}) or {}
    # Read the LoRA shape off whichever adapter this version actually holds. A
    # defender-only version has no attacker_adapter/, and reading the base model
    # from a directory that is not there recorded base_model: "" -- which is the
    # field the benchmark uses to refuse a version dimensioned for another
    # ``rl.model``, so an empty one silently disables that check.
    adapter_cfg = {}
    for role in ("attacker", "defender"):
        if role in copied:
            adapter_cfg = _read_json(
                os.path.join(dest, f"{role}_adapter", "adapter_config.json"), {}) or {}
            if adapter_cfg:
                break
    record = {
        "id": vid,
        "index": index,
        "label": label.strip() or vid,
        "notes": notes.strip(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "adapters": copied,
        # The roles this version can actually be benchmarked as. ``adapters``
        # already carries it, but a sorted list is what the UI and the benchmark
        # panel filter on, and it survives into saved run manifests.
        "roles": sorted(copied),
        "base_model": base_model or adapter_cfg.get("base_model_name_or_path", ""),
        "lora_r": adapter_cfg.get("r"),
        "lora_alpha": adapter_cfg.get("lora_alpha"),
        "rounds_done": progress.get("rounds_done"),
        "round_index": progress.get("round_index"),
        "controller": progress.get("controller"),
        "curriculum": progress.get("curriculum"),
        "training": training_summary(),
        "size_bytes": _dir_size(dest),
        **(extra or {}),
    }
    with open(os.path.join(dest, "version.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    record["dir"] = dest.replace("\\", "/")
    return record


def _load_one(vid: str) -> dict | None:
    dest = os.path.join(VERSIONS_DIR, vid)
    rec = _read_json(os.path.join(dest, "version.json"))
    if rec is None:
        return None
    rec["dir"] = dest.replace("\\", "/")
    rec["available"] = {
        role: adapter_exists(os.path.join(dest, f"{role}_adapter"))
        for role in ("attacker", "defender")
    }
    return rec


def listing() -> list:
    """Every stored version, newest first."""
    if not os.path.isdir(VERSIONS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(VERSIONS_DIR)):
        if not os.path.isdir(os.path.join(VERSIONS_DIR, name)):
            continue
        rec = _load_one(name)
        if rec is not None:
            out.append(rec)
    out.sort(key=lambda r: r.get("index", 0), reverse=True)
    return out


def get(vid: str) -> dict:
    rec = _load_one(vid) if _safe_id(vid) else None
    if rec is None:
        raise VersionError(f"no such version: {vid!r}")
    return rec


def _safe_id(vid: str) -> bool:
    """Version ids come back from the browser, so they address a directory. Only
    the shape :func:`create` produces is accepted -- no separators, no ``..``."""
    return bool(vid) and re.fullmatch(r"v\d+(-[a-z0-9-]+)?", str(vid)) is not None


def adapter_path(vid: str | None, role: str = "attacker") -> str | None:
    """Where the benchmark should load ``role`` from.

    ``None``/``"current"``/``"live"`` mean the live training checkpoint, so the
    UI's default is "benchmark what is training right now" with no special case at
    the call site. Returns ``None`` when the role is not in that version.
    """
    if vid in (None, "", "current", "live"):
        path = LIVE_ADAPTERS.get(role)
        return path if path and adapter_exists(path) else None
    rec = get(vid)
    path = os.path.join(rec["dir"], f"{role}_adapter")
    return path if adapter_exists(path) else None


def delete(vid: str) -> str:
    """Remove a version's directory. The live checkpoints are never touched."""
    rec = get(vid)                       # validates the id and that it exists
    shutil.rmtree(rec["dir"], ignore_errors=False)
    return rec["id"]


def rename(vid: str, label: str = None, notes: str = None) -> dict:
    """Update a version's human-facing fields in place (the id never changes, so
    saved benchmark results keep pointing at it)."""
    rec = get(vid)
    if label is not None:
        rec["label"] = str(label).strip() or rec["id"]
    if notes is not None:
        rec["notes"] = str(notes).strip()
    payload = {k: v for k, v in rec.items() if k not in ("dir", "available")}
    with open(os.path.join(rec["dir"], "version.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return rec


def live_status() -> dict:
    """What the un-snapshotted live checkpoints currently hold."""
    out = {"progress": _read_json(PROGRESS_FILE, {}) or {}, "adapters": {}}
    for role, path in LIVE_ADAPTERS.items():
        present = adapter_exists(path)
        out["adapters"][role] = {
            "path": path.replace("\\", "/"),
            "exists": present,
            "modified": (datetime.datetime.fromtimestamp(os.path.getmtime(path))
                         .isoformat(timespec="seconds") if present else None),
            "size_bytes": _dir_size(path) if present else 0,
        }
    out["training"] = training_summary()
    return out
