"""The HTTP API and static page behind ``python -m webui``.

Stdlib only (``http.server`` + ``yaml``, which the project already depends on):
this has to start on a GPU box with no extra ``pip install``.

Binds to 127.0.0.1 by default. Every field arriving from the browser is validated
into a **list-form argv** -- no shell, no free-form argument passthrough -- because
"a local web page can run an arbitrary command" is a real hole even on a loopback
socket. Config overrides go through :mod:`webui.configstore`, which accepts only
dotted paths that already exist in ``configs/base.yaml`` and only values that
coerce to the type already there.

Endpoints
---------
``GET  /``                     the page
``GET  /api/bootstrap``        config schema, registries, versions, run status
``GET  /api/events``           long-poll: ``?kind=train|bench&since=<seq>``
``POST /api/train/start``      spawn ``main.py`` with validated flags + overrides
``POST /api/train/stop``       terminate it (``{"force": true}`` to kill)
``POST /api/bench/start``      spawn ``benchmark.run_benchmark --events -``
``POST /api/bench/stop``
``GET  /api/versions``         list snapshots
``POST /api/versions``         snapshot the live adapters
``POST /api/versions/delete``  remove one
``POST /api/versions/rename``  relabel one
``GET  /api/runs``             past runs started from here
``GET  /api/run``              one past run's saved artifacts (``?id=``)
"""
import argparse
import contextlib
import datetime
import json
import logging
import mimetypes
import os
import posixpath
import shutil
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from webui import configstore, versions
from webui.runner import RUNS_DIR, RunError, Runner, python_exe

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
REPO_ROOT = os.path.dirname(HERE)

logger = logging.getLogger("webui")

MAX_BODY = 2 * 1024 * 1024

#: The benchmark queue. Selecting several fine-tuned versions runs ONE ordinary
#: benchmark subprocess per version, back to back, sharing one event stream so the
#: page can build the comparison as the sweep proceeds. Each entry is the kwargs
#: for a :meth:`Runner.start`; the whole list is dropped when the user stops.
_QUEUE: list = []
_QUEUE_LOCK = threading.Lock()

#: Hard cap on the legs one sweep may queue. Selecting several attacker versions
#: AND several defender versions asks for their cartesian product, and each leg
#: re-runs every (attack, defense) cell with the 3B policy on the GPU -- so a
#: 4x4 pick is 16 full benchmarks. Refused up front rather than queued quietly.
MAX_SWEEP_LEGS = 12


def _bench_finished(runner, state, _code):
    """Start the next queued version, or report the sweep done."""
    with _QUEUE_LOCK:
        if state != "finished":
            # A failed or stopped leg abandons the rest: the remaining versions
            # would face whatever went wrong with this one, and burying that in a
            # queue that kept going is how a sweep produces a table nobody can
            # trust. What has already run stays on the page.
            dropped = len(_QUEUE)
            _QUEUE.clear()
            if dropped:
                runner.bus.emit("queue_abandoned", remaining=dropped, after=state)
            return
        job = _QUEUE.pop(0) if _QUEUE else None
    if job is None:
        return
    try:
        runner.start(**job)
    except RunError as exc:
        runner.bus.emit("queue_abandoned", remaining=0, after="error", error=str(exc))


TRAIN = Runner("train")
BENCH = Runner("bench", on_end=_bench_finished)


# ---------------------------------------------------------------------------
# Argv specs. One table per CLI: what the browser may send, which flag it maps
# to, and how it is validated. Anything not in a table cannot reach the argv.
# ---------------------------------------------------------------------------

def _num(kind, lo=None, hi=None):
    def check(value):
        try:
            out = kind(value)
        except (TypeError, ValueError):
            raise ValueError(f"expected {kind.__name__}, got {value!r}") from None
        if lo is not None and out < lo:
            raise ValueError(f"must be >= {lo}, got {out}")
        if hi is not None and out > hi:
            raise ValueError(f"must be <= {hi}, got {out}")
        return str(out)
    return check


def _choice(*allowed):
    def check(value):
        out = str(value).strip()
        if out not in allowed:
            raise ValueError(f"must be one of {list(allowed)}, got {out!r}")
        return out
    return check


def _csv(allowed):
    """A comma-joined subset of ``allowed``, order preserved as the user sent it."""
    def check(value):
        items = value if isinstance(value, list) else str(value).split(",")
        items = [str(x).strip() for x in items if str(x).strip()]
        if not items:
            raise ValueError("pick at least one")
        bad = [x for x in items if x not in allowed]
        if bad:
            raise ValueError(f"unknown: {bad} (available: {list(allowed)})")
        seen, out = set(), []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return ",".join(out)
    return check


def _goal(value):
    """``untargeted_degrade=0.1`` and friends -- the exact ``--goal`` grammar."""
    spec = str(value).strip()
    gtype, _, val = spec.partition("=")
    gtype = gtype.strip()
    if gtype not in ("untargeted_degrade", "slow_degrade", "targeted_label"):
        raise ValueError(f"unknown goal type {gtype!r}")
    val = val.strip()
    if val:
        try:
            int(val) if gtype == "targeted_label" else float(val)
        except ValueError:
            raise ValueError(f"bad goal value {val!r}") from None
    return spec


def _outdir(value):
    """A relative output directory under the repo.

    Belt and braces, because this string reaches the filesystem: the syntactic
    checks reject the obvious escapes, and the resolved path is then verified to
    land inside the repo -- which is what actually decides it.

    ``os.path.isabs`` alone is not enough on Windows. Python 3.13 changed
    ``ntpath.isabs`` so a single leading slash is "root-relative" rather than
    absolute, and ``isabs("/etc/passwd")`` now returns False there; the leading
    separator is therefore rejected explicitly rather than inferred.
    """
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        raise ValueError("empty output directory")
    if raw.startswith("/") or raw.startswith("~") or os.path.isabs(raw):
        raise ValueError(f"output must be a relative path inside the repo: {raw!r}")
    parts = raw.split("/")
    if ".." in parts:
        raise ValueError(f"output must not climb out of the repo: {raw!r}")
    for part in parts:
        if part and not all(c.isalnum() or c in "._-" for c in part):
            raise ValueError(f"unsafe path segment {part!r}")
    resolved = os.path.abspath(os.path.join(REPO_ROOT, raw))
    if not (resolved == REPO_ROOT or resolved.startswith(REPO_ROOT + os.sep)):
        raise ValueError(f"output resolves outside the repo: {raw!r}")
    return raw


#: ``main.py`` -- {json key: (flag, validator) | (flag, None) for a bare switch}
TRAIN_SPEC = {
    "rounds": ("--rounds", _num(int, 1, 10_000_000)),
    "poisoners": ("--poisoners", _num(int, 1, 10_000)),
    "learn": ("--learn", _choice("attacker", "defender", "both")),
    "env": ("--env", _choice("linux", "windows")),
    "fresh": ("--fresh", None),
    "debug": ("--debug", None),
}

#: ``benchmark.run_benchmark``
BENCH_SPEC = {
    "rounds": ("--rounds", _num(int, 1, 100_000)),
    "goal": ("--goal", _goal),
    "max_poison_clients": ("--max-poison-clients", _num(int, 1, 10_000)),
    "n_clients": ("--n-clients", _num(int, 2, 10_000)),
    "baseline_knowledge": ("--baseline-knowledge", _choice("partial", "full")),
    "attack_temperature": ("--attack-temperature", _num(float, 0.0, 5.0)),
    "attack_retries": ("--attack-retries", _num(int, 0, 20)),
    "defender_temperature": ("--defender-temperature", _num(float, 0.0, 5.0)),
    "device": ("--device", _choice("cuda", "cpu")),
    "seed": ("--seed", _num(int, 0, 2**31 - 1)),
    "log_every": ("--log-every", _num(int, 1, 10_000)),
    "root_size": ("--root-size", _num(int, 1, 1_000_000)),
    "root_epochs": ("--root-epochs", _num(int, 1, 10_000)),
    "root_lr": ("--root-lr", _num(float, 0.0, 10.0)),
    "eta": ("--eta", _num(float, 0.0, 100.0)),
    "defl_delta": ("--defl-delta", _num(float, 0.0, 10.0)),
    "defl_tau": ("--defl-tau", _num(float, 0.0, 100.0)),
    "dnc_num_byzantine": ("--dnc-num-byzantine", _num(int, 0, 10_000)),
    "dnc_c": ("--dnc-c", _num(float, 0.0, 100.0)),
    "dnc_niters": ("--dnc-niters", _num(int, 1, 1000)),
    "dnc_sub_dim": ("--dnc-sub-dim", _num(int, 1, 10_000_000)),
    "multikrum_f": ("--multikrum-f", _num(int, 0, 10_000)),
    "multikrum_m": ("--multikrum-m", _num(int, 1, 10_000)),
    "lie_z": ("--lie-z", _num(float, -100.0, 100.0)),
    "lie_sign": ("--lie-sign", _num(float, -1.0, 1.0)),
    "minmax_perturbation": ("--minmax-perturbation", _choice("std", "unit_vec", "sign")),
    "minmax_gamma0": ("--minmax-gamma0", _num(float, 0.0, 1e6)),
    "minsum_bound": ("--minsum-bound", _choice("max", "min")),
    "fang_b": ("--fang-b", _num(float, 0.0, 1e6)),
    "fang_krum_f": ("--fang-krum-f", _num(int, 0, 10_000)),
    "fang_krum_lambda_mult": ("--fang-krum-lambda-mult", _num(float, 0.0, 1e6)),
    "ipm_epsilon": ("--ipm-epsilon", _num(float, -1e6, 1e6)),
    "mimic_warmup": ("--mimic-warmup", _num(int, 0, 100_000)),
    "signflip_c": ("--signflip-c", _num(float, -1e6, 1e6)),
    "noise_sigma": ("--noise-sigma", _num(float, 0.0, 1e6)),
    "scaling_gamma": ("--scaling-gamma", _num(float, -1e6, 1e6)),
    "labelflip_mode": ("--labelflip-mode", _choice("reverse", "next", "random")),
    "benign_retrain": ("--benign-retrain", None),
    "no_eval_cache": ("--no-eval-cache", None),
    "no_plot": ("--no-plot", None),
    "fresh": ("--fresh", None),
}


def _build_flags(spec: dict, payload: dict) -> list:
    """Turn the request body into argv fragments using ``spec``.

    Absent keys, ``None`` and ``""`` mean "leave it to the CLI default", which is
    what keeps a UI form from silently pinning every knob at whatever the widget
    happened to render.
    """
    out = []
    for key, (flag, validate) in spec.items():
        if key not in payload:
            continue
        value = payload[key]
        if validate is None:                    # a bare switch
            if value:
                out.append(flag)
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        try:
            out += [flag, validate(value)]
        except ValueError as exc:
            raise RunError(f"{key}: {exc}") from None
    return out


# ---------------------------------------------------------------------------
# Starting runs
# ---------------------------------------------------------------------------

def _new_run(kind: str, label: str = ""):
    """Claim a fresh run directory, and its id, by creating it.

    The stamp is second-resolution, and a multi-version sweep builds every leg's
    job in one pass -- so without the suffix all of them would take the same id and
    then overwrite each other's config, console log and results, leaving one leg's
    numbers standing in for the whole sweep. ``exist_ok=False`` is what makes the
    claim, so two callers cannot take the same directory.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{kind}-{stamp}"
    for suffix in range(0, 1000):
        run_id = base if suffix == 0 else f"{base}-{suffix + 1}"
        run_dir = os.path.join(RUNS_DIR, run_id)
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RunError(f"could not claim a run directory under {RUNS_DIR}")


def _guard_concurrency(payload: dict, other: Runner, what: str):
    """Both CLIs load a 3B model onto the GPU. Running them at once is legitimate
    on a CPU-only baseline run and a mistake everywhere else, so it needs saying
    out loud rather than being silently allowed or silently forbidden."""
    if other.running and not payload.get("force"):
        raise RunError(
            f"a {other.kind} run is already in progress and both would compete for "
            f"the same GPU. Stop it first, or resend with force=true if you know "
            f"{what} does not need the GPU (a CPU baselines-only benchmark, say).")


def _write_config(payload: dict, run_id: str, run_dir: str):
    """Materialise this run's config: base.yaml + the UI's validated overrides."""
    source = payload.get("config") or configstore.BASE_CONFIG
    if source != configstore.BASE_CONFIG:
        source = _outdir(source)             # only in-repo config paths
    if not os.path.isfile(source):
        raise RunError(f"no such config file: {source}")
    overrides = payload.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise RunError("overrides must be an object of {path: value}")
    base = configstore.load(source)
    try:
        merged, changes = configstore.apply_overrides(base, overrides)
    except configstore.OverrideError as exc:
        raise RunError(str(exc)) from None
    dest = os.path.join(run_dir, "config.yaml")
    configstore.write_run_config(merged, dest, run_id, changes, source=source)
    return dest, changes, merged


@contextlib.contextmanager
def _discard_on_failure(claimed: list):
    """Remove the run directories in ``claimed`` if the block does not launch.

    A run directory is claimed (by creating it) before the run is known to be
    valid, because the derived config has to exist on disk before the flags can be
    checked against it -- so a refusal has to take its directory back, or every
    rejected click leaves a contentless run in the history list.

    ``claimed`` is read on the way out rather than copied on the way in, so a
    sweep can keep appending to it as it builds each leg and a refusal on leg 3
    still removes legs 0-2.
    """
    try:
        yield
    except BaseException:
        for run_dir in claimed:
            shutil.rmtree(run_dir, ignore_errors=True)
        raise


def _validate_flags_against_config(payload: dict, merged: dict):
    """Refuse an impossible ``--poisoners`` / ``--learn`` HERE, not 400ms later.

    ``--learn defender`` needs a trainable defender, so it is an error under
    ``defense.mode: algorithmic`` -- and the shipped config IS algorithmic, which
    makes "train the defender" the most natural first click in the panel and a
    run that starts, prints an argparse error and dies. The rule already exists,
    once, in the CLI's own resolver, so it is REUSED rather than restated: the
    flags are applied to a throwaway copy of the config this run would use, and
    whatever ``main.py`` would have refused is refused now, in the response to
    the click, with the same wording.

    ``core.config_overrides`` is deliberately torch-free and import-light (its
    module docstring says so), so this costs the panel nothing.
    """
    import copy
    import logging as _logging

    from core.config_overrides import apply_learner_choice, apply_poisoner_count

    probe = copy.deepcopy(merged)
    # These resolvers narrate what they changed, and warn about e.g. a poisoner
    # count past half the federation. That narration belongs to the run that
    # actually applies them -- the subprocess emits all of it a moment later, so
    # echoing it from the server would put two accounts of one override in two
    # logs. Verbosity is all that races here.
    quiet = _logging.getLogger("core.config_overrides")
    before = quiet.level
    quiet.setLevel(_logging.ERROR)
    try:
        if payload.get("poisoners") not in (None, ""):
            apply_poisoner_count(probe, int(payload["poisoners"]))
        if payload.get("learn"):
            apply_learner_choice(probe, str(payload["learn"]))
    except (ValueError, TypeError) as exc:
        raise RunError(str(exc)) from None
    finally:
        quiet.setLevel(before)


def start_training(payload: dict) -> dict:
    _guard_concurrency(payload, BENCH, "training")
    mode = str(payload.get("mode", "train"))
    if mode not in ("train", "dry-run", "baseline"):
        raise RunError(f"unknown mode {mode!r}; use train | dry-run | baseline")

    # The run directory is claimed before the run is known to be startable (the
    # config has to be MATERIALISED before it can be validated), so a refusal has
    # to take its directory back with it -- otherwise every rejected click leaves a
    # contentless run in the history list.
    run_id, run_dir = _new_run("train")
    with _discard_on_failure([run_dir]):
        cfg_path, changes, merged = _write_config(payload, run_id, run_dir)
        _validate_flags_against_config(payload, merged)

    argv = [python_exe(), "-u", "main.py", "--config", cfg_path]
    argv += _build_flags(TRAIN_SPEC, payload)
    if mode == "dry-run":
        argv.append("--dry-run")
    elif mode == "baseline":
        argv.append("--baseline")

    fl = merged.get("fl", {}) or {}
    attack = merged.get("attack", {}) or {}
    meta = {
        "mode": mode,
        "config": cfg_path.replace("\\", "/"),
        "overrides": changes,
        "n_clients": fl.get("n_clients"),
        "n_compromisable": fl.get("n_compromisable"),
        "fixed_poison_clients": attack.get("fixed_poison_clients"),
        "max_poison_clients": attack.get("max_poison_clients"),
        "goal": attack.get("goal"),
        "defense_mode": (merged.get("defense", {}) or {}).get("mode"),
        # Which side --learn pinned, so the page can orient itself (defender
        # charts vs attacker charts) before the first round log arrives. None =
        # the config's own rl.learners decides.
        "learn": payload.get("learn") or None,
        "defense_algorithms": (merged.get("defense", {}) or {}).get("algorithms"),
        "curriculum": merged.get("curriculum"),
        "device": fl.get("device"),
        "target_rounds": payload.get("rounds") or fl.get("simulation_rounds"),
        "training_rounds": fl.get("training_rounds"),
        "save_every": (merged.get("rl", {}) or {}).get("save_every"),
    }
    _record_manifest(run_dir, run_id, "train", argv, meta)
    return TRAIN.start(argv, run_id=run_id, run_dir=run_dir,
                       label=payload.get("label", ""), meta=meta, tail_rounds=True)


def _selected_versions(payload: dict, plural: str = "versions",
                       single: str = "version") -> list:
    """The version ids one ROLE should sweep, in the order the user picked them.

    ``version`` (one) and ``versions`` (several) both work, so a single-version
    request stays exactly what it was before the queue existed. The key names are
    parameters because the two adapters are chosen independently -- see
    :func:`_sweep_axis`.
    """
    raw = payload.get(plural)
    if raw is None:
        raw = [payload.get(single) or "current"]
    if not isinstance(raw, list):
        raw = [raw]
    out, seen = [], set()
    for vid in raw:
        vid = str(vid).strip() or "current"
        vid = "current" if vid in ("current", "live") else vid
        if vid in seen:
            continue
        seen.add(vid)
        if vid != "current":
            versions.get(vid)         # validates the id, raises VersionError
        out.append(vid)
    if not out:
        raise RunError(f"pick at least one model version for the {single} role")
    return out


def _sweep_axis(ids: list, relevant: bool, role: str, needs: str) -> list:
    """One role's version axis, collapsed when the panel cannot tell its versions apart.

    A role's adapter only changes a leg's numbers when the panel actually RUNS
    that role: the attacker version is inert without the ``llm`` attack row, and
    the defender version is inert without the ``llm_defender`` defense column.
    Several picks on an inert axis would queue byte-identical benchmarks and then
    lay them out as a comparison, so that is refused rather than silently
    collapsed -- the user asked for a difference the panel cannot produce.

    Returns ``[None]`` for an inert axis: one leg, and no adapter flag for that
    role, which is exactly what a baselines-only or algorithmic-defense-only run
    was already doing.
    """
    if relevant:
        return list(ids)
    if len(ids) > 1:
        raise RunError(
            f"{len(ids)} {role} versions were selected but the panel has no "
            f"{needs}, so every leg would run identical numbers and the "
            f"comparison would be N copies of one result. Add the {needs} to "
            f"the panel, or pick one {role} version.")
    return [None]


def _version_label(version_id) -> str:
    """A leg's human name for one role. ``None`` is an axis the panel does not
    run at all (no ``llm`` attack, no ``llm_defender`` column), which is different
    from "the live checkpoint" and should not be labelled as it."""
    if version_id is None:
        return "n/a"
    if version_id in ("current", "live"):
        return "live checkpoint"
    return versions.get(version_id).get("label", version_id)


def _resolve_adapter(version_id, role: str, needs: str, hint: str) -> str:
    """Where this leg loads ``role`` from -- or a refusal naming what is missing.

    The silent alternative is the trap this closes. ``adapter_path`` returns
    ``None`` for a version that does not hold the role, and passing no
    ``--<role>-adapter`` does not mean "skip it": the CLI falls back to the LIVE
    ``checkpoints/<role>_adapter``. So selecting a version whose defender was
    never trained used to benchmark whatever the last training run happened to
    leave on disk and label the result with the selected version -- or, with
    nothing on disk, drop the column and leave a gap in the matrix with the
    reason only in the console. Both are worse than not starting.
    """
    path = versions.adapter_path(version_id, role)
    if path is not None:
        return path
    where = ("the live checkpoint has none" if version_id in (None, "current")
             else f"version {version_id} has none")
    raise RunError(
        f"{needs} needs a trained {role} adapter and {where} -- {hint}")


def start_benchmark(payload: dict) -> dict:
    """Queue one benchmark subprocess per swept version and start the first.

    Each leg is an ordinary ``benchmark.run_benchmark`` invocation with its own
    output directory and its own ``--attacker-adapter`` / ``--defender-adapter``
    -- the sweep is a sequence of the runs you would have typed, not a new
    evaluation mode. They run one at a time because each loads the 3B policy onto
    the GPU.

    The two adapters are picked INDEPENDENTLY, because that is how they are
    produced: ``--learn`` trains one side against a frozen opponent, so the
    defender worth evaluating and the attacker worth evaluating it against
    normally come from different snapshots. The legs are therefore the product of
    the two axes -- and an axis the panel cannot distinguish (no ``llm`` attack,
    no ``llm_defender`` defense) collapses to one leg. See :func:`_sweep_axis`.
    """
    from benchmark.attacks import AVAILABLE as ATTACKS
    from benchmark.defenses import AVAILABLE as DEFENSES

    _guard_concurrency(payload, TRAIN, "the benchmark")
    if BENCH.running:
        raise RunError("a benchmark is already in progress")

    attacks = _csv(ATTACKS)(payload.get("attacks") or ["llm"])
    defenses = _csv(DEFENSES)(payload.get("defenses") or ["fedavg"])
    wants_llm = "llm" in attacks.split(",")
    wants_llm_defender = "llm_defender" in defenses.split(",")

    attacker_axis = _sweep_axis(_selected_versions(payload),
                                wants_llm, "attacker", "'llm' row")
    defender_axis = _sweep_axis(
        _selected_versions(payload, "defender_versions", "defender_version"),
        wants_llm_defender, "defender", "'llm_defender' column")

    legs = [(a, d) for a in attacker_axis for d in defender_axis]
    if len(legs) > MAX_SWEEP_LEGS:
        raise RunError(
            f"{len(attacker_axis)} attacker x {len(defender_axis)} defender "
            f"versions is {len(legs)} full benchmarks, past the cap of "
            f"{MAX_SWEEP_LEGS}. Sweep one axis at a time: pin one side and vary "
            f"the other.")
    # Which axis this sweep is a comparison ALONG. The page needs it to know
    # whether to score the `llm` row or the `llm_defender` column, and a leg's
    # label should name the version that is actually varying.
    axis = ("both" if len(attacker_axis) > 1 and len(defender_axis) > 1
            else "defender" if len(defender_axis) > 1
            else "attacker")

    common = _build_flags(BENCH_SPEC, payload)
    jobs = []
    # A sweep claims every leg's directory in one pass, and a refusal on leg 3
    # has to take legs 0-2 with it: the sweep is not starting, so its half-built
    # run records are not history.
    claimed: list = []
    with _discard_on_failure(claimed):
        for index, (version_id, defender_version_id) in enumerate(legs):
            run_id, run_dir = _new_run("bench")
            claimed.append(run_dir)
            cfg_path, changes, merged = _write_config(payload, run_id, run_dir)

            # Validate what the BROWSER sent; trust what we built ourselves. Running
            # the server-generated default through _outdir would only re-check a path
            # this process just constructed -- and would reject it outright under a
            # test or a deployment whose runs directory is absolute.
            supplied_out = payload.get("out")
            out_dir = (_outdir(supplied_out) if supplied_out and len(legs) == 1
                       else posixpath.join(run_dir.replace("\\", "/"), "result"))

            attacker_adapter = defender_adapter = None
            if wants_llm:
                attacker_adapter = _resolve_adapter(
                    version_id, "attacker", "the 'llm' attack",
                    "train first, or drop 'llm' from the attack panel to run the "
                    "published baselines only.")
            if wants_llm_defender:
                defender_adapter = _resolve_adapter(
                    defender_version_id, "defender", "the 'llm_defender' defense",
                    "train a defender first (defense.mode: llm with --learn "
                    "defender), or drop 'llm_defender' from the defense panel.")

            argv = [python_exe(), "-u", "-m", "benchmark.run_benchmark",
                    "--events", "-", "--config", cfg_path, "--out", out_dir]
            argv += list(common)
            argv += ["--attacks", attacks, "--defenses", defenses]
            if attacker_adapter:
                argv += ["--attacker-adapter", attacker_adapter.replace("\\", "/")]
            if defender_adapter:
                argv += ["--defender-adapter", defender_adapter.replace("\\", "/")]

            att_label = _version_label(version_id)
            def_label = _version_label(defender_version_id)
            label = ({"defender": def_label,
                      "both": f"{att_label} att x {def_label} def"}
                     .get(axis, att_label))
            meta = {
                "config": cfg_path.replace("\\", "/"),
                "overrides": changes,
                "version": version_id,
                "version_label": att_label,
                "defender_version": defender_version_id,
                "defender_version_label": def_label,
                "attacker_adapter": attacker_adapter,
                "defender_adapter": defender_adapter,
                "attacks": attacks.split(","),
                "defenses": defenses.split(","),
                "out": out_dir,
                "rounds": payload.get("rounds"),
                "goal": payload.get("goal"),
                "n_clients": (merged.get("fl", {}) or {}).get("n_clients"),
                "queue": {"index": index, "total": len(legs), "axis": axis,
                          "versions": attacker_axis,
                          "defender_versions": defender_axis,
                          "version": version_id,
                          "defender_version": defender_version_id,
                          "label": label},
            }
            _record_manifest(run_dir, run_id, "bench", argv, meta)
            jobs.append({"argv": argv, "run_id": run_id, "run_dir": run_dir,
                         "label": payload.get("label", ""), "meta": meta,
                         "tail_rounds": False, "clear_bus": index == 0})

    with _QUEUE_LOCK:
        _QUEUE.clear()
        _QUEUE.extend(jobs[1:])
    try:
        return BENCH.start(**jobs[0])
    except BaseException:
        with _QUEUE_LOCK:
            _QUEUE.clear()
        raise


def _record_manifest(run_dir: str, run_id: str, kind: str, argv: list, meta: dict):
    """Write what this run is, next to its console log, so the Runs tab can list
    past runs without the server having to remember anything across restarts."""
    payload = {"run_id": run_id, "kind": kind, "argv": argv,
               "started": datetime.datetime.now().isoformat(timespec="seconds"),
               **meta}
    try:
        with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Past runs
# ---------------------------------------------------------------------------

def list_runs(limit: int = 60) -> list:
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RUNS_DIR), reverse=True)[:limit]:
        run_dir = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(run_dir):
            continue
        manifest = _read_json(os.path.join(run_dir, "run.json")) or {"run_id": name}
        manifest["dir"] = run_dir.replace("\\", "/")
        result = manifest.get("out") or os.path.join(run_dir, "result")
        manifest["has_history"] = os.path.isfile(os.path.join(result, "history.json"))
        manifest["has_console"] = os.path.isfile(os.path.join(run_dir, "console.log"))
        out.append(manifest)
    return out


def run_detail(run_id: str) -> dict:
    if not run_id or not all(c.isalnum() or c in "-_" for c in run_id):
        raise RunError(f"bad run id {run_id!r}")
    run_dir = os.path.join(RUNS_DIR, run_id)
    if not os.path.isdir(run_dir):
        raise RunError(f"no such run: {run_id}")
    manifest = _read_json(os.path.join(run_dir, "run.json")) or {"run_id": run_id}
    result = manifest.get("out") or os.path.join(run_dir, "result")
    history = _read_json(os.path.join(result, "history.json"))
    console = ""
    try:
        with open(os.path.join(run_dir, "console.log"), encoding="utf-8",
                  errors="replace") as f:
            console = f.read()[-200_000:]
    except OSError:
        pass
    return {"manifest": manifest, "history": history, "console": console,
            "dir": run_dir.replace("\\", "/")}


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Bootstrap payload
# ---------------------------------------------------------------------------

def bootstrap() -> dict:
    from benchmark.attacks import (AVAILABLE as ATTACKS, BASELINES,
                                   DEFAULT as ATTACKS_DEFAULT)
    from benchmark.defenses import AVAILABLE as DEFENSES

    return {
        "config": configstore.describe(),
        "attacks": {"available": ATTACKS, "default": ATTACKS_DEFAULT,
                    "baselines": BASELINES},
        "defenses": {"available": DEFENSES},
        "versions": versions.listing(),
        "live": versions.live_status(),
        "train": TRAIN.status(),
        "bench": BENCH.status(),
        "runs": list_runs(),
        "repo": REPO_ROOT.replace("\\", "/"),
        "python": python_exe(),
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "zero-touch-fl-webui"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # quieter than the stdlib default
        if self.path.startswith("/api/events"):
            return
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- plumbing -----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # The page is entirely self-contained; forbidding remote loads means a
        # stray URL in a config value cannot turn the panel into a beacon.
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data:; "
                         "style-src 'self' 'unsafe-inline'; script-src 'self'")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload, code: int = 200):
        body = json.dumps(payload, default=str).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, code: int, message: str):
        self._json({"error": message}, code)

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        if length > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ValueError("request body is not valid JSON") from None
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        path, query = url.path, parse_qs(url.query)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/bootstrap":
                return self._json(bootstrap())
            if path == "/api/config":
                return self._json(configstore.describe())
            if path == "/api/versions":
                return self._json({"versions": versions.listing(),
                                   "live": versions.live_status()})
            if path == "/api/runs":
                return self._json({"runs": list_runs()})
            if path == "/api/run":
                return self._json(run_detail((query.get("id") or [""])[0]))
            if path == "/api/status":
                return self._json({"train": TRAIN.status(), "bench": BENCH.status()})
            if path == "/api/events":
                return self._events(query)
        except RunError as exc:
            return self._error(400, str(exc))
        except versions.VersionError as exc:
            return self._error(404, str(exc))
        except Exception as exc:                       # noqa: BLE001 - report, don't die
            logger.exception("GET %s failed", path)
            return self._error(500, f"{type(exc).__name__}: {exc}")
        return self._error(404, f"no route for {path}")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._body()
        except ValueError as exc:
            return self._error(400, str(exc))
        try:
            if path == "/api/train/start":
                return self._json(start_training(payload))
            if path == "/api/train/stop":
                return self._json(TRAIN.stop(force=bool(payload.get("force"))))
            if path == "/api/bench/start":
                return self._json(start_benchmark(payload))
            if path == "/api/bench/stop":
                # Stopping means stopping the SWEEP, not just its current leg. The
                # queue is emptied FIRST so that a click landing in the moment
                # between two legs still cancels the rest -- and that cancellation
                # is reported even though there is no process to signal.
                with _QUEUE_LOCK:
                    dropped, _QUEUE[:] = len(_QUEUE), []
                if dropped:
                    BENCH.bus.emit("queue_abandoned", remaining=dropped,
                                   after="stopped by the user")
                try:
                    return self._json(BENCH.stop(force=bool(payload.get("force"))))
                except RunError:
                    if not dropped:
                        raise
                    return self._json({**BENCH.status(),
                                       "cancelled_queue": dropped})
            if path == "/api/versions":
                rec = versions.create(
                    label=str(payload.get("label", ""))[:120],
                    notes=str(payload.get("notes", ""))[:2000],
                    base_model=str(payload.get("base_model", ""))[:200])
                return self._json({"version": rec, "versions": versions.listing()})
            if path == "/api/versions/delete":
                vid = versions.delete(str(payload.get("id", "")))
                return self._json({"deleted": vid, "versions": versions.listing()})
            if path == "/api/versions/rename":
                rec = versions.rename(str(payload.get("id", "")),
                                      label=payload.get("label"),
                                      notes=payload.get("notes"))
                return self._json({"version": rec, "versions": versions.listing()})
            if path == "/api/config/preview":
                run_id, run_dir = "preview", os.path.join(RUNS_DIR, "_preview")
                _, changes, merged = _write_config(payload, run_id, run_dir)
                return self._json({"changes": changes, "config": merged})
        except (RunError, configstore.OverrideError) as exc:
            return self._error(400, str(exc))
        except versions.VersionError as exc:
            return self._error(400, str(exc))
        except Exception as exc:                       # noqa: BLE001
            logger.exception("POST %s failed", path)
            return self._error(500, f"{type(exc).__name__}: {exc}")
        return self._error(404, f"no route for {path}")

    # -- helpers ------------------------------------------------------------
    def _events(self, query):
        kind = (query.get("kind") or ["train"])[0]
        runner = BENCH if kind == "bench" else TRAIN
        try:
            since = int((query.get("since") or ["0"])[0])
        except ValueError:
            since = 0
        events, latest = runner.bus.since(since, timeout=20.0)
        self._json({"events": events, "seq": latest, "status": runner.status()})

    def _static(self, rel: str):
        rel = posixpath.normpath("/" + rel.replace("\\", "/")).lstrip("/")
        full = os.path.join(STATIC, *rel.split("/"))
        if not os.path.abspath(full).startswith(os.path.abspath(STATIC) + os.sep):
            return self._error(403, "outside the static root")
        try:
            with open(full, "rb") as f:
                body = f.read()
        except OSError:
            return self._error(404, f"no such asset: {rel}")
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(200, body, ctype)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Web control panel for zero-touch-fl training + benchmarking")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default: 127.0.0.1 -- loopback only)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a browser window on start")
    ap.add_argument("--debug", action="store_true", help="verbose server logging")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    os.chdir(REPO_ROOT)      # every path in this repo's CLIs is repo-relative
    os.makedirs(RUNS_DIR, exist_ok=True)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/"
    logger.info("serving %s from %s", url, REPO_ROOT)
    if args.host == "0.0.0.0":
        logger.warning(
            "bound to 0.0.0.0 -- this panel starts processes on this machine and "
            "has no authentication. Use an SSH tunnel instead of exposing it.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        for runner in (TRAIN, BENCH):
            if runner.running:
                logger.info("stopping the %s run", runner.kind)
                try:
                    runner.stop(force=True)
                except RunError:
                    pass
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
