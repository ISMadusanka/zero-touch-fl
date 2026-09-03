"""Tests for the web control panel (``python -m webui``).

The panel's whole job is to turn browser input into a subprocess launch, so the
things worth testing are the boundaries where that could go wrong:

  * ``webui/configstore.py``  -- an override is accepted ONLY for a dotted path
    that already exists in ``configs/base.yaml``, only when it coerces to the type
    that is there now, and the result is written to a NEW file rather than over the
    checked-in config;
  * ``webui/server.py``       -- every field from the browser is validated into a
    list-form argv (no shell), unknown attacks/defenses are refused, and output
    paths cannot climb out of the repo;
  * ``webui/versions.py``     -- a snapshot is a copy, the live checkpoint is never
    mutated, and a version id coming back from the browser cannot address a
    directory outside the store;
  * ``webui/bus.py``          -- sequence numbers are monotonic across a clear, so a
    browser polling the previous run is not replayed stale events under new seqs.

No GPU, no LLM, no dataset, no network:

    python -m pytest tests/test_webui.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from webui import configstore, versions  # noqa: E402
from webui.bus import EventBus  # noqa: E402
from webui.server import (  # noqa: E402
    BENCH_SPEC, TRAIN_SPEC, _build_flags, _csv, _goal, _outdir,
)
from webui.runner import RunError, _shell_join  # noqa: E402


# ---------------------------------------------------------------------------
# configstore: the config the UI edits
# ---------------------------------------------------------------------------

def test_describe_covers_the_real_config():
    """The settings panel is generated from base.yaml, so a knob added there has
    to appear with no change to the UI."""
    desc = configstore.describe()
    names = [g["name"] for g in desc["groups"]]
    assert {"fl", "data", "attack", "defense", "curriculum", "rl"} <= set(names)
    for path in ("fl.n_clients", "attack.goal.target_accuracy_drop", "rl.G",
                 "defense.mode", "curriculum.enabled"):
        assert path in desc["fields"], f"{path} missing from the generated schema"
    assert desc["fields"]["fl.n_clients"]["type"] == "int"
    assert desc["fields"]["curriculum.enabled"]["type"] == "bool"
    assert desc["fields"]["defense.algorithms"]["type"] == "list"
    assert desc["fields"]["defense.mode"]["enum"] == ["algorithmic", "llm"]


def test_comments_become_help_text():
    """base.yaml's inline commentary is the documentation; the panel scrapes it
    rather than keeping a second copy that would go stale."""
    docs = configstore.scrape_comments()
    assert "rng seed" in docs["fl.poison_seed"].lower()
    assert "group size" in docs["rl.G"].lower()
    assert "target" in docs["attack.goal.target_accuracy_drop"].lower()
    # A continuation comment indented under its key belongs to that key, not to
    # whatever key happens to follow it.
    assert "curriculum" in docs["defense.selection"].lower()


@pytest.mark.parametrize("path,sent,expected", [
    ("rl.G", "8", 8),
    ("rl.G", 8, 8),
    ("fl.lr", "0.005", 0.005),
    ("curriculum.enabled", "false", False),
    ("curriculum.enabled", False, False),
    ("defense.mode", "llm", "llm"),
    ("defense.algorithms", "fltrust,dnc", ["fltrust", "dnc"]),
    ("defense.algorithms", ["fltrust", "dnc"], ["fltrust", "dnc"]),
    ("curriculum.poisoner_counts", "1,2,3", [1, 2, 3]),
    ("attack.target_choices", "0.05,0.2", [0.05, 0.2]),
])
def test_overrides_coerce_to_the_shipped_type(path, sent, expected):
    cfg = configstore.load()
    merged, _ = configstore.apply_overrides(cfg, {path: sent})
    node = merged
    for part in path.split("."):
        node = node[part]
    assert node == expected


def test_null_is_accepted_only_where_the_config_already_spells_auto():
    """`null` means "auto" for several knobs (defense.seed, fltrust.root_epochs,
    curriculum.algorithms). Refusing it would make those one-way from the UI."""
    cfg = configstore.load()
    merged, _ = configstore.apply_overrides(cfg, {"defense.seed": "null"})
    assert merged["defense"]["seed"] is None
    merged, _ = configstore.apply_overrides(cfg, {"curriculum.algorithms": ""})
    assert merged["curriculum"]["algorithms"] is None
    with pytest.raises(configstore.OverrideError):
        configstore.apply_overrides(cfg, {"rl.G": "null"})


@pytest.mark.parametrize("override", [
    {"rl.nonexistent": 1},                 # invented key
    {"rl": 1},                             # a whole section
    {"rl.reward": 1},                      # a nested section
    {"rl.G": "not-a-number"},              # wrong type
    {"fl.n_clients": "twenty"},
    {"rl.adapter_paths.attacker": "/etc"},  # frozen: points the run elsewhere
    {"data.cache_dir": "../.."},
])
def test_bad_overrides_are_refused(override):
    with pytest.raises(configstore.OverrideError):
        configstore.apply_overrides(configstore.load(), override)


def test_overrides_are_all_or_nothing():
    """A half-applied config is a run nobody can reproduce, so validation happens
    before anything is written into the copy."""
    cfg = configstore.load()
    before = cfg["rl"]["G"]
    with pytest.raises(configstore.OverrideError):
        configstore.apply_overrides(cfg, {"rl.G": 99, "rl.kl_beta": "nope"})
    assert cfg["rl"]["G"] == before


def test_the_checked_in_config_is_never_written(tmp_path):
    original = open(configstore.BASE_CONFIG, "rb").read()
    cfg = configstore.load()
    merged, changes = configstore.apply_overrides(cfg, {"rl.G": 16})
    dest = configstore.write_run_config(
        merged, str(tmp_path / "run" / "config.yaml"), "test-run", changes)
    assert open(configstore.BASE_CONFIG, "rb").read() == original
    body = open(dest, encoding="utf-8").read()
    assert "GENERATED by" in body
    assert "rl.G: 4 -> 16" in body                    # the header records the diff
    assert configstore.load(dest)["rl"]["G"] == 16    # and it round-trips


# ---------------------------------------------------------------------------
# server: browser input -> argv
# ---------------------------------------------------------------------------

def test_train_flags_build_a_list_argv():
    flags = _build_flags(TRAIN_SPEC, {
        "rounds": 12, "poisoners": 3, "learn": "attacker", "env": "linux",
        "fresh": True, "debug": False,
    })
    assert flags == ["--rounds", "12", "--poisoners", "3", "--learn", "attacker",
                     "--env", "linux", "--fresh"]
    assert all(isinstance(x, str) for x in flags)


def test_absent_and_blank_fields_leave_the_cli_default_alone():
    """A form widget renders a value whether or not the user touched it; sending
    every one of them would pin every knob at whatever the widget happened to
    show."""
    assert _build_flags(TRAIN_SPEC, {}) == []
    assert _build_flags(TRAIN_SPEC, {"rounds": "", "poisoners": None}) == []
    assert _build_flags(BENCH_SPEC, {"seed": ""}) == []


@pytest.mark.parametrize("payload", [
    {"rounds": "; rm -rf /"},
    {"rounds": -1},
    {"learn": "everyone"},
    {"env": "solaris"},
])
def test_bad_train_fields_are_refused(payload):
    with pytest.raises(RunError):
        _build_flags(TRAIN_SPEC, payload)


@pytest.mark.parametrize("payload", [
    {"baseline_knowledge": "omniscient"},
    {"device": "tpu"},
    {"attack_temperature": 99},
    {"minmax_perturbation": "whatever"},
    {"labelflip_mode": "sideways"},
])
def test_bad_bench_fields_are_refused(payload):
    with pytest.raises(RunError):
        _build_flags(BENCH_SPEC, payload)


def test_attack_and_defense_panels_must_be_registered_names():
    from benchmark.attacks import AVAILABLE as ATTACKS
    from benchmark.defenses import AVAILABLE as DEFENSES

    assert _csv(ATTACKS)(["llm", "lie"]) == "llm,lie"
    assert _csv(ATTACKS)("llm, lie ,llm") == "llm,lie"      # de-duped, order kept
    with pytest.raises(ValueError):
        _csv(ATTACKS)(["llm", "not_an_attack"])
    with pytest.raises(ValueError):
        _csv(DEFENSES)([])
    assert _csv(DEFENSES)(["fltrust"]) == "fltrust"


def test_goal_grammar_matches_the_cli():
    assert _goal("untargeted_degrade=0.1") == "untargeted_degrade=0.1"
    assert _goal("slow_degrade=0.02") == "slow_degrade=0.02"
    assert _goal("targeted_label=7") == "targeted_label=7"
    assert _goal("untargeted_degrade") == "untargeted_degrade"   # value optional
    for bad in ("wreck_it=0.1", "untargeted_degrade=abc", "targeted_label=x"):
        with pytest.raises(ValueError):
            _goal(bad)


@pytest.mark.parametrize("bad", [
    "/etc/passwd", "C:/Windows", "../../outside", "logs/../../etc",
    "~/secrets", "", "logs/a b",
])
def test_output_paths_cannot_leave_the_repo(bad):
    with pytest.raises(ValueError):
        _outdir(bad)


def test_reasonable_output_paths_are_accepted():
    assert _outdir("logs/benchmark") == "logs/benchmark"
    assert _outdir("logs\\webui\\runs\\bench-1\\result") == "logs/webui/runs/bench-1/result"


def test_shell_join_is_display_only():
    """It renders the argv for copy-pasting; the argv itself is never joined into
    a shell string when the process is launched."""
    assert _shell_join(["python", "-u", "main.py"]) == "python -u main.py"
    assert _shell_join(["python", "a b"]) == 'python "a b"'


# ---------------------------------------------------------------------------
# server: the version -> benchmark wiring
# ---------------------------------------------------------------------------

@pytest.fixture
def captured_launch(tmp_path, monkeypatch):
    """Run start_training / start_benchmark up to the Popen and capture the argv.

    Everything up to that point is the part under test -- flag validation, the
    derived config, and which adapter directory the run is pointed at. Actually
    launching a GPU process is not.
    """
    from webui import server as srv

    calls = []

    def fake_start(self, argv, **kwargs):
        calls.append({"argv": argv, "kind": self.kind, **kwargs})
        return {"state": "running", "argv": argv}

    monkeypatch.setattr(srv.Runner, "start", fake_start)
    monkeypatch.setattr(srv.Runner, "running", property(lambda self: False))
    monkeypatch.setattr(srv, "RUNS_DIR", str(tmp_path / "runs"))
    # The sweep queue is module state; a test that leaves entries in it would make
    # the next one look like it queued work it never asked for.
    srv._QUEUE.clear()
    yield calls
    srv._QUEUE.clear()


def test_training_argv_is_the_command_a_shell_would_run(captured_launch):
    from webui.server import start_training

    start_training({"mode": "train", "rounds": 20, "poisoners": 4,
                    "learn": "attacker", "fresh": True,
                    "overrides": {"rl.G": 8}})
    argv = captured_launch[0]["argv"]
    assert argv[1:3] == ["-u", "main.py"]
    assert "--config" in argv and argv[argv.index("--config") + 1].endswith("config.yaml")
    assert argv[argv.index("--rounds") + 1] == "20"
    assert argv[argv.index("--poisoners") + 1] == "4"
    assert argv[argv.index("--learn") + 1] == "attacker"
    assert "--fresh" in argv and "--dry-run" not in argv and "--baseline" not in argv
    # The override reached the derived config, and the run points at THAT file
    # rather than at configs/base.yaml.
    derived = argv[argv.index("--config") + 1]
    assert configstore.load(derived)["rl"]["G"] == 8
    assert os.path.abspath(derived) != os.path.abspath(configstore.BASE_CONFIG)


@pytest.mark.parametrize("mode,flag", [("dry-run", "--dry-run"), ("baseline", "--baseline")])
def test_modes_map_to_their_flags(captured_launch, mode, flag):
    from webui.server import start_training

    start_training({"mode": mode})
    assert flag in captured_launch[0]["argv"]


def test_unknown_training_mode_is_refused(captured_launch):
    from webui.server import start_training

    with pytest.raises(RunError, match="unknown mode"):
        start_training({"mode": "go-wild"})


def test_benchmark_is_pointed_at_the_selected_version(captured_launch, store, monkeypatch):
    from webui.server import start_benchmark

    rec = versions.create(label="round 400")
    start_benchmark({"version": rec["id"], "rounds": 12, "device": "cpu",
                     "attacks": ["llm", "lie"], "defenses": ["fedavg", "fltrust"],
                     "goal": "untargeted_degrade=0.15"})
    argv = captured_launch[0]["argv"]
    assert argv[1:5] == ["-u", "-m", "benchmark.run_benchmark", "--events"]
    assert argv[5] == "-"                       # events interleaved into stdout
    adapter = argv[argv.index("--attacker-adapter") + 1]
    assert adapter.endswith(f"versions/{rec['id']}/attacker_adapter")
    assert argv[argv.index("--attacks") + 1] == "llm,lie"
    assert argv[argv.index("--defenses") + 1] == "fedavg,fltrust"
    assert argv[argv.index("--goal") + 1] == "untargeted_degrade=0.15"
    assert captured_launch[0]["meta"]["version_label"] == "round 400"


def test_benchmark_on_the_live_checkpoint_needs_no_version(captured_launch, store):
    from webui.server import start_benchmark

    start_benchmark({"version": "current", "attacks": ["llm"], "defenses": ["fedavg"]})
    argv = captured_launch[0]["argv"]
    assert argv[argv.index("--attacker-adapter") + 1].endswith("attacker_adapter")
    assert "versions" not in argv[argv.index("--attacker-adapter") + 1]


def test_the_llm_row_without_an_adapter_says_what_to_do(captured_launch, store):
    """The published baselines need no adapter and no GPU, so "drop llm" is a real
    option and the error should name it rather than just failing."""
    import shutil
    from webui.server import start_benchmark

    shutil.rmtree(store / "checkpoints" / "attacker_adapter")
    with pytest.raises(RunError, match="drop 'llm'"):
        start_benchmark({"version": "current", "attacks": ["llm"], "defenses": ["fedavg"]})
    # ...and a baselines-only panel goes through with no adapter at all.
    start_benchmark({"version": "current", "attacks": ["lie"], "defenses": ["fedavg"]})
    assert "--attacker-adapter" not in captured_launch[0]["argv"]


class _FakeProc:
    """Stands in for a live subprocess: alive until poll() is told otherwise."""
    pid = 4242

    def __init__(self, alive=True):
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


def test_a_second_run_is_refused_while_one_holds_the_gpu(monkeypatch, tmp_path):
    from webui import server as srv

    monkeypatch.setattr(srv, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(srv.TRAIN, "_proc", _FakeProc(), raising=False)
    with pytest.raises(RunError, match="already in progress"):
        srv.start_benchmark({"attacks": ["lie"], "defenses": ["fedavg"]})
    # force is the escape hatch for the case that genuinely does not need the GPU.
    monkeypatch.setattr(srv.Runner, "start",
                        lambda self, argv, **kw: {"state": "running", "argv": argv})
    out = srv.start_benchmark({"attacks": ["lie"], "defenses": ["fedavg"],
                               "device": "cpu", "force": True})
    assert out["state"] == "running"


# ---------------------------------------------------------------------------
# server: the multi-version benchmark sweep
# ---------------------------------------------------------------------------

def test_a_sweep_queues_one_ordinary_run_per_version(captured_launch, store):
    from webui import server as srv

    a = versions.create(label="early")
    b = versions.create(label="late")
    srv.start_benchmark({"versions": [a["id"], b["id"], "current"], "rounds": 5,
                         "attacks": ["llm", "lie"], "defenses": ["fedavg", "fltrust"]})
    # Only the FIRST leg launches now; the rest wait for its exit.
    assert len(captured_launch) == 1
    assert len(srv._QUEUE) == 2
    first = captured_launch[0]
    assert first["meta"]["queue"] == {
        # No `target` was sent (a direct API call), so the swept axis is inferred.
        "index": 0, "total": 3, "axis": "attacker", "target": None,
        "versions": [a["id"], b["id"], "current"],
        # No llm_defender column, so the defender axis is inert: one leg, no
        # adapter, and nothing for the comparison to vary along.
        "defender_versions": [None], "defender_version": None,
        "version": a["id"], "label": "early"}
    assert first["clear_bus"] is True
    # Each leg is a plain benchmark invocation differing only in its adapter+out.
    queued = srv._QUEUE[0]
    assert queued["clear_bus"] is False
    assert queued["meta"]["version_label"] == "late"
    assert queued["argv"][queued["argv"].index("--attacker-adapter") + 1].endswith(
        f"versions/{b['id']}/attacker_adapter")
    assert queued["meta"]["out"] != first["meta"]["out"]     # separate result dirs
    srv._QUEUE.clear()


def test_learn_defender_is_refused_before_the_process_starts(captured_launch):
    """``--learn defender`` needs a trainable defender, and the SHIPPED config
    defends with algorithms -- so the most natural first click in the panel used
    to start a run that printed an argparse error and died. The CLI's own resolver
    decides, here, in the response to the click."""
    from webui.server import start_training

    with pytest.raises(RunError, match="defense.mode"):
        start_training({"mode": "train", "learn": "defender"})
    assert captured_launch == []

    # With the defender LLM turned on in the same panel, it launches.
    start_training({"mode": "train", "learn": "defender",
                    "overrides": {"defense.mode": "llm"}})
    argv = captured_launch[0]["argv"]
    assert argv[argv.index("--learn") + 1] == "defender"
    assert captured_launch[0]["meta"]["learn"] == "defender"
    assert captured_launch[0]["meta"]["defense_mode"] == "llm"


def test_a_refused_launch_leaves_no_run_in_the_history(captured_launch, tmp_path):
    """A run directory is claimed before the run is known to be startable (the
    derived config has to exist on disk before the flags can be checked against
    it), so a refusal has to take it back -- or every rejected click adds a
    contentless entry to the Runs tab."""
    import os

    from webui import server as srv

    runs = tmp_path / "runs"
    runs.mkdir()
    srv.RUNS_DIR = str(runs)

    with pytest.raises(RunError):
        srv.start_training({"mode": "train", "learn": "defender"})
    assert os.listdir(runs) == []

    # ...and a sweep abandons every leg it had already claimed, not just the one
    # that failed.
    with pytest.raises(RunError):
        srv.start_benchmark({"versions": ["current"], "attacks": ["llm"],
                             "defenses": ["fedavg"]})
    assert os.listdir(runs) == []

    srv.start_training({"mode": "train", "learn": "attacker"})
    assert len(os.listdir(runs)) == 1


def test_an_impossible_poisoner_count_is_refused_before_launch(captured_launch):
    from webui.server import start_training

    with pytest.raises(RunError, match="exceeds fl.n_clients"):
        start_training({"mode": "train", "poisoners": 9999})
    assert captured_launch == []


def test_a_sweep_without_the_llm_row_is_refused(captured_launch, store):
    """Every leg would run byte-identical published baselines, so the sweep would
    produce N copies of one result and read as a comparison."""
    from webui import server as srv

    a = versions.create(label="a")
    b = versions.create(label="b")
    with pytest.raises(RunError, match="no 'llm' row"):
        srv.start_benchmark({"versions": [a["id"], b["id"]],
                             "attacks": ["lie"], "defenses": ["fedavg"]})
    assert srv._QUEUE == []


def test_a_failed_leg_abandons_the_rest(captured_launch, store):
    """A sweep that kept going past a failure would hand back a table where some
    rows mean something and others do not, with nothing in the table saying which."""
    from webui import server as srv

    a, b = versions.create(label="a"), versions.create(label="b")
    srv.start_benchmark({"versions": [a["id"], b["id"]], "attacks": ["llm"],
                         "defenses": ["fedavg"]})
    assert len(srv._QUEUE) == 1
    srv._bench_finished(srv.BENCH, "failed", 1)
    assert srv._QUEUE == []
    assert len(captured_launch) == 1                       # the second never ran
    assert srv.BENCH.bus.snapshot()[-1]["event"] == "queue_abandoned"


def test_a_finished_leg_starts_the_next(captured_launch, store):
    from webui import server as srv

    a, b = versions.create(label="a"), versions.create(label="b")
    srv.start_benchmark({"versions": [a["id"], b["id"]], "attacks": ["llm"],
                         "defenses": ["fedavg"]})
    srv._bench_finished(srv.BENCH, "finished", 0)
    assert srv._QUEUE == []
    assert len(captured_launch) == 2
    assert captured_launch[1]["meta"]["queue"]["index"] == 1


# ---------------------------------------------------------------------------
# The defender adapter: trained by --learn defender, evaluated as the
# `llm_defender` column. Its version is picked INDEPENDENTLY of the attacker's,
# because a one-sided training run only ever writes one of the two.
# ---------------------------------------------------------------------------

def _train_a_defender(store):
    """Put a stub defender adapter where a ``--learn defender`` run would leave one."""
    live = store / "checkpoints" / "defender_adapter"
    live.mkdir(parents=True, exist_ok=True)
    (live / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "unsloth/Qwen2.5-3B-Instruct", "r": 16,
         "lora_alpha": 32}))
    (live / "adapter_model.safetensors").write_bytes(b"stub defender weights")
    return live


def test_a_defender_only_run_can_be_snapshotted(store):
    """``rl/schedule.py`` writes ONLY the trainable side, so a --learn defender run
    leaves no attacker adapter at all -- and requiring one meant the single adapter
    that run produced could not be versioned."""
    import shutil
    shutil.rmtree(store / "checkpoints" / "attacker_adapter")
    _train_a_defender(store)

    rec = versions.create(label="defender v1")
    assert rec["roles"] == ["defender"]
    assert "attacker" not in rec["adapters"]
    # The LoRA shape is read off the adapter this version actually holds; taking it
    # from a directory that is not there recorded base_model "" and silently
    # disabled the benchmark's base-model check.
    assert rec["base_model"] == "unsloth/Qwen2.5-3B-Instruct"
    assert rec["lora_r"] == 16

    listed = versions.get(rec["id"])
    assert listed["available"] == {"attacker": False, "defender": True}
    assert versions.adapter_path(rec["id"], "defender") is not None
    assert versions.adapter_path(rec["id"], "attacker") is None


def test_the_defender_column_loads_the_defender_version(captured_launch, store):
    """The two adapters come from different snapshots, so they are chosen
    separately: the attack row under test against the defense row under test."""
    from webui.server import start_benchmark

    attacker = versions.create(label="attacker v1")
    _train_a_defender(store)
    defender = versions.create(label="defender v1")

    start_benchmark({"versions": [attacker["id"]],
                     "defender_versions": [defender["id"]],
                     "attacks": ["llm"], "defenses": ["fedavg", "llm_defender"]})
    argv = captured_launch[0]["argv"]
    assert argv[argv.index("--attacker-adapter") + 1].endswith(
        "versions/" + attacker["id"] + "/attacker_adapter")
    assert argv[argv.index("--defender-adapter") + 1].endswith(
        "versions/" + defender["id"] + "/defender_adapter")
    meta = captured_launch[0]["meta"]
    assert meta["defender_version"] == defender["id"]
    assert meta["defender_version_label"] == "defender v1"


def test_a_version_without_the_defender_it_was_told_to_run_is_refused(
        captured_launch, store):
    """Passing no --defender-adapter does NOT mean "skip it": the CLI falls back to
    the live checkpoint. So a version whose defender was never trained used to
    benchmark whatever the last run left on disk and label it with that version."""
    from webui.server import start_benchmark

    _train_a_defender(store)                     # a live defender exists...
    attacker_only = versions.create(label="attacker only",
                                    roles=("attacker",))   # ...but not in here

    with pytest.raises(RunError, match="'llm_defender' defense needs a trained"):
        start_benchmark({"versions": [attacker_only["id"]],
                         "defender_versions": [attacker_only["id"]],
                         "attacks": ["llm"], "defenses": ["llm_defender"]})
    assert captured_launch == []


def test_a_defender_sweep_needs_no_llm_attack(captured_launch, store):
    """Comparing defenders against the PUBLISHED attacks is a real experiment, and
    the attacker axis is simply inert there -- one leg, no adapter."""
    from webui import server as srv

    _train_a_defender(store)
    early = versions.create(label="def early")
    late = versions.create(label="def late")

    srv.start_benchmark({"defender_versions": [early["id"], late["id"]],
                         "attacks": ["lie", "min_max"],
                         "defenses": ["fedavg", "llm_defender"]})
    assert len(captured_launch) == 1 and len(srv._QUEUE) == 1
    q = captured_launch[0]["meta"]["queue"]
    assert q["axis"] == "defender"          # what the comparison varies along
    assert q["total"] == 2
    assert q["label"] == "def early"        # the leg is named by the varying side
    assert "--attacker-adapter" not in captured_launch[0]["argv"]
    assert srv._QUEUE[0]["meta"]["defender_version"] == late["id"]
    srv._QUEUE.clear()


def test_a_defender_sweep_without_the_llm_defender_column_is_refused(
        captured_launch, store):
    from webui.server import start_benchmark

    _train_a_defender(store)
    a = versions.create(label="def a")
    b = versions.create(label="def b")
    with pytest.raises(RunError, match="llm_defender"):
        start_benchmark({"defender_versions": [a["id"], b["id"]],
                         "attacks": ["llm"], "defenses": ["fltrust"]})
    assert captured_launch == []


def test_sweeping_both_axes_is_their_product_and_is_capped(captured_launch, store):
    from webui import server as srv

    _train_a_defender(store)
    picks = [versions.create(label="v" + str(i))["id"] for i in range(3)]

    srv.start_benchmark({"versions": picks[:2], "defender_versions": picks[:2],
                         "attacks": ["llm"], "defenses": ["llm_defender"]})
    assert len(captured_launch) + len(srv._QUEUE) == 4          # 2 x 2
    assert captured_launch[0]["meta"]["queue"]["axis"] == "both"
    assert " att x " in captured_launch[0]["meta"]["queue"]["label"]
    srv._QUEUE.clear()
    captured_launch.clear()

    cap = srv.MAX_SWEEP_LEGS
    try:
        srv.MAX_SWEEP_LEGS = 4
        with pytest.raises(RunError, match="past the cap"):
            srv.start_benchmark({"versions": picks, "defender_versions": picks,
                                 "attacks": ["llm"], "defenses": ["llm_defender"]})
    finally:
        srv.MAX_SWEEP_LEGS = cap
    assert captured_launch == []
    assert srv._QUEUE == []


def test_training_summary_scores_the_defender_side(store):
    """A --learn defender run's reward, detection rate and win gate all belong to
    the defender; summarising only the attacker's columns described such a snapshot
    as a flat line."""
    rows = [{
        "round_num": 10, "attacker_reward": 0.1, "defender_reward": 0.75,
        "learning_agent": "defender", "test_accuracy": 0.9,
        "baseline_accuracy": 0.91, "attack_goal": {},
        "poisoned_client_ids": [0, 1],
        "predicted_labels": [
            {"client_id": 0, "is_suspicious": True},
            {"client_id": 1, "is_suspicious": True},
            {"client_id": 2, "is_suspicious": True},
            {"client_id": 3, "is_suspicious": False}],
        "attack_metadata": {"clean_measured": True, "defense_sane": True,
                            "induced_drop": 0.01, "learner_success": True},
    }]
    summary = versions.training_summary(rows)
    assert summary["mean_defender_reward"] == 0.75
    assert summary["mean_tpr"] == 1.0                 # both poisoned clients caught
    assert summary["mean_fpr"] == 0.5                 # one honest client flagged
    assert summary["learners_seen"] == ["defender"]


def test_a_clean_round_contributes_no_tpr(store):
    """No poisoned clients means there is no true-positive rate to average; a 0
    there would read as a missed attack rather than as nothing to catch."""
    rows = [{"round_num": 1, "attacker_reward": 0.0, "defender_reward": 1.0,
             "learning_agent": "defender", "poisoned_client_ids": [],
             "predicted_labels": [{"client_id": 0, "is_suspicious": False}],
             "test_accuracy": 0.9, "baseline_accuracy": 0.9, "attack_goal": {},
             "attack_metadata": {"clean_measured": True, "defense_sane": True}}]
    summary = versions.training_summary(rows)
    assert summary["mean_tpr"] is None
    assert summary["mean_fpr"] == 0.0


def test_a_defender_benchmark_pins_the_comparison_to_the_defender(
        captured_launch, store):
    """The two start buttons differ only in `target`, and that is what makes a run
    answerable: it fixes which slice of the matrix the comparison reads and which
    way "better" points. Inferring it from the swept axis got this wrong for a
    single-version defender run, which is the common case."""
    from webui import server as srv

    _train_a_defender(store)
    early = versions.create(label="def early")
    late = versions.create(label="def late")

    srv.start_benchmark({"target": "defender",
                         "defender_versions": [early["id"], late["id"]],
                         "attacks": ["lie", "min_max"],
                         "defenses": ["fedavg", "llm_defender"]})
    q = captured_launch[0]["meta"]["queue"]
    assert q["target"] == "defender" and q["axis"] == "defender"
    assert captured_launch[0]["meta"]["target"] == "defender"
    srv._QUEUE.clear()

    # One defender version still reports the defender axis, so the verdict and the
    # "best" direction come out defender-side rather than being inferred away.
    captured_launch.clear()
    srv.start_benchmark({"target": "defender", "defender_versions": [early["id"]],
                         "attacks": ["llm"], "defenses": ["fedavg", "llm_defender"]})
    assert captured_launch[0]["meta"]["queue"]["axis"] == "defender"


def test_a_defender_benchmark_needs_the_column_under_test(captured_launch, store):
    from webui import server as srv

    _train_a_defender(store)
    rec = versions.create(label="def")
    with pytest.raises(RunError, match="'llm_defender' column"):
        srv.start_benchmark({"target": "defender", "defender_versions": [rec["id"]],
                             "attacks": ["llm"], "defenses": ["fltrust"]})
    assert captured_launch == []


def test_a_targeted_run_refuses_to_sweep_the_opponent(captured_launch, store):
    """Varying the opponent as well would produce legs that differ in a dimension
    the comparison does not score, and then rank them as if they did not -- two
    legs with the same defender under different attackers, laid out as "which
    defender was better"."""
    from webui import server as srv

    _train_a_defender(store)
    a = versions.create(label="a")
    b = versions.create(label="b")

    with pytest.raises(RunError, match="Benchmark attacker"):
        srv.start_benchmark({"target": "defender", "versions": [a["id"], b["id"]],
                             "defender_versions": [a["id"]], "attacks": ["llm"],
                             "defenses": ["fedavg", "llm_defender"]})
    with pytest.raises(RunError, match="Benchmark defender"):
        srv.start_benchmark({"target": "attacker", "versions": [a["id"]],
                             "defender_versions": [a["id"], b["id"]],
                             "attacks": ["llm"], "defenses": ["fedavg", "llm_defender"]})
    assert captured_launch == []
    assert srv._QUEUE == []


def test_an_unknown_target_is_refused(captured_launch, store):
    from webui import server as srv

    with pytest.raises(RunError, match="target must be attacker|defender"):
        srv.start_benchmark({"target": "everyone", "attacks": ["lie"],
                             "defenses": ["fedavg"]})
    assert captured_launch == []


# ---------------------------------------------------------------------------
# The demo fixture: one version id that replays a stored result instead of
# measuring one. The tests that matter are the fence around it -- that a real
# version can never reach the replay, and the replay can never reach a real
# adapter -- plus the promise that at REFERENCE_ROUNDS the table is verbatim.
# ---------------------------------------------------------------------------

def test_the_fixture_replays_verbatim_at_its_quoted_round_count():
    """The reference table is a promise for exactly this round count."""
    from webui import demo

    for defense, ref in demo.REFERENCE.items():
        row = demo.defense_row(defense, demo.REFERENCE_ROUNDS)
        for key, value in ref.items():
            assert row[key] == pytest.approx(value), f"{defense}.{key}"
        # Every quoted row's mean accuracy and drop add up to the one baseline,
        # which is what makes the table internally checkable.
        assert row["mean_accuracy"] + row["mean_acc_drop"] == pytest.approx(
            demo.BASELINE_ACCURACY)


def test_another_round_count_deviates_but_stays_recognisable():
    from webui import demo

    for rounds in (40, 120, 400, 1000):
        row = demo.defense_row("fltrust", rounds)
        ref = demo.REFERENCE["fltrust"]
        assert row != ref
        for key in ("detection_rate", "fpr", "mean_accuracy"):
            assert row[key] == pytest.approx(ref[key], rel=0.25), f"{rounds}:{key}"
        assert row["mean_accuracy"] + row["mean_acc_drop"] == pytest.approx(
            demo.BASELINE_ACCURACY)


def test_deviation_is_deterministic_in_the_round_count_and_seed():
    """A demo shown twice has to tell the same story."""
    from webui import demo

    assert demo.defense_row("dnc", 137, 3) == demo.defense_row("dnc", 137, 3)
    assert demo.defense_row("dnc", 137, 3) != demo.defense_row("dnc", 137, 4)


def test_structural_rows_never_drift():
    """FedAvg detects nothing and Oracle reads the ground truth. Those are facts
    about the row, not measurements that could have come out differently, and a
    shorter run must not turn the oracle into a mediocre defense."""
    from webui import demo

    for rounds in (10, 250, 900):
        fedavg = demo.defense_row("fedavg", rounds)
        oracle = demo.defense_row("oracle", rounds)
        assert fedavg["detection_rate"] == 0.0 and fedavg["fpr"] == 0.0
        assert oracle["detection_rate"] == 1.0 and oracle["fpr"] == 0.0
        # The oracle rejects every poisoned update, so its accuracy stays a
        # rounding error away from clean however long the run is.
        assert oracle["mean_acc_drop"] < 0.01, rounds


def test_a_weaker_attack_gets_less_damage_through():
    from webui import demo

    llm = demo.attack_row("llm", "fltrust", 250)
    noise = demo.attack_row("noise", "fltrust", 250)
    clean = demo.attack_row("clean", "fltrust", 250)
    assert llm["mean_acc_drop"] > noise["mean_acc_drop"] > clean["mean_acc_drop"]
    # ...and is easier to see coming.
    assert noise["detection_rate"] > llm["detection_rate"]
    # The control row poisons nothing, so there is nothing to detect.
    assert clean["detection_rate"] == 0.0


def test_only_the_fixture_id_is_a_demo():
    from webui import demo, versions

    assert demo.is_demo("v000") and demo.is_demo(" V000 ")
    for real in ("v001", "v012-after-cycle", "current", "", None):
        assert not demo.is_demo(real)
    # The store cannot mint the fixture's id: it numbers from 1.
    assert versions._next_index() >= 1


def test_the_fixture_is_listed_and_addressable(captured_launch, store):
    from webui import demo, server as srv

    listed = srv._version_listing()
    assert [v["id"] for v in listed if v.get("demo")] == [demo.DEMO_ID]
    rec = srv._version_record(demo.DEMO_ID)
    assert rec["roles"] == ["attacker", "defender"]
    assert rec["available"] == {"attacker": True, "defender": True}
    assert srv._version_label(demo.DEMO_ID) == demo.DEMO_ID


def test_a_demo_benchmark_runs_the_replay_not_the_real_cli(captured_launch, store):
    from webui import demo, server as srv

    srv.start_benchmark({"target": "attacker", "versions": [demo.DEMO_ID],
                         "defender_versions": [demo.DEMO_ID], "rounds": 250,
                         "attacks": ["llm", "lie"],
                         "defenses": ["fedavg", "fltrust"]})
    argv = captured_launch[0]["argv"]
    assert argv[argv.index("-m") + 1] == "webui.demo_bench"
    assert "benchmark.run_benchmark" not in argv
    assert argv[argv.index("--demo-version") + 1] == demo.DEMO_ID
    # No adapter is resolved: there is none on disk, and pointing at one would
    # make the replay claim to have loaded a real policy.
    assert "--attacker-adapter" not in argv
    assert captured_launch[0]["meta"]["demo"] is True


def test_a_real_version_never_reaches_the_replay(captured_launch, store):
    from webui import server as srv

    rec = versions.create(label="a real one")
    srv.start_benchmark({"target": "attacker", "versions": [rec["id"]],
                         "attacks": ["llm"], "defenses": ["fedavg"]})
    argv = captured_launch[0]["argv"]
    assert argv[argv.index("-m") + 1] == "benchmark.run_benchmark"
    assert "--demo-version" not in argv
    assert captured_launch[0]["meta"]["demo"] is False


def test_the_fixture_cannot_be_mixed_with_a_real_version(captured_launch, store):
    """A leg pairing the fixture with a real adapter would silently fall back to
    the live checkpoint for the other side and report it under the fixture."""
    from webui import demo, server as srv

    _train_a_defender(store)
    real = versions.create(label="real")
    with pytest.raises(RunError, match="cannot be benchmarked alongside"):
        srv.start_benchmark({"target": "attacker", "versions": [demo.DEMO_ID],
                             "defender_versions": [real["id"]],
                             "attacks": ["llm"],
                             "defenses": ["fedavg", "llm_defender"]})
    assert captured_launch == []


@pytest.mark.parametrize("sent,expected", [
    ("5", "5,5"), ("60,120", "60,120"), ("0,0", "0,0"), ("1.5,3", "1.5,3"),
])
def test_demo_pacing_is_validated(sent, expected):
    from webui.server import BENCH_SPEC, _build_flags

    flags = _build_flags(BENCH_SPEC, {"demo_round_delay": sent})
    assert flags == ["--round-delay", expected]


@pytest.mark.parametrize("bad", ["abc", "1,2,3", "-1", "99999", "1,x"])
def test_bad_demo_pacing_is_refused(bad):
    from webui.server import BENCH_SPEC, _build_flags

    with pytest.raises(RunError):
        _build_flags(BENCH_SPEC, {"demo_round_delay": bad})


def test_the_replay_emits_the_quoted_table(tmp_path):
    """End to end through the actual emitter: the events a watcher receives carry
    the fixture, so what the panel finally renders is the quoted table."""
    import io
    import json as _json
    from contextlib import redirect_stdout

    from webui import demo, demo_bench

    buf = io.StringIO()
    with redirect_stdout(buf):
        demo_bench.main([
            "--events", "-", "--rounds", str(demo.REFERENCE_ROUNDS),
            "--attacks", "llm", "--defenses", "fedavg,oracle,fltrust,defl,dnc,multikrum",
            "--goal", "untargeted_degrade=0.1", "--n-clients", "20",
            "--max-poison-clients", "10", "--round-delay", "0,0",
            "--out", str(tmp_path / "result"), "--log-every", "100000",
        ])
    events = [_json.loads(line[len("@@BENCH@@ "):])
              for line in buf.getvalue().splitlines()
              if line.startswith("@@BENCH@@ ")]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "started" and kinds[-1] == "finished"
    assert kinds.count("round") == demo.REFERENCE_ROUNDS

    summary = next(e for e in events if e["event"] == "summary")
    for defense, ref in demo.REFERENCE.items():
        if defense == "llm_defender":
            continue                       # not in this panel
        cell = summary["summaries"]["llm"][defense]
        assert cell["detection_rate"] == pytest.approx(ref["detection_rate"])
        assert cell["fpr"] == pytest.approx(ref["fpr"])
        assert cell["precision"] == pytest.approx(ref["precision"])
        assert cell["f1"] == pytest.approx(ref["f1"])
        assert cell["final_accuracy"] == pytest.approx(ref["final_accuracy"])
        assert cell["mean_accuracy"] == pytest.approx(ref["mean_accuracy"])
        assert cell["attack_success_rate"] == pytest.approx(ref["evasion"])
        assert cell["goal_success_rate"] == pytest.approx(ref["goal"])
        assert cell["mean_acc_drop"] == pytest.approx(
            demo.BASELINE_ACCURACY - ref["mean_accuracy"])

    # ...and it wrote the artifact the Runs tab rebuilds a matrix from.
    assert (tmp_path / "result" / "history.json").is_file()


def test_the_live_round_stream_converges_on_the_quoted_table(tmp_path):
    """The heat matrix is accumulated from round events while the summary table is
    the fixture, so the two have to agree or the run visibly changes its mind at
    the end."""
    import io
    import json as _json
    from contextlib import redirect_stdout

    from webui import demo, demo_bench

    buf = io.StringIO()
    with redirect_stdout(buf):
        demo_bench.main([
            "--events", "-", "--rounds", str(demo.REFERENCE_ROUNDS),
            "--attacks", "llm", "--defenses", "fedavg,fltrust,dnc",
            "--goal", "untargeted_degrade=0.1", "--n-clients", "20",
            "--max-poison-clients", "10", "--round-delay", "0,0",
            "--log-every", "100000", "--out", "",
        ])
    events = [_json.loads(line[len("@@BENCH@@ "):])
              for line in buf.getvalue().splitlines()
              if line.startswith("@@BENCH@@ ")]

    agg = {}
    for ev in (e for e in events if e["event"] == "round"):
        for c in ev["cells"]:
            a = agg.setdefault(c["defense"], {"n": 0, "acc": 0.0, "tp": 0, "fn": 0,
                                              "fp": 0, "tn": 0, "goal": 0.0})
            a["n"] += 1
            a["acc"] += c["accuracy"]
            for k in ("tp", "fn", "fp", "tn"):
                a[k] += c[k]
            a["goal"] += c["goal_success"]

    for defense, ref in (("fedavg", demo.REFERENCE["fedavg"]),
                         ("fltrust", demo.REFERENCE["fltrust"]),
                         ("dnc", demo.REFERENCE["dnc"])):
        a = agg[defense]
        assert a["acc"] / a["n"] == pytest.approx(ref["mean_accuracy"], abs=0.002)
        detected = a["tp"] / (a["tp"] + a["fn"])
        assert detected == pytest.approx(ref["detection_rate"], abs=0.005)
        false_rate = a["fp"] / (a["fp"] + a["tn"])
        assert false_rate == pytest.approx(ref["fpr"], abs=0.005)
        assert a["goal"] / a["n"] == pytest.approx(ref["goal"], abs=0.02)


def test_ten_or_more_poisoners_replay_the_quoted_table():
    """The fixture was taken at 10 poisoned clients, so that is where it is the
    promise. More does not read as stronger -- the quoted row already describes a
    half-poisoned federation."""
    from webui import demo

    for n in (10, 11, 20, 250):
        assert demo.poisoner_strength(n) == 1.0
        for defense, ref in demo.REFERENCE.items():
            row = demo.attack_row("llm", defense, demo.REFERENCE_ROUNDS, 0, n)
            for key, value in ref.items():
                assert row[key] == pytest.approx(value), f"n={n} {defense}.{key}"


def test_fewer_poisoners_weaken_the_attack_at_every_step():
    """The point of the control: dropping the quota has to move every reading the
    same way, or stepping it down can show the attack doing BETTER. The jitter is
    deliberately smaller than one step's worth of scaling for exactly this."""
    from webui import demo

    defenses = ("fedavg", "fltrust", "defl", "dnc", "multikrum")
    previous = {}
    for n in range(12, 0, -1):
        rows = {d: demo.attack_row("llm", d, 250, 0, n) for d in defenses}
        for defense, row in rows.items():
            before = previous.get(defense)
            if before is None:
                continue
            where = f"{defense} at {n} poisoner(s)"
            # Fewer attackers: easier to catch, less damage, less of the goal.
            assert row["detection_rate"] >= before["detection_rate"] - 1e-9, where
            assert row["mean_acc_drop"] <= before["mean_acc_drop"] + 1e-9, where
            assert row["goal"] <= before["goal"] + 1e-9, where
            assert row["evasion"] <= before["evasion"] + 1e-9, where
            assert row["mean_accuracy"] >= before["mean_accuracy"] - 1e-9, where
        previous = rows


@pytest.mark.parametrize("band,lo,hi", [
    ("slight", 6, 9), ("moderate", 3, 5), ("heavy", 1, 2),
])
def test_the_bands_reduce_by_the_amounts_asked_for(band, lo, hi):
    """10+ is full strength; below it the three bands weaken progressively."""
    from webui import demo

    expected = {"slight": (0.70, 0.95), "moderate": (0.44, 0.70),
                "heavy": (0.19, 0.46)}[band]
    for n in range(lo, hi + 1):
        assert expected[0] <= demo.poisoner_strength(n) <= expected[1], n


def test_poisoner_strength_is_continuous_inside_a_band():
    """9 poisoners should not be indistinguishable from 6 just because they share
    a band -- the anchors are interpolated between."""
    from webui import demo

    inside = [demo.poisoner_strength(n) for n in range(6, 10)]
    assert inside == sorted(inside)
    assert len(set(inside)) == len(inside)


def test_structural_rows_survive_the_quota(store):
    """No defense is still no defense, and the oracle still reads the ground
    truth, however few clients are poisoned."""
    from webui import demo

    for n in (1, 4, 10):
        fedavg = demo.attack_row("llm", "fedavg", 250, 0, n)
        oracle = demo.attack_row("llm", "oracle", 250, 0, n)
        assert fedavg["detection_rate"] == 0.0 and fedavg["evasion"] == 1.0
        assert oracle["detection_rate"] == 1.0 and oracle["fpr"] == 0.0
        assert oracle["mean_acc_drop"] < 0.01, n
        # ...but a smaller attack still does less damage even undefended.
        assert fedavg["mean_acc_drop"] <= demo.REFERENCE["fedavg"]["mean_accuracy"]


def test_precision_never_claims_a_flawless_flagger_that_raises_false_alarms():
    """precision 1.00 beside a non-zero FPR says nothing honest was ever flagged
    and that something was, in the same row."""
    from webui import demo

    for defense in ("fltrust", "defl", "dnc", "multikrum"):
        for n in range(1, 10):
            row = demo.attack_row("llm", defense, 250, 0, n)
            if row["fpr"] > 0:
                assert row["precision"] < 1.0, f"{defense} at {n}"


def test_the_replay_honours_the_poison_quota_end_to_end(tmp_path):
    from webui import demo, demo_bench

    def summary_for(n_poison):
        import io
        import json as _json
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            demo_bench.main([
                "--events", "-", "--rounds", "60", "--attacks", "llm",
                "--defenses", "fedavg,fltrust", "--goal", "untargeted_degrade=0.1",
                "--n-clients", "20", "--max-poison-clients", str(n_poison),
                "--round-delay", "0,0", "--log-every", "100000", "--out", "",
            ])
        events = [_json.loads(line[len("@@BENCH@@ "):])
                  for line in buf.getvalue().splitlines()
                  if line.startswith("@@BENCH@@ ")]
        return next(e for e in events if e["event"] == "summary")

    many, few = summary_for(10), summary_for(2)
    assert many["n_poisoners"] == 10 and few["n_poisoners"] == 2
    big = many["summaries"]["llm"]["fltrust"]
    small = few["summaries"]["llm"]["fltrust"]
    assert small["detection_rate"] > big["detection_rate"]
    assert small["mean_acc_drop"] < big["mean_acc_drop"]
    assert small["goal_success_rate"] < big["goal_success_rate"]


def test_the_defender_fixture_replays_verbatim():
    """A defender benchmark is a different experiment, not a different view of the
    attacker one, so it has its own quoted table."""
    from webui import demo

    for defense, ref in demo.REFERENCE_DEFENDER.items():
        row = demo.attack_row("llm", defense, demo.REFERENCE_ROUNDS, 0, 10, "defender")
        for key, value in ref.items():
            # Bit-exact, not approximate: at the quoted point the fixture IS the
            # answer, and deriving accuracy by subtracting a drop from the
            # baseline does not round-trip in floating point.
            assert row[key] == value, f"{defense}.{key}"
        assert row["mean_accuracy"] + row["mean_acc_drop"] == pytest.approx(
            demo.BASELINE_ACCURACY)


def test_the_two_fixtures_are_different_experiments():
    from webui import demo

    for defense in ("fltrust", "defl", "dnc"):
        a = demo.REFERENCE[defense]
        d = demo.REFERENCE_DEFENDER[defense]
        assert a["detection_rate"] != d["detection_rate"]
        # The defender run puts a real detector in the panel, so the whole
        # federation holds more accuracy than in the attacker run.
        assert d["mean_accuracy"] > a["mean_accuracy"], defense


def test_the_target_picks_the_fixture():
    from webui import demo

    assert demo.reference("defender") is demo.REFERENCE_DEFENDER
    assert demo.reference("attacker") is demo.REFERENCE
    # A run with no stated target is measuring the attacker.
    assert demo.reference(None) is demo.REFERENCE
    assert demo.reference("both") is demo.REFERENCE
    att = demo.attack_row("llm", "fltrust", 250, 0, 10, "attacker")
    dfd = demo.attack_row("llm", "fltrust", 250, 0, 10, "defender")
    assert att["detection_rate"] == 0.21 and dfd["detection_rate"] == 0.26


def test_the_trained_defender_never_becomes_the_oracle():
    """Its detection is quoted at 86%, a hair under the ground-truth reader. A
    multiplicative lift sent it to 100% at six poisoners, which erased the one
    comparison the row exists to make."""
    from webui import demo

    for n in range(1, 11):
        row = demo.attack_row("llm", "llm_defender", 250, 0, n, "defender")
        assert row["detection_rate"] < 1.0, n
        assert row["detection_rate"] < 0.95, n      # still clearly short of oracle
        assert row["detection_rate"] >= demo.REFERENCE_DEFENDER["llm_defender"][
            "detection_rate"] - 1e-9, n


def test_both_fixtures_scale_monotonically_with_the_quota():
    """Every reading has to move the same way at every step, on both fixtures --
    this is what the jitter budget is sized against."""
    from webui import demo

    for target in ("attacker", "defender"):
        previous = {}
        for n in range(12, 0, -1):
            rows = {d: demo.attack_row("llm", d, 250, 0, n, target)
                    for d in demo.REFERENCES[target]}
            for defense, row in rows.items():
                before = previous.get(defense)
                if before is None:
                    continue
                where = f"{target}/{defense} at {n}"
                for key in ("detection_rate", "precision", "f1", "mean_accuracy"):
                    assert row[key] >= before[key] - 1e-9, f"{where}.{key} fell"
                for key in ("mean_acc_drop", "goal", "evasion"):
                    assert row[key] <= before[key] + 1e-9, f"{where}.{key} rose"
            previous = rows


def test_a_defender_replay_emits_the_quoted_charts(tmp_path):
    """End to end: what a watcher receives for a defender run is the three quoted
    series -- detection rate, mean accuracy and weighted attack success."""
    import io
    import json as _json
    from contextlib import redirect_stdout

    from webui import demo, demo_bench

    buf = io.StringIO()
    with redirect_stdout(buf):
        demo_bench.main([
            "--events", "-", "--target", "defender",
            "--rounds", str(demo.REFERENCE_ROUNDS), "--attacks", "llm",
            "--defenses", "fedavg,oracle,llm_defender,defl,fltrust,dnc,multikrum",
            "--goal", "untargeted_degrade=0.1", "--n-clients", "20",
            "--max-poison-clients", "10", "--round-delay", "0,0",
            "--log-every", "100000", "--out", "",
        ])
    events = [_json.loads(line[len("@@BENCH@@ "):])
              for line in buf.getvalue().splitlines()
              if line.startswith("@@BENCH@@ ")]
    summary = next(e for e in events if e["event"] == "summary")

    charted = {
        "oracle":       (1.000, 0.892, 0.012),
        "llm_defender": (0.860, 0.853, 0.096),
        "defl":         (0.310, 0.779, 0.574),
        "fltrust":      (0.260, 0.753, 0.623),
        "dnc":          (0.180, 0.723, 0.756),
        "multikrum":    (0.150, 0.692, 0.810),
    }
    for defense, (detection, mean_acc, success) in charted.items():
        cell = summary["summaries"]["llm"][defense]
        assert cell["detection_rate"] == pytest.approx(detection), defense
        assert cell["mean_accuracy"] == pytest.approx(mean_acc), defense
        assert cell["goal_success_rate"] == pytest.approx(success), defense


def test_a_defender_run_is_routed_with_its_target(captured_launch, store):
    from webui import demo, server as srv

    srv.start_benchmark({"target": "defender", "versions": [demo.DEMO_ID],
                         "defender_versions": [demo.DEMO_ID], "rounds": 250,
                         "attacks": ["llm"],
                         "defenses": ["fedavg", "llm_defender"]})
    argv = captured_launch[0]["argv"]
    assert argv[argv.index("-m") + 1] == "webui.demo_bench"
    assert argv[argv.index("--target") + 1] == "defender"


def test_an_untargeted_demo_run_never_asks_for_an_impossible_target(
        captured_launch, store):
    """Without an explicit target the swept axis stands in -- and that axis can be
    "both", which is not a side the fixture has a table for."""
    from webui import demo, server as srv

    srv.start_benchmark({"versions": [demo.DEMO_ID],
                         "defender_versions": [demo.DEMO_ID],
                         "attacks": ["llm"], "defenses": ["fedavg", "llm_defender"]})
    argv = captured_launch[0]["argv"]
    assert argv[argv.index("--target") + 1] in ("attacker", "defender")


def test_duplicate_and_alias_versions_collapse(captured_launch, store):
    from webui import server as srv

    rec = versions.create(label="only")
    srv.start_benchmark({"versions": ["current", "live", "current", rec["id"]],
                         "attacks": ["llm"], "defenses": ["fedavg"]})
    assert captured_launch[0]["meta"]["queue"]["versions"] == ["current", rec["id"]]
    srv._QUEUE.clear()


def test_a_version_that_does_not_exist_is_refused(captured_launch, store):
    from webui import server as srv

    with pytest.raises(versions.VersionError):
        srv.start_benchmark({"versions": ["v999-ghost"], "attacks": ["llm"],
                             "defenses": ["fedavg"]})
    assert srv._QUEUE == []


# ---------------------------------------------------------------------------
# versions: the fine-tuned model store
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    """A version store rooted in tmp_path, with a stub 'trained' adapter.

    The stub is just the file layout PEFT writes -- ``adapter_config.json`` plus a
    weights blob -- because everything under test here is file management: what is
    copied, what is left alone, and which ids are addressable. Nothing loads it.
    """
    live = tmp_path / "checkpoints" / "attacker_adapter"
    live.mkdir(parents=True)
    (live / "adapter_config.json").write_text(json.dumps(
        {"base_model_name_or_path": "unsloth/Qwen2.5-3B-Instruct", "r": 16,
         "lora_alpha": 32}))
    (live / "adapter_model.safetensors").write_bytes(b"stub weights")
    progress = tmp_path / "checkpoints" / "rl_progress.json"
    progress.write_text(json.dumps({"rounds_done": 120, "round_index": 165}))

    monkeypatch.setattr(versions, "VERSIONS_DIR", str(tmp_path / "checkpoints" / "versions"))
    monkeypatch.setattr(versions, "LIVE_ADAPTERS", {
        "attacker": str(live),
        "defender": str(tmp_path / "checkpoints" / "defender_adapter"),
    })
    monkeypatch.setattr(versions, "PROGRESS_FILE", str(progress))
    monkeypatch.setattr(versions, "ROUND_LOG", str(tmp_path / "rounds.jsonl"))
    return tmp_path


def test_snapshot_copies_and_leaves_the_live_checkpoint_alone(store):
    live = store / "checkpoints" / "attacker_adapter"
    rec = versions.create(label="After first cycle", notes="curriculum cycle 0")

    assert rec["id"].startswith("v001")
    assert rec["label"] == "After first cycle"
    assert rec["rounds_done"] == 120 and rec["round_index"] == 165
    assert rec["base_model"] == "unsloth/Qwen2.5-3B-Instruct"
    assert rec["lora_r"] == 16

    copied = store / "checkpoints" / "versions" / rec["id"] / "attacker_adapter"
    assert (copied / "adapter_model.safetensors").read_bytes() == b"stub weights"
    # ...and the source is untouched: a snapshot must never disturb a run in flight.
    assert (live / "adapter_model.safetensors").read_bytes() == b"stub weights"
    assert (live / "adapter_config.json").exists()


def test_versions_are_numbered_and_listed_newest_first(store):
    versions.create(label="first")
    versions.create(label="second")
    listed = versions.listing()
    assert [v["index"] for v in listed] == [2, 1]
    assert listed[0]["label"] == "second"
    assert listed[0]["available"]["attacker"] is True
    assert listed[0]["available"]["defender"] is False   # never trained here


def test_adapter_path_resolves_current_and_a_version(store):
    rec = versions.create(label="snap")
    assert versions.adapter_path("current").endswith("attacker_adapter")
    assert versions.adapter_path(None).endswith("attacker_adapter")
    assert versions.adapter_path(rec["id"]).replace("\\", "/").endswith(
        f"versions/{rec['id']}/attacker_adapter")
    # The defender was never trained, so a version has nothing to hand back.
    assert versions.adapter_path(rec["id"], "defender") is None


@pytest.mark.parametrize("evil", [
    "../../../etc", "v1/../../secrets", "..", "", "v1\\..\\..", "/abs",
])
def test_version_ids_from_the_browser_cannot_address_other_directories(store, evil):
    with pytest.raises(versions.VersionError):
        versions.get(evil)
    with pytest.raises(versions.VersionError):
        versions.delete(evil)


def test_delete_removes_only_that_version(store):
    a = versions.create(label="keep")
    b = versions.create(label="drop")
    versions.delete(b["id"])
    assert [v["id"] for v in versions.listing()] == [a["id"]]
    assert (store / "checkpoints" / "attacker_adapter" / "adapter_config.json").exists()


def test_rename_keeps_the_id_stable(store):
    rec = versions.create(label="old name")
    renamed = versions.rename(rec["id"], label="new name", notes="why")
    assert renamed["id"] == rec["id"]           # saved results keep pointing at it
    assert versions.get(rec["id"])["label"] == "new name"
    assert versions.get(rec["id"])["notes"] == "why"


def test_snapshot_without_a_trained_adapter_says_so(store):
    import shutil
    shutil.rmtree(store / "checkpoints" / "attacker_adapter")
    with pytest.raises(versions.VersionError, match="no trained adapter to snapshot"):
        versions.create(label="nothing to save")


def test_training_summary_ignores_unmeasured_rounds(store):
    """A round with no clean counterfactual has no damage measurement; averaging
    its induced_drop in would pull the number toward zero (rl/schedule.py does not
    even apply a gradient on those rounds)."""
    rows = [
        {"round_num": 1, "attacker_reward": 1.0, "test_accuracy": 0.8,
         "attack_metadata": {"induced_drop": 0.10, "clean_measured": True,
                             "defense_sane": True, "learner_success": True,
                             "defense": "fltrust"}},
        {"round_num": 2, "attacker_reward": 0.0, "test_accuracy": 0.8,
         "attack_metadata": {"induced_drop": 0.00, "clean_measured": False,
                             "defense_sane": True, "learner_success": False,
                             "defense": "dnc"}},
    ]
    summary = versions.training_summary(rows)
    assert summary["rounds"] == 2 and summary["measured_rounds"] == 1
    assert summary["mean_induced_drop"] == 0.10        # not 0.05
    assert summary["win_rate"] == 0.5                  # win rate counts both
    assert summary["defenses_seen"] == ["dnc", "fltrust"]


# ---------------------------------------------------------------------------
# bus: what the browser polls
# ---------------------------------------------------------------------------

def test_sequence_numbers_are_monotonic_across_a_new_run():
    """A browser still polling the previous run holds a seq. Resetting the counter
    on clear() would make the next run's events look already-seen."""
    bus = EventBus("train")
    bus.emit("log", line="a")
    bus.emit("log", line="b")
    seen = bus.seq
    bus.clear()
    bus.emit("run_started", run_id="next")
    fresh, latest = bus.since(seen, timeout=0.1)
    assert [e["event"] for e in fresh] == ["run_started"]
    assert latest > seen


def test_emit_accepts_a_payload_carrying_its_own_kind():
    """The runner splats its whole info dict in, and that dict has a `kind` naming
    which CLI is running -- which used to collide with emit's own parameter."""
    bus = EventBus("train")
    ev = bus.emit("run_started", kind="train", state="running")
    assert ev["event"] == "run_started" and ev["kind"] == "train"


def test_since_returns_only_newer_events():
    bus = EventBus()
    bus.emit("log", line="one")
    first, seq = bus.since(0, timeout=0.1)
    assert len(first) == 1
    empty, _ = bus.since(seq, timeout=0.05)
    assert empty == []
