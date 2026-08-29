"""Spawning, watching and stopping the training / benchmark subprocesses.

The UI does not re-implement either CLI -- it runs the same command a shell would
and reads what comes back, so a run started from the browser behaves identically
to one started from a terminal and its exact argv is published to the page for
copy-pasting.

Two output channels are merged into one event bus:

* **stdout/stderr**, line by line. Benchmark runs are started with ``--events -``,
  which interleaves JSONL events into stdout behind
  :data:`benchmark.events.SENTINEL`; those lines become structured events and
  everything else becomes a ``log`` event, in true interleaved order.
* **the round log**, for training. ``main.py`` appends one JSON record per Phase-2
  round to ``logs/round_data/rounds.jsonl`` -- accuracies, the clean
  counterfactual, rewards broken into terms, per-client verdicts, the GRPO step's
  statistics. That file is the richest description of a round that exists, so
  rather than teaching the trainer to emit a second copy of it, a tailer follows
  it from the byte offset the run started at and republishes each new record.
  Training therefore needs NO changes to stream live metrics.

Only the byte offset makes the second channel safe: the round log is append-only
across every run ever made, so a tailer that started at zero would replay months
of history into the panel as if it had just happened.
"""
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time

from benchmark.events import SENTINEL
from webui.bus import EventBus

ROUND_LOG = os.path.join("logs", "round_data", "rounds.jsonl")
RUNS_DIR = os.path.join("logs", "webui", "runs")

#: How often the round-log tailer looks for new records while a run is alive.
_TAIL_INTERVAL = 0.5

#: Phase 1 has no round log -- it is honest FedAvg before the arms race starts --
#: so its progress is scraped from the two lines it prints per round.
_P1_ROUND = re.compile(r"--- Training Round (\d+)/(\d+) ---")
_P1_ACC = re.compile(r"Round (\d+) accuracy: ([0-9.]+)")
_P2_START = re.compile(r"PHASE 2:")


class RunError(RuntimeError):
    pass


class Runner:
    """Supervises at most one subprocess of one kind (``train`` or ``bench``)."""

    def __init__(self, kind: str, on_end=None):
        self.kind = kind
        self.bus = EventBus(kind)
        #: Called as ``on_end(runner, state, exit_code)`` once a run's output is
        #: fully drained. The benchmark queue uses it to launch the next version.
        self.on_end = on_end
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []
        self._stopping = False
        self.info: dict = {"kind": kind, "state": "idle"}

    # -- state --------------------------------------------------------------
    @property
    def running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def status(self) -> dict:
        proc = self._proc
        state = self.info.get("state", "idle")
        if proc is not None:
            code = proc.poll()
            if code is None:
                state = "stopping" if self._stopping else "running"
            elif state in ("running", "stopping"):
                state = "stopped" if self._stopping else (
                    "finished" if code == 0 else "failed")
                self.info["state"] = state
                self.info["exit_code"] = code
        return {**self.info, "state": state, "seq": self.bus.seq,
                "pid": (proc.pid if proc is not None else None)}

    # -- starting -----------------------------------------------------------
    def start(self, argv: list, *, run_id: str, run_dir: str, label: str = "",
              meta: dict | None = None, tail_rounds: bool = False,
              clear_bus: bool = True) -> dict:
        """Launch ``argv`` from the repo root and begin streaming it.

        ``tail_rounds`` attaches the round-log follower described in the module
        docstring -- true for training, false for the benchmark (which publishes
        its own events over stdout).

        ``clear_bus`` is false for the second and later runs of a queued sweep:
        the panel is showing every version's results side by side, so wiping the
        history between them would erase the comparison as it is being built.
        """
        with self._lock:
            if self.running:
                raise RunError(f"a {self.kind} run is already in progress")
            os.makedirs(run_dir, exist_ok=True)
            if clear_bus:
                self.bus.clear()
            self._stopping = False

            # Where the round log ends RIGHT NOW. Everything past this offset was
            # written by the run we are about to start.
            try:
                offset = os.path.getsize(ROUND_LOG)
            except OSError:
                offset = 0

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            popen_kwargs = dict(
                cwd=os.getcwd(), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                bufsize=1, text=True, encoding="utf-8", errors="replace",
            )
            if os.name == "nt":
                # Its own process group, so Ctrl-C in the terminal that started the
                # UI does not also kill the run, and so stop() can take the whole
                # tree (torch/Unsloth spawn workers).
                popen_kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True

            try:
                proc = subprocess.Popen(argv, **popen_kwargs)
            except OSError as exc:
                raise RunError(f"could not start {argv[0]!r}: {exc}") from exc

            # From here on the child is alive, so anything that raises has to take
            # it down with us. Without this an error between Popen and the reader
            # threads left an orphan holding the GPU with nobody draining its pipe.
            try:
                self._proc = proc
                self.info = {
                    "kind": self.kind, "state": "running", "run_id": run_id,
                    "run_dir": run_dir.replace("\\", "/"), "label": label,
                    "argv": argv, "command": _shell_join(argv),
                    "started": time.time(), "exit_code": None,
                    **(meta or {}),
                }
                self.bus.emit("run_started", **self.info)

                self._threads = [threading.Thread(
                    target=self._pump_output, args=(proc, run_dir), daemon=True)]
                if tail_rounds:
                    self._threads.append(threading.Thread(
                        target=self._tail_round_log, args=(proc, offset), daemon=True))
                for t in self._threads:
                    t.start()
            except BaseException:
                _kill_tree(proc)
                self._proc = None
                self.info = {"kind": self.kind, "state": "failed"}
                raise
            return self.status()

    # -- streaming ----------------------------------------------------------
    def _pump_output(self, proc: subprocess.Popen, run_dir: str):
        """Read merged stdout, split events from log lines, mirror to a file."""
        log_path = os.path.join(run_dir, "console.log")
        phase = {"name": "startup", "round": 0, "total": 0}
        try:
            sink = open(log_path, "a", encoding="utf-8", buffering=1)
        except OSError:
            sink = None
        try:
            for line in proc.stdout:
                line = line.rstrip("\n").rstrip("\r")
                if sink is not None:
                    try:
                        sink.write(line + "\n")
                    except OSError:
                        sink = None
                if line.startswith(SENTINEL):
                    payload = line[len(SENTINEL):]
                    try:
                        self.bus.publish(json.loads(payload))
                    except ValueError:
                        self.bus.emit("log", line=payload, level="warn")
                    continue
                self.bus.emit("log", line=line, level=_level(line))
                self._scrape_phase1(line, phase)
        except (OSError, ValueError):
            pass
        finally:
            if sink is not None:
                try:
                    sink.close()
                except OSError:
                    pass
        code = proc.wait()
        state = "stopped" if self._stopping else ("finished" if code == 0 else "failed")
        self.info["state"] = state
        self.info["exit_code"] = code
        self.info["ended"] = time.time()
        self.bus.emit("run_ended", state=state, exit_code=code,
                      run_id=self.info.get("run_id"),
                      elapsed=round(time.time() - self.info.get("started", time.time()), 1))
        if self.on_end is not None:
            # Fired only once the output is fully drained, so a queued follow-up
            # never starts while this run's last lines are still being published.
            try:
                self.on_end(self, state, code)
            except Exception:                             # noqa: BLE001
                logging.getLogger("webui").exception("%s on_end hook failed", self.kind)

    def _scrape_phase1(self, line: str, phase: dict):
        """Turn Phase-1's console lines into progress events.

        Phase 1 is honest FedAvg -- it writes no round log, so without this the
        panel would sit blank through the 45 rounds that produce the baseline every
        later number is measured against.
        """
        m = _P1_ROUND.search(line)
        if m:
            phase.update(name="phase1", round=int(m.group(1)), total=int(m.group(2)))
            self.bus.emit("phase1_round", round=int(m.group(1)), total=int(m.group(2)))
            return
        m = _P1_ACC.search(line)
        if m:
            self.bus.emit("phase1_accuracy", round=int(m.group(1)),
                          accuracy=float(m.group(2)), total=phase.get("total", 0))
            return
        if _P2_START.search(line):
            phase["name"] = "phase2"
            self.bus.emit("phase", phase="phase2")

    def _tail_round_log(self, proc: subprocess.Popen, offset: int):
        """Republish every round record this run appends, from ``offset`` on."""
        pending = ""
        alive = True
        while True:
            alive = proc.poll() is None
            try:
                size = os.path.getsize(ROUND_LOG)
            except OSError:
                size = offset
            if size > offset:
                try:
                    with open(ROUND_LOG, "rb") as f:
                        f.seek(offset)
                        chunk = f.read(size - offset)
                    offset = size
                except OSError:
                    chunk = b""
                pending += chunk.decode("utf-8", "replace")
                # Keep a trailing partial line for the next pass -- a round record
                # is written with one append, but a reader can still land mid-write.
                *complete, pending = pending.split("\n")
                for line in complete:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.bus.publish({"event": "train_round",
                                          "record": json.loads(line)})
                    except ValueError:
                        continue
            if not alive:
                # One more pass has just run against the final file size; stop.
                break
            time.sleep(_TAIL_INTERVAL)

    # -- stopping -----------------------------------------------------------
    def stop(self, force: bool = False) -> dict:
        """Ask the run to end. ``force`` kills immediately instead of asking first.

        A GRPO round holds a GPU model and writes checkpoints, so the default is a
        terminate (which Python turns into a normal interpreter shutdown) and only
        then, after a grace period, a kill of the whole tree.
        """
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise RunError(f"no {self.kind} run is in progress")
            self._stopping = True
            self.info["state"] = "stopping"
            self.bus.emit("run_stopping", forced=bool(force))
            threading.Thread(target=self._terminate, args=(proc, force),
                             daemon=True).start()
            return self.status()

    def _terminate(self, proc: subprocess.Popen, force: bool):
        if not force:
            try:
                proc.terminate()
            except OSError:
                pass
            deadline = time.time() + 10.0
            while time.time() < deadline and proc.poll() is None:
                time.sleep(0.25)
        if proc.poll() is None:
            _kill_tree(proc)


def _kill_tree(proc: subprocess.Popen):
    """Kill the process AND its children.

    A training run is ``python`` on top of torch/Unsloth, which fork dataloader and
    compile workers; terminating only the parent leaves those holding the GPU, and
    the next run then fails to allocate for reasons that look nothing like "the old
    run is still alive".
    """
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=20)
            return
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
            return
        except (OSError, AttributeError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _level(line: str) -> str:
    upper = line.upper()
    if " ERROR:" in upper or upper.startswith("ERROR") or "TRACEBACK" in upper:
        return "error"
    if " WARNING:" in upper or " WARN:" in upper:
        return "warn"
    return "info"


def _shell_join(argv) -> str:
    """The argv as something a human can paste into a shell."""
    out = []
    for a in argv:
        a = str(a)
        out.append(f'"{a}"' if (" " in a or not a) else a)
    return " ".join(out)


def python_exe() -> str:
    """The interpreter to launch runs with -- this one, so the UI and the run share
    a venv without the user having to configure a path."""
    return sys.executable or "python"
