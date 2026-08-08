"""Local web server behind ``python -m benchmark.ui``.

Three jobs, nothing more:

1. serve ``index.html``;
2. spawn ``python -m benchmark.run_targeted_benchmark ... --events -`` and read
   its merged stdout, splitting :data:`benchmark.events.SENTINEL` lines into
   structured events and everything else into log lines;
3. fan those out to the browser over Server-Sent Events.

The benchmark itself is untouched — the UI is a watcher, so a run started here is
the same run you would get from the terminal, and the exact argv is published as
the first event so it can be copy-pasted.

Stdlib only (``http.server`` + ``yaml``, which the project already depends on):
this has to start on a GPU box with no extra pip install.

Binds to 127.0.0.1 by default. Every field from the browser is validated into a
list-form argv — no shell, no free-form argument passthrough — because "a local
web page can run an arbitrary command" is a real hole even on a loopback socket.
"""
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from benchmark.defenses import AVAILABLE
from benchmark.events import SENTINEL

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
INDEX = HERE / "index.html"

MAX_EVENTS = 20000          # ring cap: a 100-round run is ~1.5k events + log lines
MAX_ROUNDS = 100000
SAFE_OUT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-./]*$")


# ----------------------------------------------------------------------------
# Event bus: append-only log of everything this server knows, plus a way to wait
# for more. One bus for the process; a new run clears it.
# ----------------------------------------------------------------------------
class EventBus:
    def __init__(self):
        self._cv = threading.Condition()
        self._events: list[dict] = []
        self._seq = 0

    def publish(self, event: dict) -> None:
        with self._cv:
            self._seq += 1
            event = {**event, "seq": self._seq}
            self._events.append(event)
            if len(self._events) > MAX_EVENTS:
                # Drop the oldest tail. Clients track `seq`, so a reconnect after
                # a drop simply resumes from what is still held.
                del self._events[:len(self._events) - MAX_EVENTS]
            self._cv.notify_all()

    def clear(self) -> None:
        with self._cv:
            self._events.clear()
            self._cv.notify_all()

    def since(self, seq: int, timeout: float = 15.0) -> list[dict]:
        """Events newer than ``seq``, waiting up to ``timeout`` for the first."""
        with self._cv:
            fresh = [e for e in self._events if e["seq"] > seq]
            if fresh:
                return fresh
            self._cv.wait(timeout)
            return [e for e in self._events if e["seq"] > seq]


# ----------------------------------------------------------------------------
# Run manager: at most one benchmark at a time (it owns a GPU).
# ----------------------------------------------------------------------------
class RunManager:
    def __init__(self, bus: EventBus, config_path: str, demo: bool = False,
                 python_exe: str | None = None):
        self.bus = bus
        self.config_path = config_path
        self.demo = demo
        self.python = python_exe or sys.executable
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self.run_id = 0
        self.state = "idle"          # idle | starting | running | done | error | cancelled
        self.argv: list[str] = []

    # -- introspection ------------------------------------------------------
    def status(self) -> dict:
        return {"state": self.state, "run_id": self.run_id, "demo": self.demo,
                "argv": self.argv, "command": shlex.join(self.argv) if self.argv else ""}

    @property
    def busy(self) -> bool:
        return self.state in ("starting", "running")

    # -- lifecycle ----------------------------------------------------------
    def start(self, params: dict) -> dict:
        with self._lock:
            if self.busy:
                raise RuntimeError("a benchmark is already running — stop it first")
            argv = self._build_argv(params)
            self.run_id += 1
            self.state = "starting"
            self.argv = argv
            self._stop_flag.clear()
            self.bus.clear()
            self.bus.publish({"event": "run_started", "run_id": self.run_id,
                              "command": shlex.join(argv), "params": params,
                              "demo": self.demo, "cwd": str(REPO_ROOT)})
            target = self._demo_loop if self.demo else self._spawn_loop
            self._thread = threading.Thread(target=target, args=(argv,), daemon=True)
            self._thread.start()
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._stop_flag.set()
            proc = self._proc
        if proc is not None and proc.poll() is None:
            self._log("[ui] stopping the benchmark process…")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._log("[ui] it ignored terminate — killing it")
                proc.kill()
        return self.status()

    # -- plumbing -----------------------------------------------------------
    def _log(self, line: str, stream: str = "ui") -> None:
        self.bus.publish({"event": "log", "line": line, "stream": stream})

    def _finish(self, state: str) -> None:
        self.state = state
        self.bus.publish({"event": "run_finished", "run_id": self.run_id, "state": state})

    def _build_argv(self, p: dict) -> list[str]:
        """Validate the browser's parameters into an argv. Rejects anything odd —
        this is the trust boundary between a web page and a subprocess."""
        rounds = _as_int(p.get("rounds"), "rounds", 1, MAX_ROUNDS)
        label = _as_int(p.get("label"), "label", 0, 999)
        ids = p.get("client_ids") or []
        if not isinstance(ids, list) or not ids:
            raise ValueError("pick at least one client for the attacker to control")
        ids = [_as_int(c, "client id", 0, 9999) for c in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate client ids")
        defenses = p.get("defenses") or []
        if not isinstance(defenses, list) or not defenses:
            raise ValueError("pick at least one defense")
        unknown = [d for d in defenses if d not in AVAILABLE]
        if unknown:
            raise ValueError(f"unknown defense(s): {unknown}")

        argv = [self.python, "-u", "-m", "benchmark.run_targeted_benchmark",
                "--rounds", str(rounds),
                "--label", str(label),
                "--poison-client-ids", ",".join(str(c) for c in ids),
                "--defenses", ",".join(defenses),
                "--config", self.config_path,
                "--events", "-"]

        if p.get("poison_clients") not in (None, ""):
            budget = _as_int(p["poison_clients"], "poison budget", 1, len(ids))
            argv += ["--poison-clients", str(budget)]
        if p.get("attack_temperature") not in (None, ""):
            argv += ["--attack-temperature",
                     str(_as_float(p["attack_temperature"], "attack temperature", 0.0, 5.0))]
        if p.get("log_every") not in (None, ""):
            argv += ["--log-every", str(_as_int(p["log_every"], "log-every", 1, 10000))]
        out = (p.get("out") or "").strip()
        if out:
            if not SAFE_OUT.match(out) or ".." in out or Path(out).is_absolute():
                raise ValueError(f"output directory {out!r} must be a simple relative path")
            argv += ["--out", out]
        else:
            argv += ["--out", ""]                 # explicit "write nothing"
        if p.get("no_plot"):
            argv.append("--no-plot")
        if p.get("fresh"):
            argv.append("--fresh")
        return argv

    def _spawn_loop(self, argv: list[str]) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        try:
            proc = subprocess.Popen(
                argv, cwd=str(REPO_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as e:
            self._log(f"[ui] could not start the benchmark: {e}")
            self._finish("error")
            return
        with self._lock:
            self._proc = proc
            self.state = "running"
        for raw in proc.stdout:                        # merged stdout+stderr, in order
            line = raw.rstrip("\n")
            if line.startswith(SENTINEL):
                try:
                    self.bus.publish(json.loads(line[len(SENTINEL):]))
                except json.JSONDecodeError:
                    self._log(line, stream="child")    # torn line: show it raw
            elif line:
                self._log(line, stream="child")
        code = proc.wait()
        with self._lock:
            self._proc = None
        # The exit code is the whole signal: every failure path in the benchmark
        # goes through sys.exit("ERROR: …") or an exception, both non-zero.
        if self._stop_flag.is_set():
            self._log("[ui] run cancelled")
            self._finish("cancelled")
        elif code == 0:
            self._finish("done")
        else:
            self._log(f"[ui] benchmark exited with code {code}")
            self._finish("error")

    # -- demo -------------------------------------------------------------
    def _demo_loop(self, argv: list[str]) -> None:
        """Replay a plausible run without torch, a GPU or an adapter.

        This exists so the dashboard can be opened and checked on a laptop. It
        emits the same events the real command does — nothing in the UI knows the
        difference — but the numbers are synthetic and say so.
        """
        from benchmark.ui.demo import demo_events
        with self._lock:
            self.state = "running"
        self._log("[ui] DEMO MODE — synthetic numbers, no model is being run")
        for delay, ev in demo_events(argv):
            if self._stop_flag.is_set():
                self._log("[ui] run cancelled")
                self._finish("cancelled")
                return
            time.sleep(delay)
            if ev.get("event") == "log":
                self._log(ev["line"], stream="child")
            else:
                self.bus.publish(ev)
        self._finish("done")


def _as_int(v, what: str, lo: int, hi: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a whole number, got {v!r}") from None
    if not lo <= n <= hi:
        raise ValueError(f"{what} must be between {lo} and {hi}, got {n}")
    return n


def _as_float(v, what: str, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {v!r}") from None
    if not lo <= x <= hi:
        raise ValueError(f"{what} must be between {lo} and {hi}, got {x}")
    return x


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
def _read_meta(config_path: str) -> dict:
    """Defaults for the form, read off the same config the benchmark will use."""
    meta = {"config": config_path, "n_clients": 20, "n_compromisable": 1,
            "default_pool": [0], "default_label": 0, "eval_poison_clients": 1,
            "n_classes": 10, "device": "cuda", "config_error": None,
            "defenses": list(AVAILABLE),
            "default_defenses": ["fedavg", "oracle", "llm_defender", "fltrust",
                                 "defl", "dnc", "multikrum"]}
    try:
        cfg = yaml.safe_load(open(REPO_ROOT / config_path, encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        meta["config_error"] = f"{config_path}: {e}"
        return meta
    fl = cfg.get("fl", {}) or {}
    attack = cfg.get("attack", {}) or {}
    goal = attack.get("goal", {}) or {}
    n_clients = int(fl.get("n_clients", 20))
    n_comp = max(1, min(int(fl.get("n_compromisable", n_clients)), n_clients))
    meta.update({
        "n_clients": n_clients,
        "n_compromisable": n_comp,
        "default_pool": list(range(n_comp)),
        "n_classes": int((cfg.get("data", {}) or {}).get("n_classes", 10)),
        "eval_poison_clients": int(attack.get("eval_poison_clients", 1)),
        "device": fl.get("device", "cuda"),
        # The UI's own default is label 0 (the class training derived from client
        # 0's shard); the config's `label` is only the fallback, so show both.
        "config_label": goal.get("label"),
        "target_label_from_client": attack.get("target_label_from_client"),
        "target_class_drop": goal.get("target_class_drop"),
        "max_collateral": goal.get("max_collateral"),
    })
    return meta


class Handler(BaseHTTPRequestHandler):
    server_version = "zero-touch-fl-benchmark-ui"
    bus: EventBus
    runs: RunManager
    config_path: str

    # Quieter than the default per-request stderr spam (SSE reconnects are noisy).
    def log_message(self, fmt, *args):
        if self.path.startswith("/api/stream"):
            return
        sys.stderr.write("[ui] %s - %s\n" % (self.address_string(), fmt % args))

    def handle_one_request(self):
        """Swallow the reset the browser causes every time it drops an SSE
        connection (reload, navigate away). The stdlib logs a full traceback for
        it, which fills the console with alarming noise about nothing."""
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    # -- helpers ------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # -- routes -------------------------------------------------------------
    def do_GET(self):                                     # noqa: N802 - stdlib API
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            except OSError as e:
                self._send(500, f"cannot read {INDEX}: {e}".encode(), "text/plain")
        elif path == "/api/meta":
            self._json(200, {**_read_meta(self.config_path), **self.runs.status()})
        elif path == "/api/status":
            self._json(200, self.runs.status())
        elif path == "/api/stream":
            self._stream()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):                                    # noqa: N802 - stdlib API
        path = self.path.split("?", 1)[0]
        if path == "/api/run":
            try:
                self._json(200, self.runs.start(self._body()))
            except (ValueError, RuntimeError) as e:
                self._json(400, {"error": str(e)})
        elif path == "/api/stop":
            self._json(200, self.runs.stop())
        else:
            self._json(404, {"error": "not found"})

    def _stream(self):
        """Server-Sent Events. Replays from ``?from=`` so a reload keeps the run."""
        try:
            since = int((self.path.split("from=", 1) + ["0"])[1].split("&")[0])
        except (IndexError, ValueError):
            since = 0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                events = self.bus.since(since, timeout=15.0)
                if events:
                    since = events[-1]["seq"]
                    chunk = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
                else:
                    chunk = ": keep-alive\n\n"            # keeps proxies/browsers happy
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return                                        # browser navigated away


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Live web UI for the targeted-poisoning benchmark")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--config", default="configs/targeted.yaml",
                    help="config the runs use (also supplies the form's defaults)")
    ap.add_argument("--python", default=None,
                    help="interpreter to run the benchmark with (default: this one)")
    ap.add_argument("--demo", action="store_true",
                    help="replay a synthetic run instead of spawning the benchmark — "
                         "for checking the dashboard without a GPU or an adapter")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    bus = EventBus()
    runs = RunManager(bus, args.config, demo=args.demo, python_exe=args.python)
    Handler.bus, Handler.runs, Handler.config_path = bus, runs, args.config

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    url = f"http://{args.host}:{args.port}/"
    print(f"[ui] benchmark dashboard on {url}")
    print(f"[ui] repo root: {REPO_ROOT}")
    print(f"[ui] config:    {args.config}" + ("   (DEMO MODE)" if args.demo else ""))
    print("[ui] Ctrl-C to quit")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] shutting down")
        runs.stop()
        httpd.shutdown()


if __name__ == "__main__":
    main()
