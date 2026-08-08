"""Structured event stream for a benchmark run — what `benchmark.ui` watches.

The CLI stays the source of truth for a run: the UI does not re-implement the
benchmark behind a web server, it *watches* one. So `run_targeted_benchmark`
optionally emits one JSON object per line describing what just happened, and any
watcher (the UI server, a `tee`, a notebook) reads them back.

Destinations
------------
``""``/``None``   emit nothing — the default, so plain CLI output is byte-identical
``"-"``/``stdout``  stdout, each line prefixed with :data:`SENTINEL`
*any other value*   that file path, one JSON object per line

The sentinel exists so events and human-readable logging can share one pipe: a
watcher spawning the CLI reads stdout, treats sentinel lines as events and
everything else as log text, and gets both in their true interleaved order
without a second channel to synchronize.

Emitting is best-effort and never raises. A 100-round GPU run must not die
because a watcher hung up its pipe or a disk filled.
"""
import json
import sys

SENTINEL = "@@BENCH@@ "


class EventEmitter:
    """Writes JSONL events to a file, to stdout (sentinel-prefixed), or nowhere.

    ``enabled`` is False for the no-op emitter, so callers can skip building
    expensive payloads:  ``if em.enabled: em.emit("round", **heavy())``.
    """

    def __init__(self, dest: str | None = None):
        self.dest = dest or ""
        self._fh = None
        self._stdout = False
        self._broken = False
        if not self.dest:
            return
        if self.dest in ("-", "stdout"):
            self._stdout = True
        else:
            try:
                self._fh = open(self.dest, "w", encoding="utf-8", buffering=1)
            except OSError as e:                       # unwritable path: run anyway
                print(f"[events] cannot open {self.dest}: {e}", file=sys.stderr)
                self._broken = True

    @property
    def enabled(self) -> bool:
        return bool(self.dest) and not self._broken

    def emit(self, event: str, **fields) -> None:
        """Write one ``{"event": <name>, ...}`` line. Never raises."""
        if not self.enabled:
            return
        try:
            line = json.dumps({"event": event, **fields}, default=_fallback)
        except (TypeError, ValueError) as e:           # a payload we can't serialize
            line = json.dumps({"event": "warning",
                               "message": f"undumpable {event} event: {e}"})
        try:
            if self._stdout:
                sys.stdout.write(SENTINEL + line + "\n")
                sys.stdout.flush()
            else:
                self._fh.write(line + "\n")
        except (OSError, ValueError):                  # watcher hung up / disk full
            self._broken = True

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def _fallback(o):
    """Last-resort JSON coercion for stray numpy/torch scalars in a payload."""
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    return str(o)
