"""Append-only event bus: everything the server knows about the current run.

One bus per run kind (``train`` / ``bench``), so a benchmark started while a
training run is still going does not interleave into the training panel.

Every event gets a monotonic ``seq``. Clients poll ``/api/events?since=<seq>``,
which blocks in :meth:`EventBus.since` until there is something newer or the
timeout expires — a long-poll, which needs no extra dependency and survives the
proxy hops an SSE stream sometimes does not.

The bus is capped: a 200-round, 10-attack benchmark is ~200 round events plus a
few thousand log lines, so the cap only bites on pathological runs. Clients track
``seq``, so a reconnect after a drop resumes from whatever is still held and the
gap is visible rather than silent.
"""
import threading

MAX_EVENTS = 20000


class EventBus:
    def __init__(self, name: str = ""):
        self.name = name
        self._cv = threading.Condition()
        self._events: list[dict] = []
        self._seq = 0
        self._dropped = 0

    # -- producing ----------------------------------------------------------
    def publish(self, event: dict) -> dict:
        """Append one event, stamped with the next ``seq``. Returns it."""
        with self._cv:
            self._seq += 1
            event = {**event, "seq": self._seq}
            self._events.append(event)
            if len(self._events) > MAX_EVENTS:
                cut = len(self._events) - MAX_EVENTS
                del self._events[:cut]
                self._dropped += cut
            self._cv.notify_all()
            return event

    def emit(self, kind: str, /, **fields) -> dict:
        """``kind`` is positional-only: callers splat whole payload dicts in here
        (``bus.emit("run_started", **info)``) and those dicts legitimately carry a
        ``kind`` of their own -- the runner's, naming which CLI is running."""
        return self.publish({"event": kind, **fields})

    def clear(self) -> None:
        """Start a new run: drop history but KEEP the sequence counter climbing, so
        a browser still polling the old run gets fresh events rather than replaying
        seq numbers it has already seen."""
        with self._cv:
            self._events.clear()
            self._dropped = 0
            self._cv.notify_all()

    # -- consuming ----------------------------------------------------------
    def since(self, seq: int, timeout: float = 20.0) -> tuple[list[dict], int]:
        """Events newer than ``seq``, waiting up to ``timeout`` for the first one.

        Returns ``(events, latest_seq)``. An empty list means the timeout expired
        with nothing new — the client re-polls; that is the keep-alive.
        """
        with self._cv:
            fresh = [e for e in self._events if e["seq"] > seq]
            if not fresh:
                self._cv.wait(timeout)
                fresh = [e for e in self._events if e["seq"] > seq]
            return fresh, self._seq

    @property
    def seq(self) -> int:
        with self._cv:
            return self._seq

    def snapshot(self) -> list[dict]:
        with self._cv:
            return list(self._events)
