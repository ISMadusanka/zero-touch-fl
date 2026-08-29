"""Web control panel for the zero-touch-fl testbed.

``python -m webui`` serves a local page that drives the two things this repo does
— GRPO training (``main.py``) and the attack x defense benchmark
(``benchmark.run_benchmark``) — and streams what they are doing back to the
browser as they run.

Design rule: **the UI is a watcher, not a second implementation.** Every run it
starts is the same subprocess you would launch from a shell, with the exact argv
published to the page so it can be copy-pasted. Nothing about a run's behaviour
lives here; the CLIs stay the source of truth.

Modules
-------
``bus``         append-only event bus with sequence numbers + long-poll
``configstore`` reads ``configs/base.yaml`` (values, types and its inline comments
                as help text) and writes validated per-run derived configs
``versions``    the fine-tuned-model version store: snapshots of the LoRA adapters
                that a benchmark can then be pointed at
``runner``      spawns / supervises / stops the training and benchmark subprocesses
                and turns their output into events
``server``      the HTTP + Server-Sent-Events API and the static page
"""

__all__ = ["bus", "configstore", "runner", "server", "versions"]
