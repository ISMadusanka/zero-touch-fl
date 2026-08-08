"""Live web UI for the targeted-poisoning benchmark.

    python -m benchmark.ui

Opens a local dashboard that runs

    python -m benchmark.run_targeted_benchmark --rounds N --label L --poison-client-ids ...

as a child process and streams what happens — per-round target-class recall for
every defense, who each defense flagged, the attacker's plan — then shows the
same summary table the CLI prints.

The UI does not reimplement the benchmark. It spawns the real command with
``--events -`` and reads its structured stream (``benchmark.events``), so what
you see is exactly what the CLI would have produced, and the command it ran is
shown so it can be copy-pasted into a terminal.
"""
