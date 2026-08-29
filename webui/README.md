# Web control panel

```bash
python -m webui
```

Opens `http://localhost:8090` — a local page that drives the two things this repo
does, GRPO **training** (`main.py`) and the attack × defense **benchmark**
(`benchmark.run_benchmark`), and streams what they are doing back to the browser
as they run.

```bash
python -m webui --port 9000        # a different port
python -m webui --no-browser       # do not open a browser window
python -m webui --debug            # verbose server logging
```

Stdlib only (`http.server` + `yaml`, which the project already depends on), so it
starts on a GPU box with no extra `pip install`.

## The one design rule

**The UI is a watcher, not a second implementation.** Every run it starts is the
same subprocess you would launch from a shell, and the exact argv is printed on
the page so you can copy-paste it. Nothing about how a run *behaves* lives here;
the CLIs stay the source of truth, and a result produced through the panel is the
same result the terminal would have produced.

That rule is what decides how the live metrics work:

- **Training** needed no changes at all. `main.py` already appends one JSON record
  per Phase-2 round to `logs/round_data/rounds.jsonl` — accuracies, the clean
  counterfactual, rewards broken into terms, per-client verdicts, the GRPO step's
  statistics. It is the richest description of a round that exists, so rather than
  teaching the trainer to emit a second copy, the server **tails that file from the
  byte offset the run started at** and republishes each new record. Phase 1, which
  writes no round log, is scraped from the two lines it prints per round.
- **The benchmark** grew an opt-in `--events` flag (`benchmark/events.py`) that
  emits a JSONL description of each round. Without the flag, CLI output is
  byte-identical to before.

## Views

### Training

Mode (`train` / `dry-run` / `baseline`), `--rounds`, `--poisoners`, `--learn`,
`--env`, `--fresh`, `--debug` — plus **every knob in `configs/base.yaml`**, with
the YAML's own inline comments as the help text.

While it runs: accuracy against the clean counterfactual, induced drop against the
target and the win gate, the attacker reward split into its damage / stealth /
malformed / collab terms, GRPO step health (loss, mean reward, reward spread,
zero-advantage fraction), detection TPR/FPR, attack potency, the live client grid
(TP / FN / FP / TN per client, with the attacker's pool marked), the current
curriculum block, and the console.

### Model versions

Training overwrites `checkpoints/attacker_adapter` **in place** every
`rl.save_every` rounds. That is right for resuming and wrong for evaluating: by
the time a benchmark finishes, the adapter it was told to load has moved on, and
there is no way to ask "how did round 400 compare to round 900?".

A **version** is a copy of the adapter directories taken at a moment you choose,
stored under `checkpoints/versions/<id>/` next to the resume state and a record of
what training looked like when it was taken (rounds done, mean reward, mean induced
drop, win rate, potency, the base model it is dimensioned for). A snapshot only
*reads* `checkpoints/` — a run in flight is never disturbed.

### Benchmark

The attack panel, the defense panel, the goal, the poison quota, the federation
size, adversary knowledge, and every per-attack and per-defense hyperparameter the
CLI accepts. Pick **one version** or **several**: several run as one sweep — an
ordinary benchmark subprocess per version, back to back, each with its own
`--attacker-adapter` and its own output directory.

While it runs, per round: which clients were poisoned, which the focused defense
flagged (and how far the attack moved each one's weights), each defense's accuracy
and drop for that round, the goal-achieved strip, and an attack × defense heat
matrix filling in. At the end: a plain-language verdict, the full sortable matrix,
and — for a sweep — a **version comparison** over the `llm` row.

### Run history

Every run started from the panel keeps its resolved config, console log and (for a
benchmark) `history.json` under `logs/webui/runs/<run_id>/`. Opening one rebuilds
its matrix from the saved per-round records.

## Safety

The panel starts processes on the machine it runs on and **has no authentication**,
so it binds to `127.0.0.1` by default. To reach it on a remote GPU box, tunnel:

```bash
ssh -i <key> -L 8090:localhost:8090 <user>@<server>
```

`--host 0.0.0.0` works and logs a warning; prefer the tunnel.

Within that boundary the request handling is still defensive, because a local page
is not a trusted caller:

- Every field is validated into a **list-form argv** — no shell, no free-form
  argument passthrough.
- A config override is accepted only for a dotted path that **already exists** in
  `configs/base.yaml`, and only when it coerces to the type already there. The
  result is written to the run's own directory and passed as `--config`;
  **`configs/base.yaml` is never modified.**
- Output directories must be relative and inside the repo; version ids must match
  the shape the store generates, so they cannot address another directory.
- The page is self-contained under a strict CSP — no remote scripts, styles or
  fonts.

## Layout

| file | what it does |
|---|---|
| `bus.py` | append-only event bus with sequence numbers + long-poll |
| `configstore.py` | reads `base.yaml` (values, types, inline comments) and writes validated per-run configs |
| `versions.py` | the fine-tuned-model version store |
| `runner.py` | spawns / supervises / stops the subprocesses and turns their output into events |
| `server.py` | the HTTP API, the argv specs, and the benchmark sweep queue |
| `static/` | the page: `index.html`, `styles.css`, `charts.js` (canvas primitives), `app.js` |

Tests: `python -m pytest tests/test_webui.py` — no GPU, no LLM, no dataset, no
network.
