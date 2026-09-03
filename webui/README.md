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
malformed / collab terms, the defender's reward with its TPR/FPR, GRPO step health
(loss, mean reward, reward spread, zero-advantage fraction), attack potency, the
live client grid (TP / FN / FP / TN per client, with the attacker's pool marked),
the current curriculum block, and the console.

**Which side is learning reorders the readout.** `--learn` picks one policy to
optimize and leaves the other frozen, so half of those numbers describe a policy
that is not moving. The KPI strip therefore leads with the learner's block —
defender reward and detection under `--learn defender`, induced drop and attacker
reward under `--learn attacker` — and marks the other side `frozen`. Nothing is
hidden; the run is just not described by its frozen half.

**Training the defender** needs `defense.mode: llm`: under the shipped
`algorithmic` the server defends with FLTrust / DeFL / DnC / Multi-Krum, which
have no parameters, so there is no defender policy to put an optimizer on. That is
the CLI's rule, not the panel's, and the panel **applies the CLI's own resolver**
(`core/config_overrides.py`) to the run's config before spawning anything — so the
combination is refused in the response to the click, with the message `main.py`
would have printed, rather than by a process that starts and dies. The selector
says so inline and offers the one-click `defense.mode: llm`.

### Model versions

Training overwrites the adapter it is training — `checkpoints/attacker_adapter`,
`checkpoints/defender_adapter`, or both — **in place** every `rl.save_every`
rounds. That is right for resuming and wrong for evaluating: by the time a
benchmark finishes, the adapter it was told to load has moved on, and there is no
way to ask "how did round 400 compare to round 900?".

A **version** is a copy of the adapter directories taken at a moment you choose,
stored under `checkpoints/versions/<id>/` next to the resume state and a record of
what training looked like when it was taken (rounds done, both sides' mean rewards,
mean induced drop, detection TPR/FPR, win rate, which side those rounds were
training, the base model it is dimensioned for). A snapshot only *reads*
`checkpoints/` — a run in flight is never disturbed.

A version holds **whichever adapters exist**, and `roles` records which. That
matters because `rl/schedule.py` writes only the side `--learn` named: an
attacker-only run leaves no defender adapter and a `--learn defender` run leaves no
attacker one. The table's `holds` column shows it, and the benchmark panel will not
let you select a version for a role it does not hold.

### Benchmark

The attack panel, the defense panel, the goal, the poison quota, the federation
size, adversary knowledge, and every per-attack and per-defense hyperparameter the
CLI accepts.

**Two start buttons, one per side under test.** *Benchmark attacker* and
*Benchmark defender* run the same `benchmark.run_benchmark` over the same panel;
what differs is the `target`, and that is what makes a run answerable. It fixes
which version axis may be swept, which slice of the matrix the comparison reads
(the `llm` row vs the `llm_defender` column) and which way *better* points — an
attacker wants the drop it caused to be large, a defender wants the drop it
allowed to be small. A defender run adds `llm_defender` to the defense panel if it
is not already there, rather than sending a defense list the page does not show.

**The two adapters are picked independently** — an attacker row and a defender row
— because that is how they are produced: `--learn` trains one side against a frozen
opponent, so the defender worth evaluating and the attacker worth evaluating it
against normally come from different snapshots. The attacker version feeds the
`llm` attack row (`--attacker-adapter`); the defender version feeds the
`llm_defender` defense column (`--defender-adapter`).

Pick **one** or **several** on the target's axis: several run back to back as
ordinary benchmark subprocesses, each with its own output directory. The opponent
is held fixed — selecting several there is refused and names the other button,
because those legs would differ in a dimension the comparison does not score and
then be ranked as if they did not. An axis the panel cannot distinguish collapses
to one leg (the attacker version is inert without the `llm` row, the defender
version without the `llm_defender` column), and selecting several on an inert axis
is refused too, since the legs would be byte-identical. A direct API call may omit
`target`, in which case the swept axis is inferred and both axes may vary at once;
`MAX_SWEEP_LEGS` caps that product.

A role's adapter is resolved **before** anything launches, and a version that does
not hold it is a refusal, not a fallback. This is load-bearing: omitting
`--defender-adapter` does not mean "skip it", it means the CLI falls back to the
live `checkpoints/defender_adapter`, so a silently-unresolved defender used to
benchmark whatever the last training run left on disk and label the result with the
version you picked.

While it runs, per round: which clients were poisoned, which the focused defense
flagged (and how far the attack moved each one's weights), each defense's accuracy
and drop for that round, the goal-achieved strip, and an attack × defense heat
matrix filling in. At the end: a plain-language verdict, the full sortable matrix,
and — for a sweep — a **version comparison** along the swept axis: the `llm` row
for an attacker sweep (more drop is better), the `llm_defender` column for a
defender sweep (less drop allowed is better, shown with FPR and F1). The plain
language verdict leads with the side under test, so a defender run opens with how
the trained defender ranked rather than with which attack hurt most.

If the CLI drops the `llm_defender` column because no defender adapter was found,
the page says so — a missing row and a row never asked for look identical in a
matrix.

### The demo fixture

One version id, `v000`, is a **fixture** rather than a snapshot: it has no adapter
directory behind it, and a benchmark aimed at it runs `webui/demo_bench.py`
instead of `benchmark.run_benchmark`. It exists so the whole benchmark path — the
live theatre, the heat matrix filling in, the console, the closing table, the
saved `history.json`, the Runs tab — can be walked through on a laptop with no
GPU, no dataset and no trained adapter.

It is fenced off rather than woven in:

- the replay is a **separate program**. The real CLI has no idea it exists and so
  cannot be put into a mode where it invents numbers. The two share only the event
  protocol (`benchmark/events.py`) and the log shape, which is what lets the page
  render both with no branch on the client;
- `v000` is outside the range the store can mint (`_next_index` starts at 1), so
  no real snapshot collides with it;
- a leg pairing the fixture with a real version is **refused**, because the
  fixture has no adapter and the run would quietly fall back to the live
  checkpoint for the other side and report it under the fixture's name.

`webui/demo.py` holds the result it replays: a per-defense table for the `llm`
row, quoted at 250 rounds and reproduced verbatim at exactly that count. Ask for a
different number and it is perturbed instead — deterministically in the round
count and seed, and by more on a shorter run, because a shorter run really is
noisier. Structural facts do not drift: FedAvg still detects nothing, Oracle still
reads the ground truth, and `mean_acc_drop` is always recomputed from the one
baseline so the identity the report checks still holds. Every other attack row is
that table moved toward the clean baseline by the attack's strength.

The table is a fixture, not a measurement, and its columns are not all derivable
from one another the way `benchmark.metrics` derives them from confusion counts.
So the replay does not try to reproduce it by feeding synthetic rounds through the
real metrics: it emits the table as the authoritative `summary`, and generates a
per-round stream that converges on it — the accuracy trajectory ends at
`final_accuracy` and averages `mean_accuracy`, detection converges on
`detection_rate`/`fpr`, and the goal score on its rate. So the heat matrix agrees
with the closing table on all four of its metrics instead of visibly changing its
mind at the end.

Rounds are paced so the run unfolds rather than appearing all at once — **3–8 s
each** by default, which puts a 250-round replay at 12–33 minutes. *Demo round
delay* under **Attack & defense hyperparameters** takes a `MIN,MAX` in seconds;
`0,0` finishes in about a second.

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
