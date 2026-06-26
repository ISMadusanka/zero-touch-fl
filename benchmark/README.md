# Defense Benchmark

Pits the **trained attacker** against a panel of defenses — the **trained
defender LLM** plus established baselines — for *N* attack rounds, and reports
**how much of the attack each defense detected** and how well it preserved model
accuracy.

This package is **purely additive**: it reuses the existing FL components
read-only (`model/`, `data/`, `clients/`, `server/`, `agents/`, `detector/`,
`rl/policy.py`, `rl/env.py`, `metrics/`) and modifies **none** of them.

## Run it

On the GPU box (needs torch / unsloth / peft + the trained adapters in `checkpoints/`):

```bash
python -m benchmark.run_benchmark --rounds 200
# choose defenses / tune FLTrust / save outputs:
python -m benchmark.run_benchmark --rounds 200 \
    --defenses fedavg,oracle,fltrust,llm_defender \
    --attack-temperature 0.7 --root-size 100 --eta 1.0 --out logs/benchmark
```

Output: a console table + `logs/benchmark/benchmark.{json,csv}` + per-round
`history.json` + a 4-panel graph `benchmark.png` (accuracy per round, rolling
detection-rate, rolling FPR, and attack-strength per round). Re-plot a saved run
without re-running (the 200 rounds are slow + need the GPU):

```bash
python -m benchmark.plot --history logs/benchmark/history.json   # -> benchmark.png
```

Disable graphing with `--no-plot`. Example table shape:

```
defense       detect%  FPR    prec  F1    final_acc  mean_acc  acc_drop  atk_thru
------------  -------  -----  ----  ----  ---------  --------  --------  --------
fedavg        0.0%     0.0%   0.00  0.00  0.71       0.70      +0.10     100.0%
oracle        100.0%   0.0%   1.00  1.00  0.80       0.80      +0.00     0.0%
llm_defender  ...      ...    ...   ...   ...        ...       ...       ...
fltrust       ...      ...    ...   ...   ...        ...       ...       ...
```

## The defenses

| name | what it is |
|---|---|
| `fedavg` | **No defense** — plain FedAvg over all clients (flags nobody). Lower bound + the state the attacker observes each round. Always included. |
| `oracle` | Cheats by flagging exactly the ground-truth poisoned clients. **Upper bound** on detection + robustness. |
| `llm_defender` | Your **trained defender adapter** — the model under test. |
| `fltrust` | **FLTrust** (Cao et al., NDSS 2021): trust-bootstrapped robust aggregation using a small clean root dataset. |

## The metrics

**Detection** (per-client accept/reject vs the ground-truth poisoned set, via
`metrics.compute.confusion_counts`):
- `detect%` / `detection_rate` = **recall / TPR** = the fraction of
  poisoned-client rounds the defense flagged → *"how much of the attack it caught."*
- `FPR` = honest clients wrongly flagged; `precision`, `F1`.

**Robustness** (the resulting model under attack):
- `final_acc`, `mean_acc` (test accuracy), `acc_drop` = mean accuracy lost vs the
  clean Phase-1 baseline (lower is better).
- `atk_thru` / `attack_success_rate` = fraction of rounds a poisoned client
  slipped through.

### Reading the detection numbers (important caveat)

Detection metrics are **only loosely comparable** across defense *styles*, so
read them with care:

- The `llm_defender` emits an **explicit binary** flag (what it was trained to
  do). FLTrust does **not** flag — it assigns a **continuous trust weight**; we
  *derive* a reject decision (`is_suspicious` when trust = 0, exactly the clients
  ReLU drops from the aggregate). So FLTrust's TPR/FPR/F1 measure a *proxy*, not a
  classifier it optimises.
- Each defense scores detection against **its own diverged global** (FLTrust uses
  cos-to-`g0`; the LLM defender uses features vs *its* current model). The globals
  are identical only at round 1, so per-round detection is strictly
  apples-to-apples only early on; the 200-round aggregate `detect%`/`F1` blends
  "detector quality" with "how far each model drifted."

**Therefore: treat `acc_drop` / `final_acc` (robustness) as the primary,
cross-paradigm comparison** — it's well-defined for every defense regardless of
whether it flags or re-weights — and use `detect%` as a secondary, within-style
signal. For FLTrust the per-round `info.trust_sum` and the verdict confidences
also give a softer view than the binary flag.

## How the comparison is kept fair

- The env (`rl/env.py`) is reused **only as a generator**: each round it samples
  the poisoned subset and builds the benign + attacker-poisoned client updates.
- The trained attacker produces **one** attack per round; the **same** set of
  client updates is fed to **every** defense. We *vary the defense and hold the
  attack fixed.* The attacker plans against the **frozen Phase-1 benign weights**
  plus the no-defense **scalar accuracy** (one-round lag) — it does not see each
  defense's evolving global. (Making the attack adapt per-defense would change the
  threat model and break the held-fixed comparison.)
- Each defense evolves its **own** global model, so robustness reflects how that
  defense's model fares over the run. A round a defense **skips** (produces no new
  global — e.g. FLTrust with all-zero trust) keeps its previous model; that
  round's accuracy is its held accuracy and still counts toward `mean_acc`. The
  count of such rounds is reported as `skipped_rounds` (JSON/CSV).
- **Assumes `benign_retrain_each_round: false`** (the project default — frozen
  benign replay), so the benign client updates are identical across all defenses
  each round (the source of the "same attack to everyone" fairness). The runner
  warns if it is `true`.
- Runs in the project's native regime (1-of-5 poisoned by default per
  `configs/base.yaml`), which keeps the LLM defender in-distribution. FLTrust
  still works here: benign client deltas point toward good weights and so align
  with the server's root update, while poison points away (negative cosine → zero
  trust).

## FLTrust knobs

`--root-size` (clean root samples, default 100, the paper's default) · `--root-epochs`
(server local epochs `R_l`, default 1) · `--root-lr` (default `fl.lr`) · `--eta`
(global learning rate, default 1.0). The root set is carved once from the clean
MNIST train set with a fixed seed.

## Adding another defense (later)

1. Create `benchmark/defenses/<name>.py` with a `class <Name>(Defense)` that
   implements `step(updates, poisoned_ids) -> StepResult` (return a new global
   state_dict + per-client `DetectionVerdict`s).
2. Register it in `benchmark/defenses/__init__.py` (`build_defenses` + `AVAILABLE`).

Good next candidates: **Krum / Multi-Krum**, **coordinate-wise Median**,
**Trimmed-Mean** (Yin et al.), **Norm-clipping**, **FLDetector**. Krum/Median/
Trimmed-Mean are robust aggregators — derive a reject signal the same way FLTrust
does (a client is "rejected" when it's excluded / trimmed from the aggregate).

## Tests

- `tests/test_benchmark.py` — torch-free (metrics + report). Runs anywhere:
  `python tests/test_benchmark.py`.
- `tests/test_fltrust.py` — the FLTrust trust/aggregation math (needs torch; run
  on the GPU box): `python tests/test_fltrust.py`.
