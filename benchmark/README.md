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

# Pick the attack goal the attacker aims for (fixed for the whole run, no per-round
# sampling). The trained attacker generalizes across targets, so you can evaluate it
# at any requested drop and read each defense's acc_drop against that target:
python -m benchmark.run_benchmark --rounds 200 --goal 'untargeted_degrade=0.1'
python -m benchmark.run_benchmark --rounds 200 --goal 'untargeted_degrade=0.3'
#   forms: untargeted_degrade=<drop> | slow_degrade=<drop> | targeted_label=<label>
#   default (no --goal): attack.goal from configs/base.yaml
# The chosen goal is printed in the report header and saved to benchmark.json.

# to chnage outcomes
python -m benchmark.run_benchmark --rounds 10 --seed 1
python -m benchmark.run_benchmark --rounds 10 --seed 2
python -m benchmark.run_benchmark --rounds 10 --seed 3

# choose defenses / tune FLTrust + DeFL + DnC + Multi-Krum / save outputs:
python -m benchmark.run_benchmark --rounds 200 \
    --defenses fedavg,oracle,fltrust,llm_defender,defl,dnc,multikrum,ensemble \
    --attack-temperature 0.7 --root-size 100 --eta 1.0 \
    --defl-delta 0.05 --defl-tau 2.5 --dnc-c 1.0 --dnc-sub-dim 10000 \
    --multikrum-m 4 --out logs/benchmark
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
defl          ...      ...    ...   ...   ...        ...       ...       ...
dnc           ...      ...    ...   ...   ...        ...       ...       ...
multikrum     ...      ...    ...   ...   ...        ...       ...       ...
```

## The defenses

| name | what it is |
|---|---|
| `fedavg` | **No defense** — plain FedAvg over all clients (flags nobody). Lower bound + the state the attacker observes each round. Always included. |
| `oracle` | Cheats by flagging exactly the ground-truth poisoned clients. **Upper bound** on detection + robustness. |
| `llm_defender` | Your **trained defender adapter** — the model under test. |
| `fltrust` | **FLTrust** (Cao et al., NDSS 2021): trust-bootstrapped robust aggregation using a small clean root dataset. |
| `defl` | **DeFL** (Yan et al., AAAI 2023): CLP-aware defense. Inspects the DNN layer-by-layer via a Federated Gradient Norm Vector (FGNV) to (a) detect the *critical learning period*, (b) flag malicious clients by per-layer outlier voting (MOUD-Vote), then hard-remove them during the CLP and soft-down-weight them after via a per-client Bayesian (Beta) trust. **Needs no clean root set and no LLM.** |
| `dnc` | **DnC** (Shejwalkar & Houmansadr, NDSS 2021): Divide-and-Conquer spectral aggregator. Subsamples dimensions, centers the updates, projects them onto their top singular vector, and filters out the `c·m` clients that project furthest (the spectral outliers), then averages the rest. **Needs no clean root set and no LLM** (assumes a known #malicious `m`). |
| `multikrum` | **Multi-Krum** (Blanchard et al., NeurIPS 2017): distance-based robust aggregator. Scores each client by the sum of squared distances to its `n−f−2` closest peers, keeps the `m` lowest-scoring (most-central) updates and averages them; drops the rest. **Needs no clean root set and no LLM** (assumes a known #Byzantine `f`). Krum = `m=1`. |
| `ensemble` | **All of the classical defenses together.** Every member above (default `fltrust,multikrum,dnc,defl`) scores the SAME updates against the SAME global model and votes; a client is rejected once `--ensemble-vote` members agree (`majority` = ⌈n/2⌉, or `any`/`all`/an int), and the survivors are FedAvg-averaged. **No LLM.** This is also the defense `python main.py --freeze defender` trains the attacker against — see the repo README. Configure members/vote here or under `defense:` in the config. |

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
- `defl` reports its **MOUD-Vote** decision as the flag (matching the paper's
  Table 2 detector). Note its **aggregation** decouples from that flag *after* the
  CLP: a flagged-malicious client is then only soft-down-weighted (Beta trust), not
  removed, so it can still be `detected` (counts toward TPR) yet leak a little into
  the model. As with FLTrust, read DeFL's `acc_drop` as the primary signal.
- `dnc` is an aggregator, not a classifier: its flag is **derived** (a client is
  `is_suspicious` exactly when it is filtered out of the aggregate, i.e. among the
  `c·m` highest spectral-outlier scores). By construction it removes a *fixed*
  `c·m` clients per round, so its TPR/FPR are pinned to that budget — read
  `acc_drop` as the primary signal here too.
- `multikrum` is likewise an aggregator with a derived flag (`is_suspicious` = not
  among the `m` selected). It drops a *fixed* `n−m` clients per round (default
  `m=n−f` → drops `f`), so its TPR/FPR are pinned to that budget — read `acc_drop`
  as the primary signal.

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

## DeFL knobs

`--defl-delta` (CLP relative-rise threshold δ, paper default **0.05** — larger ⇒
fewer early rounds declared critical ⇒ fewer hard removals) · `--defl-tau` (MOUD
per-layer outlier z-threshold, default **2.5** — lower ⇒ more aggressive flagging,
higher TPR but higher FPR). DeFL takes **no** root set: it derives everything from
the per-layer FGNV of the submitted updates. A "layer" = one module (its weight +
bias grouped), so MnistNet has L = 2 layers. With the default eval budget of 1
poisoned client, MOUD's adaptive vote isolates the single outlier; the Beta trust
then keeps the run robust to the occasional false positive.

## DnC knobs

`--dnc-num-byzantine` (assumed #malicious `m`; default = the eval poison budget
`--max-poison-clients` clamped to a benign majority — a hyperparameter / assumed
adversary budget, **not** per-round ground truth) ·
`--dnc-c` (filtering fraction `c`, paper iid default **1** → drop `c·m` clients each
round) · `--dnc-niters` (subsampling iterations, default **1**) · `--dnc-sub-dim`
(subsample dimension `b`, default **10000**, the paper's value — clamped to the
model's dimension, so for the tiny MnistNet (d=970) all coordinates are used and no
subsampling occurs). DnC centers by the client-mean, which cancels the global
reference, so it runs on the absolute weights directly and its aggregate is FedAvg
over the surviving clients.

## Multi-Krum knobs

`--multikrum-f` (assumed #Byzantine `f`; default = the eval poison budget, same
assumed-budget hyperparameter as DnC — **not** ground truth) · `--multikrum-m`
(#updates selected and averaged; default `n−f`, i.e. drop the `f` worst — set `1` for
plain Krum, or `n−f−2` for the paper's strong-resilience bound). The score always
sums each client's `n−f−2` closest squared distances (paper Eq. 5); for `n=5, f=1`
that is the 2 closest. Pairwise distances are global-invariant, so Multi-Krum runs on
absolute weights directly and its aggregate is FedAvg over the selected clients. Note
the paper assumes `n ≥ 2f+3` for its guarantee (with `n=5` that caps `f` at 1).

## Ensemble knobs

`--ensemble-members` (comma-separated; default `defense.members` from the config,
else `fltrust,multikrum,dnc,defl`) · `--ensemble-vote` (`majority` | `any` | `all` |
an int; default `defense.vote`, else `majority`). Each member is built with the
same knobs its standalone panel entry would get, so `--multikrum-f`, `--dnc-c`,
`--defl-tau`, `--root-size` … apply to the ensemble's copies too.

`majority` rather than `any` because Multi-Krum and DnC drop a FIXED quota of
clients every round even when nobody is malicious: under `any` their standing
quota alone would evict honest clients from the average every round. `oracle`,
`llm_defender` and `ensemble` are rejected as members (ground-truth cheating,
the model under test, and recursion respectively).

Detection read-out: `is_suspicious` = the vote passed the threshold, and
`confidence` = the fraction of members that agreed with the decision. Like its
members' flags this is a DERIVED aggregator signal, so read `acc_drop` as the
primary metric (see the caveat above).

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
- `tests/test_defl_logic.py` — DeFL's layer grouping, CLP rule, MOUD-Vote and Beta
  model. Torch-free, runs anywhere: `python tests/test_defl_logic.py`.
- `tests/test_defl.py` — DeFL's FGNV norms + end-to-end step (CLP hard-removal vs
  post-CLP soft-weighting; needs torch; run on the GPU box): `python tests/test_defl.py`.
- `tests/test_dnc_logic.py` — DnC's removal-count / lowest-k keep / intersection +
  fallback / subsampling. Torch-free: `python tests/test_dnc_logic.py`.
- `tests/test_dnc.py` — DnC's spectral scoring + end-to-end step (needs torch; run
  on the GPU box): `python tests/test_dnc.py`.
- `tests/test_multikrum_logic.py` — Multi-Krum's neighbour/selection counts, score
  formula and lowest-m selection. Torch-free: `python tests/test_multikrum_logic.py`.
- `tests/test_multikrum.py` — Multi-Krum's pairwise distances + end-to-end step
  (needs torch; run on the GPU box): `python tests/test_multikrum.py`.
- `tests/test_ensemble.py` — the ensemble's vote rules/confidence plus its
  end-to-end detection over the real members (needs torch, runs on CPU):
  `python tests/test_ensemble.py`.
- `tests/test_frozen_defender.py` — the `--freeze defender` training mode wired to
  this ensemble (needs torch, runs on CPU): `python tests/test_frozen_defender.py`.
