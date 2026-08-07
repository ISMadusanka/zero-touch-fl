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

# Pick the dataset (default: data.dataset from the config). It resolves exactly the
# way main.py does — same datasets.<name> overrides, Phase-1 state read from
# checkpoints/<dataset>/, results written to logs/<dataset>/benchmark/. The attacker
# adapter is shared, so this evaluates the same trained policy on the other task.
python -m benchmark.run_benchmark --rounds 200 --dataset cifar10

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
    --defenses fedavg,oracle,fltrust,llm_defender,defl,dnc,multikrum \
    --attack-temperature 0.7 --root-size 100 --eta 1.0 \
    --defl-delta 0.05 --defl-tau 2.5 --dnc-c 1.0 --dnc-sub-dim 10000 \
    --multikrum-m 4 --out logs/mnist/benchmark

# sweep the adversary's size: 1 .. fl.n_clients (20) poisoners per round.
# Evaluation is NOT limited to fl.n_compromisable (5) the way training is — the
# controllable pool is widened to match the quota, so `10` really does poison
# exactly 10 clients (clients 0..9 in that widened case).
for k in 1 3 5 10 15 20; do
  python -m benchmark.run_benchmark --rounds 50 --max-poison-clients $k \
      --out logs/mnist/benchmark/poisoners_$k
done
```

### `--max-poison-clients` (the exact eval poison quota)

Valid range is **1 .. `fl.n_clients`** (20 by default). Two things to keep in mind:

- It is an **exact quota, not a ceiling.** The attacker chooses which clients and
  their plans, but not how many: `--max-poison-clients 10` means exactly 10
  effective poisoned updates each completed round. The report labels this as an
  exact quota and each summary retains `mean_poisoned` as an audit field.
- **Past ~50% the defenses have no guarantee left.** Multi-Krum (needs `n ≥ 2f+3`),
  DnC and DeFL all assume the adversary is a minority; at or above half the
  federation, weak detection is the expected result rather than a defect. FLTrust is
  the exception — its trust comes from the server's clean root set, not from the
  client population. At `--max-poison-clients 20` there are no honest updates at all,
  so FPR/precision are degenerate and only the accuracy columns mean anything. The
  run warns about each of these thresholds as it starts.

Going above `fl.n_compromisable` also means the policy is being evaluated outside the
threat model it was fitted to (it trained against a 5-client foothold). That is a
legitimate generalization test, and the run says so — raise `fl.n_compromisable` and
retrain if you want it in-distribution.

Output: a console table + `logs/<dataset>/benchmark/benchmark.{json,csv}` + per-round
`history.json` + a 4-panel graph `benchmark.png` (accuracy per round, rolling
detection-rate, rolling FPR, and attack-strength per round). Re-plot a saved run
without re-running (the 200 rounds are slow + need the GPU):

```bash
python -m benchmark.plot --dataset mnist                        # -> benchmark.png
```

Disable graphing with `--no-plot`. Example table shape:

```
defense       detect%  FPR    prec  F1    final_acc  mean_acc  acc_drop  def_cost  atk_drop  atk_thru  atk_succ
------------  -------  -----  ----  ----  ---------  --------  --------  --------  --------  --------  --------
fedavg        0.0%     0.0%   0.00  0.00  0.71       0.70      +0.10     +0.00     +0.10     100.0%    100.0%
oracle        100.0%   0.0%   1.00  1.00  0.80       0.80      +0.00     +0.00     +0.00     0.0%      0.0%
llm_defender  ...      ...    ...   ...   ...        ...       ...       ...       ...       ...       ...
fltrust       ...      ...    ...   ...   ...        ...       ...       ...       ...       ...       ...
defl          ...      ...    ...   ...   ...        ...       ...       ...       ...       ...       ...
dnc           ...      ...    ...   ...   ...        ...       ...       ...       ...       ...       ...
multikrum     ...      ...    ...   ...   ...        ...       ...       ...       ...       ...       ...

Attack strength: mean poison perturbation = 1.4x the honest update it replaced
```

## The defenses

| name | what it is |
|---|---|
| `fedavg` | **No defense** — plain FedAvg over all clients (flags nobody). Lower bound + the state the attacker observes each round. Always included. |
| `oracle` | Cheats by flagging exactly the ground-truth poisoned clients. **Upper bound** on detection + robustness. |
| `llm_defender` | Your **trained defender adapter** — the model under test. The only column needing a *defender* checkpoint: it is **skipped with a warning** (not an error) when none exists, which is the normal case under `defense.mode: algorithmic`, where the defender LLM is disabled and only the attacker trains. Point at one with `--defender-adapter <dir>`, or set `defense.mode: llm` and train one. |
| `fltrust` | **FLTrust** (Cao et al., NDSS 2021): trust-bootstrapped robust aggregation using a small clean root dataset. |
| `defl` | **DeFL** (Yan et al., AAAI 2023): CLP-aware defense. Inspects the DNN layer-by-layer via a Federated Gradient Norm Vector (FGNV) to (a) detect the *critical learning period*, (b) flag malicious clients by per-layer outlier voting (MOUD-Vote), then hard-remove them during the CLP and soft-down-weight them after via a per-client Bayesian (Beta) trust. **Needs no clean root set and no LLM.** |
| `dnc` | **DnC** (Shejwalkar & Houmansadr, NDSS 2021): Divide-and-Conquer spectral aggregator. Subsamples dimensions, centers the updates, projects them onto their top singular vector, and filters out the `c·m` clients that project furthest (the spectral outliers), then averages the rest. **Needs no clean root set and no LLM** (assumes a known #malicious `m`). |
| `multikrum` | **Multi-Krum** (Blanchard et al., NeurIPS 2017): distance-based robust aggregator. Scores each client by the sum of squared distances to its `n−f−2` closest peers, keeps the `m` lowest-scoring (most-central) updates and averages them; drops the rest. **Needs no clean root set and no LLM** (assumes a known #Byzantine `f`). Krum = `m=1`. |

## The metrics

**Detection** (per-client accept/reject vs the ground-truth poisoned set, via
`metrics.compute.confusion_counts`):
- `detect%` / `detection_rate` = **recall / TPR** = the fraction of
  poisoned-client rounds the defense flagged → *"how much of the attack it caught."*
- `FPR` = honest clients wrongly flagged; `precision`, `F1`.

**Robustness** (the resulting model under attack):
- `final_acc`, `mean_acc` (test accuracy), `acc_drop` = mean accuracy lost vs the
  clean Phase-1 baseline (lower is better).
- `def_cost` / `mean_defense_cost` and `atk_drop` / `mean_attack_drop` = the two
  halves of `acc_drop`. See **Attribution** below; `atk_drop` is the honest
  robustness number.
- `atk_thru` / `attack_success_rate` = fraction of rounds a poisoned client
  slipped through.
- `atk_succ` / `goal_success_rate` = **weighted** attack success against the goal's
  requested drop: each round scores `min(1, atk_drop / target)` and those are
  averaged, so with `--goal 'untargeted_degrade=0.1'` a round where the attack cost
  the model 0.1 accuracy counts as 100%, one that cost 0.05 counts as 50%, and one
  that cost nothing counts as 0%. Overshoot is capped at 100% and an *improvement*
  floors at 0% (never negative). `goal_full_success_rate` (JSON/CSV) keeps the
  all-or-nothing reading — the fraction of rounds that reached the target in full —
  and each round's own weight is saved in `history.json` as `goal_success`.

**Attack strength** — `mean_poison_ratio`, printed under the table as
`Attack strength: mean poison perturbation = Nx the honest update it replaced`:
`‖poisoned − benign‖ / ‖benign − global‖`, i.e. the attacker's edit measured against
the honest update it replaced. **Read this before anything else in the table.**
Every other column is about the defense and quietly assumes there was an attack to
defend against. When the perturbation is well below the spread between honest
non-IID clients (< `0.05`), a robust aggregator ranks a poisoned update as *more
central* than a real one — it survives every filter and moves the global by nothing.
`0% detected / 100% throughput` then means "there was nothing to detect", not "the
defense failed", and the run says so explicitly (a `WARNING` in the log and under
the table). Panel 4 of `benchmark.png` plots it per round.

### Attribution: `acc_drop` = `def_cost` + `atk_drop`

Measuring every defense against one global baseline conflates *what the attack cost*
with *what the defense costs by itself*. They are not close to equal: a defense that
rejects most of an honest non-IID federation loses several points of accuracy on a
round with no poison in it at all, and against a single baseline the attacker is
credited for all of it. FLTrust is the standard case — it can exclude every poisoner
in 80% of rounds and still be reported as suffering a successful attack.

So each round every defense is *also* run on the **unpoisoned** updates
(`Defense.probe` — a real `step` whose effects on the global model and cross-round
memory are rolled back), giving that defense's own clean counterfactual:

| column | meaning |
|---|---|
| `def_cost` | `baseline − clean_accuracy` — the price of running this defense on an honest federation |
| `atk_drop` | `clean_accuracy − post_accuracy` — what the **attack** actually cost |
| `acc_drop` | their sum: the total loss vs the clean baseline |

`atk_succ` is scored on `atk_drop`. This is the same definition the training reward
already uses (`rl.env.FLArmsRaceEnv.clean_reference_accuracy`) — the benchmark was
the odd one out. It costs one extra aggregation + test-set evaluation per defense per
round; `--no-clean-counterfactual` skips it and reverts every drop to the
single-baseline measurement, in which case `def_cost`/`atk_drop` read `n/a`.

Per round, `history.json` carries `clean_accuracy`, `attack_drop` and
`poison_ratios` alongside the confusion counts.

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

**Therefore: treat `atk_drop` / `final_acc` (robustness) as the primary,
cross-paradigm comparison** — it's well-defined for every defense regardless of
whether it flags or re-weights, and unlike `acc_drop` it does not include the
accuracy each defense costs itself — and use `detect%` as a secondary, within-style
signal. For FLTrust the per-round `info.trust_sum` and the verdict confidences
also give a softer view than the binary flag.

One more trap worth naming, because it makes the whole table look flat: under the
project default `benign_retrain_each_round: false`, the weight-averaging defenses
(`fedavg`, `oracle`, `dnc`, `multikrum`) rebuild their global from the same frozen
benign weights every round, so **their model carries no memory between rounds** — an
N-round run measures N *independent one-shot attacks* and damage cannot accumulate.
The giveaway is `mean_acc == final_acc`. Only FLTrust (`w ← w + η·g`) and, partly,
DeFL (Beta counts) integrate across rounds. The runner logs this at startup; set
`benign_retrain_each_round: true` if you want a degradation goal that compounds.

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
- Runs in the project's native regime (5-of-5 controllable clients poisoned by default per
  `configs/base.yaml`), which keeps the LLM defender in-distribution. FLTrust
  still works here: benign client deltas point toward good weights and so align
  with the server's root update, while poison points away (negative cosine → zero
  trust).

## FLTrust knobs

`--root-size` (clean root samples, default 100, the paper's default) · `--root-epochs`
(server local epochs `R_l`, default 1) · `--root-lr` (default `fl.lr`) · `--eta`
(global learning rate, default 1.0). The root set is carved once from the clean
train set of the selected `--dataset`, with a fixed seed.

## DeFL knobs

`--defl-delta` (CLP relative-rise threshold δ, paper default **0.05** — larger ⇒
fewer early rounds declared critical ⇒ fewer hard removals) · `--defl-tau` (MOUD
per-layer outlier z-threshold, default **2.5** — lower ⇒ more aggressive flagging,
higher TPR but higher FPR). DeFL takes **no** root set: it derives everything from
the per-layer FGNV of the submitted updates. A "layer" = one module (its weight +
bias grouped), so MnistNet has L = 2 layers. The Beta trust then keeps the run
robust to the occasional false positive.

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
- `tests/test_attack_attribution.py` — the `def_cost`/`atk_drop` split, `Defense.probe`
  non-destructiveness, perturbation measurement and the negligible-poison floor (needs
  torch): `python tests/test_attack_attribution.py`.
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
