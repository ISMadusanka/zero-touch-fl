# Attack x Defense Benchmark

Runs a panel of **attacks** against a panel of **defenses** for *N* rounds and
reports the matrix: how much accuracy each attack cost each defense, and how much
of each attack each defense detected.

The attack panel puts the **trained attacker** side by side with the published
state-of-the-art untargeted poisoning attacks — LIE, Min-Max, Min-Sum, Fang,
Fang-Krum, IPM, Mimic, plus the classic Byzantine baselines — so the question
"is the learned policy actually better than what is already published?" is one
table rather than a cross-run comparison. The defense panel is the **trained
defender LLM** plus established robust aggregators.

Every attack in a round poisons the **same clients**, sees the **same honest
updates**, and is scored over the **same rounds**. Only the crafted updates differ.

This package is **purely additive**: it reuses the existing FL components
read-only (`model/`, `data/`, `clients/`, `server/`, `agents/`, `detector/`,
`rl/policy.py`, `rl/env.py`, `metrics/`) and modifies **none** of them.

## Run it

On the GPU box (needs torch / unsloth / peft + the trained adapters in `checkpoints/`):

```bash
python -m benchmark.run_benchmark --rounds 200

# the attack panel (default = llm + every published baseline). Add the no-attack
# control row to get each defense's clean accuracy as a denominator:
python -m benchmark.run_benchmark --rounds 200 --attacks clean,llm,lie,min_max,min_sum,fang,ipm,mimic

# just the trained policy, as before (identical output to the old single-attack report):
python -m benchmark.run_benchmark --rounds 200 --attacks llm

# published baselines only — no adapter, no GPU:
python -m benchmark.run_benchmark --rounds 20 --attacks lie,min_max,fang --device cpu

# give the baselines the omniscient view their papers assume (a stronger adversary
# than the trained attacker, which only ever sees its own clients):
python -m benchmark.run_benchmark --rounds 200 --baseline-knowledge full

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
    --multikrum-m 4 --out logs/benchmark

# sweep the adversary's size: 1 .. fl.n_clients (20) poisoners per round.
# Evaluation is NOT limited to fl.n_compromisable (10) the way training is — the
# controllable pool is resized to match the quota, so `15` really does poison
# exactly 15 clients (clients 0..14 in that widened case).
# Under a fixed poisoned set (attack.fixed_poison_clients, the shipped default)
# --max-poison-clients resizes that set: the first k clients, the same every round.
for k in 1 3 5 10 15 20; do
  python -m benchmark.run_benchmark --rounds 50 --max-poison-clients $k \
      --out logs/benchmark/poisoners_$k
done
```

> **Note.** Attack rows other than `llm` need no adapter and no GPU, so a
> baselines-only panel is a fast CPU sanity check of the whole pipeline:
>
> ```bash
> python -m benchmark.run_benchmark --rounds 5 --attacks lie,min_max,fang \
>     --defenses fedavg,oracle,multikrum --device cpu
> ```

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
threat model it was fitted to (it trained against a 10-client foothold). That is a
legitimate generalization test, and the run says so — raise `fl.n_compromisable` and
retrain if you want it in-distribution.

### Unusable attacker rounds (`--attack-retries`)

A single attacker generation can come back unusable: truncated mid-JSON at
`rl.max_new_tokens`, or a plan that parses fine but changes no weight (`scale`
with `factor: 1.0`, an unknown operator, a target no layer matches).
`select_and_apply` will not label byte-identical benign weights as poison, so such
a round produces fewer effective plans than the quota and cannot be measured.

That is sampling noise, not a broken run, so the round's action is **resampled**
up to `--attack-retries` times (default **3**; retries are drawn at temperature
≥ 0.7 even under `--attack-temperature 0`, since a greedy redraw would return the
identical text). The run log names the cause — including whether the generation hit
the token cap — and quotes the offending output.

A round with no usable action after every retry is **skipped for the WHOLE panel**:
logged at ERROR, excluded from every (attack, defense) cell — including the published
baselines, which would otherwise be averaged over more rounds than the policy is — and
the run continues. It is not scored,
because feeding the panel an all-honest round whose ground truth claims `budget`
poisoners would depress `detect%` and `acc_drop` for an attack that never happened.
The report header and each summary's `rounds` therefore count **measured** rounds,
and `history.json` records `requested_rounds` / `measured_rounds` /
`unusable_attack_rounds`. Persistent skips mean the attacker output is too long for
its budget (raise `rl.max_new_tokens`) or the adapter is degenerate — not a defense
result.

Output: console tables + `logs/benchmark/benchmark.{json,csv}` (one row per
(attack, defense) cell) + per-round `history.json` + graphs. With several attacks
the graphs are `benchmark_attacks.png` (the attack x defense comparison: undefended
accuracy per round, an acc_drop heatmap, a detection heatmap, and acc_drop grouped
bars) plus one per-attack `benchmark_<attack>.png` in the original 4-panel layout
(accuracy per round, rolling detection-rate, rolling FPR, attack strength). A
single-attack run writes just `benchmark.png`, as before.

Re-plot a saved run without re-running (the 200 rounds are slow + need the GPU) —
the history is nested by attack for a matrix run and flat for a single-attack one,
and the plotter detects which it is, so older saved runs still work:

```bash
python -m benchmark.plot --history logs/benchmark/history.json
```

Disable graphing with `--no-plot`. Example table shape (one per attack, under the
matrix grids):

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
- `atk_thru` / `attack_success_rate` = fraction of rounds a poisoned client
  slipped through.
- `atk_succ` / `goal_success_rate` = **weighted** attack success against the goal's
  requested drop: each round scores `min(1, acc_drop / target)` and those are
  averaged, so with `--goal 'untargeted_degrade=0.1'` a round that cost the model
  0.1 accuracy counts as 100%, one that cost 0.05 counts as 50%, and one that cost
  nothing counts as 0%. Overshoot is capped at 100% and an *improvement* floors at
  0% (never negative). `goal_full_success_rate` (JSON/CSV) keeps the all-or-nothing
  reading — the fraction of rounds that reached the target in full — and each
  round's own weight is saved in `history.json` as `goal_success`.

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

## The attacks

The second axis. `--attacks` selects the panel; the trained policy (`llm`) is one
row among the published state-of-the-art untargeted attacks, and every row is run
over the same rounds, against the same honest updates, poisoning the same clients.

| name | what it is |
|---|---|
| `llm` | **The trained attacker adapter — the system under test.** Observes compact per-layer statistics of its clients' benign updates and emits an attack plan in the operator DSL (`agents/attack_ops.py`). The only row needing a GPU + checkpoint; drop it from `--attacks` for a baselines-only run. |
| `lie` | **LIE / "A Little Is Enough"** (Baruch et al., NeurIPS 2019). The canonical *stealthy* attack: all colluders send `mu + z*sigma`, the coordinate-wise honest mean shifted by a fraction of the population's own standard deviation. `z` is the paper's `z^max`, derived from `n` and the poison count, not a knob. |
| `min_max` | **AGR-agnostic Min-Max** (Shejwalkar & Houmansadr, NDSS 2021). Solves for the largest perturbation whose distance to the furthest honest update stays inside the honest cloud's own diameter. Needs no knowledge of the server's aggregation rule. |
| `min_sum` | **AGR-agnostic Min-Sum** (same paper). Same shape, but the constraint bounds the *sum* of squared distances to the honest updates. Neither dominates the other, which is why the paper reports both. |
| `fang` | **Fang et al.** (USENIX Security 2020), the variant tailored to coordinate-wise **trimmed mean / median**: per coordinate, place the compromised values just outside the honest range on the side that opposes the honest direction. Each compromised client draws its own value, so this row is stochastic across rounds. |
| `fang_krum` | **Fang et al.**, the variant tailored to **Krum / Multi-Krum** — the rule the `multikrum` column runs. All colluders send `mu - lambda*s`, with `lambda` halved from the paper's derived upper bound until a *simulated* Krum selects one of them. The simulation reuses `benchmark/defenses/multikrum.py`, so the attack is solved against exactly the code it faces. |
| `ipm` | **Inner Product Manipulation** (Xie et al., UAI 2019). Sends `-epsilon * mu`: perfectly collinear with the honest consensus and therefore invisible to norm/distance filters, while cancelling the honest progress. The default `epsilon=0.1` is the paper's stealthy setting. |
| `mimic` | **Mimic** (Karimireddy et al., ICLR 2022). Every colluder copies one carefully chosen benign client's update — so the malicious update *is* an honest update and no anomaly detector can flag it — over-weighting that client's data distribution. Designed for heterogeneous data, which is what `data.iid: false` gives this project. |
| `sign_flip` | Classic sign-flipping Byzantine baseline: `-c *` the client's own honest update. At `c=1` with `m` of `n` poisoners the aggregate keeps only `(n-2m)/n` of the honest progress. |
| `noise` | Classic Gaussian/random Byzantine baseline (Blanchard et al., NeurIPS 2017). `sigma` is expressed as a multiple of the honest per-coordinate standard deviation so it is scale-free. Note it is **zero-mean, so it averages out under FedAvg** — needing a large `sigma` to do damage while being easy to detect is the classic finding, not a bug. |
| `scaling` | Boosting / model replacement (Bagdasaryan et al., AISTATS 2020), read untargeted: `gamma *` the client's own honest update. The control for "does the defense do the easy thing?" — norm clipping and distance filters exist specifically for this. Opt-in. |
| `label_flip` | Untargeted **data** poisoning (Biggio et al., ICML 2012; Fang et al. Sec. 3): the compromised clients train honestly on relabelled data. A different threat model, not a stronger attack — nothing about the resulting update is anomalous, because it is a real gradient of a wrong objective. Opt-in; **forces `--benign-retrain`** so the honest updates come from the same process. |
| `clean` | **Not an attack — the no-attack control row.** Nothing is poisoned, so each defense's row is its clean accuracy. Opt-in, and worth adding: it is the per-defense denominator for every other row (`clean - attack` for the same defense), and because its ground truth is empty it doubles as a clean-round false-alarm rate. |

Default panel: `llm,lie,min_max,min_sum,fang,fang_krum,ipm,mimic,sign_flip,noise`.
`scaling`, `label_flip` and `clean` are opt-in via `--attacks`.

### Adversary knowledge (`--baseline-knowledge`)

Every published attack estimates the honest population from the updates it can
see. Which updates those are is a threat-model choice, so it is one flag for the
whole panel:

- **`partial`** (default) — only the **compromised** clients' honest updates. This
  is the like-for-like comparison: it is exactly what the trained attacker
  observes, so a difference between rows is a difference in what the adversary
  *did*, not in what it *knew*.
- **`full`** — every client's honest updates. Omniscient, and the primary setting
  most of the papers state their attack in, so it is the fairer reading of the
  *literature*. A strictly stronger adversary than the one `llm` faces.

Under `partial` the baselines still get **more** raw information than the trained
attacker (full weight vectors versus compacted per-layer statistics), so the
comparison does not flatter the policy.

A few attacks need at least two visible honest updates to have a population to
estimate (`lie`, `min_max`, `min_sum`, `mimic`). At `--max-poison-clients 1` with
`partial` knowledge they degenerate to submitting the honest mean; that is logged
loudly rather than reported as a weak attack. Use `--baseline-knowledge full` there.

### Attack knobs

`--lie-z` / `--lie-sign` · `--minmax-perturbation {std,unit_vec,sign}` ·
`--minmax-gamma0` · `--minsum-bound {max,min}` · `--fang-b` · `--fang-krum-f` ·
`--fang-krum-lambda-mult` · `--ipm-epsilon` · `--mimic-warmup` · `--signflip-c` ·
`--noise-sigma` · `--scaling-gamma` · `--labelflip-mode {reverse,next,random}`.

Every default is the value its paper states (or, where the paper gives an absolute
value that only makes sense for its own model, a scale-free equivalent — see the
module docstring in `benchmark/attacks/`). `fang_krum` is *AGR-tailored*, so it is
given the same assumed `f` the `multikrum` column is configured with: knowing the
rule and its parameters is that attack's threat model, not an unfair peek.

## Reading the matrix

```
ACC_DROP  — mean test accuracy the attack cost this defense
attack     fedavg   oracle   fltrust  defl     dnc      multikrum
---------  -------  -------  -------  -------  -------  ---------
clean      +0.000   +0.000   +0.004   +0.004   -0.000   -0.001
lie        -0.001   -0.002*  +0.004   +0.001   +0.003   +0.000
fang       +0.018*  -0.002   +0.004   +0.022   +0.025*  +0.078*
mimic      +0.004   -0.002   +0.004   +0.029*  +0.021   +0.036
...
  (* = strongest attack in that defense's column)
```

- **`ACC_DROP` is the primary result.** It is well defined for every defense
  regardless of whether it flags or re-weights, and for every attack regardless of
  how it crafts. The `*` marks the strongest attack per defense column — the
  comparison the benchmark exists to make. Control rows (`clean`) are excluded
  from that mark.
- **`DETECT%` is the stealth axis.** A strong attack that every defense flags and
  a weak one that none does are different results, and the two grids together are
  the honest summary. Read the per-defense caveats in
  [Reading the detection numbers](#reading-the-detection-numbers-important-caveat)
  first — they apply per cell.
- **`ATK_SUCC`** weights each round by `min(1, acc_drop / target)`, so it is
  `ACC_DROP` rescaled by the goal. Useful for "did it meet the brief", not for
  ranking attacks that all overshoot or all fall short.
- `acc_drop` is measured against the **clean Phase-1 baseline**, a fixed number
  from before the run. That is what makes it comparable ACROSS rows. For "how much
  did this attack cost *this defense*", add the `clean` row and subtract.

### The single-round caveat (read this before comparing to published numbers)

Phase 2 runs **simulated rounds off a frozen anchor** (`fl.freeze_global_in_phase2`,
the shipped default): every round sends the clients the same Phase-1 model, and each
defense's global is that round's aggregate — it does not carry damage into the next
round. So this benchmark measures **per-round attack strength against a fixed
reference**, not end-of-training degradation.

That matters for how the rows compare. LIE, Min-Max, Min-Sum and IPM are designed to
be *individually negligible and cumulatively fatal* — they bound their perturbation
by the honest population's own spread precisely so it survives many rounds. Measured
one round at a time against a converged model, they will read as near-zero
`acc_drop`, while blatant attacks read higher. That is a real property of the
measurement, not evidence that the stealthy attacks are weak in a real deployment,
and it is why `DETECT%` is reported beside `ACC_DROP` rather than under it.

The regime is **identical for every row**, including `llm` — which is also the
regime the policy was trained and rewarded in — so the comparison between rows is
sound. It is the comparison to a *paper's* end-to-end numbers that does not transfer.

### Per-round variation (`--benign-retrain`)

By default the honest clients **replay** their frozen Phase-1 weights, so the honest
updates are byte-identical in every round. Within a round that is what guarantees
every attack and defense sees the same cohort; across rounds it means a
deterministic attack (`lie`, `min_max`, `min_sum`, `ipm`, `mimic`, `sign_flip`,
`scaling`, `fang_krum`) reproduces itself exactly and only the stochastic rows
(`llm`, `fang`, `noise`) vary with `--rounds`. Pass `--benign-retrain` to retrain
the honest clients each round instead; the within-round guarantee is unaffected,
since `begin_round()` draws the honest updates once and hands them to everyone.

## How the comparison is kept fair

- The env (`rl/env.py`) is reused **only as a generator**: each round it builds the
  controllable pool and the honest client updates.
- **The same clients are poisoned in every row.** The round's poisoned set is fixed
  ONCE: when `llm` is in the panel it is the policy's own committed selection and
  every baseline poisons exactly those clients; otherwise it is the first `budget`
  of the controllable pool (which is what the shipped `attack.fixed_poison_clients`
  regime produces anyway). An attack that returns any other client ids is a hard
  error, not a warning.
- **The same honest updates.** All attacks craft from one `begin_round()`, so the
  benign half of the cohort is byte-identical across rows.
- **The same rounds.** A round the trained attacker cannot produce a usable action
  for is skipped for the **whole panel**, not just for `llm` — otherwise the
  baselines would be averaged over more rounds than the policy. `run_info` records
  `measured_rounds` / `unusable_attack_rounds`.
- **Vary the defense, hold the attack fixed** (along a row): one attack produces one
  cohort per round and every defense sees it.
- **Each attack gets its own defense instances**, because a defense's global model
  and its cross-round memory (DeFL's Beta trust) are shaped by the attack it faced.
  One shared panel would let rows contaminate each other.
- Each attack observes **its own** undefended (`fedavg`) accuracy as the reference
  the next round's prompt sees — never another row's damage.
- Detection is scored against the shared ground-truth poisoned set with
  `metrics.compute.confusion_counts`, exactly as the live system scores it.

## Cost

A matrix run is `n_attacks x n_defenses` test-set passes per round. Two things keep
that manageable:

- **The accuracy cache** memoises test accuracy by the exact bytes of the global
  model. Under frozen benign replay a deterministic attack produces a byte-identical
  global every round, so those repeats collapse to one pass each. It cannot change a
  number — a hit means the model is bit-for-bit one already measured — and
  `--no-eval-cache` turns it off. The run logs its hit rate.
- **`llm_defender` costs one LLM generation per round *per attack*.** Drop it from
  `--defenses` for a fast algorithmic-only matrix; the run warns when it is in a
  multi-attack panel.

## FLTrust knobs

`--root-size` (clean root samples, default 100, the paper's default) · `--root-epochs`
(server local epochs `R_l`, default 1) · `--root-lr` (default `fl.lr`) · `--eta`
(global learning rate, default 1.0). The root set is carved once from the clean
5G-NIDD train split with a fixed seed. It is drawn **uniformly**, as in the paper —
not class-balanced — so on a dataset this imbalanced the root set is mostly benign
and UDPFlood traffic, which is a real property of FLTrust's threat model (the
server's clean data is just clean, not curated) and shapes the trust direction
`g0` it derives.

## DeFL knobs

`--defl-delta` (CLP relative-rise threshold δ, paper default **0.05** — larger ⇒
fewer early rounds declared critical ⇒ fewer hard removals) · `--defl-tau` (MOUD
per-layer outlier z-threshold, default **2.5** — lower ⇒ more aggressive flagging,
higher TPR but higher FPR). DeFL takes **no** root set: it derives everything from
the per-layer FGNV of the submitted updates. A "layer" = one module (its weight +
bias grouped), so NiddNet has L = 2 layers (`net.0`, `net.2`). The Beta trust then
keeps the run robust to the occasional false positive.

## DnC knobs

`--dnc-num-byzantine` (assumed #malicious `m`; default = the eval poison budget
`--max-poison-clients` clamped to a benign majority — a hyperparameter / assumed
adversary budget, **not** per-round ground truth) ·
`--dnc-c` (filtering fraction `c`, paper iid default **1** → drop `c·m` clients each
round) · `--dnc-niters` (subsampling iterations, default **1**) · `--dnc-sub-dim`
(subsample dimension `b`, default **10000**, the paper's value — clamped to the
model's dimension, so for the tiny NiddNet (d=681) all coordinates are used and no
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

Good next candidates: **coordinate-wise Median**, **Trimmed-Mean** (Yin et al.),
**Norm-clipping**, **FLDetector**. Median/Trimmed-Mean are robust aggregators —
derive a reject signal the same way FLTrust does (a client is "rejected" when it's
excluded / trimmed from the aggregate).

## Adding another attack (later)

1. Create `benchmark/attacks/<name>.py` with a `class <Name>(DeltaAttack)` that
   implements `craft_deltas(ctx) -> {client_id: flat malicious delta}` for exactly
   `ctx.poisoned_ids`. Work in flat delta space (`ctx.known_deltas()`,
   `ctx.deltas_for(...)`); the base class converts back to the absolute weights
   clients submit. Subclass `Attack` directly instead if the attack produces weights
   some other way (`label_flip` retrains, so it does).
2. Register it in `benchmark/attacks/__init__.py` (`build_attacks` + `AVAILABLE`,
   and `DEFAULT` if it belongs in the standard panel), give it a `citation`, and add
   a colour in `benchmark/plot.py`.
3. Read `ctx.known_ids`, never `ctx.honest` directly — that is what makes
   `--baseline-knowledge` mean something.
4. Never mutate anything reachable from `ctx`: the honest updates are shared by
   every attack and every defense in the round.

## Tests

- `tests/test_benchmark_attacks.py` — every attack against the formula its paper
  states (LIE's `z^max`, Min-Max/Min-Sum feasibility, Fang's out-of-range placement,
  Fang-Krum actually landing in Krum's selection, Mimic's chosen client), plus the
  flat-vector plumbing and the registry. Needs torch:
  `python tests/test_benchmark_attacks.py`.
- `tests/test_benchmark_matrix.py` — the matrix invariants: same poisoned clients in
  every row, same honest updates, per-attack defense instances, a skipped round
  dropped for the whole panel, and the accuracy cache being a pure memo. Needs torch:
  `python tests/test_benchmark_matrix.py`.
- `tests/test_benchmark.py` — torch-free (metrics + report). Runs anywhere:
  `python tests/test_benchmark.py`.
- `tests/test_benchmark_retry.py` — the harness's unusable-attacker-action path:
  resample, then skip the round instead of aborting the run (needs torch):
  `python tests/test_benchmark_retry.py`.
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
