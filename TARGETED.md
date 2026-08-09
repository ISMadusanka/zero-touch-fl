# Targeted Label Poisoning

A second, **separate** experiment alongside the untargeted one. Same arms race, same
GRPO machinery, different objective:

> **Untargeted** (`configs/base.yaml`): *"cut global accuracy by 20%."*
> **Targeted** (`configs/targeted.yaml`): *"make the model misclassify label L —
> and leave every other class working."*

The two never share a config, a LoRA adapter, a log directory or a report. They do
share one thing on purpose: the Phase-1 honest-training checkpoint, so both are
measured from the identical clean baseline.

**One insider, one label.** As shipped, the attacker controls exactly **one** client
(client 0, `fl.n_compromisable: 1`) and attacks exactly **one** class — the class
client 0's own non-IID shard is dominated by. That label is not written in the config:
the split is non-IID, so which classes client 0 owns depends on the partition RNG and is
only knowable at runtime. `data/target_label.py` measures it right after partitioning,
pins the run to it, and prints it (see [§3](#which-label-derived-from-client-0s-own-data)).

```bash
python train_targeted.py                                  # train
python -m benchmark.run_targeted_benchmark --rounds 100    # evaluate the same label
python -m benchmark.ui                                     # ...or evaluate from a live dashboard
```

---

## 1. What the attack actually does

### The 17 numbers that matter

The global model ends in `Linear(16, 10)` — `net.4`, weight shape `[10, 16]`:

```
logit[c] = net.4.weight[c, :] · hidden + net.4.bias[c]
```

Row `c` **is** class `c`, by construction. So the only parameters specific to class 2
are **row 2 of `net.4.weight` (16 numbers) + `net.4.bias[2]` (1 number)** — 17 of the
model's 970. There is no identification step, no saliency analysis, no probing data:
the architecture hands you the mapping. `agents.attack_ops.output_layer_keys` reads it
off the `state_dict` at runtime, so it stays correct if the model changes — including
for a CNN, where the conv stack below the head is shared across all classes and has no
per-class rows at all.

Everything *below* the head is shared. Editing it damages every class, which is exactly
what the reward punishes.

### Confining an operator to one row

Every operator in the attack DSL now accepts an optional `rows` list:

```json
{"op": "scale", "target": "net.4", "rows": [2], "factor": -6.0}
```

That scales row 2 of `net.4.weight` **and** entry 2 of `net.4.bias`, leaving the other
nine rows and the whole hidden layer byte-identical. Omit `rows` and the operator
behaves exactly as it always did. Rows that address nothing (row 99 of a 10-row tensor)
make the op **invalid** rather than silently falling back to the whole tensor — that
fallback would quietly turn a mis-specified targeted attack into an untargeted one.

### Dilution: why the factors are large

FedAvg averages **all 20 clients**, so poisoning `k` of them applies the edit at
strength `k/20`. Scaling the target row by `f` on `k` clients leaves the aggregated row
at `((n-k) + k·f)/n` of its honest value, so

```
f = 1 − n/k        drives the aggregated row to exactly zero
```

| poisoned clients `k` | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `f` that zeroes the aggregate row | −19 | −9 | −5.67 | −4 | −3 |

This table is computed per round and handed to the attacker in its prompt
(`federation.row_zero_factor`). Without it the policy has no way to calibrate magnitude
and reliably under-shoots: a "big-looking" factor of −1 moves a 20-client average by 10%
and does nothing.

Going *past* `f` drives the logit negative, which is what actually stops the class being
predicted.

### Why this is stealthy — and where it isn't

Concentrating the whole perturbation into 17 of 970 weights keeps the **whole-model**
update norm small, which is what most robust-aggregation defenses key on. That is the
attack's advantage.

But the defender in this system also scores **each layer separately**
(`detector/features.py`: per-layer `l2_norm`, `rel_norm`, `cos_to_median`,
`sign_agreement`). `net.4` is only 170 parameters, so a huge edit to 17 of them is
loud *in that layer* even when the whole model looks clean. The attacker's real problem
is therefore: **how hard can I hit one row before the per-layer view gives me away?**
Spreading the edit over several clients (smaller `f` each) is the obvious lever, and the
reward's stealth and client-cost terms price that trade-off.

---

## 2. How the reward works

`rl/rewards.py`. Everything is computed from the round's **clean counterfactual** — the
per-class recall the aggregate reaches with *no poison at all*
(`FLArmsRaceEnv.clean_reference_eval`) — so an attack that hits its goal scores the same
every round it hits it, regardless of what happened before.

```
reward = alpha · drop_term(target_class_drop, effective_target)   # destroy class L
       − eta   · collateral_cost                                   # ...and ONLY class L
       + beta  · stealth                                           # evade the defender
       − gamma · malformed_fraction                                # don't waste clients
       − delta · client_cost                                       # use as few as possible
       + zeta  · collab_bonus                                      # coordinate if using several
```

Only the first two terms differ from the untargeted reward.

### `target_class_drop` — the damage term

`clean_recall[L] − post_recall[L]`, normalized by `effective_target` and shaped by the
existing `drop_term` (linear to the goal, then saturating — so overshoot breaks ties but
never turns the objective into "destroy the model").

`effective_target = min(target_class_drop, clean_recall[L])`, floored at 0.05.

**Why the clamp matters.** MNIST per-class recall for this tiny MLP is uneven — some
classes sit near 0.95, others near 0.60. A fixed 0.80 target would be *unreachable* for
a class that only had 0.62 to lose, so that label would be permanently unwinnable and
rewards would be incomparable across labels within one run. With the clamp, "goal met"
means the same thing — *this class is destroyed* — for every label.

### `collateral_cost` — what makes it targeted

```
collateral      = mean over c ≠ L of max(0, clean_recall[c] − post_recall[c])
collateral_cost = clip(collateral / max_collateral, 0, 3.0)
```

Only *losses* count: a class that happens to improve cannot offset one that was
destroyed. The mean (not the max) keeps the signal smooth, which GRPO needs.

**This term is the whole experiment.** Without it, the cheapest way to crush class 2's
recall is to crush every class — that scores identically on the damage term — so the
policy would simply rediscover the untargeted attack. With `eta = 1.0` and
`max_collateral = 0.05`:

| rollout | damage | collateral | stealth | ≈ reward |
|---|---|---|---|---|
| surgical (class L destroyed, rest intact) | ~1.1 | ~0.0 | high | **≈ +1.3** |
| indiscriminate (whole model wrecked) | ~1.1 | capped −3.0 | low | **≈ −1.9** |
| no-op / unparseable | 0 | 0 | 0 | **≈ −1.0** |

The cap at 3.0 is deliberate: an uncapped negative from one catastrophic rollout would
swamp the GRPO group's reward spread and kill the gradient for every *other* rollout in
the group.

`tests/test_targeted.py::test_surgical_attack_beats_indiscriminate_destruction` pins
this ordering.

### The win-gate

`rl/switch.py`. A targeted round counts as an attacker win only when **all three** hold:

1. every poisoned client evaded detection (`attacker_min_evaded`),
2. `target_class_drop ≥ win_fraction · effective_target` (default 0.6), **and**
3. `collateral ≤ max_collateral`.

Condition 3 is not optional. Without it a model-wrecking round would count as a
targeted "win", the schedule would freeze the attacker at that checkpoint, and the
arms-race ratchet would lock in exactly the behaviour we are training out of it.

---

## 3. How it is trained

Identical GRPO loop to the untargeted run — `G` rollouts per state, group-normalized
advantage, freeze-and-alternate schedule, opponent league. **Nothing about GRPO
changed**; only the scalar it optimizes.

### Which label: derived from client 0's own data

The poisoner is client 0 and the target class is *its* class. With a non-IID split that
class is decided at runtime, not in the config:

```yaml
fl:
  n_compromisable: 1              # the attacker controls client 0 and nothing else
attack:
  target_label_from_client: 0     # read the label off THAT client's shard
  max_poison_clients: 1           # one poisoned client per round
  sample_budget_in_training: false
```

Before Phase 2 starts, `data/target_label.py::resolve_client_target_label` reads client
0's label histogram (`data/mnist_loader.py::client_label_counts` — indexes the dataset's
`targets` through the shard's index list, so it costs no data loading), takes its
most-represented class, and pins three fields:

| field | set to | why |
|---|---|---|
| `attack.goal.label` | the derived class | what the reward, win-gate and prompt read |
| `attack.target_labels` | `[that class]` | nothing else is trained |
| `attack.sample_target_in_training` | `false` | no per-round redraw |

All three matter. Leaving `sample_target_in_training: true` would make
`rl/env.py::_round_goal` redraw a *different* label every round — correct when training a
label-agnostic policy, wrong here, where the attack must stay on the one class the
insider actually holds data for.

With the shipped settings (`n_clients: 20`, `noniid_bias: 0.5`, `poison_seed: 0`) the
FLTrust round-robin puts client 0 in group 0, so the derived label is **0**, at ~49% of
its 3045 samples. Change the seed, the bias or the client count and it can change — which
is the whole reason it is measured instead of hardcoded.

The derivation is logged at startup, before the first round:

```
============================================================
TARGET LABEL DERIVED AT RUNTIME from client 0's non-IID shard
  client 0 holds 3045 samples across 10 class(es)
  label histogram (label:count)  0:1479  1:204  2:174  3:204  4:155  5:144  6:167  7:187  8:162  9:169
  runner-up: label 1 (204 samples)
  >>> TARGET LABEL = 0   (1479 samples = 48.6% of client 0's data) <<<
  pinned: attack.goal.label=0, target_labels=[0], sample_target_in_training=False (same label every round)
============================================================
```

and every round's log line then carries it as `TGT[0]`, so a run is never ambiguous about
what it attacked. `--debug` also records the full derivation under
`config_summary.target_label_derivation` in `logs/targeted/debug.json`.

To go back to the label-agnostic policy (train over several classes so evaluation can ask
for any of them), set `target_label_from_client: null` and restore
`sample_target_in_training: true` with the `target_labels` list. Labels **6–9 are then
deliberately excluded from training** so evaluating on them measures whether the policy
learned the procedure or memorized special cases.

### One poisoned client

`n_compromisable: 1` makes `rl/env.py::begin_round` expose a controllable pool of exactly
`[0]`, and `max_poison_clients: 1` caps the round budget at one client. The attacker's
prompt therefore offers `controllable_client_ids: [0]`, `max_poison_clients: 1`, and a
dilution table with a single entry — `row_zero_factor["1"] = -19`. A selection naming any
other client is dropped by `AttackerAgent.select_and_apply` (it filters to the pool), so
client 0 is the only client that can ever be poisoned.

Two reward terms go inert at one client and are left in place for the multi-client
configuration: `delta` (penalty for using more clients than needed) and `zeta` (bonus for
coordinated multi-client roles) both require `n_used > 1`.

### Round flow

```
env.begin_round()                     # draws this round's label + poison budget,
                                      #   measures the clean per-class counterfactual
attacker prompt  ──► G rollouts       # each: client selection + per-client row-targeted plan
  each rollout ──► apply_plan ──► FedAvg ──► evaluate_per_class ──► reward
GRPO step on the group-normalized advantages
commit the best-scoring rollout ──► win-gate ──► maybe freeze & hand off to the defender
```

Per-class evaluation costs **nothing extra**: `FedServer.evaluate_per_class` produces the
overall accuracy and the ten recalls from the same single pass over the test set.

### Watching a run

```
Round 312 [learn=attacker ph=4.7 WIN]: acc 0.7810->0.7003 (clean_ref=0.7814 drop=+0.0811)
  | TGT[3] 0.812->0.041 (drop=+0.771 bar=0.487 of tgt=0.812) collat=0.004/0.050 goal=MET
  | att_reward=1.284 def_reward=0.220 | grpo_loss=-0.0113 mean_r=0.842 zero_adv=0.00 step
```

`TGT[3] 0.812->0.041` is the headline: class 3's recall collapsed. `collat=0.004/0.050`
says the other nine classes lost 0.4 points of mean recall against a 5-point tolerance —
this was a genuinely targeted hit. Note overall accuracy only fell 8 points; on 10
classes, destroying one **cannot** cost more than ~10. Judging a targeted run by overall
accuracy will always make a total success look like a near-miss.

**`bar` is the number that matters, not `tgt`.** The win-gate compares the drop against
`win_fraction × effective_target` (0.6 × 0.812 = 0.487 here), not against the effective
target itself. A round reading `drop=+0.400 bar=0.300 of tgt=0.500` has *cleared* the
gate. `goal=MET|no` states the verdict outright so there is nothing to infer.

Every round also lands in `logs/targeted/round_data/rounds.jsonl` under
`attack_metadata.targeted` with the full `per_class_clean` / `per_class_post` vectors.

---

## 4. Running it

### Train

```bash
python train_targeted.py
```

Needs the GPU box. Pins `--config configs/targeted.yaml --run-name targeted`; every
other `main.py` flag still works.

```bash
python train_targeted.py --rounds 5000      # absolute round budget
python train_targeted.py --debug            # 3 fully-logged rounds, prompts + outputs
python train_targeted.py --dry-run --env linux   # CPU smoke test via Ollama, no training
```

Resume is automatic — rerun the same command and it continues from
`checkpoints/targeted/rl_progress.json`.

### Where things land

| path | what |
|---|---|
| `checkpoints/global_model.pt`, `client_updates.pt`, `baseline.json` | Phase-1 baseline — **shared** with the untargeted run |
| `checkpoints/targeted/attacker_adapter/`, `defender_adapter/` | targeted LoRA adapters |
| `checkpoints/targeted/fl_state.pt`, `rl_progress.json` | Phase-2 resume state |
| `logs/targeted/round_data/rounds.jsonl` | per-round record incl. per-class recalls |
| `logs/targeted/metrics/`, `logs/targeted/debug.json` | metrics + `--debug` dump |

Nothing here collides with the untargeted run's `checkpoints/attacker_adapter/` or
`logs/`.

### Evaluate

```bash
python -m benchmark.run_targeted_benchmark --rounds 100
```

With no flags this evaluates **exactly what training attacked**: it re-derives the label
from `attack.target_label_from_client`'s shard (same seed → same partition → same class)
and uses `attack.eval_poison_clients` (1) as the budget. The derivation block is printed
here too, so a report can always be traced back to its label.

The two knobs the experiment is parameterised on:

| flag | meaning |
|---|---|
| `--label L` | which class the attack must make the model misclassify (0–9). **Overrides** the runtime derivation — use it to probe generalization to a class the policy was not trained on |
| `--poison-clients k` | how many clients the attacker may poison per round (it chooses *which*, from the `n_compromisable` pool — so `k > 1` needs a wider pool in the config) |

Both are fixed for the whole run — no per-round sampling at evaluation. Other useful
flags: `--defenses`, `--rounds`, `--attack-temperature`, `--target-class-drop`,
`--max-collateral`, `--out`, `--attacker-adapter`.

`--poison-client-ids` names **which** clients the attacker controls instead of taking
the config's insider prefix `[0 .. fl.n_compromisable)`:

```bash
python -m benchmark.run_targeted_benchmark --poison-client-ids 0,3,7 --rounds 100
```

Naming ids also sets the round budget to how many were named, unless `--poison-clients`
narrows it. This is `fl.n_compromisable`'s *identity* knob, not its size knob: the
trained policy is the insider on the config's client, so compromising a different one
measures generalization the same way `--label` does. It is logged as a warning when the
named set differs from the config's.

### Watch a run in a browser

```bash
python -m benchmark.ui                    # opens http://127.0.0.1:8420
```

A local dashboard for the same command. Pick the rounds, the compromised clients (from
the client list — client 0 by default, the config's insider) and the target label
(0 by default, the class training derived from client 0's shard), choose the defense
panel, and press Run. While it runs you get, per round: the target class's recall for
every defense against the clean reference, the other classes' mean recall, the current
per-class bar chart, a per-client caught/missed/false-alarm map, and a feed of what the
attacker did and which defenses saw it. When it finishes, the same summary and
per-class tables the CLI prints. Every chart has a table view, and the exact argv is
shown so a run can be reproduced in a terminal.

The UI **runs the real command** — it spawns
`python -m benchmark.run_targeted_benchmark … --events -` and reads its structured
event stream (`benchmark/events.py`), so it cannot drift from what the CLI does, and
`--events` off by default leaves plain CLI output unchanged.

| flag | |
|---|---|
| `--port` / `--host` | default `8420` on loopback |
| `--config` | the config runs use, and where the form's defaults come from |
| `--python` | interpreter to run the benchmark with (default: the one serving the UI) |
| `--demo` | replay a synthetic run instead of spawning the benchmark — for checking the dashboard on a machine with no GPU, adapter or torch. The numbers are invented and the page says so |
| `--no-browser` | don't open a browser tab |

Only one run at a time (it owns the GPU); **Stop** terminates it. On a remote GPU box,
forward the port — `ssh -L 8420:127.0.0.1:8420 <box>` — rather than binding `--host
0.0.0.0`: the page can start processes.

Check whether the single-insider policy transfers to other classes:

```bash
for L in 0 1 2 3 4 5 6 7 8 9; do
  python -m benchmark.run_targeted_benchmark --label $L \
      --rounds 50 --out logs/targeted/benchmark_l$L
done
```

Every label except client 0's own is held out by construction here: a run trained with
`target_label_from_client` sees one class only, so all nine others measure generalization.

---

## 5. Reading the benchmark output

The report leads with the target class against the others, per defense (example below
from a wider-pool run: `--label 2 --poison-clients 3`):

```
====================================================================
TARGETED POISONING BENCHMARK — 100 attack rounds
(goal: misclassify label 2; clean overall acc = 0.782; poisoned clients per round = 3)
====================================================================
defense       detect%  FPR    F1    TGT_final  TGT_mean  others_mean  collat  overall  atk_thru  tgt_succ
------------  -------  -----  ----  ---------  --------  -----------  ------  -------  --------  --------
fedavg          0.0%   0.0%   0.00      0.017     0.031        0.771   0.004    0.699    100.0%     96.0%
oracle        100.0%   0.0%   1.00      0.903     0.901        0.775   0.000    0.782      0.0%      0.0%
llm_defender   72.0%   3.1%   0.79      0.244     0.288        0.769   0.006    0.731     28.0%     26.0%
fltrust        41.0%   8.0%   0.52      0.101     0.140        0.758   0.011    0.712     59.0%     55.0%

TARGETED ATTACK WORKED (undefended / fedavg): class 2 recall 0.903 -> 0.017 (lost +0.886);
other classes 0.775 -> 0.771 (lost +0.004)

PER-CLASS RECALL — FINAL model  (* = attack target, class 2)
defense        0      1     *2      3      4      5      6      7      8      9    TARGET  others
------------  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ------  ------
clean         0.918  0.961  0.903  0.788  0.812  0.601  0.874  0.845  0.669  0.703   0.903   0.775
fedavg        0.921  0.960  0.017  0.784  0.809  0.598  0.871  0.844  0.665  0.700   0.017   0.771
oracle        0.918  0.961  0.903  0.788  0.812  0.601  0.874  0.845  0.669  0.703   0.903   0.775
llm_defender  0.915  0.958  0.244  0.781  0.807  0.596  0.869  0.841  0.663  0.698   0.244   0.769
```

*(illustrative shape, not measured results)*

How to read it:

- **`clean` row** — the unpoisoned reference. Every other row is judged against it.
- **`fedavg` row** — the no-defense world, so it isolates what the *attack* did.
  Column `*2` collapsed to 0.017 while every other column moved by ≤0.004. That is the
  result: **only the targeted class broke.**
- **`oracle`** — perfect detection, so it should match `clean` exactly. If it doesn't,
  something is wrong with the harness, not the defense.
- **`TGT_final` vs `others_mean`** — the same story as two numbers. A defense is working
  when `TGT_final` stays near the `clean` row.
- **`collat`** — mean recall the other classes lost per round. ~0 = perfectly surgical.
- **`overall`** — note it only fell from 0.782 to 0.699. This is why the untargeted
  report is the wrong instrument here.
- **`tgt_succ`** — fraction of rounds the targeted goal was met, scored with the *same*
  bar training used (enough of the target destroyed **and** collateral in tolerance).

Also written to `--out`:

| file | contents |
|---|---|
| `targeted_benchmark.json` | full summaries incl. `per_class_clean/final/mean` |
| `targeted_benchmark.csv` | same, per-class fields flattened into columns |
| `history.json` | per-round record per defense |
| `targeted.png` | 4 panels: target-class recall over rounds, other-classes recall, final per-class bar chart, rolling detection rate |

Re-plot without re-running the benchmark:

```bash
python -m benchmark.targeted_plot --history logs/targeted/benchmark/history.json
```

---

## 6. Configuration reference

`configs/targeted.yaml`, the fields that differ from `base.yaml`:

```yaml
fl:
  n_compromisable: 1          # SINGLE INSIDER: the pool is clients [0 .. n-1] = [0]

attack:
  goal:
    type: "targeted_label"
    label: 2                  # FALLBACK ONLY — replaced at runtime by the derived label
    target_class_drop: 0.50   # how much of the class's recall to destroy
                              #   (clamped per round to that class's clean recall)
    max_collateral: 0.05      # mean recall the OTHER classes may lose
  target_label_from_client: 0 # derive the label from client 0's own shard (null = off)
  sample_target_in_training: false     # no per-round redraw (forced false when derived)
  target_labels: [0,1,2,3,4,5]         # only used when the derivation is off
  max_poison_clients: 1                # training budget: one poisoned client per round
  sample_budget_in_training: false     # nothing to randomize at a cap of 1
  eval_poison_clients: 1               # benchmark default (--poison-clients)

data:
  n_classes: 10               # width of the per-class breakdown; also the partition's
                              #   group count, so it decides which class client 0 owns

rl:
  adapter_paths:              # SEPARATE from the untargeted run
    attacker: "checkpoints/targeted/attacker_adapter"
    defender: "checkpoints/targeted/defender_adapter"
  reward:
    attacker:
      eta: 1.0                # collateral-damage penalty — set to 0 and the policy
                              #   just relearns the untargeted attack
```

### Tuning notes

- **Attack too destructive?** Raise `eta` or lower `max_collateral`.
- **Attack never lands?** The likely cause is under-shooting the dilution. Check the
  `TGT[...]` line: if `post_recall ≈ clean_recall` the factors are too timid. At one
  poisoned client of 20 the factor that zeroes the aggregated row is **−19**, and the
  policy has to reach past it; a timid −3 moves the average by 20% and does nothing.
  Raising `n_compromisable` / `max_poison_clients` / `eval_poison_clients` (back above 1)
  gives the attacker more aggregate share and smaller per-client factors.
- **Attacking the wrong class?** The label comes from client 0's data, not the config —
  read the `TARGET LABEL DERIVED AT RUNTIME` block at the top of the log. It moves with
  `fl.poison_seed`, `data.noniid_bias` and `fl.n_clients`, since those decide the split.
- **`zero_adv=1.00` on many rounds?** The group has no reward spread. `resample_temperature`
  and `scoring_opponent_temperature` are the existing levers.
- **Want source→target (`2 → 7`) instead of `2 → anything`?** Not implemented. It needs a
  second term rewarding the fraction of class-2 samples predicted *as 7*, and an attack
  that boosts row 7 as well as suppressing row 2.

### CUDA OOM in `reference_token_logprobs` / `policy_token_logprobs`

Fixed — but worth knowing what it was, because it bounds how long prompts can get.

`LLMPolicy._completion_token_logprobs` used to run `log_softmax(logits.float())` over the
**whole** sequence and then keep only the completion's rows. At Qwen2.5's ~152k vocab
that intermediate is `seq_len × 152k × 4` bytes — **~2.8 GB on a 4.8k-token prompt** —
allocated, indexed once, and discarded. It now slices to the completion's positions
*before* the fp32 `log_softmax`, which is an exact identity (`log_softmax` normalizes over
the vocab independently at each position), and asks the model to run its LM head on only
the tail of the sequence when the installed transformers supports it. That frees
~2.4–4.3 GB. `tests/test_logprob_memory.py` pins the new path bit-for-bit against the old
computation.

The targeted prompt is ~520 tokens larger than the untargeted one (row/class explanation,
dilution table, `output_layer` + `federation` fields), which is what pushed an
already-wasteful code path over the edge. If you still hit OOM, in order of preference:

1. `rl.max_new_tokens` — the completion length now sets the fp32 tensor's size. 1024 is
   generous for the attacker's JSON; 512 halves it.
2. `rl.load_in_4bit: true` — ~4 GB off the base model.
3. `attacker_agent.detail_precision` — fewer decimals in `client_update_stats`, which is
   the bulk of the prompt (~3.8k of the ~4.8k tokens) and predates the targeted work.

A crash mid-run loses at most `rl.save_every` (25) rounds — rerun `python train_targeted.py`
and it resumes from `checkpoints/targeted/rl_progress.json`.

### CUDA OOM with a small allocation ("tried to allocate 332 MiB")

When the *failed allocation is small* but the GPU is full, the problem is almost never
this process. Read PyTorch's message carefully — it reports two different things in
near-identical wording:

```
Process 851739 has 23.78 GiB memory in use.            <- ANOTHER process
Including non-PyTorch memory, this process has 7.33 GiB memory in use.   <- ours
```

Training itself needs ~7 GiB for Qwen2.5-3B in bf16. If the totals add up to the card's
capacity, something else is holding the rest — usually a previous run that crashed
without releasing its CUDA context:

```bash
nvidia-smi
```

Kill the stale PID and restart. Warnings now print a `[GPU … | other processes ~N GiB]`
line so this is visible without decoding the OOM text.

Related fix: an OOM during generation used to trip the *sticky* fallback from KV-cached
generation to the manual no-cache decoder. That made things strictly worse — the manual
decoder re-runs the full forward for every generated token, so it needs more memory per
step and failed immediately afterwards, on a slower path, for the rest of the run. OOM is
now treated as transient (free the cache, retry the same path once, surface it if it
recurs); the sticky fallback is reserved for genuine kernel-incompatibility errors, which
is what it was written for.

---

## 7. What changed in the codebase

New:

| file | purpose |
|---|---|
| `configs/targeted.yaml` | the targeted experiment's config |
| `train_targeted.py` | training entry point |
| `benchmark/run_targeted_benchmark.py` | evaluation entry point |
| `benchmark/targeted_report.py` | per-class report + verdict line |
| `benchmark/targeted_plot.py` | 4-panel targeted figure |
| `benchmark/events.py` | opt-in per-round JSONL event stream (`--events`) |
| `benchmark/ui/` | the live dashboard: `server.py` (stdlib HTTP + SSE), `index.html`, `demo.py` |
| `tests/test_targeted.py` | 24 tests, CPU-only, no LLM |

Modified — all additive; passing none of the new arguments reproduces the old behaviour
exactly (`tests/test_targeted.py::test_untargeted_reward_is_unchanged_by_the_new_arguments`):

| file | change |
|---|---|
| `core/types.py` | `ClassEval` (overall + per-class recall + support) |
| `server/fed_server.py` | `evaluate_per_class`; `evaluate` unchanged, now sharing one pass |
| `rl/env.py` | per-class clean counterfactual, `evaluate_updates_full`, `commit_full`, per-round label sampling, `pool_override` (name the compromised clients instead of taking the prefix) |
| `rl/rewards.py` | `targeted_terms`, `goal_drop`, `goal_label`; `eta`/`clean_eval`/`post_eval` on `attacker_reward` |
| `rl/switch.py` | collateral condition in the win-gate |
| `rl/turns.py`, `rl/schedule.py`, `rl/inference.py`, `rl/baseline.py` | thread the per-class evals through; targeted logging |
| `agents/attack_ops.py` | `rows` on every operator; `output_layer_keys` |
| `agents/attacker_agent.py` | `TARGETED_SYSTEM_PROMPT`; `output_layer` + `federation` observation |
| `storage/checkpoint.py` | `set_rl_dir` so RL artifacts stay per-experiment |
| `main.py` | `--run-name` for log/checkpoint isolation |
| `benchmark/metrics.py`, `benchmark/harness.py` | per-class accumulation (inert when the goal is untargeted); optional `on_start`/`on_round` observers for a live watcher |

One visible side effect on the untargeted benchmark: the per-defense per-round
`"Global model test accuracy: …"` line is now logged at DEBUG instead of INFO, because
the benchmark switched to `evaluate_per_class`. Results are unaffected.

---

## 8. Tests

```bash
python tests/test_targeted.py
```

24 tests, CPU only, no download, no LLM. They cover: per-class eval consistency with
plain accuracy; `rows` confining an operator to one class (and *not* silently widening
when it addresses nothing); `output_layer_keys` on the real model; an end-to-end check
that suppressing a row stops that class being predicted with other logits bit-identical;
the effective-target clamp; collateral counting losses only; **surgical scoring far above
indiscriminate**; the win-gate's collateral and evasion conditions; per-round label
sampling; and a regression guard that the untargeted reward is unchanged.
