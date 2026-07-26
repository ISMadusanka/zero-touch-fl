# Targeted Label Poisoning

A second, **separate** experiment alongside the untargeted one. Same arms race, same
GRPO machinery, different objective:

> **Untargeted** (`configs/base.yaml`): *"cut global accuracy by 20%."*
> **Targeted** (`configs/targeted.yaml`): *"make the model misclassify label 2 —
> and leave every other class working."*

The two never share a config, a LoRA adapter, a log directory or a report. They do
share one thing on purpose: the Phase-1 honest-training checkpoint, so both are
measured from the identical clean baseline.

```bash
python train_targeted.py                                          # train
python -m benchmark.run_targeted_benchmark --label 2 --poison-clients 3   # evaluate
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

### Training across labels 0–5

```yaml
attack:
  sample_target_in_training: true
  target_labels: [0, 1, 2, 3, 4, 5]
```

Each round `rl/env.py::_round_goal` redraws `goal.label` from `target_labels` and leaves
the rest of the goal alone. The label the attacker must hit therefore **changes every
round**, and it travels in the prompt (`attack_goal.label`, plus
`output_layer.row_for_target_label`).

That is the entire mechanism behind generalization: because the label is never constant,
memorizing "row 2" scores badly. The only strategy that works across rounds is *read the
label off the goal and attack that class's row*. At evaluation you pin one label and the
policy applies the learned procedure to it.

Labels **6–9 are deliberately excluded from training** — evaluating on them measures
whether the policy learned the procedure or just memorized six special cases.

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
  | TGT[3] 0.812->0.041 (drop=+0.771/0.812) collat=0.004/0.050
  | att_reward=1.284 def_reward=0.220 | grpo_loss=-0.0113 mean_r=0.842 zero_adv=0.00 step
```

`TGT[3] 0.812->0.041` is the headline: class 3's recall collapsed. `collat=0.004/0.050`
says the other nine classes lost 0.4 points of mean recall against a 5-point tolerance —
this was a genuinely targeted hit. Note overall accuracy only fell 8 points; on 10
classes, destroying one **cannot** cost more than ~10. Judging a targeted run by overall
accuracy will always make a total success look like a near-miss.

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
python -m benchmark.run_targeted_benchmark --label 2 --poison-clients 3 --rounds 100
```

The two knobs the experiment is parameterised on:

| flag | meaning |
|---|---|
| `--label L` | which class the attack must make the model misclassify (0–9) |
| `--poison-clients k` | how many clients the attacker may poison per round (it chooses *which*) |

Both are fixed for the whole run — no per-round sampling at evaluation. Other useful
flags: `--defenses`, `--rounds`, `--attack-temperature`, `--target-class-drop`,
`--max-collateral`, `--out`, `--attacker-adapter`.

Sweep a few settings:

```bash
for L in 0 1 2 3 4 5 6 7 8 9; do
  python -m benchmark.run_targeted_benchmark --label $L --poison-clients 3 \
      --rounds 50 --out logs/targeted/benchmark_l$L
done
```

Labels 6–9 are the held-out generalization check.

---

## 5. Reading the benchmark output

The report leads with the target class against the others, per defense:

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
attack:
  goal:
    type: "targeted_label"
    label: 2                  # EVAL default; training overrides per round
    target_class_drop: 0.80   # how much of the class's recall to destroy
                              #   (clamped per round to that class's clean recall)
    max_collateral: 0.05      # mean recall the OTHER classes may lose
  sample_target_in_training: true      # redraw `label` every round
  target_labels: [0,1,2,3,4,5]         # the classes trained on
  eval_poison_clients: 1               # benchmark default (--poison-clients)

data:
  n_classes: 10               # width of the per-class breakdown

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
  `TGT[...]` line: if `post_recall ≈ clean_recall` the factors are too timid. Raising
  `eval_poison_clients` / `max_poison_clients` gives the attacker more aggregate share.
- **`zero_adv=1.00` on many rounds?** The group has no reward spread. `resample_temperature`
  and `scoring_opponent_temperature` are the existing levers.
- **Want source→target (`2 → 7`) instead of `2 → anything`?** Not implemented. It needs a
  second term rewarding the fraction of class-2 samples predicted *as 7*, and an attack
  that boosts row 7 as well as suppressing row 2.

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
| `tests/test_targeted.py` | 24 tests, CPU-only, no LLM |

Modified — all additive; passing none of the new arguments reproduces the old behaviour
exactly (`tests/test_targeted.py::test_untargeted_reward_is_unchanged_by_the_new_arguments`):

| file | change |
|---|---|
| `core/types.py` | `ClassEval` (overall + per-class recall + support) |
| `server/fed_server.py` | `evaluate_per_class`; `evaluate` unchanged, now sharing one pass |
| `rl/env.py` | per-class clean counterfactual, `evaluate_updates_full`, `commit_full`, per-round label sampling |
| `rl/rewards.py` | `targeted_terms`, `goal_drop`, `goal_label`; `eta`/`clean_eval`/`post_eval` on `attacker_reward` |
| `rl/switch.py` | collateral condition in the win-gate |
| `rl/turns.py`, `rl/schedule.py`, `rl/inference.py`, `rl/baseline.py` | thread the per-class evals through; targeted logging |
| `agents/attack_ops.py` | `rows` on every operator; `output_layer_keys` |
| `agents/attacker_agent.py` | `TARGETED_SYSTEM_PROMPT`; `output_layer` + `federation` observation |
| `storage/checkpoint.py` | `set_rl_dir` so RL artifacts stay per-experiment |
| `main.py` | `--run-name` for log/checkpoint isolation |
| `benchmark/metrics.py`, `benchmark/harness.py` | per-class accumulation (inert when the goal is untargeted) |

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
