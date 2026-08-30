# System Architecture — Label-Flip Poisoning vs a GRPO-Trained Defender LLM

This document describes the round loop, the attack, the defender contract, the
verifiable reward, the GRPO training schedule, and the checkpoint layout.

**The attack is not a policy.** It is a deterministic, detection-adaptive
label-flipping schedule that the environment runs directly
(`agents/label_flip_attacker.py`). The defender LLM is the only trainable agent.
An earlier design had an attacker LLM emitting weight-space operator plans; that
is gone — see [Why the poison is in the data](#why-the-poison-is-in-the-data).

## Components

| Layer | Module | Role |
|-------|--------|------|
| Model | `model/mnist_net.py` | `MnistNet`, ~970 params. State_dict keys: `net.2.weight [16,49]`, `net.2.bias [16]`, `net.4.weight [10,16]`, `net.4.bias [10]`. The schema the defender operates over. |
| Data | `data/mnist_loader.py` | MNIST load + per-client partition: IID, or the FLTrust non-IID bias-`q` scheme (`partition_noniid_fltrust`). |
| Data | `data/round_sampler.py` | A fresh slice of each client's own shard per round. |
| Attack | `data/label_flip.py` | The symmetric flip `y → 9−y`, the flipped-dataset wrapper, and the seeded choice of which examples to flip. |
| Attack | `agents/label_flip_attacker.py` | The detection-adaptive ladder: which clients flip, how much, and how it reacts to the verdicts. |
| Clients | `clients/benign_client.py` | Honest local SGD → `ModelUpdate`. |
| Clients | `clients/malicious_client.py` | The same local SGD, on flipped labels. |
| Server | `server/fed_server.py`, `server/aggregation.py` | Global model + eval; FedAvg over non-flagged clients. |
| Features | `detector/features.py` | Per-client, per-layer statistical feature vectors (no decisions). |
| Defender (default) | `agents/defender_agent.py` | Prompt + parse of per-client benign/malicious labels. The trainable policy. |
| Defender (alt) | `server/algo_defender.py` | Published algorithms (FLTrust / DeFL / DnC / Multi-Krum from `benchmark/defenses/`); one per round, producing both the verdicts and the aggregate. Active under `defense.mode: algorithmic`. |
| RL | `rl/*` | Environment, reward, turn, GRPO, schedule, policy, baseline, inference. |
| Metrics | `metrics/*` | Ground-truth confusion/TPR/FPR/ASR/APR (research evaluation). |

## Round loop (Phase 2)

```
reset env from Phase-1 checkpoint (per-client benign weights, global, baseline acc)
for each round:
  1. every client gets a fresh slice of its OWN shard        # data/round_sampler.py
  2. every client trains HONESTLY from the current global    # -> honest_updates
  3. THE ATTACK: the ladder's current fraction f is applied to each poisoned
     client's own per-round sample count -> n_flip labels. Each poisoned client
     RE-TRAINS the same data with n_flip labels flipped (y -> 9-y).
     Ground truth = the clients that ended up with >= 1 flipped label.
  4. build the full update list (poisoned clients swapped into the honest cohort)
  5. CLEAN COUNTERFACTUAL: aggregate the all-honest cohort and evaluate it
  6. DEFEND:
     llm (default) — detector/features -> per-client stat vectors -> DEFENDER LLM
       emits a benign/malicious label + confidence per client (input: features
       ONLY), then FedAvg over the clients labelled benign
     algorithmic — env.defend(): the round's algorithm emits per-client verdicts
       AND its own aggregate (FLTrust trust-weighting, DeFL Beta+CLP,
       DnC/Multi-Krum filtered mean) -> new global (None -> keep prev)
  7. evaluate the defended aggregate
  8. defender_reward (ground truth = the poisoned set)       <- train-time only
  9. GRPO update for the defender; write round log + metrics
 10. FEED THE LADDER the committed round's verdicts: caught -> step down,
     missed -> hold, caught at the floor -> reset to the top
```

Steps 2 and 3 produce **two updates for each poisoned client** — one honest, one
poisoned — from the same starting model and the same examples. That is what makes
the counterfactual in step 5 exact: the two branches differ by nothing but the
labels.

**Frozen anchor (`fl.freeze_global_in_phase2`, default true).** Phase 2 does not
continue the federation. Each round is an independent episode branching off the
same Phase-1 final model: the aggregate in step 7 is measured and then *discarded*,
so round *n+1* sends the clients exactly the model round *n* did. Without it the
attack's own damage becomes the next round's starting point, and a round's numbers
depend on the wreckage left by every round before it. Set it to `false` to restore
a continuing federation.

**Local training is mandatory.** `fl.benign_retrain_each_round` is forced on: the
poison IS the poisoned clients' local training on mislabelled data, so there is no
frozen state_dict to replay and no weight edit to apply. Replaying frozen honest
weights while the poisoned clients trained fresh would also be a giveaway for
entirely the wrong reason (staleness, not the labels).

## The attack (`agents/label_flip_attacker.py`, `data/label_flip.py`)

### Why the poison is in the data

A weight-space edit (scale ×2, flip the signs, add noise) is a post-hoc transform
of a vector the client already computed. Its magnitude is a free parameter, so an
attacker can dial it to whatever a detector tolerates, and the resulting update
need not resemble anything a real client would ever send — which makes "did the
detector catch it" a question about outlier arithmetic rather than about
poisoning.

A label flip has exactly **one** knob: how many of the client's own examples carry
a wrong label. Everything downstream is honest — same examples, same batch size,
same epochs, same learning rate — so the update is a real SGD trajectory of a real
(if wrong) objective and lands inside the honest update distribution by
construction.

### The flip

`symmetric`: `y → (n_classes − 1) − y`. On MNIST that is `0↔9, 1↔8, … 4↔5`. The
standard label-flipping attack in the FL-poisoning literature (Fang et al., USENIX
Security 2020; Tolpegin et al., ESORICS 2020): deterministic, maximally wrong
under an ordinal reading of the label space, and it perturbs every class rather
than one pair, so no class is left as a clean anchor.

`flip_label` mirrors the **type** of the input (a Python `int` for torchvision's
MNIST, a 0-dim tensor for a `TensorDataset`) because `default_collate` refuses to
stack a batch mixing the two — which is every partially-poisoned batch.

**Which** examples are flipped is a seeded, deterministic function of
`(run seed, client id, round index)`, and the flipped loader's shuffle uses a
generator seeded the same way. So a poisoned client's whole local training — the
mislabelled examples *and* the order SGD visits them in — is reproducible, and a
resumed run reproduces the interrupted one's poison exactly.

### The ladder

```
start:  flip start_fraction (1.0) of the poisoned client's round data
CAUGHT      -> step down step_fraction (0.1):  100% -> 90% -> 80% -> ...
NOT CAUGHT  -> HOLD. Send the same level again, unchanged.
CAUGHT at floor_fraction (0.5) -> RESET to start_fraction and descend again.
```

On a 1000-example round that is exactly 1000, 900, 800, 700, 600, 500, then back
to 1000.

- **Levels are an integer index**, not a running float subtraction: `fraction =
  start − level × step`. Repeatedly doing `f -= 0.1` from 1.0 lands on
  `0.7999999999999999` and `0.5000000000000001`, so "am I at the floor?" — the test
  that decides when the cycle resets — would be at the mercy of binary rounding.
- **A floor off the step grid is snapped up** to the nearest reachable level (with
  a log line), so the ladder can never descend past what was configured.
- **The fraction is applied per client**, to that client's own per-round sample
  count, so a non-IID partition (which gives clients unequal shards) still flips
  the same *proportion* everywhere.
- **The ladder advances exactly once per committed round** — `env.record_detection`
  is guarded against a second call. The GRPO loop scores `G` rollouts against the
  same cohort, and letting each advance the ladder would make the attack schedule
  depend on `rl.G`, a sampling hyperparameter, rather than on whether the defense
  caught anything.

**Why a floor.** Below about half the client's data the honest gradient dominates
its own update: the round stops being an attack and becomes noise, and neither
"caught" nor "missed" carries information.

**Why a reset.** Parked at the floor, the defender would spend the rest of the run
being trained on one static, barely-detectable setting. The reset forces it to
keep re-proving itself across the whole range.

### What "caught" means

`attack.schedule.caught_rule` — `all` (default, every poisoned client flagged),
`majority`, or `any`. Ground truth is what **actually shipped** flipped labels,
not the configured set: a client whose level rounded to zero flips sent an honest
update, and holding the defense responsible for not flagging it would step the
ladder on a detection that could not have happened.

### Effective poison only

A configured client counts as poisoned **only if at least one of its labels was
flipped**. A ladder level that rounds to zero on a small shard leaves the client
genuinely honest, so it is excluded from `poisoned_ids`, from the defender's
reward, and from the ladder's feedback. `poisoned_ids` may therefore be empty — a
clean round, on which the defender wins by staying quiet.

## Defense (`defense.mode`)

The shipped config is **`defense.mode: llm`** — the trainable defender.
`algorithmic` swaps in `AlgorithmicDefender` over the pool named by
`defense.algorithms` (default `fltrust, defl, dnc, multikrum`, the same classes the
benchmark panel uses). The curriculum `select()`s one per round; without one it
`choose()`s from a dedicated RNG.

- **`python main.py` refuses to train under `algorithmic`**, with a message saying
  why: the attack is a fixed schedule and the algorithms are not policies, so there
  is no gradient to compute. It is for `--dry-run`, `--baseline` and the benchmark.
- **Fixed for the round.** Set in `env.begin_round()`, so the clean counterfactual
  and the commit face the same defense.
- **Verdicts + aggregate together.** `env.defend(updates, commit=)` returns both;
  `env.evaluate_state` / `env.commit_state` consume the state. For the derived-flag
  defenses (FLTrust trust ≤ 0, DnC/Multi-Krum "dropped", DeFL MOUD-Vote) read
  `induced_drop` as the primary signal and TPR/FPR loosely — same caveat as
  `benchmark/README.md`.
- **Scoring is side-effect free.** `commit=False` snapshots and restores the
  algorithm's cross-round state (`Defense.state_snapshot` / `state_restore`: DeFL's
  Beta counts + `S(t-1)`, DnC's subsampling RNG).
- Each round log carries `attack_metadata.defense` (the algorithm name, or
  `"llm"`); rounds are only comparable within one defense.

## Training curriculum (`curriculum:`, `rl/curriculum.py`)

Which **defense algorithm** each Phase-2 round faces, under `defense.mode:
algorithmic`. It replaces `defense.selection`'s uniform draw, whose per-algorithm
counts are even only in expectation and which re-rolled every round — so a
stateful algorithm's memory advanced on ~1 round in 4, scattered.

```
for algorithm in defense.algorithms:            # fltrust, defl, dnc, multikrum
    curriculum.rounds_per_block (10) consecutive rounds defended by it
# ...then the cycle repeats.
```

- **INACTIVE under `defense.mode: llm`.** There is no algorithm axis, and the
  attack strength is the ladder's, not a schedule's. `build_training_curriculum`
  returns `None` and logs why.
- **`n_poisoners` survives on `CurriculumSlot` as a constant** —
  `len(attack.poison_client_ids)` — so a round log names the attack size without
  re-deriving it from the config. It is not an axis the curriculum varies. (The
  two-axis machinery is kept so a future experiment can sweep something else
  without rewriting the sweep.)
- **One slot per round, consumed in `env.begin_round()`.** The between-phase benign
  FL round does not go through it, so a block always gets its full 10 rounds.
- **Position is one integer**, saved in `checkpoints/rl_progress.json` next to
  `rounds_done` / `controller` / `attacker`, so a resume continues mid-block. Older
  progress files fall back to `rounds_done`, which counts exactly the same rounds.
- `curriculum.enabled: false` (or no `curriculum:` block) restores the random draw.

## Defender-LLM contract (classification)

- **Input** (`agents/defender_agent.build_user_prompt`): per-client features from
  `detector/features.compute_client_features` — **only** features, never the ground
  truth, and never the ladder's current level.
  - Per layer (one per model layer; e.g. `net.2`, `net.4` for MnistNet): `l2_norm`,
    `rel_norm` (vs the median over all clients), `cos_to_median` (vs the
    coordinate-wise median over all clients — references include the scored client
    itself, not leave-one-out; with a benign majority the median is honest either
    way), `sign_agreement` (fraction of coords matching the median sign).
  - Whole model: `l2_norm`, `rel_norm`, `cos_to_mean`, `max_pairwise_cos`
    (FoolsGold), `dnc_score` (SVD spectral outlier).
- **Output**: `{"clients": [{client_id, is_suspicious, confidence}, ...]}` → one
  `DetectionVerdict` per client (missing/garbled entries default benign). A short
  free-text `reason` per client is **off by default** to save generation tokens
  (informational only — never used by the reward/metrics); re-enable it with
  `emit_reason: true` in `configs/defender_agent.yaml`.
- **`confidence` is certainty in THIS verdict**, not a suspicion score: 1.0 = "I am
  sure of the label I just gave". `DetectionVerdict.p_malicious` is the optional,
  explicitly calibrated P(malicious), which the algorithmic defenses supply and
  which carries a hard contract — `p ≥ 0.5` **iff** `is_suspicious`. See
  `core/types.py` and `tests/test_p_malicious_calibration.py`.

## Verifiable reward (`rl/rewards.py`)

### The defender's reward — the only trained signal

Confidence-weighted **soft-F1** vs the poisoned set (or `clip(TPR − λ·FPR)` with
`mode: tpr_minus_fpr`). Continuous by design: GRPO's advantage *is* the
within-group reward spread, and a hard hit/miss score would tie whenever several
rollouts agree on the flags.

On a **clean round** (empty poisoned set — a ladder level that rounded to zero
flips) F1 is undefined and would score a flawless defender 0, training it to
invent detections. There the reward is `1 − mean soft P(malicious)` instead: it is
rewarded for staying quiet.

`_soft_malicious_prob` prefers a producer's calibrated `p_malicious` over the
`(is_suspicious, confidence)` reconstruction. The reconstruction is correct for the
LLM defender (which is asked for exactly that certainty) and **wrong** for a
threshold filter, whose decision boundary is not at 0.5 — under it the soft signal
ran backwards over the entire accepted half of the cohort.

### The attack measurement — reported, never trained on

`attack_effectiveness(clean, post, goal) = drop_term(clean − post, target)`.

- **`drop = clean_reference_accuracy − post_accuracy`** — measured against **this
  round's clean counterfactual**: the accuracy the aggregate reaches with the same
  clients trained on the same data with their real labels
  (`FLArmsRaceEnv.clean_reference_accuracy`, one extra test-set evaluation per
  round, cached). It is *not* the previous round's post-attack accuracy: in a
  continuing federation the attack's own damage becomes the next round's starting
  point, so a previous-round reference measures the round-over-round *change* and a
  repeated, equally damaging attack reads as ≈0 from the second round on.

  **When there is no counterfactual to measure** — the round's defense refused to
  aggregate even the unpoisoned updates (FLTrust zeroing every trust score, DeFL
  removing everyone in a CLP) — `clean_reference_accuracy` returns
  `current_accuracy` as a placeholder and sets `clean_reference_measured = False`.
  `RoundContext.clean_measured` carries it into the round log as
  `clean_measured: false`. **Slice those rounds out before reporting damage.**

  **When the defense aggregated, but only after rejecting the honest majority** —
  the counterfactual exists and looks perfectly ordinary, but it is a reading of the
  defense's own false-positive rate. This is tested on the **unpoisoned** cohort,
  where every flag is by definition a false positive, which is what keeps a
  genuinely strong attack (which legitimately provokes flags) from being mistaken
  for a broken defense. Exposed as `clean_defense_sane` → `RoundContext.defense_sane`
  → `defense_sane: false` in the round log.
- **`drop_term(drop, target)`** is `x = drop/target`: linear on `−0.5 ≤ x ≤ 1`
  (hitting the goal scores exactly 1.0), then `1 + 0.5·(x−1)/(x−1+1)` above —
  strictly increasing, asymptotic to 1.5 — and `−0.5 − 0.25·u/(u+1)` with
  `u = −0.5 − x` below, strictly *decreasing*, asymptotic to −0.75. **No flat region
  at either end**, so two rounds whose damage differs always differ here too. Both
  saturations are fast (4× the target buys < 0.4 extra), so 1.0 keeps meaning "hit
  the requested drop".
- **`attack_was_damaging`** (`rl/switch.py`) is the reported pass/fail:
  `drop ≥ win_fraction × target` (0.06 at the shipped 0.6 × 0.10). Shared with the
  metrics tracker's `attack_success` so the two cannot drift apart.

These exist to answer a question the defender's own reward cannot: *was the thing
it got good at catching actually an attack?* A run where the defender wins every
round but `induced_drop` sits at zero is a defender that learned to detect a
formality.

## GRPO + schedule

- **`rl/grpo.py`**: sample `G` completions; reward each; advantage
  `A_i = (r_i − mean)/max(std, std_floor)`; loss
  `(1/Σ_i L_i)·Σ_i Σ_t [ −A_i·logπ(o_i,t) + β·KL_t ]` with the k3 KL estimator
  against the **frozen base model** (adapters disabled). Single-iteration ⇒ no
  clipping needed. **Token-level normalization**, not sequence-mean: the defender
  emits one verdict object per client, so output length scales with `fl.n_clients`
  and a rollout that omits clients is shorter — the old per-rollout mean rewarded
  terser (more incomplete) verdicts.
  - **Degeneracy is decided by absolute reward SPREAD** (`min_reward_spread`,
    0.02), not by `std < 1e-6`. The reward is a soft F1 over per-client
    confidences, so two behaviourally identical verdict-sets routinely differ by a
    hair; treating that as signal meant z-scoring sampling noise up to
    full-magnitude advantages.
  - **`resample_on_zero_advantage`** re-draws the group once at a higher
    temperature; **`skip_zero_advantage`** then skips the optimizer step entirely
    rather than stepping on a pure KL-to-base signal (which is active un-learning).
  - **Where the spread comes from:** all `G` rollouts classify the *same* cohort, so
    the only source of variation is `rl.temperature` (default 1.0). At 0 every
    rollout is identical and no step is ever taken.
- **`rl/policy.py`**: one Unsloth `Qwen2.5-3B-Instruct` base (bf16 LoRA by default;
  4-bit QLoRA optional via `rl.load_in_4bit`) + one PEFT LoRA adapter (`defender`).
  `set_adapter` selects the active policy; `disable_adapter` exposes the base as the
  KL reference. The multi-adapter machinery is kept — it costs nothing with one
  adapter and is what would let a second trainable agent be added back.
- **`rl/turns.py`**: `DefenderTurn` binds one round to the defender policy. It does
  not sample an attack — the env has already produced the round's poisoned updates
  by the time `begin_round` returns, which is exactly the property the old design
  needed a frozen-attacker sample to obtain. Its `commit()` is where
  `env.record_detection` fires.
- **`rl/schedule.py`**: defender phases. A phase runs until the defender sustains a
  win (`success_streak` consecutive rounds catching every poisoned client without
  over-flagging) or hits `max_phase_rounds`; the adapter is then snapshotted into
  the league and persisted. This still earns its place because the opponent is not
  static: winning a phase means the ladder has been driven down, so the phase
  boundary checkpoints the defender at each rung it has actually mastered.
  - **The committed rollout is an on-policy draw** (`rl.commit_selection: sample`),
    not the argmax. Committing the best-of-G would corrupt the attack schedule as
    well as the metrics: the ladder reacts to the committed verdicts, so it would be
    calibrated to the policy's luckiest sample rather than its actual behaviour. The
    argmax is still logged as `best_index` next to `committed_index`.
  - The **league** is a bounded ring buffer (`league_max_snapshots`, default 10,
    oldest evicted) of past defender checkpoints; each is a full CPU copy of the
    adapter's LoRA tensors (~115 MB at `lora_r: 16`), so an unbounded pool OOMs a
    long run. With no opponent policy left there is nothing to swap them into — they
    are retained for inspection, not played against.
- **Between-phase benign FL round** (`fl_interlude_between_phases`, default on):
  before every phase after the first, one honest FL round runs exactly like Phase 1
  (`FLArmsRaceEnv.run_benign_fl_round`) — all clients retrain from the current
  global with real labels, FedAvg into a new global, and the freshly trained
  per-client weights replace the stored references. **No effect while
  `fl.freeze_global_in_phase2` is true**: advancing the shared global is precisely
  what a simulated round must not do, so it skips and logs.

## Modes (`main.py`)

| Mode | Flag | Uses | GPU |
|------|------|------|-----|
| Train | *(default)* | `rl/policy.py` + `rl/schedule.py` (GRPO) | yes |
| Dry-run | `--dry-run` | `rl/inference.py` (frozen Ollama/OpenAI), full loop, no updates | no |
| Baseline | `--baseline` | `rl/baseline.py` — no LLM at all | no |

The attack needs no model in any mode. Under `defense.mode: llm`, `--dry-run`
makes one LLM call per round (the defender) and `--baseline` uses a fixed
norm/sign heuristic in its place; under `algorithmic` neither calls a model at all.
Training requires `llm`.

The ladder adapts in every mode — `--dry-run` and `--baseline` call
`env.record_detection` exactly as training does, so their logs show the same
saw-tooth a training run would produce.

All three honour `--rounds N` (an absolute budget overriding `fl.simulation_rounds`)
and `--config <path>`. `--debug` without `--rounds` caps how many rounds **this
run** adds on top of `rounds_done`, so debugging a resumed run still executes
rounds instead of exiting immediately.

## Checkpoints & resume

- `checkpoints/global_model.pt`, `client_updates.pt`, `baseline.json` — Phase 1.
- `checkpoints/defender_adapter/` — the LoRA adapter
  (`adapter_model.safetensors` + `adapter_config.json`).
- `checkpoints/fl_state.pt` — the live Phase-2 FL state (evolving global + per-client
  weights + accuracy + round index).
- `checkpoints/rl_progress.json` — resume state:
  `{"rounds_done", "round_index", "controller", "curriculum", "attacker"}`.
  - `rounds_done` — the GRPO-step counter.
  - `round_index` — the FL round-number counter, so round labels and
    `logs/round_data/rounds.jsonl` continue instead of restarting.
  - `controller` — the `PhaseController` snapshot (`learner`, `phase_index`,
    `phase_round`, `streak`, `capped`).
  - `curriculum` — the defense sweep's position.
  - **`attacker` — the label-flip ladder's `level` and `cycle`.** Without it a
    restart rewinds the attack to full poison and replays strengths the defender is
    already past, making the schedule depend on how often the run happened to crash.
    Saved with the grid (`start`/`step`/`n_levels`) so a config edit between runs is
    detectable rather than silently re-interpreting a saved level as a different
    fraction.

  Written together with the adapter on the `rl.save_every` cadence and on exit.
  Older files with any field missing still load (each falls back safely; a
  controller that says `learner: "attacker"` — from before the attacker LLM was
  removed — continues as the defender with a warning).
- **Not** persisted: the in-memory checkpoint league (it restarts empty).

## Logs

Per-round records are **appended to a JSONL stream** (one JSON object per line)
rather than written one file per round. At the configured `fl.simulation_rounds` a
file-per-round sink produced millions of tiny files across two directories,
exhausting inodes; appending is O(1) per round and stays greppable. `monitor.py`
and `visualize_rounds.py` read the JSONL **and** legacy `round_NNN.json` files, so
logs from older runs still load.

- `logs/system.log` — run log.
- `logs/round_data/rounds.jsonl` — per round: `attack_goal`, `poisoned_client_ids`,
  `predicted_labels`, accuracies, `attack_effectiveness`, `defender_reward`,
  `learning_agent`, and an `attack_metadata` block with:
  - `flip_fraction` / `flip_plan` / `n_flipped` — **the attack**: the ladder level
    the round was sent at, and how many labels that meant per client.
  - `ladder` — where it moved after seeing the verdicts (`caught`, `caught_rule`,
    `event` ∈ `hold`/`step_down`/`reset`, `next_flip_fraction`, `level`, `cycle`).
    The run's whole attack schedule is recoverable from its own logs.
  - `clean_accuracy` / `induced_drop` — the counterfactual and the damage.
  - `clean_measured` / `defense_sane` — **slice both `false` cases out before
    reporting damage**; they measure the defense, not the attack.
  - `attack_effectiveness` / `attack_damaging` — the normalized damage and whether
    it cleared the reporting bar.
  - `defense` — which algorithm faced the round, or `"llm"`. Rounds are only
    comparable within one defense.
  - `curriculum`, `phase_index`, `phase_round`, `learner_success`, and a `train`
    sub-block (loss, mean reward, reward spread, zero-advantage fraction,
    `committed_index` vs `best_index`, `stepped`, `resampled`).
- `logs/metrics/rounds.jsonl` + `summary.json` — ground-truth confusion / TPR /
  FPR / ASR / APR. `summary.json`'s `aggregate` covers every round; its `per_round`
  block is the retained tail (`MetricsTracker.keep_rounds`, default 2000) — the full
  history is in `rounds.jsonl`.
- `logs/debug.json` (`--debug`) — structured event stream, capped at the most recent
  `DebugLogger.MAX_EVENTS` events (the file is rewritten in full each round, so an
  unbounded buffer made long debug runs O(n²)). `events_dropped` records how many
  were evicted.
- `logs/visualizations/report.html` — `python visualize_rounds.py`.
