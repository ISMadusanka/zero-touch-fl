# System Architecture — LLM-Direct Adversarial FL with GRPO

This document describes the redesigned system: the round loop, the attacker /
defender contracts, the verifiable rewards, the GRPO training schedule, and the
checkpoint layout. It supersedes the old feedback/episodic-memory design.

## Components

| Layer | Module | Role |
|-------|--------|------|
| Model | `model/mnist_net.py` | `MnistNet`, ~970 params. State_dict keys: `net.2.weight [16,49]`, `net.2.bias [16]`, `net.4.weight [10,16]`, `net.4.bias [10]`. The schema both LLMs operate over. |
| Data | `data/mnist_loader.py` | MNIST load + per-client partition: IID, or the FLTrust non-IID bias-`q` scheme (`partition_noniid_fltrust`). |
| Clients | `clients/benign_client.py` | Honest local SGD → `ModelUpdate`. |
| Server | `server/fed_server.py`, `server/aggregation.py` | Global model + eval; FedAvg over non-flagged clients. |
| Features | `detector/features.py` | Per-client, per-layer statistical feature vectors (no decisions). |
| Attacker | `agents/attacker_agent.py` + `agents/attack_ops.py` | Prompt (layer stats) + parse/apply of an attack plan (operator DSL). |
| Defender | `agents/defender_agent.py` | Prompt + parse of per-client benign/malicious labels. |
| Defense ensemble | `server/defense_ensemble.py` | The defender-LLM-free path (`--freeze defender`): FLTrust + Multi-Krum + DnC + DeFL run as detectors, rejections unioned. |
| RL | `rl/*` | Environment, rewards, turns, GRPO, schedule, policy, baseline, inference. |
| Metrics | `metrics/*` | Ground-truth confusion/TPR/FPR/ASR/APR (research + reward source). |

## Round loop (Phase 2)

Leader–follower (Stackelberg): the attacker moves, the defender best-responds to
the realized updates.

```
reset env from Phase-1 checkpoint (per-client benign weights, global, baseline acc)
for each round:
  1. honest updates for all N clients             # retrain from global, or replay Phase-1 weights
  2. expose the attacker's controllable pool [0..n_compromisable) + a poison budget b
     (b = randint(1, max_poison_clients) in training; fixed = eval budget at eval time)
  3. ATTACKER LLM → SELECT <= b clients from the pool + a per-client attack plan
     (input: round, controllable_client_ids, max_poison_clients, per-client LAYER STATS, acc, goal)
     → apply_plan(benign_i, plan_i) → poisoned weights for the CHOSEN clients
  4. build full update list (poisoned ∪ honest)
  5. detector/features → per-client per-layer stat vectors
  6. DEFENDER LLM → benign/malicious label + confidence per client   (input: features ONLY)
  7. FedAvg over clients labelled benign → new global (None if all flagged → keep prev)
  8. evaluate global accuracy
  9. attacker_reward + defender_reward (ground truth = the CHOSEN poisoned set)  ← train-time only
 10. GRPO update for the learning agent; write round log + metrics
```

The attacker may poison at most `min(max_poison_clients, n_compromisable)` of its
pool — always a strict minority of the `N` clients — so the defender's robust
feature references (coordinate-wise median, MAD) always see an honest majority.
Different GRPO rollouts may pick different subsets, so each rollout is rewarded
against its OWN chosen set; the committed rollout's set becomes the round's ground
truth.

**Arms-race handoff.** The attacker trains (defender frozen) until it wins 3
committed rounds in a row, then a single **benign FL round** runs (step 1 above
for all clients, no attacker/detector — the refreshed client weights + global are
kept), and the defender trains (attacker frozen) until it wins 3 in a row, then
another benign FL round runs, and so on. Every GRPO phase after the first starts
from a freshly advanced client state (see "Between-phase benign FL round" below).

## Attacker contract (client selection + attack-plan DSL)

- **Input** (`agents/attacker_agent.build_user_prompt`): `round`,
  `current_global_accuracy`, `attack_goal`, `controllable_client_ids` (the pool it
  may touch), `max_poison_clients` (this round's budget), and `client_update_stats`
  — per-layer + whole-model **statistics of each pool client's HONEST UPDATE**
  `Δ = local − global` (`agents/attack_ops.delta_details`): `rel_update`
  (‖Δ‖/‖G‖), `rms_delta`, `energy_frac`, `sign_flip_frac`, `std_ratio`,
  `absmean_ratio`, and whole-model `cos_to_global`. Every value is normalized
  against the **global model only** — never a median/mean/pairwise reference over
  other clients (those are defender-only; a partial-insider attacker cannot
  observe the honest majority) and never a pool baseline (the budget makes the
  pool unstable and it is the poison target). This keeps the observation
  dimensionless, architecture-independent, and budget-invariant. No raw weights.
- **Output**: a single JSON object
  `{"clients": [ {"id": <pool id>, "operations": [ {op, target, ...params}, ... ]}, ... ]}`.
  The attacker **selects which** pool clients to poison (≤ budget) and gives **each
  its own plan**. Operators (`agents/attack_ops.py`, 10): `scale`, `sign_flip`,
  `add_gaussian_noise`, `mask`, `clip`, `add_constant`, `permute`,
  `scale_neurons`, `blend_random`, `quantize`. `target` is `"all"`, a layer name,
  or a full parameter key — the exact names come from `client_update_stats` (for
  MnistNet e.g. `"net.2"` / `"net.4.weight"`). Operations apply in order.
- **Selection + application** (`agents/attacker_agent.select_and_apply` →
  `attack_ops.apply_plan`): filters ids to the pool, dedups, **truncates to the
  budget**; per chosen client deep-copies its benign weights, applies its plan,
  skips unknown ops / bad params / bad targets (counted, not fatal), then scrubs
  NaN/Inf and clamps to `±max_weight_abs`. PyTorch does all arithmetic — the LLM
  only emits the selection + plans. Shorthand inputs (`{"operations": [...]}`
  shared plan, `{"clients":[ids], "operations":[...]}`) are accepted for
  robustness, and parsing never raises.
- **Effective poison only.** A selected client counts as poisoned **only if its
  weights actually changed**. Unparseable output, an empty plan, ops that were all
  skipped as invalid, and arithmetic no-ops (`scale factor=1.0`) all leave the
  client's weights byte-identical to its honest update — the server receives an
  honest update, so there is nothing to detect. Those clients are excluded from
  the returned `poisoned_ids` and charged to `n_malformed` (a wasted client, a
  reward penalty) instead. `poisoned_ids` may therefore be **empty**: a clean
  round where the attacker selected clients and achieved nothing. Counting them as
  poisoned made the ASR metric report 100% success for an attack that did nothing
  and trained the defender to flag undetectable clients.

## Defender contract (classification)

- **Input** (`agents/defender_agent.build_user_prompt`): per-client features
  from `detector/features.compute_client_features` — **only** features, never the
  ground truth.
  - Per layer (one per model layer; e.g. `net.2`, `net.4` for MnistNet): `l2_norm`,
    `rel_norm` (vs the median over all clients), `cos_to_median` (vs the
    coordinate-wise median over all clients — references include the scored client
    itself, not leave-one-out; with a benign majority the median is honest either
    way), `sign_agreement` (fraction of coords matching the median sign — catches
    sign-flip/targeted attacks that preserve norm).
  - Whole model: `l2_norm`, `rel_norm`, `cos_to_mean`, `max_pairwise_cos`
    (FoolsGold), `dnc_score` (SVD spectral outlier).
- **Output**: `{"clients": [{client_id, is_suspicious, confidence}, ...]}`
  → one `DetectionVerdict` per client (missing/garbled entries default benign). A
  short free-text `reason` per client is **off by default** to save generation
  tokens (informational only — never used by the reward/metrics); re-enable it
  with `emit_reason: true` in `configs/defender_agent.yaml`.

## Verifiable rewards (`rl/rewards.py`)

Both continuous, so GRPO group advantages don't collapse.

- **Attacker**: `α·drop_term(drop, target) + β·stealth − γ·malformed_frac
  − δ·client_cost + ζ·diversity`, with `target` from the goal, `stealth` =
  confidence-weighted evasion over the chosen clients (0 when nothing was
  poisoned), `malformed_frac = n_malformed / n_selected` (the waste penalty,
  normalized over the clients the attacker *selected*),
  `client_cost = (n_used−1)/(n_compromisable−1)` (the **use-fewer-clients**
  penalty, 0 for a single client), and `diversity = 1 − mean pairwise cosine` of
  the chosen clients' perturbations (the **collaboration** bonus, 0 for a single
  client) — rewarding coordinated, distinct multi-client attacks over identical
  Sybil-like clones. See `rl/rewards.py` (`attacker_reward`, `drop_term`,
  `perturbation_diversity`).

  - **`drop = clean_reference_accuracy − post_accuracy`** — the damage measured
    against **this round's clean counterfactual**: the accuracy the aggregate
    reaches with no poison (`FLArmsRaceEnv.clean_reference_accuracy`, one extra
    test-set evaluation per round, cached). It is *not* the previous round's
    post-attack accuracy. With `benign_retrain_each_round: false` the environment
    is memoryless — every round's global is rebuilt from frozen benign weights
    plus that round's poison — so a previous-round reference measured the
    round-over-round *change*: an identical devastating attack scored high once
    and ≈0 forever after, which put the schedule's `success_streak` gate
    permanently out of reach. Against the clean counterfactual an attack that hits
    its target scores the same every time, in either retrain mode.
  - **`drop_term(drop, target)`** is `x = drop/target` floored at `−0.5`, linear
    up to `x = 1` (hitting the goal scores exactly 1.0), then
    `1 + 0.5·(x−1)/(x−1+1)` — strictly increasing but asymptotic to 1.5. Same
    range and same value at the goal as the old hard `clip(x, −0.5, 1.5)`; the
    difference is that there is **no flat region**. The flat region was a training
    failure: once the policy reliably overshot `1.5·target`, every rollout in a
    GRPO group tied, the advantage spread collapsed to zero, and `grpo_step`
    skipped the update — the attacker stopped learning exactly when it got good.
    Saturation is fast (4× the target buys < 0.4 extra), so the objective stays
    "hit the requested drop", not "destroy the model".
- **Defender** (train-time ground truth): confidence-weighted **soft-F1** vs the
  poisoned set (or `clip(TPR − λ·FPR)`). On a **clean round** (empty poisoned set
  — the attacker's plans were all no-ops) F1 is undefined and would score a
  flawless defender 0, training it to invent detections; there the reward is
  `1 − mean soft P(malicious)` instead, i.e. it is rewarded for staying quiet.

## GRPO + schedule

- **`rl/grpo.py`**: sample `G` completions; reward each; advantage
  `A_i = (r_i − mean)/(std + ε)`; loss
  `mean_i[ −A_i·mean_t logπ(o_i,t) + β·mean_t KL_t ]` with the k3 KL estimator
  against the **frozen base model** (adapters disabled). Single-iteration ⇒ no
  clipping needed. Reports the zero-advantage-group fraction (stall signal).
- **`rl/policy.py`**: one Unsloth `Qwen2.5-3B-Instruct` base (bf16 LoRA by
  default; 4-bit QLoRA optional via `rl.load_in_4bit`) + two PEFT LoRA
  adapters (`attacker`, `defender`). `set_adapter` selects the active policy;
  `disable_adapter` exposes the base as the KL reference.
- **`rl/schedule.py`**: freeze-and-alternate — train attacker `K_a` rounds
  (defender frozen, greedy), then defender `K_d` rounds (attacker frozen,
  greedy), repeat. The best-scoring sampled action is committed to advance the
  env. An **opponent league** snapshots adapters periodically and, with
  probability `league_prob`, makes a phase face a random past snapshot. The league
  is a **bounded ring buffer** (`league_max_snapshots`, default 10, oldest
  evicted): each snapshot is a full CPU copy of one adapter's LoRA tensors
  (~115 MB for Qwen2.5-3B at `lora_r: 16`), so an unbounded pool grew past 20 GB
  of host RAM within ~10k rounds and eventually OOM'd the box.
- **Switch trigger** (`best_response` mode): a phase freezes-and-switches as soon
  as the learner wins `success_streak` (default **3**) committed rounds **in a
  row**. `min_phase_rounds` is set equal to `success_streak`, so there is no
  extra minimum-length gate — 3 consecutive wins is the whole condition (a
  `max_phase_rounds` cap still forces a switch if the win never comes).
- **Between-phase benign FL round** (`fl_interlude_between_phases`, default on):
  before EVERY phase after the first — i.e. right before the incoming learner
  starts GRPO — the schedule runs ONE honest FL round exactly like Phase 1
  (`FLArmsRaceEnv.run_benign_fl_round`): all benign clients retrain from the
  current global, FedAvg into a new global, and the freshly trained per-client
  weights REPLACE the frozen benign references. So each new attacker/defender
  phase (and the frozen opponent + aggregator) trains against an advanced client
  state rather than the same frozen Phase-1 weights. The interlude gets its own
  sequential round number and is logged to `logs/system.log`,
  `logs/round_data/rounds.jsonl` (`attack_metadata.event="benign_fl_round"`)
  and `logs/debug.json`.

## Single-learner mode (`--freeze`)

`main.py --freeze <attacker|defender>` trains ONE agent for the whole run and
bypasses the schedule above entirely: no `PhaseController`, no switching, no
between-phase FL interlude, no opponent league (the opponent never changes, so
snapshots would only cost host RAM). `rl/schedule._train_single_learner` runs the
same round body as the alternating schedules — same prompts, same `grpo_step`, same
rewards, same win-gate — so `learner_success`, ASR/TPR/FPR and the logs stay
directly comparable across modes.

- **`--freeze defender`** deactivates the defender LLM. `server/defense_ensemble.py`
  runs FLTrust, Multi-Krum, DnC and DeFL (config: `defense.algorithms`) over the
  round's updates as pure detectors and **unions their rejections** — one flag from
  any algorithm drops that client from FedAvg, so a poisoner has to evade all of
  them at once. Details that make it train correctly:
  - The verdict's confidence is the **fraction of algorithms that flagged** the
    client (benign-and-cleared = confidence 1.0). The per-algorithm confidences are
    not comparable (FLTrust `1 − trust` ∈ [0,1] vs Multi-Krum/DnC raw unbounded
    outlier scores vs DeFL's vote fraction), and `rl/rewards._soft_malicious_prob`
    reads confidence as a probability — the consensus fraction is a well-defined
    [0,1] stand-in that gives the stealth term a gradient.
  - `FLArmsRaceEnv.clean_reference_accuracy` runs the **same defense** over the
    all-honest updates. Multi-Krum and DnC drop a fixed `f`/`c·m` clients every
    round and DeFL always flags at least one, so an all-accepted reference would
    charge the attacker's `drop` with damage the *defense* caused.
  - GRPO scores `G` rollouts per round, so `verdicts(..., commit=False)` snapshots
    and restores each defense's cross-round state (DeFL's CLP total + Beta trust
    counts, DnC's subsampling RNG); only the committed round advances it.
- **`--freeze attacker`** is the mirror: the defender LLM trains against the frozen
  attacker adapter. No algorithmic defenses are involved.

Only the learner's adapter is written, and `storage.save_progress` **merges** rather
than overwrites — a `--freeze` run advances `rounds_done`/`round_index` while leaving
the stored `controller` untouched, so a later plain `main.py` run resumes the
alternating arms race in the phase it stopped in.

## Modes (`main.py`)

| Mode | Flag | Uses | GPU |
|------|------|------|-----|
| Train | *(default)* | `rl/policy.py` + `rl/schedule.py` (GRPO) | yes |
| Single-learner | `--freeze <agent>` | as Train, but one learner and no switching | yes |
| Dry-run | `--dry-run` | `rl/inference.py` (frozen Ollama/OpenAI), full loop, no updates | no |
| Baseline | `--baseline` | `rl/baseline.py` best-of-N fixed actions, no LLM | no |

All honour `--rounds N` (an absolute budget overriding `fl.simulation_rounds`)
and `--config <path>`. `--debug` without `--rounds` caps how many rounds **this
run** adds on top of `rounds_done`, so debugging a resumed run still executes
rounds instead of exiting immediately. `--freeze` composes with Train and Dry-run
(it is ignored, with a warning, in `--baseline`, which uses no LLM agents).

## Checkpoints & resume

- `checkpoints/global_model.pt`, `client_updates.pt`, `baseline.json` — Phase 1.
- `checkpoints/attacker_adapter/`, `checkpoints/defender_adapter/` — LoRA adapters
  (`adapter_model.safetensors` + `adapter_config.json`).
- `checkpoints/rl_progress.json` — resume state:
  `{"rounds_done", "round_index", "controller"}`. `rounds_done` is the GRPO-step
  counter; `round_index` is the FL round-number counter (so round labels and
  `logs/round_data/rounds.jsonl` continue instead of restarting from the first
  Phase-2 round); `controller` is the `PhaseController` snapshot
  (`learner`, `phase_index`, `phase_round`, `streak`, `capped`) so the arms-race
  schedule continues where it left off. Written together with the adapters on the
  `rl.save_every` cadence and on exit; fields not supplied are **merged, not
  cleared**, so a single-learner run (which has no controller) advances the round
  counters without wiping the arms-race schedule. Older files with only
  `rounds_done` still load (the missing fields fall back:
  `round_index`→`rounds_done`, `controller`→a fresh schedule).
- Rerunning resumes: adapters + `rounds_done` + `round_index` + `controller` are all
  reloaded. The env still re-derives its weights from the Phase-1 baseline (the model
  state lives in the adapters), but the round counter and schedule pick up in place.
  **Not** persisted: the in-memory opponent **league** (snapshots restart empty).

## Logs

Per-round records are **appended to a JSONL stream** (one JSON object per line)
rather than written one file per round. At the configured `fl.simulation_rounds`
a file-per-round sink produced millions of tiny files across two directories,
exhausting inodes and making the directories unusable; appending is O(1) per round
and stays greppable. `monitor.py` and `visualize_rounds.py` read the JSONL **and**
legacy `round_NNN.json` files, so logs from older runs still load.

- `logs/system.log` — run log.
- `logs/round_data/rounds.jsonl` — per round: `attack_goal`,
  `poisoned_client_ids`, `predicted_labels`, accuracies, `attacker_reward`,
  `defender_reward`, `learning_agent`, and an `attack_metadata` block with
  `clean_accuracy` / `induced_drop` (the counterfactual and the damage the reward
  actually uses), `n_malformed`, `budget`, `n_used`, a `defense` sub-block
  (`mode: "llm_defender"`, or `mode: "algorithmic"` plus the per-algorithm flags
  under `--freeze defender`), plus a `train` sub-block (loss, mean reward,
  zero-advantage fraction).
- `logs/metrics/rounds.jsonl` + `summary.json` — ground-truth confusion / TPR /
  FPR / ASR / APR. `summary.json`'s `aggregate` covers every round; its
  `per_round` block is the retained tail (`MetricsTracker.keep_rounds`, default
  2000) — the full history is in `rounds.jsonl`.
- `logs/debug.json` (`--debug`) — structured event stream, capped at the most
  recent `DebugLogger.MAX_EVENTS` events (the file is rewritten in full each
  round, so an unbounded buffer made long debug runs O(n²)). `events_dropped`
  records how many were evicted.
- `logs/visualizations/report.html` — `python visualize_rounds.py`.
