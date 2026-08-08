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
| Defender (default) | `server/algo_defender.py` | **Active defense.** Rotating pool of published algorithms (FLTrust / DeFL / DnC / Multi-Krum from `benchmark/defenses/`); one drawn per round, producing both the verdicts and the aggregate. |
| Defender (LLM) | `agents/defender_agent.py` | Prompt + parse of per-client benign/malicious labels. **Disabled** unless `defense.mode: llm`. |
| RL | `rl/*` | Environment, rewards, turns, GRPO, schedule, policy, baseline, inference. |
| Metrics | `metrics/*` | Ground-truth confusion/TPR/FPR/ASR/APR (research + reward source). |

## Round loop (Phase 2)

Leader–follower (Stackelberg): the attacker moves, the defender best-responds to
the realized updates.

```
reset env from Phase-1 checkpoint (per-client benign weights, global, baseline acc)
for each round:
  0. FIX this round's defense algorithm + poison quota b from the training CURRICULUM
     (algorithmic mode; both held for the whole 10-round block — see below.
      No curriculum: the algorithm is drawn per defense.selection and
      b = randint(1, max_poison_clients))
  1. honest updates for all N clients               # retrain from global, or replay Phase-1 weights
  2. expose the attacker's controllable pool [0..n_compromisable) + exact poison quota b
     (curriculum block's count in training; fixed = eval budget at eval time)
  3. ATTACKER LLM → SELECT exactly b clients from the pool + a per-client attack plan
     (input: round, controllable_client_ids, max_poison_clients, per-client LAYER STATS, acc, goal)
     → apply_plan(benign_i, plan_i) → poisoned weights for the CHOSEN clients
  4. build full update list (poisoned ∪ honest)
  5. DEFEND:
     algorithmic (default) — env.defend(): the round's algorithm emits per-client
       verdicts AND its own aggregate (FLTrust trust-weighting, DeFL Beta+CLP,
       DnC/Multi-Krum filtered mean) → new global (None → keep prev)
     llm — detector/features → per-client stat vectors → DEFENDER LLM emits a
       benign/malicious label + confidence per client (input: features ONLY),
       then FedAvg over the clients labelled benign
  6. evaluate global accuracy
  7. attacker_reward + defender_reward (ground truth = the CHOSEN poisoned set)  ← train-time only
  8. GRPO update for the learning agent; write round log + metrics
```

The attacker must poison exactly the round's quota `b` (always clamped to the
controllable pool) — normally a strict minority of the `N` clients — so the defender's robust
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
With the defender LLM disabled there is no defender phase: every phase is an
attacker phase, still bounded by the same win-streak / cap gate, so the benign FL
interlude keeps firing between them.

## Defense (`defense.mode`, `server/algo_defender.py`)

The shipped config is **`defense.mode: algorithmic`** — the defender LLM is off.
`AlgorithmicDefender` holds the pool named by `defense.algorithms` (default
`fltrust, defl, dnc, multikrum`, the same classes the benchmark panel uses). The
training curriculum `select()`s one per round; without a curriculum it
`choose()`s one from a dedicated RNG, so the draw never shifts the env's poison /
budget / target sampling.

- **Fixed for the round.** Set in `env.begin_round()`, so the clean
  counterfactual, all `G` scored rollouts and the commit face the same defense —
  GRPO advantages stay meaningful.
- **Verdicts + aggregate together.** `env.defend(updates, commit=)` returns both;
  `env.evaluate_state` / `env.commit_state` consume the state. The verdicts are
  what `defender_reward`, the metrics tracker and the attacker's evasion term see.
  For the derived-flag defenses (FLTrust trust ≤ 0, DnC/Multi-Krum "dropped",
  DeFL MOUD-Vote) read `induced_drop` as the primary signal and TPR/FPR loosely —
  same caveat as `benchmark/README.md`.
- **Scoring is side-effect free.** `commit=False` snapshots and restores the
  algorithm's cross-round state (`Defense.state_snapshot` / `state_restore`:
  DeFL's Beta counts + `S(t-1)`, DnC's subsampling RNG).
- **Attacker-only training.** `rl/schedule.py` builds one optimizer, keeps the
  `PhaseController` on `learners=("attacker",)`, skips league/curriculum opponent
  swaps, and never writes `checkpoints/defender_adapter/`. `main.py` doesn't even
  create the defender LoRA adapter. Set `defense.mode: llm` to restore the
  two-sided race — the defender adapter resumes untouched.
- Each round log carries `attack_metadata.defense` (the algorithm name, or
  `"llm"`); rounds are only comparable within one defense.

## Training curriculum (`curriculum:`, `rl/curriculum.py`)

Which (defense, #poisoners) pair each Phase-2 round faces. Enabled by default; it
replaces `defense.selection` **and** `attack.sample_budget_in_training`, whose two
independent uniform draws split the rounds evenly only in expectation (per-cell
counts are `Binomial(rounds, 1/20)`, sd ≈ 3.1 per 200 rounds) and re-rolled the
pair every round, so the policy never trained contiguously in any one regime.

```
for algorithm in defense.algorithms:            # fltrust, defl, dnc, multikrum
    for k in curriculum.poisoner_counts:        # 1, 2, 3, 4, 5
        curriculum.rounds_per_block (10) consecutive GRPO rounds at (algorithm, k)
# 4 x 5 x 10 = 200 rounds per cycle; every algorithm gets exactly 50, 10 per k.
```

- **One slot per round, consumed in `env.begin_round()`.** The between-phase
  benign FL round does not go through it, so a block always gets its full 10
  attacker rounds; the phase machinery runs on top of the sweep unchanged.
- **Position is one integer**, saved in `checkpoints/rl_progress.json` next to
  `rounds_done` / `controller`, so a resume continues mid-block. Older progress
  files fall back to `rounds_done`, which counts exactly the same rounds.
- **The attack target is pinned** (`attack.goal.target_accuracy_drop: 0.02`,
  `sample_target_in_training: false`) so a block's rounds — and one block against
  the next — are comparable: the win gate is `win_fraction x the round's target`
  and the reward is normalized by it, so a moving target would change what
  "success" means inside a block. The value is the **reward's scale**, not an
  aspiration: at the old `0.10`, against 1–5 of 20 clients facing a robust
  aggregator, the observed drops of ~0.01 mapped to ~0.1 of reward — less than the
  collaboration bonus and only ~3x the stealth term, so the reward nominally led
  with damage while ranking it third. Raise it back if you widen the threat model.
- Each round log carries `attack_metadata.curriculum`
  (`algorithm`, `n_poisoners`, `cycle`, `block`, `position`, `block_round`).
- `curriculum.enabled: false` (or no `curriculum:` block) restores the random draws.

## Attacker contract (client selection + attack-plan DSL)

- **Input** (`agents/attacker_agent.build_user_prompt`): `round`,
  `current_global_accuracy`, `attack_goal`, `controllable_client_ids` (the pool it
  may touch), `max_poison_clients` (this round's exact quota; legacy key name), and `client_update_stats`
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
  The attacker **selects exactly the quota** of pool clients to poison and gives **each
  its own plan**. Operators (`agents/attack_ops.py`, 10): `scale`, `sign_flip`,
  `add_gaussian_noise`, `mask`, `clip`, `add_constant`, `permute`,
  `scale_neurons`, `blend_random`, `quantize`. `target` is `"all"`, a layer name,
  or a full parameter key — the exact names come from `client_update_stats` (for
  MnistNet e.g. `"net.2"` / `"net.4.weight"`). Operations apply in order.
- **Selection + application** (`agents/attacker_agent.select_and_apply` →
  `attack_ops.apply_plan`): filters ids to the pool, dedups, truncates excess ids,
  and fills an under-sized usable selection to the **exact quota** with remaining
  pool ids and an emitted plan; per chosen client deep-copies its benign weights, applies its plan,
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

## Defender-LLM contract (classification) — only under `defense.mode: llm`

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
  + ζ·diversity·stealth`, with `target` from the goal, `stealth` =
  calibrated evasion over the chosen clients (`1 − p_malicious`, 0 when nothing was
  poisoned), `malformed_frac = n_malformed / n_selected` (the waste penalty,
  normalized over the clients the attacker *selected*), and
  `diversity = 1 − mean pairwise cosine` of
  the chosen clients' perturbations (the **collaboration** bonus, 0 for a single
  client, and **gated on stealth** so a well-coordinated attack that got caught earns
  nothing for its coordination) — rewarding coordinated, distinct multi-client attacks over identical
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

    **When there is no counterfactual to measure** — the round's defense refused to
    aggregate even the unpoisoned updates (FLTrust zeroing every trust score, DeFL
    removing everyone in a CLP) — `clean_reference_accuracy` returns
    `current_accuracy` as a placeholder and sets `clean_reference_measured = False`.
    Callers must check it: `RoundContext.clean_measured` carries it to the schedule,
    which passes `skip_update=True` to `grpo_step` so the round still advances the
    environment but applies **no gradient**, and the round log records
    `clean_measured: false`. Returning the fallback silently was a real training bug:
    it is the previous round's post-attack accuracy, and when the poisoned round also
    failed to aggregate the post accuracy was that same number, so `drop` was
    identically `+0.0000` by construction. About a quarter of the rounds in a
    recorded run looked like a measured "the attack achieved nothing".
  - **`drop_term(drop, target)`** is `x = drop/target`: linear on `−0.5 ≤ x ≤ 1`
    (hitting the goal scores exactly 1.0), then `1 + 0.5·(x−1)/(x−1+1)` above —
    strictly increasing, asymptotic to 1.5 — and `−0.5 − 0.25·u/(u+1)` with
    `u = −0.5 − x` below, strictly *decreasing*, asymptotic to −0.75. Same value at
    the goal and the same linear region as the old hard `clip(x, −0.5, 1.5)`; the
    difference is that there is **no flat region at either end**.

    Flat regions are a specific training failure, because GRPO's advantage *is* the
    within-group reward spread: once every rollout lands in the same flat region they
    tie, the spread collapses, and `grpo_step` skips the update. The overshoot flat
    came first — the attacker stopped learning exactly when it got good at overshooting
    `1.5·target`. The **backfire** flat matters once `target` is small: at the
    configured `0.02`, `x < −0.5` means "the attack made the model more than 1pp
    *better* than the clean counterfactual", which several rounds in a recorded run
    were well past, and every such rollout used to score exactly `−0.5`.

    Both saturations are fast (4× the target buys < 0.4 extra), so the objective stays
    "hit the requested drop", not "destroy the model", and a catastrophic backfire
    stays bounded-bad instead of dominating.
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
  greedy), repeat. **With the algorithmic defense the rotation is attacker-only**
  (`learners=("attacker",)`): one optimizer, no opponent generator, no league
  swaps, and the defender adapter is neither loaded nor saved.
  The best-scoring sampled action is committed to advance the
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

## Modes (`main.py`)

| Mode | Flag | Uses | GPU |
|------|------|------|-----|
| Train | *(default)* | `rl/policy.py` + `rl/schedule.py` (GRPO) | yes |
| Dry-run | `--dry-run` | `rl/inference.py` (frozen Ollama/OpenAI), full loop, no updates | no |
| Baseline | `--baseline` | `rl/baseline.py` best-of-N fixed actions, no LLM | no |

All three run against whichever defense `defense.mode` selects. Under
`algorithmic` the defender LLM is never called, so `--dry-run` makes one LLM call
per round (the attacker) instead of two, and `--baseline` scores its fixed
actions against the round's real algorithm rather than the built-in norm
heuristic (which is only used under `defense.mode: llm`).

All three honour `--rounds N` (an absolute budget overriding `fl.simulation_rounds`)
and `--config <path>`. `--debug` without `--rounds` caps how many rounds **this
run** adds on top of `rounds_done`, so debugging a resumed run still executes
rounds instead of exiting immediately.

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
  `rl.save_every` cadence and on exit. Older files with only `rounds_done` still
  load (the missing fields fall back: `round_index`→`rounds_done`, `controller`→a
  fresh schedule).
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
  actually uses), `n_malformed`, `budget`, `n_used`, `defense` (which algorithm
  faced the round, or `"llm"` — rounds are only comparable within one defense),
  plus a `train` sub-block (loss, mean reward, zero-advantage fraction).
- `logs/metrics/rounds.jsonl` + `summary.json` — ground-truth confusion / TPR /
  FPR / ASR / APR. `summary.json`'s `aggregate` covers every round; its
  `per_round` block is the retained tail (`MetricsTracker.keep_rounds`, default
  2000) — the full history is in `rounds.jsonl`.
- `logs/debug.json` (`--debug`) — structured event stream, capped at the most
  recent `DebugLogger.MAX_EVENTS` events (the file is rewritten in full each
  round, so an unbounded buffer made long debug runs O(n²)). `events_dropped`
  records how many were evicted.
- `logs/visualizations/report.html` — `python visualize_rounds.py`.
