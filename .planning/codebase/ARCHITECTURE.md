<!-- refreshed: 2026-08-01 -->
# Architecture

**Analysis Date:** 2026-08-01

## System Overview

```text
┌──────────────────────────────────────────────────────────────────┐
│                   Phase 1: Honest FedAvg Training                │
│                  (45 rounds — all clients benign)                │
│              Checkpoint: global model, per-client weights        │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│          Phase 2: Adversarial RL Arms Race (Simulation)          │
├──────────────────────┬──────────────────────┬───────────────────┤
│  Attacker LLM        │  Defender LLM        │  Classical Defenses│
│ `agents/            │ `agents/             │ `server/defense_   │
│  attacker_agent.py` │  defender_agent.py`  │  ensemble.py`      │
│                     │                      │ (FLTrust, Krum,    │
│ Selects clients     │ Classifies benign/   │  DnC, DeFL)        │
│ + attack plans      │  malicious           │ (--freeze defender)│
└──────────┬───────────┴──────────┬───────────┴───────────┬────────┘
           │                      │                       │
           ▼                      ▼                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              FL Round Loop — FLArmsRaceEnv                       │
│           `rl/env.py`: Protocol, state, aggregation             │
├──────────────────────────────────────────────────────────────────┤
│  1. Honest updates (all N clients)           [clients/          │
│  2. Attacker chooses pool subset + plans     [agents/           │
│  3. Apply attack plans to benign weights                        │
│  4. Build updates list (poisoned ∪ honest)                      │
│  5. Extract per-client feature vectors       [detector/        │
│  6. Defender classifies OR algo. verdicts                       │
│  7. FedAvg aggregates non-flagged clients    [server/          │
│  8. Evaluate accuracy & compute rewards      [metrics/          │
│  9. GRPO update (training agent only)        [rl/grpo.py]      │
│ 10. Commit round, advance model state                           │
└───────────────────────┬──────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────────┐ ┌──────────────┐ ┌─────────────┐
│  Attacker RL  │ │ Defender RL  │ │ No Learner  │
│  Training     │ │ Training     │ │ (--dry-run) │
│  (3-round     │ │ (3-round     │ │             │
│   success     │ │  success     │ │             │
│   streak →    │ │  streak →    │ │             │
│   freeze,     │ │  freeze,     │ │             │
│   switch)     │ │  switch)     │ │             │
└───────────────┘ └──────────────┘ └─────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **Benign Client** | Honest local SGD training on data shard | `clients/benign_client.py` |
| **FedServer** | Holds global model, provides evaluation | `server/fed_server.py` |
| **FedAvgAggregator** | Weights averaging for non-flagged clients | `server/aggregation.py` |
| **Attacker LLM** | Selects clients from pool, generates per-client attack plans | `agents/attacker_agent.py` |
| **Attack Ops** | Parses & applies attack plan operators to client weights | `agents/attack_ops.py` |
| **Defender LLM** | Classifies each client benign/malicious from features | `agents/defender_agent.py` |
| **Defense Ensemble** | FLTrust / Multi-Krum / DnC / DeFL detectors (--freeze defender) | `server/defense_ensemble.py` |
| **Feature Extractor** | Computes per-client, per-layer statistical vectors | `detector/features.py` |
| **FL Environment** | Round protocol, state management, clean counterfactual | `rl/env.py` |
| **GRPO Step** | Single-iteration policy gradient update with KL penalty | `rl/grpo.py` |
| **Reward Computation** | Verifiable rewards from ground-truth poisoned set | `rl/rewards.py` |
| **RL Schedule** | Arms-race switching (attacker/defender alternation) | `rl/schedule.py` & `rl/switch.py` |
| **Turns** | Bind one FL round to one learning agent + frozen opponent | `rl/turns.py` |
| **Metrics** | TPR/FPR/ASR computation, confusion matrix | `metrics/compute.py` & `metrics/tracker.py` |

## Pattern Overview

**Overall:** Leader-follower (Stackelberg) adversarial RL arms race with verifiable rewards.

**Key Characteristics:**
- **Two phases:** Phase 1 (honest training checkpoint), Phase 2 (RL adversarial loop)
- **Partial insider threat:** Attacker controls only first `n_compromisable` clients (default 5 of 20); others always honest
- **Client selection:** Attacker **chooses which** of its pool to poison (≤ budget per round), optimizing to use fewer clients
- **Attack plan DSL:** 11 operators (`scale_delta`, `scale`, `sign_flip`, `mask`, `add_gaussian_noise`, `add_constant`, `permute`, `scale_neurons`, `blend_random`, `clip`, `quantize`)
- **Delta-space vs weight-space:** `scale_delta` (delta-space, aimable) dominates; others (weight-space, loss-prone) available for shaping
- **Pure agents:** Attacker/Defender are prompt builders + output parsers; generation is owned by policy or inference
- **Verifiable rewards:** Ground truth = attacker's actually chosen poisoned set → both agents get exact reward signal
- **Arms-race schedule:** Success-gated switching with min/max phase lengths, curriculum learning via league snapshots
- **Frozen base + LoRA:** Qwen2.5-3B-Instruct base, separate LoRA adapters per agent
- **GRPO:** Single-iteration group-relative policy gradient; no critic network, explicit KL penalty
- **Clean counterfactual:** Attack damage measured against per-round clean run (no poison), not absolute accuracy drop

## Layers

**Data Layer:**
- Purpose: Load MNIST and partition to clients
- Location: `data/mnist_loader.py`
- Contains: IID partition, non-IID partition (FLTrust bias-q scheme)
- Depends on: Torch, MNIST dataset
- Used by: FLArmsRaceEnv (during benign retraining), BenignClient instantiation

**Client Layer:**
- Purpose: Benign local SGD training on client data shards
- Location: `clients/benign_client.py`
- Contains: BenignClient (local training loop, SGD optimizer, loss tracking)
- Depends on: PyTorch, data loaders
- Used by: FLArmsRaceEnv (honest update generation in each round)

**Server/Aggregation Layer:**
- Purpose: Global model state, evaluation, weight aggregation
- Location: `server/fed_server.py` (model state, eval), `server/aggregation.py` (averaging)
- Contains: FedServer (model holder), FedAvgAggregator (coordinate-wise mean)
- Depends on: PyTorch, test loader
- Used by: FLArmsRaceEnv, metrics/benchmarking

**Attack/Defense Agent Layer:**
- Purpose: LLM action selection (attack plans, classification) via prompt building & parsing
- Location: `agents/attacker_agent.py`, `agents/defender_agent.py`, `agents/attack_ops.py`
- Contains: AttackerAgent (client selection + per-client plan generation), DefenderAgent (per-client benign/malicious classification), attack operator interpreter
- Depends on: LLM backend (via policy/inference), JSON parsing, PyTorch (for apply_plan)
- Used by: RL turns/rollout loops, inference paths

**Feature/Detector Layer:**
- Purpose: Extract statistics for defender input; algorithmically detect anomalies (--freeze defender)
- Location: `detector/features.py`, `server/defense_ensemble.py`
- Contains: compute_client_features (per-layer + whole-model stats), FLTrust, Multi-Krum, DnC, DeFL
- Depends on: Update dictionaries, numpy for statistics
- Used by: Defender LLM prompt building, algorithmic defense verdicts, metrics

**RL/Training Layer:**
- Purpose: Environment, reward, policy training, schedule
- Location: `rl/` directory (env.py, grpo.py, rewards.py, turns.py, schedule.py, policy.py, inference.py)
- Contains: FLArmsRaceEnv (round protocol), GRPO step, reward functions, turns (agent bindings), schedule (switching logic), policy (LoRA trainer)
- Depends on: PyTorch, HF Transformers, Unsloth, vLLM, environment state
- Used by: main.py round loop, benchmarking

**Storage/Checkpoint Layer:**
- Purpose: Save/load model state, adapter weights, progress tracking
- Location: `storage/checkpoint.py`
- Contains: state save/load, adapter save/load, progress tracking
- Depends on: PyTorch, file I/O
- Used by: main.py (resume/fresh), GRPO training (checkpoints)

**Metrics/Instrumentation Layer:**
- Purpose: Compute TPR/FPR/ASR/APR, track round-by-round metrics, generate logs
- Location: `metrics/compute.py`, `metrics/tracker.py`, `core/debug.py`
- Contains: MetricsTracker, compute functions, structured debugging
- Depends on: Ground-truth labels, predictions, environment state
- Used by: main.py, reward computation, logging

## Data Flow

### Primary Round Flow (Leader-Follower, Stackelberg)

1. **env.begin_round()** (`rl/env.py:begin_round`)
   - Build honest updates for all N clients (from global model)
   - Expose attacker's controllable pool + this round's poison budget
   - Return RoundContext with pool_benign weights, budget, current accuracy

2. **Attacker LLM action** (training: G-rollout GRPO loop; committed: greedy inference)
   - AttackerAgent.build_user_prompt() serializes pool stats, budget, goal
   - Generator samples N (G rollouts in training, 1 in committed)
   - AttackerAgent.select_and_apply() parses JSON, filters to pool, truncates to budget, applies operators
   - Return: poisoned_by_client dict, chosen_ids list, n_malformed count

3. **Build full updates** (`rl/env.py:build_updates`)
   - Merge poisoned_by_client (from attacker) + honest updates
   - Return ModelUpdate objects for all clients

4. **Feature extraction** (`detector/features.py:compute_client_features`)
   - Per-client, per-layer statistics (no decisions)
   - Inputs: updates dict, global weights
   - Output: feature dict keyed by client_id

5. **Defender action** (if LLM enabled; else algo ensemble)
   - **LLM path:** DefenderAgent.build_user_prompt() formats features
     - Generator samples N
     - DefenderAgent.parse() extracts per-client verdicts
   - **Algo path:** defense.verdicts() runs FLTrust/Multi-Krum/DnC/DeFL
   - Return: DetectionVerdict per client

6. **Evaluate (no commit)** (`rl/env.py:evaluate_updates`)
   - Apply verdicts to filter clients
   - FedAvg aggregate
   - Evaluate accuracy
   - Return: post_accuracy, client drop count
   - Used for: rollout scoring (training)

7. **Compute clean reference** (`rl/env.py:clean_reference_accuracy`)
   - Run FedAvg with NO poison (hypothetical)
   - Evaluate accuracy under same defense
   - Used for: measuring attacker damage (drop = clean - post)

8. **Compute rewards** (`rl/rewards.py`)
   - **Attacker:** `α·drop_term + β·stealth_gate(stealth) − γ·malformed_frac − δ·client_cost + ζ·diversity`
     - drop = clean_ref − post_accuracy
     - stealth = detector confidence over chosen clients, gated on achieving target drop
     - client_cost penalizes using more than 1 client
     - diversity rewards distinct, coordinated multi-client attacks
   - **Defender:** `1 − (FPR·w_fp + FNR·w_fn)` balanced accuracy
   - Return: scalar reward per rollout

9. **GRPO update (training rounds only)** (`rl/grpo.py:grpo_step`)
   - Compute group-relative advantages (z-scores within G)
   - Per-rollout loss = `−advantage · mean_t logπ + β·KL`
   - Backward pass on adapter only
   - Optional: resample on zero-advantage, skip on degenerate group

10. **Commit round** (`rl/env.py:commit`)
    - Set env.poisoned_ids to chosen_ids
    - Apply verdicts (filter flagged clients)
    - FedAvg aggregate
    - Update global model
    - Advance clean counterfactual state
    - Log round (RoundLog)

### Arms-Race Schedule (Between-Phase Flow)

1. **Attacker trains** (frozen defender)
   - Round loop above, attacker gets reward, GRPO updates
   - Success gate: 3 consecutive rounds where `drop ≥ target_drop * win_fraction`

2. **On attacker win** → phase freeze + benign interlude
   - Run ONE honest FL round (all clients, no attacker/defender)
   - Refresh per-client benign weights, update global
   - Checkpoint

3. **Defender trains** (frozen attacker)
   - Same round loop, defender gets reward, GRPO updates
   - Success gate: 3 consecutive rounds of high TPR (configurable)

4. **On defender win** → phase freeze + benign interlude + repeat

**State Management:**
- Global model weights (updated per round)
- Per-client benign weights (frozen from Phase 1, unless benign_retrain_each_round)
- LoRA adapter states (per learner, checkpointed every N rounds)
- League snapshots (ring buffer of past adapter states for curriculum)
- Round log & metrics (streamed to disk)

## Key Abstractions

**ModelUpdate:**
- Purpose: Represents one client's weight submission
- Examples: `clients/benign_client.py:train()` returns ModelUpdate; attack_ops applies plan to create poisoned ModelUpdate
- Pattern: Immutable after creation; client_id + state_dict + metadata

**DetectionVerdict:**
- Purpose: Represents defender's classification for one client
- Examples: DefenderAgent.parse() returns list of DetectionVerdict; defense ensemble produces them
- Pattern: client_id, is_suspicious (bool), confidence (float 0-1), reason (optional)

**RoundContext:**
- Purpose: Read-only observation given to attacker agent
- Examples: env.begin_round() returns RoundContext
- Pattern: Exposes pool_benign, budget, current_accuracy, attack_goal; poisoned_ids set at commit

**AttackerAgent/DefenderAgent:**
- Purpose: Pure prompt builders + output parsers
- Examples: build_user_prompt() serializes state; parse() extracts JSON; select_and_apply() applies operators
- Pattern: No internal LLM calls; generation owned by policy/inference layer

**FLArmsRaceEnv:**
- Purpose: Encapsulates round protocol, state, aggregation, eval
- Examples: begin_round() → build_updates() → evaluate_updates() → set_committed_poison() → commit()
- Pattern: Stateful object; per-round observations + committed state

**AttackerTurn / DefenderTurn:**
- Purpose: Bind one FL round to one learning agent + frozen opponent
- Examples: rl/turns.py; used by GRPO sampler to score G rollouts
- Pattern: Implements turn.messages() (system/user prompts) and turn.reward(completion_text)

## Entry Points

**main.py** (`main.py`)
- Triggers: `python main.py --env linux` (full GRPO), `--freeze defender`, `--dry-run`, `--baseline`, `--fresh`
- Responsibilities:
  - Parse args (env, freeze, rounds, etc.)
  - Load data, instantiate clients, server, agents
  - Create FLArmsRaceEnv with optional defense ensemble
  - Instantiate policy (LoRA trainer) or inference generator
  - Create schedule (attack → defend → ... cycle)
  - Call main loop (grpo_main or run_inference)
- Dependencies: All major components

**run_inference()** (`rl/inference.py`)
- Triggers: --dry-run, --baseline modes
- Responsibilities: Frozen-LLM round loop, no weight updates
- Used for: Sanity-checking plumbing on CPU, validating prompts/parsing

**grpo_main()** (in main.py)
- Triggers: Normal run (--env linux) or single-agent training (--freeze)
- Responsibilities: RL round loop with GRPO training
- Loop: for each phase (attacker or defender): sample G rollouts, compute rewards, GRPO step, checkpoint, check success gate

**infer.py** (`infer.py`)
- Triggers: Manual one-off inference (not part of main training loop)
- Responsibilities: Single attacker or defender action for testing

## Architectural Constraints

- **Partial insider threat:** Attacker may poison only `n_compromisable` of `n_clients` (strict minority), ensuring honest majority for robust feature references
- **Ground-truth reward:** Both agents' rewards come from the **committed** poisoned set (the attacker's actual chosen clients), not a hypothetical all-poisoned counterfactual
- **Frozen base model:** Qwen2.5-3B-Instruct base is never finetuned; only LoRA adapters per agent are trained (memory efficiency)
- **No episodic memory:** LLMs learn purely via RL + in-context exemplars in the prompt; no feedback loop of past rounds (simplicity, reproducibility)
- **Per-client distinct plans:** Attack plans are per-client, not shared Sybils; detected by cosine similarity / sign-agreement metrics
- **League curriculum:** Snapshots of past adapters in a ring buffer (max 10) create a moving-target curriculum
- **Single-phase defense:** In --freeze defender mode, only ONE algorithm judges each round (not all unioned) to avoid flagging too many honest clients
- **Clean counterfactual:** Attacker damage is measured against the per-round clean run (same global model, same defense, no poison), not the previous global accuracy

## Anti-Patterns

### Naive weight-space operators

**What happens:** Operators like `scale`, `add_constant`, `add_gaussian_noise` are applied directly to the client's weight vector `W`, so the server sees `W' − global` as the submitted update. Since W ≈ global + small_delta, an absolute edit looks like a giant update.

**Why it's wrong:** On a positively-homogeneous ReLU MLP, scaling all weights by factor f buys `rel_update ~ |f − 1|`, which is enormous for tiny model damage. At `f=1.5` on an honest rel_update of 0.2, the submitted rel_update is ~0.3 (3x honest size) but model predictions barely change — maximum suspicion, no leverage.

**Do this instead:** Use `scale_delta` (`W' = global + factor·(W − global)`), which pushes along the direction the round is learning in. It buys ~100× more damage per unit of update norm and is **aimable**: the submitted `rel_update` is exactly `|factor|` times the honest `rel_update` shown in the prompt.

### Sybil-like client cloning

**What happens:** When poisoning multiple clients, give them identical or near-identical plans.

**Why it's wrong:** The defender extracts cosine similarity and sign-agreement metrics per-client. Identical clones have `max_pairwise_cos ≈ 1.0` (very suspicious) and are flagged as Sybils.

**Do this instead:** Give each client a **distinct, coordinated role**. E.g., one client scales the first layer while another shifts a different layer; together they move the model in a consistent direction but individually look independent.

### Unioning all defense algorithms

**What happens:** In --freeze defender mode, applying all four algorithms (FLTrust, Multi-Krum, DnC, DeFL) and flagging any client that **any** algorithm flags.

**Why it's wrong:** On the repo's Phase-1 checkpoint, a **clean round with no attacker** flags 14/20 honest clients (fltrust 12, multikrum 1, dnc 1, defl 2), leaving FedAvg only 6 clients and dropping accuracy 0.782 → 0.757. This removes all room for the attacker to fit an attack through, and the accuracy swing from *which* honest clients happen to be dropped (sd ≈ 0.012) dwarfs the real attack damage (≈ 0.0003), making the reward signal mostly noise.

**Do this instead:** Judge with **one algorithm per round** (rotate through them, `defense.mode: single`). Clean accuracy stays within 0.01 of the undefended baseline, preserving a gap for attacks and clean reward signal.

### Defending with ground-truth client set

**What happens:** Showing the defender the true poisoned client ids or expecting it to learn from episodic feedback.

**Why it's wrong:** The production defender doesn't know which clients are poisoned — it sees only features. Training on ground truth breaks the abstraction and makes the learned policy brittle to out-of-distribution attacks.

**Do this instead:** Build the prompt from **features only** (per-layer stats, no labels). The reward is computed from ground truth (training signal), but the observation is oracle-free.

## Error Handling

**Strategy:** Graceful degradation with explicit fallbacks.

**Patterns:**
- **Unparseable LLM output:** extract_json() handles invalid JSON; extract_selection() returns None if no valid JSON found; both fall back to benign (no-op) behavior, counted as malformed
- **Invalid operator targets:** If an operator target (layer name, full key) doesn't exist in the client's weights, skip it and increment skip counter (not fatal)
- **NaN/Inf in computed weights:** Clamp to `±max_weight_abs` after applying all operators (e.g., `max_weight_abs=100.0`)
- **All clients flagged:** env.commit() keeps the previous global model (no update) rather than aggregating empty set
- **Zero-advantage group:** GRPO detects degeneracy (all rollouts same reward); optionally resamples at higher temperature or skips the gradient step (guards against un-learning)

## Cross-Cutting Concerns

**Logging:** `logging` module + structured `core/debug.py` debug logging. Round-by-round data written to `logs/round_data/<round_id>.json`. Summary stats to `logs/system.log`.

**Validation:** 
- Attacker's chosen client ids are filtered to the controllable pool; budget is enforced
- Operator params validated before apply (e.g., `factor` is a number)
- Defender verdicts checked for presence of all client ids (missing default to benign)
- Feature vectors checked for NaN/Inf

**Authentication/Authorization:** Not applicable (single-user research testbed).

**Determinism/Reproducibility:** RNG seeded from `fl.poison_seed` controls non-IID partition, budget sampling, algorithm selection. LoRA initialization is seeded via pytorch seed. Inference uses controlled temperature.

---

*Architecture analysis: 2026-08-01*
