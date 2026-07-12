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
  `scale_neurons`, `blend_random`, `quantize`. `target` is `"all"`, a layer
  group (`"net.2"`), or a full key (`"net.4.weight"`). Operations apply in order.
- **Selection + application** (`agents/attacker_agent.select_and_apply` →
  `attack_ops.apply_plan`): filters ids to the pool, dedups, **truncates to the
  budget**; per chosen client deep-copies its benign weights, applies its plan,
  skips unknown ops / bad params / bad targets (counted, not fatal), then scrubs
  NaN/Inf and clamps to `±max_weight_abs`. PyTorch does all arithmetic — the LLM
  only emits the selection + plans. A chosen client with an empty/unusable plan
  falls back to benign weights and counts as *malformed* (a wasted client). If
  nothing parses, one benign client is selected as a fallback; parsing never
  raises. Shorthand inputs (`{"operations": [...]}` shared plan, `{"clients":[ids],
  "operations":[...]}`) are accepted for robustness.

## Defender contract (classification)

- **Input** (`agents/defender_agent.build_user_prompt`): per-client features
  from `detector/features.compute_client_features` — **only** features, never the
  ground truth.
  - Per layer (`net.2`, `net.4`): `l2_norm`, `rel_norm` (vs median), `cos_to_median`,
    `sign_agreement` (fraction of coords matching the median sign — catches
    sign-flip/targeted attacks that preserve norm).
  - Whole model: `l2_norm`, `rel_norm`, `cos_to_mean`, `max_pairwise_cos`
    (FoolsGold), `dnc_score` (SVD spectral outlier).
- **Output**: `{"clients": [{client_id, is_suspicious, confidence, reason}, ...]}`
  → one `DetectionVerdict` per client (missing/garbled entries default benign).

## Verifiable rewards (`rl/rewards.py`)

Both continuous, so GRPO group advantages don't collapse.

- **Attacker**: `α·clip(drop/target, -0.5, 1.5) + β·stealth − γ·malformed_frac
  − δ·client_cost + ζ·diversity`, with `drop = prev_acc − post_acc`, `target` from
  the goal, `stealth` = confidence-weighted evasion over the chosen clients,
  `client_cost = (n_used−1)/(n_compromisable−1)` (the **use-fewer-clients**
  penalty, 0 for a single client), and `diversity = 1 − mean pairwise cosine` of
  the chosen clients' perturbations (the **collaboration** bonus, 0 for a single
  client) — rewarding coordinated, distinct multi-client attacks over identical
  Sybil-like clones. See `rl/rewards.py` (`attacker_reward`,
  `perturbation_diversity`).
- **Defender** (train-time ground truth): confidence-weighted **soft-F1** vs the
  poisoned set (or `clip(TPR − λ·FPR)`).

## GRPO + schedule

- **`rl/grpo.py`**: sample `G` completions; reward each; advantage
  `A_i = (r_i − mean)/(std + ε)`; loss
  `mean_i[ −A_i·mean_t logπ(o_i,t) + β·mean_t KL_t ]` with the k3 KL estimator
  against the **frozen base model** (adapters disabled). Single-iteration ⇒ no
  clipping needed. Reports the zero-advantage-group fraction (stall signal).
- **`rl/policy.py`**: one Unsloth `Qwen2.5-1.5B-Instruct` base (bf16 LoRA by
  default; 4-bit QLoRA optional) + two PEFT LoRA adapters (`attacker`, `defender`).
  `set_adapter` selects the active policy; `disable_adapter` exposes the base as
  the KL reference.
- **`rl/schedule.py`**: freeze-and-alternate — train attacker `K_a` rounds
  (defender frozen, greedy), then defender `K_d` rounds (attacker frozen,
  greedy), repeat. The best-scoring sampled action is committed to advance the
  env. An **opponent league** snapshots adapters periodically and, with
  probability `league_prob`, makes a phase face a random past snapshot.
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
  `logs/round_data/round_NNN.json` (`attack_metadata.event="benign_fl_round"`)
  and `logs/debug.json`.

## Throughput (per-round cost)

This is a 1M+ round arms race, so each round is engineered to minimize GPU work:

- **Unsloth fast inference** (`rl.unsloth_fast_generation`) engages the fused
  fast-generation kernel for sampling; falls back to HF KV-cached generate, and
  again to a no-cache decoder, on any incompatibility.
- **Structured decoding** (`rl.stop_on_json`) stops each rollout as soon as it
  emits one complete top-level JSON object — both agents emit exactly one, so no
  tokens are wasted past the closing brace. `rl.max_new_tokens` (512) is only the
  ceiling. See `first_json_object_end` + the `StoppingCriteria` in `rl/policy.py`.
- **Batched opponent scoring**: an attacker round scores its `G` rollouts with a
  SINGLE batched frozen-defender generation (`LLMPolicy.generate_many` +
  `AttackerTurn.reward_batch`) instead of `G` sequential calls. (Defender rounds
  score by parsing only — no generation per rollout — so nothing to batch.)
- **Cheap reward eval**: the per-rollout reward accuracy is measured on a fixed
  `rl.reward_eval_samples` (1500) subsample of the preloaded test set
  (`FedServer.preload_test_set` caches it as one device tensor pair); the committed
  round uses the full 10k. A constant subsample bias cancels in the group-relative
  advantage, so the gradient is unchanged.

## Modes (`main.py`)

| Mode | Flag | Uses | GPU |
|------|------|------|-----|
| Train | *(default)* | `rl/policy.py` + `rl/schedule.py` (GRPO) | yes |
| Dry-run | `--dry-run` | `rl/inference.py` (frozen Ollama/OpenAI), full loop, no updates | no |
| Baseline | `--baseline` | `rl/baseline.py` best-of-N fixed actions, no LLM | no |

## Checkpoints & resume

- `checkpoints/global_model.pt`, `client_updates.pt`, `baseline.json` — Phase 1.
- `checkpoints/attacker_adapter/`, `checkpoints/defender_adapter/` — LoRA adapters
  (`adapter_model.safetensors` + `adapter_config.json`).
- `checkpoints/rl_progress.json` — rounds completed.
- Rerunning resumes: adapters + progress are reloaded; the env restarts from the
  Phase-1 baseline and replays forward.

## Logs

- `logs/system.log` — run log.
- `logs/round_data/round_NNN.json` — per round: `attack_goal`,
  `poisoned_client_ids`, `predicted_labels`, accuracies, `attacker_reward`,
  `defender_reward`, `learning_agent`, and a `train` sub-block (loss, mean reward,
  zero-advantage fraction).
- `logs/metrics/round_NNN.json` + `summary.json` — ground-truth confusion / TPR /
  FPR / ASR / APR.
- `logs/visualizations/report.html` — `python visualize_rounds.py`.
