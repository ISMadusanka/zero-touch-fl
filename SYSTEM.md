# System Architecture — LLM-Direct Adversarial FL with GRPO

This document describes the redesigned system: the round loop, the attacker /
defender contracts, the verifiable rewards, the GRPO training schedule, and the
checkpoint layout. It supersedes the old feedback/episodic-memory design.

## Components

| Layer | Module | Role |
|-------|--------|------|
| Model | `model/mnist_net.py` | `MnistNet`, ~970 params. State_dict keys: `net.2.weight [16,49]`, `net.2.bias [16]`, `net.4.weight [10,16]`, `net.4.bias [10]`. The schema both LLMs operate over. |
| Data | `data/mnist_loader.py` | MNIST load + per-client IID/non-IID partition. |
| Clients | `clients/benign_client.py` | Honest local SGD → `ModelUpdate`. |
| Server | `server/fed_server.py`, `server/aggregation.py` | Global model + eval; FedAvg over non-flagged clients. |
| Features | `detector/features.py` | Per-client, per-layer statistical feature vectors (no decisions). |
| Attacker | `agents/attacker_agent.py` + `agents/weight_codec.py` | Prompt + parse of raw poisoned weights. |
| Defender | `agents/defender_agent.py` | Prompt + parse of per-client benign/malicious labels. |
| RL | `rl/*` | Environment, rewards, turns, GRPO, schedule, policy, baseline, inference. |
| Metrics | `metrics/*` | Ground-truth confusion/TPR/FPR/ASR/APR (research + reward source). |

## Round loop (Phase 2)

Leader–follower (Stackelberg): the attacker moves, the defender best-responds to
the realized updates.

```
reset env from Phase-1 checkpoint (per-client benign weights, global, baseline acc)
for each round:
  1. poisoned_ids = rng.sample(clients, k)        # k = clamp(round(poison_fraction·n), 1, (n-1)//2)
  2. honest updates for all clients               # retrain from global, or replay Phase-1 weights
  3. ATTACKER LLM → raw poisoned weights for poisoned_ids   (input: round, benign weights, acc, goal)
  4. build full update list (poisoned ∪ honest)
  5. detector/features → per-client per-layer stat vectors
  6. DEFENDER LLM → benign/malicious label + confidence per client   (input: features ONLY)
  7. FedAvg over clients labelled benign → new global (None if all flagged → keep prev)
  8. evaluate global accuracy
  9. attacker_reward + defender_reward (ground truth = poisoned_ids)   ← train-time only
 10. GRPO update for the learning agent; write round log + metrics
```

`k` is clamped to keep a strict benign majority, because the defender's feature
references (coordinate-wise median, MAD) assume most clients are honest.

## Attacker contract (raw weights)

- **Input** (`agents/attacker_agent.build_user_prompt`): `round`,
  `current_global_accuracy`, `attack_goal`, `poisoned_client_ids`,
  `output_schema` (exact layer shapes + lengths), and `benign_weights`
  (per-poisoned-client flat arrays).
- **Output**: a single JSON object keyed by client id; each value is
  `{layer_key: [flat floats of the exact length]}`.
- **Hardening** (`agents/weight_codec.parse_round`): validates shape/count,
  scrubs NaN/Inf, clamps to `±max_weight_abs`, accepts nested or flat arrays,
  and on any unrecoverable block **falls back to that client's benign weights**
  and counts it `malformed`. Parsing never raises; malformed output costs reward
  instead of crashing training.

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

- **Attacker**: `α·clip(drop/target, -0.5, 1.5) + β·evasion_rate − γ·malformed_frac`,
  with `drop = prev_acc − post_acc`, `target` from the goal, `evasion_rate` =
  fraction of poisoned clients not flagged.
- **Defender** (train-time ground truth): confidence-weighted **soft-F1** vs the
  poisoned set (or `clip(TPR − λ·FPR)`).

## GRPO + schedule

- **`rl/grpo.py`**: sample `G` completions; reward each; advantage
  `A_i = (r_i − mean)/(std + ε)`; loss
  `mean_i[ −A_i·mean_t logπ(o_i,t) + β·mean_t KL_t ]` with the k3 KL estimator
  against the **frozen base model** (adapters disabled). Single-iteration ⇒ no
  clipping needed. Reports the zero-advantage-group fraction (stall signal).
- **`rl/policy.py`**: one Unsloth `gpt-oss-20b` 4-bit base + two PEFT LoRA
  adapters (`attacker`, `defender`). `set_adapter` selects the active policy;
  `disable_adapter` exposes the base as the KL reference.
- **`rl/schedule.py`**: freeze-and-alternate — train attacker `K_a` rounds
  (defender frozen, greedy), then defender `K_d` rounds (attacker frozen,
  greedy), repeat. The best-scoring sampled action is committed to advance the
  env. An **opponent league** snapshots adapters periodically and, with
  probability `league_prob`, makes a phase face a random past snapshot.

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
