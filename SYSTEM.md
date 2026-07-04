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

## Attacker contract (client selection + attack-plan DSL)

- **Input** (`agents/attacker_agent.build_user_prompt`): `round`,
  `current_global_accuracy`, `attack_goal`, `controllable_client_ids` (the pool it
  may touch), `max_poison_clients` (this round's budget), and
  `client_layer_details` — per-layer **statistics** (shape, mean, std, min, max,
  L2 norm, abs-mean) of **each pool client's** benign weights. No raw weights.
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
- **`rl/policy.py`**: one Unsloth `Llama-3.2-3B-Instruct` 4-bit base + two PEFT LoRA
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
