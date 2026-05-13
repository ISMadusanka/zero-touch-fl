# Memory
## Short-term memory (recent_history): 
The LLM is given exactly the last 5 rounds of history. (In the code: "recent_history": self.history[-5:])

## Long-term associative memory (similar_past_experiences): 
The LLM is also given the top 3 most similar past rounds retrieved from the FAISS vector database. The agent creates a vector embedding of the current situation (accuracies, detections, etc.) and asks the FAISS index to find past rounds that looked similar.


# Agent feedbacks
## Attacker Agent Feedback

- baseline_accuracy: The model's clean accuracy from Phase 1, serving as a
  baseline reference.

- current_accuracy: The test accuracy of the global model after the last
  round's aggregation (to measure if the previous attack successfully degraded
  performance).

- was_detected: A boolean flag indicating whether the attacker's last attack
  was caught by the defender. The LLM is only prompted to adapt its strategy
  if this flag is True.

- recent_history: A summary of the attacker's outcomes over the past 5 rounds.
  
  Each entry contains:
    - Round number
    - Strategy used
    - Whether it was detected
    - Accuracy after the round

- similar_past_experiences: Up to 3 relevant past episodes retrieved from the
    attacker's FAISS vector memory (based on a hash of the current state) to help
  it recall strategies that worked in similar situations.

## Defender Agent Feedback
- update_features: Statistical features computed from all client updates in the 
  current round, which include:
    - l2_norms: The L2 norm of each client's weight delta.
    - cosine_similarities: The cosine similarity of each update compared to the
      mean update (not the global model).
    - dnc_scores: Spectral outlier scores (SVD projection²) per client.
    - fltrust_scores: ReLU(cosine_similarity) trust scores per client.
    - foolsgold_max_cs: Maximum pairwise cosine similarity per client.
    - mean_pairwise_distance: Average pairwise L2 distance between updates.

- attack_passed_through: A boolean flag indicating whether an attack evaded 
  detection in the previous round. The LLM is only prompted to adapt its strategy
  if this flag is True (i.e., its previous defense failed).

- all_clients_flagged: A boolean flag indicating whether the previous round's
  thresholds were so strict that every client was flagged, causing the round
  to be skipped. When True, the LLM must loosen thresholds.

- tpr_recent: True positive rate (recall) over the last 5 rounds (0.0–1.0).

- fpr_recent: False positive rate over the last 5 rounds (0.0–1.0).

- accuracy_preservation_rate: current_accuracy / baseline_accuracy (0.0–1.0).

- recent_history: A summary of the defender's outcomes over the past 5 rounds. 
  Each entry contains:
    - Round number
    - Strategy used
    - Whether an attack passed through
    - Whether all clients were flagged
    - Detailed verdicts produced for each client (suspicious flag, confidence, and reason)
    - tpr_recent, fpr_recent, and accuracy_preservation_rate at that point

- similar_past_experiences: Up to 3 relevant past episodes retrieved from the defender's FAISS vector memory to help it adapt thresholds based on historical failures.


## defence and attack agents feedbacks
Read the agents and their current feedback contracts. Here's the mapping.

## What each agent currently sees

**Attacker** (from [SYSTEM_PROMPT in attacker_agent.py:25](agents/attacker_agent.py:25)): `baseline_accuracy`, `current_accuracy`, `was_detected` (single-round bool), `attack_success_rate_recent`, `fpr_recent`, `accuracy_preservation_rate`, `recent_history`, `similar_past_experiences`.

**Defender** (from [SYSTEM_PROMPT in defender_agent.py:22](agents/defender_agent.py:22)): `update_features` (l2_norms, cosine_similarities, dnc_scores, fltrust_scores, foolsgold_max_cs, mean_pairwise_distance), `attack_passed_through` (single-round bool), `all_clients_flagged`, `tpr_recent`, `fpr_recent`, `accuracy_preservation_rate`, `recent_history`, `similar_past_experiences`.

Both agents now receive windowed aggregate KPIs (over the last 5 rounds) in addition to the single-round binary signals and 5-round history.

## Recommendation

| Metric | Attacker | Defender | Why |
|---|---|---|---|
| **Attack Success Rate** (cumulative / windowed) | ✅ | ❌ | Attacker's persistent outcome KPI; tells it "my regime is being shut down" vs "I'm consistently slipping through." For the defender, ASR = 1 − TPR — redundant. |
| **TPR / Recall** (cumulative / windowed) | ❌ | ✅ | Defender's core "am I catching them" KPI. For the attacker it's just 1 − ASR — drop one to avoid redundancy. |
| **FPR** (cumulative / windowed) | ✅ | ✅ | **Defender:** explicitly told to "minimize false positives"; right now it only learns about FPs via the extreme `all_clients_flagged` signal — surfacing FPR closes the loop directly. **Attacker:** adversarial intel — a sloppy defender (high FPR) means aggressive attacks blend in; a precise defender (low FPR) forces subtlety. |
| **Accuracy Preservation Rate** | ✅ | ✅ | **Attacker:** damage gauge, normalized against baseline so the LLM doesn't have to compute `current / baseline`. **Defender:** collateral-damage gauge — over-aggressive flagging skips aggregation and tanks accuracy; APR makes that visible. |
| Raw TP/FN/FP/TN counts | ❌ | ❌ | Already encoded in the rates above. At single-round granularity (one attacker, n−1 honest) they're noisy and clutter the prompt. |
| Recall as a separate metric | ❌ | ❌ | Alias of TPR — pick one name in the prompt so the LLM doesn't think they're independent signals. |

## Two practical notes on how to surface them

1. **Use a trailing window, not per-round.** With one attacker per round, single-round TPR is just 0/1 — no signal. Aggregating over the last 5–10 rounds (matching the existing `history[-5:]` convention) gives the LLM a trajectory.

2. **Embed per-round metrics into each history entry** *and* surface the windowed aggregate at the top of the prompt. The LLM already sees 5 history entries — adding `attack_success`, `apr`, etc. to each gives it the trend for free, while the top-level aggregate gives the headline number.

## Net feedback contract

- **Attacker feedback**: keep current fields + add `attack_success_rate_recent`, `fpr_recent`, `accuracy_preservation_rate` (current and recent average).
- **Defender feedback**: keep current fields + add `tpr_recent`, `fpr_recent`, `accuracy_preservation_rate`.

This gives each agent exactly the metrics that drive its own objective, with no cross-redundancy between TPR and ASR.


# Defense Strategies

## Available Methods (all adaptive — no hardcoded thresholds)

All defenses use: `threshold = median ± sensitivity × MAD`
where MAD = Median Absolute Deviation (robust to outliers).
The LLM tunes a single `sensitivity` parameter (default 2.0).

| Method | Paper | What it Detects | Threshold Formula |
|--------|-------|-----------------|-------------------|
| `norm_threshold` | Sun et al. (2019) | Large-magnitude updates | `median(norms) + s×MAD(norms)` |
| `dnc` | Shejwalkar & Houmansadr (NDSS 2021) | Spectral outliers (SVD) | `median(scores) + s×MAD(scores)` |
| `fltrust` | Cao et al. (NDSS 2021) | Low-trust direction divergence | `median(TS) - s×MAD(TS)` |
| `foolsgold` | Fung et al. (RAID 2020) | Sybil/colluding similarity | `median(weights) - s×MAD(weights)` |
| `flame` | Nguyen et al. (USENIX Sec 2022) | Clustering-based outliers | HDBSCAN majority cluster |

## Adaptation Feedback Matrix

| Scenario | Attacker Feedback | Defender Feedback | Gap? |
|---|---|---|---|
| Defender flags wrong client only | ✅ `was_detected=False` → keep strategy | ✅ `attack_passed=True` → adapt | No |
| Defender flags real attacker only | ✅ `was_detected=True` → adapt | ✅ `attack_passed=False` → keep strategy | No |
| Defender flags real attacker + innocents | ✅ `was_detected=True` → adapt | ⚠️ `attack_passed=False` → keep strategy | YES — defender keeps an over-aggressive strategy that harms model quality |
| Defender flags nobody | ✅ `was_detected=False` → keep strategy | ✅ `attack_passed=True` → adapt | No |

# client flaggiings by defend agent
how to flag it, what should we do rather than all client aggregation if all clients are flaged 