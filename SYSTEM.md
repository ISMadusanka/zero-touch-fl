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
  current round, which include  :
    - l2_norms: The L2 norm of each client's weight delta.
    - cosine_similarities: The cosine similarity of each update compared to the 
      global model.
    - pairwise_distances: The average pairwise L2 distance between the updates.
    
- attack_passed_through: A boolean flag indicating whether an attack evaded 
  detection in the previous round. The LLM is only prompted to adapt its strategy
   if this flag is True (i.e., its previous defense failed).

- recent_history: A summary of the defender's outcomes over the past 5 rounds. 
  Each entry contains:
    - Round number
    - Strategy used
    - Whether an attack passed through
    - Detailed verdicts produced for each client (suspicious flag, confidence,   and reason).

- similar_past_experiences: Up to 3 relevant past episodes retrieved from the defender's FAISS vector memory to help it adapt thresholds based on historical failures.


## defence and attack agents feedbacks
Read the agents and their current feedback contracts. Here's the mapping.

## What each agent currently sees

**Attacker** (from [SYSTEM_PROMPT in attacker_agent.py:25](agents/attacker_agent.py:25)): `baseline_accuracy`, `current_accuracy`, `was_detected` (single-round bool), `recent_history`, `similar_past_experiences`.

**Defender** (from [SYSTEM_PROMPT in defender_agent.py:22](agents/defender_agent.py:22)): `update_features` (l2_norms, cosines, pairwise), `attack_passed_through` (single-round bool), `all_clients_flagged`, `recent_history`, `similar_past_experiences`.

Both agents reason on *single-round binary signals* + 5-round history. They have no aggregate sense of how their regime is performing over time. That's exactly the gap the metrics can fill.

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


# Defend strategies

# Defense Strategies

| Scenario | Attacker Feedback | Defender Feedback | Gap? |
|---|---|---|---|
| Defender flags wrong client only | ✅ `was_detected=False` → keep strategy | ✅ `attack_passed=True` → adapt | No |
| Defender flags real attacker only | ✅ `was_detected=True` → adapt | ✅ `attack_passed=False` → keep strategy | No |
| Defender flags real attacker + innocents | ✅ `was_detected=True` → adapt | ⚠️ `attack_passed=False` → keep strategy | YES — defender keeps an over-aggressive strategy that harms model quality |
| Defender flags nobody | ✅ `was_detected=False` → keep strategy | ✅ `attack_passed=True` → adapt | No |

# client flaggiings by defend agent
how to flag it, what should we do rather than all client aggregation if all clients are flaged 