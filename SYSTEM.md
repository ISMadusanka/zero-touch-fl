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