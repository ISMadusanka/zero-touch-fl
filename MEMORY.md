# Memory System

## Overview
The agent uses hybrid memory: **short-term** for immediate context within an episode, **long-term** for cross-episode pattern learning via FAISS vector database.

---

## Short-term Memory (recent_history)

**What it stores:**
- Last 5 rounds from current episode
- Per-round: accuracy, attack_detected, agent_action, anomaly_score, accuracy_delta, defense_mechanism, aggregation_method

**Details:**
- Type: Fixed-size deque (max 5 rounds)
- Location: `self.history` in agent classes
- Access: `self.history[-5:]`
- Lifetime: Per-episode only (cleared on episode end)
- Persistence: NOT saved to disk
- Footprint: ~10-25 KB per episode

---

## Long-term Memory (similar_past_experiences)

**What it stores:**
- Top 3 most similar past rounds via FAISS vector database
- Per-experience: episode_id, round, accuracy, attack_detected, past_action, past_outcome, anomaly_score, similarity_score
- Dense vector embeddings of round state features

**Details:**
- Type: FAISS vector index (Facebook AI Similarity Search)
- Location: `storage/vector_store.py`
- Retrieval: K=3 nearest neighbors by cosine similarity
- Embedding source: [accuracy, accuracy_delta, attack_detected, anomaly_score, num_clients, ...]
- Lifetime: Persists across episodes
- Capacity: ~100k rounds before degradation
- Persistence: Saved to disk as checkpoint after each episode
- Footprint: ~3-6 KB per embedding (300-600 MB for 100k experiences)

**Query process:**
1. Extract current round features → create embedding
2. Search FAISS for k=3 nearest neighbors
3. Return ranked by similarity (highest first)

**Index types:**
- Flat: For <50k experiences (O(n) query)
- HNSW: For >50k experiences (O(log n) query, faster)

---

## Configuration (base.yaml)

```yaml
short_term_memory:
  capacity: 5
  enabled: true

long_term_memory:
  enabled: true
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  index_type: "flat"              # Switch to "hnsw" for >50k experiences
  retrieval_count: 3
  similarity_threshold: 0.75
  checkpoint_dir: "./storage/checkpoints"
  max_capacity: 100000            # Trigger pruning above this
```

---

## Memory Flow

**Per Round:**
1. Fetch short-term: last 5 rounds from `self.history`
2. Create embedding from current state features
3. Query FAISS: retrieve top 3 similar past rounds
4. Combine both → format for LLM prompt
5. LLM makes decision based on combined memory
6. Execute action → observe outcome

**End of Episode:**
1. Create experience record (features + outcome + action)
2. Append to short-term buffer (auto-evict if >5)
3. Embed experience → add to FAISS index
4. Save checkpoint: `faiss_index.bin` + `metadata.json`

---

## Monitoring

| Metric | Expected | Alert |
|--------|----------|-------|
| Short-term size | 0-5 rounds | Always ≤ 5 |
| Long-term size | 0-100k | > 100k triggers pruning |
| Query similarity | 0.7-0.95 | < 0.6 = weak match |
| Checkpoint size | 0-600 MB | > 800 MB = cleanup |
| Query latency | 1-50 ms | > 100 ms = performance issue |

---

## Summary

### Short-Term Memory:
✓ Last 5 rounds from current episode  
✓ Immediate trend detection  
✗ NOT persistent (cleared on episode end)  

### Long-Term Memory:
✓ Cross-episode patterns (thousands of past rounds)  
✓ Vector embeddings + similarity search  
✓ Persistent (disk checkpoint)  
✓ Efficient retrieval (FAISS)  

**Result:** Agent learns from immediate context + historical patterns for informed decisions.

---

## Implementation

**Code Integration:**
- Defender Agent: `agents/defender_agent.py` - uses both memory types for defense decisions
- Attacker Agent: `agents/attacker_agent.py` - uses both memory types for attack strategy
- Vector Store: `storage/vector_store.py` - manages FAISS lifecycle + checkpoints

**Error Handling:**
- First round (no history): Return empty → agent handles gracefully
- FAISS failure: Log warning → continue with short-term only
- Index full: Trigger pruning (remove oldest low-similarity experiences)
- Missing checkpoint: Initialize new index

**Pruning Strategy (when max_capacity reached):**
1. Remove experiences with similarity < threshold
2. Remove experiences older than N episodes  
3. Keep only top-performing actions
