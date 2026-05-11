# Memory System

The system uses **hybrid memory**: short-term (recency-based) and long-term (similarity-based). Both the Attacker and Defender agents maintain independent instances of each.

---

## Short-Term Memory (`recent_history`)

A sliding window of the **last 5 round outcomes** from `self.history[-5:]`, providing immediate temporal context to the LLM.

### Attacker Agent — stored fields per round

```python
{
    "round":          int,    # Global round number (e.g. 4, 5, 6)
    "strategy": {
        "attack_type": str,   # "sign_flip" | "noise_injection" | "scaling"
        "params":      dict,  # e.g. {"scale": 3.0} or {}
        "reasoning":   str    # LLM's explanation
    },
    "was_detected":   bool,   # True if defender caught this attack
    "accuracy_after": float   # Global model accuracy after this round
}
```

### Defender Agent — stored fields per round

```python
{
    "round":    int,          # Global round number
    "strategy": {
        "method":    str,     # "norm_threshold" | "cosine_threshold" | "combined"
        "params":    dict,    # e.g. {"threshold": 2.0} or {"norm_threshold": 1.5, "cosine_threshold": 0.6}
        "reasoning": str      # LLM's explanation
    },
    "attack_passed_through": bool,  # True if attacker evaded detection (defense failed)
    "verdicts": [                   # One entry per client
        {
            "client_id":  int,
            "suspicious": bool,
            "confidence": float,    # 0.0–1.0
            "reason":     str       # e.g. "norm_ratio=2.5>2.0"
        }
    ]
}
```

### Properties

- **Window size:** 5 entries sent to LLM (full list grows unbounded)
- **Selection:** Chronological (most recent)
- **Persistence:** In-memory only — lost on restart
- **Purpose:** Detect recent trends (e.g. repeated detection, accuracy drift)

---

## Long-Term Memory (`similar_past_experiences`)

A FAISS vector index storing **every past round** as a 384-dim embedding. At query time, the **top 3 most similar** past rounds are retrieved and sent to the LLM.

### What is stored per entry

Each entry consists of two parts:

**Part 1 — Vector embedding (for similarity search):**
- The round outcome dict is serialized to JSON (`json.dumps(data, sort_keys=True)`)
- Encoded by `all-MiniLM-L6-v2` sentence-transformer → **384-dim `float32` vector**
- Stored in the FAISS `IndexFlatL2` index

**Part 2 — Metadata (the actual data returned to the LLM):**

Attacker Agent — stored parameters per entry:
```python
{
    "round":          int,    # Global round number
    "strategy": {
        "attack_type": str,   # "sign_flip" | "noise_injection" | "scaling"
        "params":      dict,  # e.g. {"scale": 3.0} or {}
        "reasoning":   str    # LLM's explanation for choosing this attack
    },
    "was_detected":   bool,   # Whether the defender caught this attack
    "accuracy_after": float   # Model accuracy after this round
}
```

Defender Agent — stored parameters per entry:
```python
{
    "round":    int,          # Global round number
    "strategy": {
        "method":    str,     # "norm_threshold" | "cosine_threshold" | "combined"
        "params":    dict,    # e.g. {"threshold": 2.0}
        "reasoning": str      # LLM's explanation for choosing this defense
    },
    "attack_passed_through": bool,  # Whether the attacker evaded detection
    "verdicts": [                   # Detection result for each client
        {
            "client_id":  int,
            "suspicious": bool,
            "confidence": float,    # 0.0–1.0
            "reason":     str       # e.g. "norm_ratio=2.5>2.0"
        }
    ]
}
```

> These are the **same fields** stored in short-term memory. The difference is that long-term memory also has the vector embedding, enabling similarity-based retrieval instead of recency-based.

### Retrieval process

```
Current context dict
       ↓
json.dumps(sort_keys=True)
       ↓
SentenceTransformer encode → 384-dim vector
       ↓
FAISS IndexFlatL2 search(k=3)
       ↓
Top 3 nearest metadata dicts → sent to LLM as "similar_past_experiences"
```

### Properties

- **Retrieval count:** 3 nearest neighbors
- **Distance metric:** L2 (Euclidean)
- **Embedding model:** `all-MiniLM-L6-v2` (384 dimensions)
- **Selection:** Semantic similarity (not recency)
- **Persistence:** Saved to disk after each round (`index.faiss` + `metadata.json`)
- **Fallback:** Brute-force numpy search if FAISS is not installed
- **Purpose:** Recall what worked in similar past situations across all runs

---

## Short-Term vs Long-Term

| Aspect | Short-Term | Long-Term |
|--------|-----------|-----------|
| Sent to LLM | Last 5 rounds | Top 3 similar rounds |
| Selection | Recency | Vector similarity |
| Persisted | No | Yes (disk) |
| Scope | Current run | All runs (cumulative) |
| Purpose | Recent trends | Historical patterns |

---

## Code References

| Component | File |
|-----------|------|
| Attacker memory | `agents/attacker_agent.py` — `self.history`, `self.memory` |
| Defender memory | `agents/defender_agent.py` — `self.history`, `self.memory` |
| Embedding | `agents/embedder.py` — `embed()`, `get_dimension()` |
| Vector store | `storage/vector_store.py` — `add()`, `search()`, `save()` |
