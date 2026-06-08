# Memory System

The system uses **hybrid memory**: short-term (recency-based) and long-term (similarity-based). Both the Attacker and Defender agents maintain independent instances of each.

> Every field documented below is verified against the actual code in
> `agents/attacker_agent.py`, `agents/defender_agent.py`, `agents/embedder.py`,
> `storage/vector_store.py`, `clients/malicious_client.py`, `attacks/*.py`,
> and `main.py`.

---

## Short-Term Memory (`recent_history`)

A Python list (`self.history`) that grows unboundedly over the run. Only the
**last 5 entries** (`self.history[-5:]`) are included in the LLM prompt as
`recent_history`.

### Attacker Agent — stored fields per round

Built in `attacker_agent.py:record_outcome()` (lines 118–165), called from
`main.py` (lines 323–333):

```python
{
    "round":          int,      # Global round number (e.g. 4, 5, 6)
    "strategy": {
        "attack_type": str,     # "sign_flip" | "noise_injection" | "scaling" | "gaussian_noise"
        "params":      dict,    # e.g. {"scale": 3.0}, {"c": 2.0, "k": 50}, {"sigma": 1.0}
        "reasoning":   str      # LLM's explanation for choosing this attack
    },
    "was_detected":   bool,     # True if defender caught this attack
    "accuracy_after": float,    # Global model test accuracy after this round's aggregation

    # ── Windowed KPIs (recomputed AFTER this round via MetricsTracker) ──
    "attack_success_rate_recent": float,  # Trailing 5-round ASR (0.0–1.0)
    "fpr_recent":                float,  # Trailing 5-round defender FPR (0.0–1.0)
    "accuracy_preservation_rate": float,  # current_accuracy / baseline_accuracy (0.0–1.0)

    # ── Optional: present only when the attack populates last_metadata ──
    "attack_metadata": {        # sign_flip, noise_injection, scaling provide this
                                # gaussian_noise does NOT (no last_metadata attribute)
        "k":            int|str,  # Number of targeted weights, or "all"
        "total_params": int,      # Total model parameters

        # sign_flip uses these keys:
        "flipped_per_layer":          dict,  # {layer_name: count}
        "flipped_indices_per_layer":  dict,  # {layer_name: [int, ...]}
        "avg_flipped_grad_magnitude":   float,  # mean |grad| of flipped weights
        "avg_unflipped_grad_magnitude": float,  # mean |grad| of unflipped weights

        # noise_injection & scaling use these keys instead:
        "affected_per_layer":          dict,  # {layer_name: count}
        "affected_indices_per_layer":  dict,  # {layer_name: [int, ...]}
        "avg_targeted_grad_magnitude":   float,  # mean |grad| of targeted weights
        "avg_untargeted_grad_magnitude": float   # mean |grad| of untargeted weights
    }
}
```

> When `k = "all"` (no selective targeting), only `k`, `total_params`, and
> the `*_per_layer` count dict are present. The `*_indices_per_layer` and
> gradient magnitude fields are only populated for the selective (top-k) case.

### Defender Agent — stored fields per round (production-ready)

Built in `defender_agent.py:record_outcome()` (lines 190–218), called from
`main.py` (lines 334–360):

> **All fields are production-observable — no oracle feedback.** The defender
> stores the same signal types it receives during decision-making, ensuring
> FAISS memory entries are structurally compatible with production queries.

```python
{
    "round":    int,              # Global round number
    "strategy": {
        "method":    str,         # "norm_threshold" | "dnc" | "fltrust" | "foolsgold" | "flame"
        "params":    dict,        # e.g. {"sensitivity": 2.0}
        "reasoning": str          # LLM's explanation for choosing this defense
    },
    "verdicts": [                   # One entry per client (serialized DetectionVerdict)
        {
            "client_id":  int,
            "suspicious": bool,
            "confidence": float,    # 0.0–1.0
            "reason":     str       # e.g. "norm=5.1234, threshold=3.4567 (median=2.0 + 2.0×MAD=0.7)"
        }
    ],
    "all_clients_flagged":   bool,  # True if ALL clients were flagged (round was skipped)

    # ── Production-observable signals ──
    "accuracy_delta":             float,  # Round-over-round accuracy change
    "accuracy_trend":             float,  # 5-round linear accuracy slope
    "accuracy_volatility":        float,  # 5-round accuracy std dev
    "accuracy_preservation_rate": float,  # current_accuracy / baseline_accuracy (0.0–1.0)
    "flag_rate":                  float,  # Fraction of clients flagged this round (0.0–1.0)
    "rounds_skipped_recent":      int,    # Count of all-flagged rounds in last 5
    "method_consensus": {                 # Per-client consensus score (0–5)
        0: int,  # client 0: flagged by N out of 5 methods
        1: int,  # client 1: ...
        ...
    },
    "client_flag_history": {              # Per-client cumulative flag count (last 10 rounds)
        0: int,  # client 0: flagged in N of last 10 rounds
        1: int,  # client 1: ...
        ...
    }
}
```

### Properties

| Property | Value |
|---|---|
| Window sent to LLM | Last 5 entries (`self.history[-5:]`) |
| Selection method | Chronological (most recent) |
| Persistence | **In-memory only** — lost on restart |
| Growth | Unbounded (full list kept, only last 5 sent to LLM) |
| Purpose | Detect recent trends (e.g. accuracy drift, flag rate trajectory, consensus patterns) |

---

## Long-Term Memory (`similar_past_experiences`)

A FAISS vector index storing **every past round** as a 384-dim semantic
embedding. At query time, the **top 3 most similar** past rounds are retrieved
and sent to the LLM as `similar_past_experiences`.

### Architecture

Each agent owns an independent `VectorStore` instance (`self.memory`),
initialized with the embedding dimension and a disk persistence path:

```python
# attacker_agent.py:87–90
self.memory = VectorStore(
    dimension=get_dimension(),  # 384
    persist_path=config.get("memory", {}).get("persist_path"),
)

# defender_agent.py:153–156
self.memory = VectorStore(
    dimension=get_dimension(),  # 384
    persist_path=config.get("memory", {}).get("persist_path"),
)
```

### What is stored per entry

Each entry consists of two parts:

**Part 1 — Vector embedding (for similarity search):**
- The round outcome dict (same `entry` stored in `self.history`) is serialized
  to a deterministic JSON string: `json.dumps(data, sort_keys=True, default=str)`
- Encoded by `all-MiniLM-L6-v2` SentenceTransformer → **384-dim `float32` vector**
- Stored in the FAISS `IndexFlatL2` index (or numpy fallback array)

**Part 2 — Metadata (the actual data returned to the LLM):**
The metadata is the **exact same dict** as stored in `self.history` (documented
above). Both short-term and long-term memory store identical data structures —
the difference is retrieval method (recency vs. similarity).

### Attacker long-term entry

Identical to the short-term entry documented above, including all windowed KPIs
and optional `attack_metadata`.

### Defender long-term entry

Identical to the short-term entry documented above, including all production
signals (`accuracy_delta`, `flag_rate`, `method_consensus`, `client_flag_history`, etc.).

> **Structural compatibility**: Because the defender's history entries and
> decision context both use production-observable signals (not oracle metrics),
> the FAISS query vectors and stored vectors embed structurally similar dicts.
> This ensures meaningful similarity search in both simulation and production.

### Write path (recording)

Both agents call `record_outcome()` at the end of each round, which:
1. Builds the `entry` dict
2. Appends to `self.history` (short-term)
3. Embeds the entry via `embed(entry)` → 384-dim vector
4. Calls `self.memory.add(vec, entry)` (adds to FAISS index + metadata list)
5. Calls `self.memory.save()` (persists to disk immediately)

```python
# attacker_agent.py:161–164
vec = self._make_vector(entry)   # embed(entry) → 384-dim
self.memory.add(vec, entry)      # FAISS index + metadata
self.memory.save()               # persist to disk

# defender_agent.py:214–216
vec = self._make_vector(entry)
self.memory.add(vec, entry)
self.memory.save()
```

### Read path (retrieval)

When the LLM is consulted (in `_ask_llm()`), the agent:
1. Embeds the **current decision context** (not the outcome entry)
2. Searches FAISS for the 3 nearest neighbors
3. Returns the metadata dicts of those neighbors

**Attacker query vector** — the context dict embedded for FAISS search:

```python
# attacker_agent.py:170–173  (only when self.history is non-empty)
query_vec = self._make_vector(context)
# where context = {
#     "baseline_accuracy": float,
#     "current_accuracy":  float,
#     "was_detected":      bool | None,
#     "attack_success_rate_recent": float,
#     "fpr_recent":        float,
#     "accuracy_preservation_rate": float,
# }
similar = self.memory.search(query_vec, k=3)
```

**Defender query vector** — the context dict embedded for FAISS search:

```python
# defender_agent.py:222–224  (only when self.history is non-empty)
query_vec = self._make_vector(context)
# where context = {
#     "update_features":       dict,   # l2_norms, cosines, dnc, fltrust, foolsgold, mean_pairwise
#     "accuracy_delta":        float | None,
#     "accuracy_trend":        float,
#     "accuracy_volatility":   float,
#     "accuracy_preservation_rate": float,
#     "flag_rate":             float,
#     "all_clients_flagged":   bool | None,
#     "rounds_skipped_recent": int,
#     "method_consensus":      dict[int, int],
#     "client_flag_history":   dict[int, int],
# }
similar = self.memory.search(query_vec, k=3)
```

> **Structural alignment**: Unlike the previous design where query vectors
> and stored vectors embedded different dict structures, the production
> defender now stores the same signal types it queries with. This improves
> FAISS similarity search quality.

### Disk persistence

The `VectorStore.save()` method writes two files to the configured `persist_path`:

| Agent | Config key | Default path | Files on disk |
|---|---|---|---|
| Attacker | `configs/attacker_agent.yaml` → `memory.persist_path` | `checkpoints/attacker_memory/` | `index.faiss` + `metadata.json` |
| Defender | `configs/defender_agent.yaml` → `memory.persist_path` | `checkpoints/defender_memory/` | `index.faiss` + `metadata.json` |

- **`index.faiss`**: The FAISS `IndexFlatL2` index (binary). If FAISS is unavailable, stored as `vectors.npy` instead.
- **`metadata.json`**: JSON array of all entry dicts (the metadata returned to the LLM on search).

On startup, the `VectorStore.__init__()` calls `_load()` which reads both files
back from disk, restoring the full index and metadata from previous runs.

> [!WARNING]
> After switching to production signals, old defender FAISS memory is
> **incompatible** (different entry structure). Clear `checkpoints/defender_memory/`
> before running with the new code.

### Properties

| Property | Value |
|---|---|
| Retrieval count | 3 nearest neighbors (`k=3`) |
| Distance metric | L2 (Euclidean) via `IndexFlatL2` |
| Embedding model | `all-MiniLM-L6-v2` (384 dimensions, shared singleton) |
| Selection method | Semantic similarity (not recency) |
| Persistence | Saved to disk after **every** round |
| Cross-run | Yes — index is loaded on startup, accumulates across runs |
| Fallback | Brute-force numpy L2 search if FAISS is not installed |
| Purpose | Recall what worked in similar past situations across all runs |

---

## Short-Term vs Long-Term Comparison

| Aspect | Short-Term (`recent_history`) | Long-Term (`similar_past_experiences`) |
|--------|-------------------------------|----------------------------------------|
| Sent to LLM as | `recent_history` | `similar_past_experiences` |
| Entries sent | Last 5 rounds | Top 3 most similar rounds |
| Selection method | Recency (chronological) | Vector similarity (semantic) |
| Data structure | Same entry dict | Same entry dict |
| Persisted to disk | No (in-memory only) | Yes (`index.faiss` + `metadata.json`) |
| Survives restart | No | Yes (loaded on init) |
| Scope | Current run only | All runs (cumulative) |
| Purpose | Recent trends & trajectories | Historical pattern recall |
| When populated | On `record_outcome()` | On `record_outcome()` (same call) |
| When read | Every `_ask_llm()` call | Every `_ask_llm()` call |

---

## Data Flow Diagram

```text
main.py — end of each round
    │
    ├─► attacker_agent.record_outcome(round, strategy, was_detected, accuracy,
    │       attack_metadata, attack_success_rate_recent, fpr_recent, apr)
    │       │
    │       ├─► entry = {round, strategy, was_detected, accuracy_after,
    │       │            attack_success_rate_recent, fpr_recent, apr,
    │       │            ?attack_metadata}
    │       ├─► self.history.append(entry)           ← SHORT-TERM
    │       ├─► vec = embed(entry)                   ← 384-dim embedding
    │       ├─► self.memory.add(vec, entry)           ← LONG-TERM (FAISS)
    │       └─► self.memory.save()                    ← persist to disk
    │
    └─► defender_agent.record_outcome(round, strategy, verdicts,
            all_clients_flagged, accuracy_delta, accuracy_trend,
            accuracy_volatility, accuracy_preservation_rate, flag_rate,
            rounds_skipped_recent, method_consensus, client_flag_history)
            │
            ├─► entry = {round, strategy, verdicts, all_clients_flagged,
            │            accuracy_delta, accuracy_trend, accuracy_volatility,
            │            accuracy_preservation_rate, flag_rate,
            │            rounds_skipped_recent, method_consensus,
            │            client_flag_history}
            ├─► self.history.append(entry)           ← SHORT-TERM
            ├─► vec = embed(entry)                   ← 384-dim embedding
            ├─► self.memory.add(vec, entry)           ← LONG-TERM (FAISS)
            └─► self.memory.save()                    ← persist to disk
```

```text
agent._ask_llm(context) — when LLM is consulted
    │
    ├─► query_vec = embed(context)                   ← embed current situation
    ├─► similar = self.memory.search(query_vec, k=3) ← LONG-TERM retrieval
    │
    └─► user_msg = json.dumps({
            ...context fields...,
            "recent_history": self.history[-5:],     ← SHORT-TERM (last 5)
            "similar_past_experiences": similar,      ← LONG-TERM (top 3)
        })
```

---

## Code References

| Component | File | Key variables / methods |
|---|---|---|
| Attacker short-term | `agents/attacker_agent.py` | `self.history`, `record_outcome()`, `self.history[-5:]` in `_ask_llm()` |
| Attacker long-term | `agents/attacker_agent.py` | `self.memory` (VectorStore), `_make_vector()`, `memory.search(k=3)` |
| Defender short-term | `agents/defender_agent.py` | `self.history`, `record_outcome()`, `self.history[-5:]` in `_ask_llm()` |
| Defender long-term | `agents/defender_agent.py` | `self.memory` (VectorStore), `_make_vector()`, `memory.search(k=3)` |
| Embedding | `agents/embedder.py` | `embed()` → `all-MiniLM-L6-v2`, `get_dimension()` → 384 |
| Vector store | `storage/vector_store.py` | `add()`, `search()`, `save()`, `_load()` |
| Production signals | `metrics/production_signals.py` | `compute_accuracy_delta()`, `compute_flag_rate()`, `compute_client_flag_history()`, etc. |
| Cross-method consensus | `detector/anomaly_detector.py` | `compute_consensus()` — runs all 5 methods, returns per-client 0–5 scores |
| Attacker config | `configs/attacker_agent.yaml` | `memory.persist_path: "checkpoints/attacker_memory"` |
| Defender config | `configs/defender_agent.yaml` | `memory.persist_path: "checkpoints/defender_memory"` |
| Record calls | `main.py` | Lines 323–333 (attacker), lines 334–360 (defender) |
| Attack metadata | `attacks/sign_flip.py`, `attacks/noise_injection.py`, `attacks/scaling.py` | `self.last_metadata` (captured via `malicious_client.py:31`) |
| No metadata | `attacks/gaussian_noise.py` | Does **not** set `last_metadata` — `getattr(attack, "last_metadata", {})` returns `{}` |
