# System Architecture & Agent Feedback Contract

> Every field documented below is verified against the actual code in
> `main.py`, `agents/attacker_agent.py`, `agents/defender_agent.py`,
> `detector/anomaly_detector.py`, `metrics/tracker.py`, and
> `metrics/production_signals.py`.

---

## Memory Architecture

### Short-term memory (`recent_history`)
Each agent maintains a `self.history` list. The last 5 entries are included
in the LLM prompt as `recent_history` (see `self.history[-5:]` in both
`_ask_llm` methods).

### Long-term associative memory (`similar_past_experiences`)
Each agent owns a FAISS `VectorStore`. When the LLM is consulted, the agent
embeds the current context with SentenceTransformers (`all-MiniLM-L6-v2`,
384-dim) and retrieves the top 3 most similar past round entries from the
FAISS index. These are included in the prompt as `similar_past_experiences`.

---

## Attacker Agent Feedback

> **Simulation-only component.** The attacker agent does not exist in
> production — it generates diverse attacks to train and evaluate the
> defender. It retains oracle feedback (ground truth) for this purpose.

### When does the LLM get consulted?
- **First round**: always (no prior strategy exists).
- **Last attack was detected** (`was_detected == True`): the LLM is asked to adapt.
- **Last attack passed through** (`was_detected == False`): the current strategy is **reused without consulting the LLM**.

Source: [attacker_agent.py:94–116](agents/attacker_agent.py)

### Decision context (`attacker_context` → `decide()`)

Built at [main.py:211–219](main.py) using windowed metrics computed **before** the current round:

| Field | Type | Source | Description |
|---|---|---|---|
| `baseline_accuracy` | `float` | `baseline_accuracy` (Phase 1 saved value) | Clean model accuracy before any attacks. Fixed reference. |
| `current_accuracy` | `float` | `server.evaluate(test_loader)` after the **previous** round's aggregation | Test accuracy of the global model after the last round. |
| `was_detected` | `bool \| None` | `last_attack_detected` from previous round (`None` on first round) | Whether the attacker's **last** attack was caught by the defender. |
| `attack_success_rate_recent` | `float` | `windowed["attack_success_rate"]` — `MetricsTracker.get_windowed_metrics(5)` | Fraction of rounds (last 5) where the attack evaded detection (`fn > 0`). 0.0–1.0. |
| `fpr_recent` | `float` | `windowed["fpr"]` — `MetricsTracker.get_windowed_metrics(5)` | Defender's false positive rate over the last 5 rounds (`FP / (FP + TN)`). |
| `accuracy_preservation_rate` | `float` | `windowed["accuracy_preservation_rate"]` — `MetricsTracker.get_windowed_metrics(5)` | `current_accuracy / baseline_accuracy` from the most recent completed round. |

### What gets sent to the LLM prompt

Constructed in `_ask_llm()` at [attacker_agent.py:167–196](agents/attacker_agent.py):

```json
{
    "baseline_accuracy": <from context>,
    "current_accuracy": <from context>,
    "was_detected": <from context>,
    "attack_success_rate_recent": <from context, default 0.0>,
    "fpr_recent": <from context, default 0.0>,
    "accuracy_preservation_rate": <from context, default 1.0>,
    "recent_history": <self.history[-5:]>,
    "similar_past_experiences": <top 3 from FAISS>
}
```

### History entries (`record_outcome()`)

Recorded at [main.py:323–333](main.py), stored by [attacker_agent.py:118–165](agents/attacker_agent.py).
Each entry in `self.history` (and FAISS) contains:

| Field | Type | Source | Description |
|---|---|---|---|
| `round` | `int` | `round_num` | Global round number. |
| `strategy` | `dict` | `attack_strategy` (LLM output) | The attack strategy used: `{attack_type, params, reasoning}`. |
| `was_detected` | `bool` | `attack_detected` — `malicious_verdict.is_suspicious` | Whether the malicious client was flagged this round. |
| `accuracy_after` | `float` | `current_accuracy` — `server.evaluate(test_loader)` | Test accuracy after this round's aggregation. |
| `attack_success_rate_recent` | `float` | `windowed_after["attack_success_rate"]` (recomputed **after** this round) | Trailing 5-round attack success rate at end of this round. |
| `fpr_recent` | `float` | `windowed_after["fpr"]` (recomputed **after** this round) | Trailing 5-round FPR at end of this round. |
| `accuracy_preservation_rate` | `float` | `windowed_after["accuracy_preservation_rate"]` (recomputed **after** this round) | APR at end of this round. |
| `attack_metadata` | `dict \| absent` | `malicious_update.metadata["attack_metadata"]` | Optional. Attack-specific data (e.g. `k`, `total_params`, `flipped_per_layer` / `affected_per_layer`, gradient stats). Only present if the attack populates it. |

### LLM system prompt tells the attacker about

Defined at [attacker_agent.py:18–64](agents/attacker_agent.py):
- Available attack types (from config)
- Parameter ranges for `sign_flip`, `noise_injection`, `scaling`, `gaussian_noise`
- How to interpret `attack_success_rate_recent`, `fpr_recent`, `accuracy_preservation_rate`
- Strategic guidance: adapt when caught, exploit high FPR, use `k` parameter for stealth
- Must respond with JSON: `{attack_type, params, reasoning}`

---

## Defender Agent Feedback (Production-Ready)

> **All signals are production-observable.** The defender agent uses ZERO
> ground truth about which clients are malicious. Oracle metrics (TPR, FPR,
> `attack_passed_through`) are computed by `MetricsTracker` for researcher
> evaluation only — the defender never sees them.

### When does the LLM get consulted?
- **First round**: **NOT** consulted — uses initial strategy from config (default: `norm_threshold`, sensitivity `2.0`).
- **All clients were flagged last round** (`all_clients_flagged == True`): the LLM is asked to loosen thresholds.
- **Accuracy dropped** (`accuracy_delta < -0.01`): the LLM is asked to adapt (proxy for "attack passed through").
- **Accuracy stable and not all flagged**: the current strategy is **reused without consulting the LLM**.

Source: [defender_agent.py:161–188](agents/defender_agent.py)

### Decision context (`defender_context` → `decide()`)

Built at [main.py:250–281](main.py) using production-observable signals:

| Field | Type | Source | Description |
|---|---|---|---|
| `update_features` | `dict` | `detector.get_features(updates, current_global)` → `_summarize_features()` | Statistical features of all client updates in the **current** round (see table below). |
| `accuracy_delta` | `float \| None` | `compute_accuracy_delta(current, previous)` | Round-over-round accuracy change. Negative = model degraded. `None` on first round. **Primary proxy for `attack_passed_through`.** |
| `accuracy_trend` | `float` | `compute_accuracy_trend(accuracy_history, 5)` | Linear slope of accuracy over last 5 rounds. Positive = improving. |
| `accuracy_volatility` | `float` | `compute_accuracy_volatility(accuracy_history, 5)` | Std dev of accuracy over last 5 rounds. High = unstable system. |
| `accuracy_preservation_rate` | `float` | `current_accuracy / baseline_accuracy` | How well the model retains pre-attack performance. |
| `flag_rate` | `float` | `compute_flag_rate(n_flagged, n_total)` | Fraction of clients flagged last round. **Proxy for FPR when combined with accuracy signals.** |
| `all_clients_flagged` | `bool \| None` | `last_all_clients_flagged` from previous round (`None` on first round) | Whether **every** client was flagged last round (round was skipped). |
| `rounds_skipped_recent` | `int` | `compute_rounds_skipped(history, 5)` | Count of all-flagged (skipped) rounds in last 5. |
| `method_consensus` | `dict[int, int]` | `detector.compute_consensus(updates, current_global)` | Per-client score (0–5): how many of the 5 detection methods flag each client. **Strongest proxy for TPR.** |
| `client_flag_history` | `dict[int, int]` | `compute_client_flag_history(history, 10)` | Per-client flag count over last 10 rounds. Persistent offenders = likely malicious. |

### `update_features` breakdown

Computed by `AnomalyDetector.get_features()` → `_summarize_features()` at [anomaly_detector.py:81–190](detector/anomaly_detector.py). All values are keyed by client ID:

| Feature | Type | Description |
|---|---|---|
| `l2_norms` | `{client_id: float}` | L2 norm of each client's weight delta (`update − global`). |
| `cosine_similarities` | `{client_id: float}` | Cosine similarity of each client's delta with the **mean** delta. |
| `dnc_scores` | `{client_id: float}` | SVD spectral outlier scores — squared projection onto top singular vector. |
| `fltrust_scores` | `{client_id: float}` | `ReLU(cosine_similarity)` trust scores per client. |
| `foolsgold_max_cs` | `{client_id: float}` | Maximum pairwise cosine similarity per client (Sybil detection signal). |
| `mean_pairwise_distance` | `float` | Average pairwise L2 distance between all client updates (scalar). |

### What gets sent to the LLM prompt

Constructed in `_ask_llm()` at [defender_agent.py:219–244](agents/defender_agent.py):

```json
{
    "update_features": <from context — the summarized feature dict>,
    "accuracy_delta": <from context>,
    "accuracy_trend": <from context, default 0.0>,
    "accuracy_volatility": <from context, default 0.0>,
    "accuracy_preservation_rate": <from context, default 1.0>,
    "flag_rate": <from context, default 0.0>,
    "all_clients_flagged": <from context>,
    "rounds_skipped_recent": <from context, default 0>,
    "method_consensus": <from context>,
    "client_flag_history": <from context>,
    "recent_history": <self.history[-5:]>,
    "similar_past_experiences": <top 3 from FAISS>
}
```

### History entries (`record_outcome()`)

Recorded at [main.py:334–360](main.py), stored by [defender_agent.py:190–218](agents/defender_agent.py).
Each entry in `self.history` (and FAISS) contains:

| Field | Type | Source | Description |
|---|---|---|---|
| `round` | `int` | `round_num` | Global round number. |
| `strategy` | `dict` | `defend_strategy` (LLM output or initial default) | The defense strategy used: `{method, params: {sensitivity}, reasoning}`. |
| `verdicts` | `list[dict]` | Serialized `DetectionVerdict` objects | Per-client verdicts: `{client_id, suspicious, confidence, reason}` for every client. |
| `all_clients_flagged` | `bool` | `n_flagged == len(verdicts)` | Whether every client was flagged (round was skipped). |
| `accuracy_delta` | `float` | `compute_accuracy_delta(current, previous)` | Accuracy change this round. |
| `accuracy_trend` | `float` | `compute_accuracy_trend(history, 5)` | 5-round accuracy slope at end of this round. |
| `accuracy_volatility` | `float` | `compute_accuracy_volatility(history, 5)` | 5-round accuracy std dev at end of this round. |
| `accuracy_preservation_rate` | `float` | `current_accuracy / baseline_accuracy` | APR at end of this round. |
| `flag_rate` | `float` | `compute_flag_rate(n_flagged, n_total)` | Fraction of clients flagged this round. |
| `rounds_skipped_recent` | `int` | `compute_rounds_skipped(history, 5)` | Skipped rounds count at end of this round. |
| `method_consensus` | `dict[int, int]` | `detector.compute_consensus()` | Cross-method consensus scores for this round. |
| `client_flag_history` | `dict[int, int]` | `compute_client_flag_history(history, 10)` | Cumulative client flag counts at end of this round. |

### LLM system prompt tells the defender about

Defined at [defender_agent.py:14–139](agents/defender_agent.py):
- All 6 `update_features` fields and their meaning
- How to interpret `accuracy_delta`, `accuracy_trend`, `accuracy_volatility`
- How to use `flag_rate` + `accuracy_delta` as a proxy for FPR
- How to interpret `method_consensus` (strongest TPR proxy)
- How to use `client_flag_history` for persistent offender detection
- The `all_clients_flagged` and `rounds_skipped_recent` emergency signals
- 5 available defense methods with adaptive thresholds
- Strategy guidance: when to switch methods, how to adjust sensitivity
- Must respond with JSON: `{method, params: {sensitivity}, reasoning}`

---

## Side-by-Side Comparison

### What each agent sees AT DECISION TIME

| Signal | Attacker | Defender |
|---|---|---|
| `baseline_accuracy` | ✅ | ❌ |
| `current_accuracy` | ✅ | ❌ |
| `was_detected` (previous round) | ✅ | ❌ |
| `update_features` (current round stats) | ❌ | ✅ |
| `accuracy_delta` (round-over-round change) | ❌ | ✅ |
| `accuracy_trend` (5-round slope) | ❌ | ✅ |
| `accuracy_volatility` (5-round std dev) | ❌ | ✅ |
| `accuracy_preservation_rate` (5-round APR) | ✅ | ✅ |
| `flag_rate` (clients flagged / total) | ❌ | ✅ |
| `all_clients_flagged` | ❌ | ✅ |
| `rounds_skipped_recent` | ❌ | ✅ |
| `method_consensus` (0–5 per client) | ❌ | ✅ |
| `client_flag_history` (10-round counts) | ❌ | ✅ |
| `attack_success_rate_recent` (5-round ASR) | ✅ | ❌ |
| `fpr_recent` (5-round FPR, oracle) | ✅ | ❌ |
| `recent_history` (last 5 history entries) | ✅ | ✅ |
| `similar_past_experiences` (top 3 FAISS) | ✅ | ✅ |
| `attack_metadata` (in history entries only) | ✅ | ❌ |

### What each agent stores IN HISTORY

| History Field | Attacker | Defender |
|---|---|---|
| `round` | ✅ | ✅ |
| `strategy` | ✅ (attack strategy) | ✅ (defense strategy) |
| `was_detected` | ✅ | ❌ |
| `accuracy_after` | ✅ | ❌ |
| `verdicts` | ❌ | ✅ |
| `all_clients_flagged` | ❌ | ✅ |
| `accuracy_delta` | ❌ | ✅ |
| `accuracy_trend` | ❌ | ✅ |
| `accuracy_volatility` | ❌ | ✅ |
| `accuracy_preservation_rate` | ✅ | ✅ |
| `flag_rate` | ❌ | ✅ |
| `rounds_skipped_recent` | ❌ | ✅ |
| `method_consensus` | ❌ | ✅ |
| `client_flag_history` | ❌ | ✅ |
| `attack_metadata` | ✅ (when present) | ❌ |
| `attack_success_rate_recent` | ✅ | ❌ |
| `fpr_recent` (oracle) | ✅ | ❌ |

### Oracle vs Production-Ready Signals

| Signal Type | Attacker Agent | Defender Agent |
|---|---|---|
| **Oracle signals** (require `malicious_ids`) | ✅ Uses: `was_detected`, `attack_success_rate_recent`, `fpr_recent` | ❌ **Never sees oracle signals** |
| **Production signals** (no ground truth needed) | N/A (simulation-only) | ✅ Uses: `accuracy_delta`, `flag_rate`, `method_consensus`, etc. |
| **Rationale** | Simulation-only component — oracle feedback drives realistic attack evolution | Must work identically in simulation and production |

---

## Windowed Metrics Pipeline (Researcher Evaluation Only)

> [!IMPORTANT]
> These oracle metrics are computed for **researcher evaluation** (measuring
> TPR, FPR, ASR). They are fed to the **attacker agent** (simulation-only)
> but are **never fed to the defender agent**.

Computed by `MetricsTracker.get_windowed_metrics(window=5)` at [metrics/tracker.py:117–148](metrics/tracker.py):

### Metric Formulas

The core metrics are calculated over the target window (default: last 5 rounds). For a single round, an attack is defined as "successful" if `FN > 0` (at least one malicious client evaded detection).

- **Attack Success Rate (ASR)**: `(rounds with attack success) / (total rounds in window)`
- **True Positive Rate (TPR)**: `sum(TP) / (sum(TP) + sum(FN))` calculated across all clients over the window.
- **False Positive Rate (FPR)**: `sum(FP) / (sum(FP) + sum(TN))` calculated across all clients over the window.
- **Accuracy Preservation Rate (APR)**: `current_accuracy / baseline_accuracy` where `current_accuracy` is the global model test accuracy after the most recent round in the window.

### Pipeline Flow

```text
MetricsTracker.update(round, verdicts, accuracy)
    → compute_round_metrics()         [metrics/compute.py]
        → confusion_counts()           TP, FN, FP, TN per round
        → attack_success = (fn > 0)    attack evaded detection?
        → tpr, fpr, apr                per-round rates

MetricsTracker.get_windowed_metrics(5)
    → aggregate over last 5 RoundMetrics
    → returns {attack_success_rate, tpr, fpr, accuracy_preservation_rate}
```

**Timing**: In `main.py`, windowed metrics are computed **twice** per round:
1. **Before the round** (line 209): used in `attacker_context` for decision-making. These reflect history up to the previous round.
2. **After the round** (line 321): used in `attacker_agent.record_outcome()` for attacker history. These include the current round's data.

---

## Production Signals Pipeline

Computed by helper functions in [metrics/production_signals.py](metrics/production_signals.py):

### Signal Computation

```text
main.py — BEFORE defender decides (Step 3):
    → accuracy_delta = current_accuracy - previous_accuracy
    → accuracy_trend = linear_slope(accuracy_history[-5:])
    → accuracy_volatility = std_dev(accuracy_history[-5:])
    → flag_rate = n_flagged_last / n_clients
    → rounds_skipped = count all_flagged in defender_history[-5:]
    → client_flag_history = per-client flag counts over history[-10:]
    → method_consensus = detector.compute_consensus()
        → runs all 5 methods, counts per-client flags (0–5)

main.py — AFTER round completes (Step 7):
    → same signals recomputed with current round's data included
    → stored in defender_agent.record_outcome()
```

### Cross-Method Consensus

Computed by `AnomalyDetector.compute_consensus()` at [anomaly_detector.py:84–130](detector/anomaly_detector.py):

```text
For each of [norm_threshold, dnc, fltrust, foolsgold, flame]:
    Run method with default sensitivity=2.0
    For each client: if flagged, increment consensus[client_id]
Result: {client_id: 0–5}
```

---

## Adaptation Feedback Matrix

| Scenario | Attacker Signal | Attacker Action | Defender Signal | Defender Action |
|---|---|---|---|---|
| Defender catches attacker, accuracy stable | `was_detected=True` | Consult LLM → adapt | `accuracy_delta ≥ -0.01` | Keep strategy |
| Attack passes through, accuracy drops | `was_detected=False` | Keep strategy | `accuracy_delta < -0.01` | Consult LLM → adapt |
| Attack passes through, accuracy stable | `was_detected=False` | Keep strategy | `accuracy_delta ≥ -0.01` | Keep strategy ⚠️ |
| Defender flags ALL clients | `was_detected=True` | Consult LLM → adapt | `all_clients_flagged=True` | Consult LLM → loosen |
| Defender over-flags (high flag_rate, accuracy drops) | `was_detected=True/False` | Based on detection | `accuracy_delta < -0.01`, `flag_rate > 0.5` | Consult LLM → may switch method |

> [!WARNING]
> **Row 3 gap**: When an attack passes through but doesn't immediately degrade
> accuracy (e.g. a slow backdoor), the defender sees `accuracy_delta ≥ -0.01`
> and keeps its strategy. The `method_consensus` signal partially addresses
> this — if consensus is high for a client, the LLM can see the risk in the
> context even when accuracy hasn't dropped yet. However, the defender only
> consults the LLM when accuracy actually drops or all clients are flagged.

---

## Defense Strategies

### Available Methods (all adaptive — no hardcoded thresholds)

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

### Aggregation behavior

- **All methods except `fltrust`**: aggregation uses FedAvg over non-flagged clients.
- **`fltrust`**: uses trust-weighted aggregation (Cao et al.) — continuous trust scores as weights, magnitude normalization. Does **not** filter by verdicts.
- **All clients flagged**: aggregation returns `None` → global model unchanged, round is skipped.

Source: [server/aggregation.py](server/aggregation.py)