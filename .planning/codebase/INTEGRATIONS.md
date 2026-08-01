# External Integrations

**Analysis Date:** 2026-08-01

## APIs & External Services

**LLM Inference Backends:**

**OpenAI (Primary inference backend):**
- Service: OpenAI API (gpt-4o-mini, gpt-4o, or configurable model)
- What it's used for: LLM-direct attack planning (attacker agent) and defense verdicts (defender agent)
- SDK/Client: `openai>=1.0` Python SDK
- Auth: `OPENAI_API_KEY` environment variable
- Implementation: `agents/llm_client.py` — `OpenAILLMClient` class
  - Chat completions with JSON-mode response parsing (`response_format: {"type": "json_object"}`)
  - Temperature, max_tokens configurable per request
  - Default model: `gpt-4o-mini` (configurable in `create_llm_client()`)
- Used by: `agents/attacker_agent.py`, `agents/defender_agent.py` (Phase 2 LLM agents)

**Ollama (Local LLM alternative):**
- Service: Local Ollama server (self-hosted, no external auth required)
- What it's used for: Alternative to OpenAI for LLM-direct attack/defense (same agent interface)
- SDK/Client: `requests` library (HTTP REST client)
- Auth: None (local service)
- Connection: `ollama_base_url` (default `http://localhost:11434`)
- API Endpoint: `POST /api/generate` (Ollama generate API)
- Implementation: `agents/llm_client.py` — `OllamaLLMClient` class
  - Extracts JSON from raw text output (handles markdown code fences, chain-of-thought text)
  - Temperature, `num_predict` (token limit) configurable
  - Default model: `deepseek-r1:70b` (configurable in `create_llm_client()`)
- Used by: Same agent interface as OpenAI (swappable via `llm_backend` flag)
- Fallback strategy: If Ollama connection fails, returns empty dict; logs connection error

**Backend Selection (Runtime):**
- Controlled by: `--env` flag in `main.py`
  - `python main.py --env linux` uses config-specified backend (default: OpenAI)
  - `python main.py --dry-run` uses frozen-LLM inference via `agents/llm_client.py`
- Factory: `create_llm_client(backend, model, temperature, ollama_base_url)` in `agents/llm_client.py`

## Data Storage

**Datasets:**
- **MNIST**: Downloaded and cached from PyTorch/TorchVision
  - Source: `torchvision.datasets.MNIST`
  - Local cache: `data/mnist_raw/` (configured in `configs/base.yaml` as `data.data_dir`)
  - Partitioning: IID or FLTrust non-IID (configured via `data.iid` and `data.noniid_bias`)

**Model Checkpoints:**
- Storage: Local filesystem
  - Directory: `checkpoints/` (managed in `storage/checkpoint.py`)
  - Phase 1 checkpoints: `checkpoints/global_model.pt`, `checkpoints/client_updates.pt`, `checkpoints/baseline.json`
  - Phase 2 RL checkpoints: LoRA adapters (safetensors format) for "attacker" and "defender"
  - Paths configured in `configs/base.yaml`:
    - `rl.adapter_paths.attacker: "checkpoints/attacker_adapter"`
    - `rl.adapter_paths.defender: "checkpoints/defender_adapter"`
  - FL state: `checkpoints/fl_state.pt` (Phase-2 live model + per-client weights for resume)
  - Progress: `checkpoints/rl_progress.json` (RL training resume state)

**File Storage:**
- Local filesystem only
  - Logs: `logs/system.log`, `logs/round_data/` (per-round detailed data)
  - Metrics: `logs/metrics.json` (per-round evaluations)

**Caching:**
- None (no Redis, memcached, or external cache)

## Authentication & Identity

**Auth Provider:**
- OpenAI: API key-based (bearer token in `Authorization: Bearer` header, handled by `openai` SDK)
- Ollama: None (local service, no authentication)

**Secrets Management:**
- `OPENAI_API_KEY` must be set in environment before runtime
- No external secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)
- No `.env` file integration (users must export env vars manually or in shell profile)

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Datadog, or external error aggregation)
- Errors logged to `logs/system.log` via Python's `logging` module

**Logging:**
- **Framework:** Python `logging` module
- **Handlers:**
  - File: `logs/system.log` (mode "a" — append)
  - Console: stdout with UTF-8 encoding
- **Format:** `"%(asctime)s [%(name)s] %(levelname)s: %(message)s"`
- **Verbosity Control:**
  - Default: `logging.INFO`
  - Debug mode: `--debug` flag reduces third-party library noise (sets transformers, unsloth, peft, etc. to WARNING)
  - Structured logging via `core.debug` module for per-round detailed pictures
- **Log Files Per Round:**
  - Round data logged to `logs/round_data/` (round-by-round metrics, verdicts, rewards)

**Metrics:**
- Tracked in-memory via `MetricsTracker` (`metrics/tracker.py`)
- Saved to `logs/metrics.json` per round
- No external metrics collection (Prometheus, Grafana, etc.)

## CI/CD & Deployment

**Hosting:**
- Not applicable (research testbed, single-machine execution)

**CI Pipeline:**
- None (no GitHub Actions, GitLab CI, Jenkins)
- Test suite in `tests/` (run manually with pytest)

## Environment Configuration

**Required Environment Variables:**
- `OPENAI_API_KEY` - For OpenAI LLM backend (if using `--env linux` without Ollama)

**Optional Environment Variables:**
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - Set by `main.py` at startup; can be pre-set to optimize memory
- `UNSLOTH_DISABLE_FAST_GENERATION=1` - Set by `rl/policy.py` to disable incompatible Unsloth kernel

**Config File Paths:**
- `configs/base.yaml` - Master config (required)
- `configs/attacker_agent.yaml` - Attacker prompt and behavior (loaded by `AttackerAgent`)
- `configs/defender_agent.yaml` - Defender prompt and behavior (loaded by `DefenderAgent`)

**Secrets Location:**
- `OPENAI_API_KEY` - Must be in environment; typically set via shell `export` or CI/CD secrets manager (if deploying to a service)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None (no callbacks to external services)

## Data Flow - LLM Integration Example

```
Round begins:
  1. Attacker LLM (via OpenAI/Ollama) receives:
     - Controllable client IDs, poison budget, per-client weight statistics
     - Current global accuracy, attack goal
  2. Attacker outputs: Client selection + per-client attack plan (operators)
  3. Plan interpreter applies ops to benign weights → poisoned client weights
  4. Server aggregates (Defender LLM flags malicious clients if enabled)
  5. Defender LLM (via OpenAI/Ollama) receives:
     - Per-client per-layer statistical features
  6. Defender outputs: Benign/malicious classification per client
  7. Server FedAvg-aggregates non-flagged clients
  8. Both agents rewarded based on verifiable ground truth (known poison set)
  9. If training (not --dry-run): GRPO step updates LoRA adapters
```

---

*Integration audit: 2026-08-01*
