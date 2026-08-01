# Technology Stack

**Analysis Date:** 2026-08-01

## Languages

**Primary:**
- Python 3.9+ - Entire codebase including FL simulation, RL training, agents, and server logic

## Runtime

**Environment:**
- PyTorch 2.0+ (CUDA-enabled for GPU training)
- Python 3.9+ with pip package management

**Package Manager:**
- pip
- Lockfile: `requirements.txt` (present)

## Frameworks

**Core ML/RL Stack:**
- **PyTorch** 2.0+ - Tensor computation and model training (CPU/GPU)
- **Unsloth** 2026.6.9+ - 4-bit QLoRA loading, Qwen2.5 optimizations, fast language model inference
- **Transformers** 4.45+ - Pre-trained language models, Qwen2.5-3B-Instruct base
- **PEFT** 0.13+ - Parameter-Efficient Fine-Tuning (LoRA adapters)
- **Accelerate** - Distributed training and multi-GPU support
- **BitAndBytes** - 4-bit quantization for QLoRA training

**Data & Datasets:**
- **TorchVision** - MNIST dataset loading and image transforms
- **NumPy** - Numerical computing and array operations

**Federated Learning:**
- Custom GRPO (Group Relative Policy Optimization) implementation in `rl/grpo.py`
- Custom FedAvg aggregator in `server/aggregation.py`
- Algorithmic defenses: FLTrust (NDSS'21), Multi-Krum (NeurIPS'17), DnC (NDSS'21), DeFL (AAAI-23)

**LLM Inference Backends:**
- **OpenAI Python SDK** 1.0+ - Chat completions API integration
- **Requests** - HTTP client for Ollama REST API calls

**Utilities:**
- **PyYAML** - Configuration file parsing (`configs/*.yaml`)
- **Matplotlib** - Visualization and plotting

## Key Dependencies

**Critical for Training:**
- `unsloth>=2026.6.9` - Provides Unsloth optimizations for Qwen2.5-3B; must be ≥2026.6.9 to avoid RoPE cos/sin broadcast crashes with Transformers 5.3
- `transformers>=4.45` - HuggingFace model loading and generation
- `peft>=0.13` - LoRA adapter management (two adapters: "attacker", "defender" on same base)
- `torch>=2.0` - GPU tensor operations
- `accelerate` - Distributed training primitives

**Critical for Inference:**
- `openai>=1.0` - OpenAI API client (requires `OPENAI_API_KEY` env var)
- `requests` - Ollama HTTP client at configurable `ollama_base_url`

**Infrastructure:**
- `bitsandbytes` - 4-bit quantization (QLoRA mode: `load_in_4bit: true` in config)
- `torchvision` - Dataset utilities and transforms
- `numpy` - Array math
- `pyyaml` - YAML parsing
- `matplotlib` - Plotting

## Configuration

**Environment Variables:**
- `OPENAI_API_KEY` - Required for `llm_backend: "openai"` mode (OpenAI API authentication)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - Set in `main.py` to reduce CUDA fragmentation OOMs
- `UNSLOTH_DISABLE_FAST_GENERATION=1` - Set in `rl/policy.py` to disable Unsloth's fused fast-KV inference (can be incompatible with Transformers 5.3)

**Configuration Files:**
- `configs/base.yaml` - Master configuration: FL params, attack/defense config, RL hyperparameters, LLM defaults
- `configs/attacker_agent.yaml` - Attacker LLM prompt and behavior
- `configs/defender_agent.yaml` - Defender LLM prompt and behavior

**Key Config Parameters (from `configs/base.yaml`):**
- **FL Training:** `training_rounds: 45`, `simulation_rounds: 2000000`, `n_clients: 20`, `local_epochs: 3`, `batch_size: 64`
- **LLM:** `ollama_base_url: "http://localhost:11434"`, `ollama_model: "qwen2.5:3b"` (for local inference)
- **RL:** `model: "unsloth/Qwen2.5-3B-Instruct"`, `max_seq_len: 8192`, `load_in_4bit: false` (bf16 LoRA default), `lora_r: 16`, `lora_alpha: 32`
- **Checkpointing:** `save_every: 25`, `league_snapshot_every: 100`

## Platform Requirements

**Development/Training:**
- CUDA 13+ capable GPU (recommended 12GB+ VRAM for bf16 LoRA; 4-bit QLoRA tighter on memory)
- Linux recommended (tested on Linux; Windows support via WSL2)
- Python 3.9+
- ~500MB+ disk for MNIST data, ~115MB per LoRA adapter snapshot

**Inference (CPU-compatible modes):**
- CPU-only: `--dry-run` mode or `--baseline` (uses OpenAI or Ollama, no GPU training)
- Local Ollama server running on `http://localhost:11434` (if using Ollama backend)

**Production Deployment:**
- Not applicable (research testbed)

---

*Stack analysis: 2026-08-01*
