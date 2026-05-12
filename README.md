# Zero-Touch Federated Learning

A research-oriented federated learning simulation with LLM-powered adversarial agents.

## Overview

Two-phase system on MNIST:

1. **Phase 1 (Rounds 1–3):** 5 clients train honestly via FedAvg. State is checkpointed.
2. **Phase 2 (Round 4+):** Client 0 becomes adversarial. An LLM attacker agent selects model poisoning attacks, while an LLM defender agent tunes anomaly detection — forming an adaptive arms race.

## LLM Backends

The system supports two LLM backends, selected at runtime via the `--env` flag:

| Flag             | Backend | Model (default)     | Requires                        |
|------------------|---------|---------------------|---------------------------------|
| `--env windows`  | OpenAI  | `gpt-4o-mini`       | `OPENAI_API_KEY` env variable   |
| `--env linux`    | Ollama  | `deepseek-r1:70b`   | Ollama server running locally   |

## Setup

### Windows (OpenAI)

```bash
pip install -r requirements.txt
set OPENAI_API_KEY=sk-proj-your_actual_key_here
# or in PowerShell:
$env:OPENAI_API_KEY="sk-proj-your_actual_key_here"
```

### Linux (Ollama)

```bash
pip install -r requirements.txt

# Install and start Ollama (if not already running)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# Pull the model
ollama pull deepseek-r1:70b
```

> **Note:** The Ollama base URL defaults to `http://localhost:11434`. To use a
> remote Ollama server, update `llm.ollama_base_url` in `configs/base.yaml`.

## Usage

```bash
# Windows / OpenAI (default)
python main.py

# Linux / Ollama
python main.py --env linux

# Force fresh Phase 1 training
python main.py --fresh

# Combine flags
python main.py --env linux --fresh

python visualize_rounds.py
```

TO SEE THE VISUALIZATIONS

On server run: 
  - cd ~/ruh-fyp-2026/fl/zero-touch-fl/logs/visualizations
  - python -m http.server 8084
		
On win cmd run: 
   - ssh -i C:\fl\server\isuru -L 8084:netslabsmosv1:8084 ruh_fyp_26@netslabsv1.ucd.ie

open : http://localhost:8084/ on win browser



## Configuration

Models and settings for each agent are in `configs/`:

- **`base.yaml`** — FL hyperparameters and global Ollama defaults
- **`attacker_agent.yaml`** — Attacker LLM model, temperature, available attacks
- **`defender_agent.yaml`** — Defender LLM model, temperature, initial strategy

Each agent config specifies both an OpenAI model (`llm.model`) and an Ollama
model (`llm.ollama_model`). The `--env` flag determines which one is used.

## Project Structure

```
core/          Shared types & interfaces (zero business logic)
model/         Tiny MLP (~805 params)
data/          MNIST loading & partitioning
clients/       Benign and malicious client implementations
attacks/       Model poisoning plugins (sign_flip, noise, scaling, gaussian_noise)
agents/        LLM-powered attacker & defender agents
  llm_client.py  ← Backend-agnostic LLM abstraction (OpenAI / Ollama)
detector/      Statistical anomaly detection
server/        Central server & FedAvg aggregation
storage/       Checkpointing & FAISS vector store
configs/       YAML configuration files
logs/          System logs & per-round JSON data
```

## Agent Dynamics

- **Attacker** adapts only when caught by the defender
- **Defender** adapts only when an attack passes through
- Both use FAISS-backed episodic memory persisted to disk
