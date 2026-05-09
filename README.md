# Zero-Touch Federated Learning

A research-oriented federated learning simulation with LLM-powered adversarial agents.

## Overview

Two-phase system on MNIST:

1. **Phase 1 (Rounds 1–3):** 5 clients train honestly via FedAvg. State is checkpointed.
2. **Phase 2 (Round 4+):** Client 0 becomes adversarial. An LLM attacker agent selects model poisoning attacks, while an LLM defender agent tunes anomaly detection — forming an adaptive arms race.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
for windows set OPENAI_API_KEY=sk-proj-xxxxxxxx
powershell $env:OPENAI_API_KEY="sk-proj-your_actual_key_here"
```

## Usage

```bash
# First run: trains Phase 1, then runs simulation
python main.py

# Subsequent runs: loads checkpoint, skips training
python main.py

# Force fresh training
python main.py --fresh
```

## Project Structure

```
core/          Shared types & interfaces (zero business logic)
model/         Tiny MLP (~805 params)
data/          MNIST loading & partitioning
clients/       Benign and malicious client implementations
attacks/       Model poisoning plugins (sign_flip, noise, scaling)
agents/        LLM-powered attacker & defender agents
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
