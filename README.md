# Zero-Touch Federated Learning — LLM-Direct Adversarial RL

A research testbed where two LLMs **directly generate** the attack and the
defense in a federated-learning arms race on MNIST, and are **reinforcement-
trained** against each other with verifiable rewards.

## Overview

Two phases:

1. **Phase 1 (`training_rounds`):** All clients train honestly via FedAvg. The
   global model, each client's weights, and the baseline accuracy are
   checkpointed.
2. **Phase 2 (`simulation_rounds`):** A **random subset** of clients is poisoned
   each round.
   - **Attacker LLM** — input: round number, per-layer **statistics** of the
     poisoned clients' benign weights, current global accuracy, and a
     configurable attack goal. Output: an **attack plan** — an ordered list of
     primitive weight operators (scale, sign_flip, mask, add_gaussian_noise,
     clip, add_constant, permute, scale_neurons, blend_random, quantize). A
     deterministic interpreter applies the plan to the benign weights to produce
     the poisoned weights sent to the server.
   - **Defender LLM** — input: per-client, per-layer statistical feature vectors.
     Output: a direct **benign/malicious classification** per client.
   - The server FedAvg-aggregates the clients the defender did not flag.
   - Because we know the ground-truth poisoned set, both agents get an exact
     **verifiable reward**, and both are trained online with **GRPO**.

There are no hardcoded attack plugins, no hardcoded detector rules, and no
episodic-memory feedback loop — the LLMs learn everything via RL.

## Why GRPO (not PPO / DPO)

The reward is an exactly-computable scalar per round (RLVR regime). **DPO** is
offline/preference-pair based and discards reward magnitude. **PPO** needs a
learned value/critic network whose per-token credit assignment buys nothing for
a single terminal-reward emission. **GRPO** is critic-free: sample `G`
completions per state and use the group-normalized reward as the advantage.
Rewards are deliberately **continuous** so the within-group spread (and thus the
gradient) doesn't collapse on the tiny action space. The game is **Stackelberg**
(attacker leads, defender best-responds), so we train with a **freeze-and-
alternate** schedule plus an **opponent league** to damp co-adaptation cycling.

## LLM handling (read before training)

- **Training happens on a GPU machine — not through Ollama.** Ollama and the
  OpenAI API are **inference-only** and cannot fine-tune.
- The policy is **one frozen `Llama-3.2-3B-Instruct` base loaded via Unsloth in
  16-bit (bf16) LoRA by default** (`rl.load_in_4bit: false` — the bf16 base is
  ~6.4GB and is higher-fidelity than 4-bit NF4; set `true` to fall back to 4-bit
  QLoRA on small-VRAM GPUs), with **two LoRA adapters** over it — `attacker` and
  `defender`. This is exactly "separate checkpoints on the same LLM": the base is
  never duplicated; each policy is its own small adapter, saved independently to
  `checkpoints/attacker_adapter/` and `checkpoints/defender_adapter/`.
  (Loaded with `attn_implementation="sdpa"` by default — see `rl.attn_implementation`;
  `"eager"` is the most compatible fallback, `"flash_attention_2"` the fastest if
  its sm_120 wheel is installed.)
- GRPO is implemented directly in [`rl/grpo.py`](rl/grpo.py) (environment-coupled
  reward, KL penalty to the frozen base) on top of Unsloth + PEFT — no TRL
  trainer dependency required.
- **Resume**: rerun the same command. Existing adapters and
  `checkpoints/rl_progress.json` are reloaded and training continues.
- **`gpt-4o-mini` (OpenAI) and Ollama `llama3.2` are inference/baseline only** —
  used by `--dry-run` and `--baseline`. They are **not** fine-tuned.
- **Serving a trained adapter**: use **vLLM** (multi-LoRA hot-swap), or **merge**
  the adapter into the base and export a single GGUF for Ollama (Llama 3.2 is
  supported by Ollama as `llama3.2`; merging gives up the two-swappable-adapters
  design but is the simplest serving path).
- **Hardware**: Llama-3.2-3B QLoRA fits comfortably on a single ~8 GB+ GPU
  (~4–6 GB floor) — your 5090 (31 GB) has ample headroom. The attacker emits a
  short attack plan (tens of tokens), so generation is fast and `rl.max_new_tokens`
  can stay small (512).

## Setup

### GPU machine (training)

```bash
pip install -r requirements.txt   # installs unsloth/peft/transformers/bitsandbytes
# Llama-3.2-3B-Instruct is downloaded from Hugging Face on first run (unsloth/Llama-3.2-3B-Instruct)
```

### CPU machine (logic dry-run / baseline only)

```bash
pip install torch torchvision numpy pyyaml matplotlib openai requests
# For --dry-run you also need an Ollama server with llama3.2:
#   ollama serve & ; ollama pull llama3.2:3b
```

## Usage

```bash
# Full GRPO training (GPU). Phase 1 runs once, then the RL arms race.
python main.py --env linux

# Quick smoke run: few rounds
python main.py --env linux --rounds 8

# Logic dry-run — full round loop with a FROZEN LLM via Ollama (no training, no GPU)
python main.py --env linux --dry-run --rounds 4

# Reward-harness sanity — best-of-N over fixed actions, NO LLM at all
python main.py --baseline --rounds 10

# Force fresh Phase-1 training
python main.py --env linux --fresh

# Visualize results
python visualize_rounds.py

# MONITOR LLM training
python monitor.py                 # prints a health report + saves logs/monitor/health.png
python monitor.py --window 50     # smooth over a larger recent window for long runs
```

## RUN INFERENCE ON TRAINED LLMS
```
# Attacker — using the REAL system prompt it was trained with (recommended):
python infer.py --adapter attacker --role \
  --prompt '{"round":5,"current_global_accuracy":0.8,"attack_goal":{"type":"untargeted_degrade","target_accuracy_drop":0.2},"poisoned_client_ids":[0],"benign_layer_details":{}}'

# Defender — real system prompt, feature JSON as the user message:
python infer.py --adapter defender --role \
  --prompt '{"client_ids":[0,1,2,3,4],"features":{}}'

# Free-form prompt, no system prompt at all:
python infer.py --adapter attacker --prompt "Describe a stealthy model-poisoning attack."

# Sample several completions (temperature > 0):
python infer.py --adapter attacker --role --prompt '...' --n 4 --temperature 1.0

# Interactive — load the 3B model ONCE, then keep prompting (best for exploring):
python infer.py --adapter defender --role --interactive

# Pipe a prompt from stdin / a file:
cat my_prompt.json | python infer.py --adapter attacker --role
```

Tune everything (poison fraction, attack goal, GRPO group size `G`, KL,
alternation lengths `K_a`/`K_d`, learning rate, league) in
[`configs/base.yaml`](configs/base.yaml).

### Viewing visualizations on a remote server

```bash
# On server:
cd <repo>/logs/visualizations && python -m http.server 8084
# On your machine:
ssh -i <key> -L 8084:<server>:8084 <user>@<server>
# Open http://localhost:8084/report.html
```

## Configuration

- **`configs/base.yaml`** — FL hyperparameters, `poison_fraction` / `poison_seed`
  / `benign_retrain_each_round`, the `attack.goal`, the `rl:` block (GRPO + LoRA
  + league + reward weights), and inference LLM defaults.
- **`configs/attacker_agent.yaml`** — attacker goal fallback, layer-detail
  precision, poisoned-weight clamp, attacker adapter path.
- **`configs/defender_agent.yaml`** — defender defaults, defender adapter path.

Attack goals (configurable; `untargeted_degrade` is the first experiment):
`untargeted_degrade` (target accuracy drop), `slow_degrade` (per-round drop),
`targeted_label` (per-class — scaffolded).

## Project structure

```
core/         Shared types (ModelUpdate, DetectionVerdict, RoundLog) + aggregator interface
model/        Tiny MLP (~970 params) — the schema both LLMs operate over
data/         MNIST loading & partitioning
clients/      Honest client local training
server/       Central server + FedAvg aggregation
detector/     features.py — per-client per-layer statistical feature extractor
agents/       attacker_agent.py / defender_agent.py (pure prompt+parse), attack_ops.py (operator DSL), llm_client.py
rl/           env, rewards, turns, inference (dry-run), policy (Unsloth+LoRA), grpo, schedule, baseline
metrics/      Ground-truth confusion/TPR/FPR/ASR/APR (research evaluation + reward source)
storage/      Phase-1 checkpoint + RL progress
configs/      YAML configuration
logs/         system.log, round_data/, metrics/, visualizations/
```

See [`SYSTEM.md`](SYSTEM.md) for the full architecture (round loop, reward
definitions, feature spec, GRPO schedule, checkpoint layout) and
[`DATA_PARTION.md`](DATA_PARTION.md) for the data partitioning.


benchmark
python -m benchmark.run_benchmark --rounds 200