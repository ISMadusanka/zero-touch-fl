# Zero-Touch Federated Learning — LLM-Direct Adversarial RL

A research testbed where two LLMs **directly generate** the attack and the
defense in a federated-learning arms race on MNIST, and are **reinforcement-
trained** against each other with verifiable rewards.

## Overview

Two phases:

1. **Phase 1 (`training_rounds`):** All clients train honestly via FedAvg. The
   global model, each client's weights, and the baseline accuracy are
   checkpointed.
2. **Phase 2 (`simulation_rounds`):** The attacker is a **partial insider** — it
   controls only the first `n_compromisable` clients (default 5 of 20) and
   **chooses which of them to poison** each round, up to a per-round budget,
   optimizing to use **as few clients as possible** (and to coordinate them when
   it uses more than one).
   - **Attacker LLM** — input: round number, its `controllable_client_ids`, this
     round's `max_poison_clients` budget, per-layer **statistics** of each pool
     client's benign weights, current global accuracy, and a configurable attack
     goal. Output: a **client selection + a per-client attack plan** — for each
     chosen client, an ordered list of primitive weight operators (scale,
     sign_flip, mask, add_gaussian_noise, clip, add_constant, permute,
     scale_neurons, blend_random, quantize). A deterministic interpreter applies
     each plan to that client's benign weights to produce the poisoned weights
     sent to the server.
   - **Defender LLM** — input: per-client, per-layer statistical feature vectors.
     Output: a direct **benign/malicious classification** per client.
   - The server FedAvg-aggregates the clients the defender did not flag.
   - Because we know the ground-truth poisoned set, both agents get an exact
     **verifiable reward**, and both are trained online with **GRPO**.
   - A selected client counts as poisoned only if its plan **actually changed its
     weights**; no-ops (unparseable output, empty plans, ops all skipped as
     invalid, `scale factor=1.0`) send honest weights, so they are charged as
     *wasted* clients rather than entering the ground truth. The attacker's damage
     is scored against the round's **clean counterfactual** — the accuracy the
     aggregate reaches with no poison at all — so hitting the goal scores the same
     every round it is hit.

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
- The policy is **one frozen `Qwen2.5-3B-Instruct` base loaded via Unsloth in
  bf16 LoRA by default** (4-bit QLoRA is available via `rl.load_in_4bit: true` —
  use it only when GPU memory is tight; on a 5090 bf16 is faster as it skips the
  per-matmul dequant), with **two LoRA adapters** over it — `attacker` and
  `defender`. This is exactly "separate checkpoints on the same LLM": the base is
  never duplicated; each policy is its own small adapter, saved independently to
  `checkpoints/attacker_adapter/` and `checkpoints/defender_adapter/`.
  (Loaded with `attn_implementation="eager"` by default — see `rl.attn_implementation`;
  Qwen2.5 also supports `"sdpa"`.)
- GRPO is implemented directly in [`rl/grpo.py`](rl/grpo.py) (environment-coupled
  reward, KL penalty to the frozen base) on top of Unsloth + PEFT — no TRL
  trainer dependency required.
- **Resume**: rerun the same command. Existing adapters and
  `checkpoints/rl_progress.json` are reloaded and training continues where it left
  off — not just the trained adapters and the round count (`rounds_done`), but also
  the FL round number (`round_index`, so round labels and `logs/round_data` keep
  advancing instead of restarting from the first Phase-2 round) and the arms-race
  schedule (`controller`: which agent is learning, phase index, win streak). The
  in-memory opponent league is the one piece not persisted (it restarts empty). Old
  progress files that only hold `rounds_done` still load (the rest falls back safely).
- **Switching the base model invalidates old adapters.** A LoRA adapter is
  dimensioned for the exact base it was trained on, so adapters from a previous
  base (e.g. the earlier `Qwen2.5-1.5B-Instruct` run) **cannot** load onto
  `Qwen2.5-3B-Instruct`. If `checkpoints/attacker_adapter/` or
  `checkpoints/defender_adapter/` exist from an old base, delete them (and
  `checkpoints/rl_progress.json` + `checkpoints/fl_state.pt`) and retrain from
  scratch. The Phase-1 MNIST checkpoint (`global_model.pt`, `client_updates.pt`,
  `baseline.json`) is model-agnostic and can stay.
- **`gpt-4o-mini` (OpenAI) and Ollama `qwen2.5` are inference/baseline only** —
  used by `--dry-run` and `--baseline`. They are **not** fine-tuned.
- **Serving a trained adapter**: use **vLLM** (multi-LoRA hot-swap), or **merge**
  the adapter into the base and export a single GGUF for Ollama (Qwen2.5 is
  supported by Ollama as `qwen2.5`; merging gives up the two-swappable-adapters
  design but is the simplest serving path).
- **Hardware**: Qwen2.5-3B fits comfortably on a single GPU — ~6 GB of weights
  in the default bf16 LoRA (or ~2–3 GB floor under 4-bit QLoRA), so your 5090
  (31 GB) has ample headroom for either. Generation is short: the attacker emits a
  client selection + per-client plans, and the defender emits one verdict per
  client (20 clients). `rl.max_new_tokens` defaults to 1024 to fit the defender's
  full verdict list without truncation.

## Setup

### GPU machine (training)

```bash
pip install -r requirements.txt   # installs unsloth/peft/transformers/bitsandbytes
# Qwen2.5-3B-Instruct is downloaded from Hugging Face on first run (unsloth/Qwen2.5-3B-Instruct)
```

### CPU machine (logic dry-run / baseline only)

```bash
pip install torch torchvision numpy pyyaml matplotlib openai requests
# For --dry-run you also need an Ollama server with qwen2.5:
#   ollama serve & ; ollama pull qwen2.5:3b
```

## Usage

```bash
# Full GRPO training (GPU). Phase 1 runs once, then the RL arms race.
python main.py --env linux

# Attacker vs the CLASSICAL defenses — the defender LLM is switched OFF and the
# defense becomes a fixed ensemble of FLTrust + Multi-Krum + DnC + DeFL voting
# together. No attacker/defender switching: only the attacker is GRPO-trained.
python main.py --env linux --freeze defender

# Quick smoke run: few rounds. --rounds is an ABSOLUTE budget overriding
# fl.simulation_rounds, so on a resumed run the rounds already done count toward it.
python main.py --env linux --rounds 8

# Run against a different config file
python main.py --env linux --config configs/my_experiment.yaml

# Verbose DEBUG run (GPU) — print EVERYTHING for one short run, to the console AND
# logs/debug.json: the exact attacker/defender LLM prompts + raw outputs, each
# poisoning step (attack plan + per-layer poisoned-weight deltas), the per-round
# FL fine-tuning data, and all rewards / GRPO advantages / commit outcomes.
# Third-party library noise is silenced. Without --rounds it runs 3 MORE rounds
# than are already done (each round is logic-identical — it just stops early), so
# debugging a resumed run still executes rounds.
python main.py --env linux --debug
python main.py --env linux --debug --rounds 6     # debug a longer stretch

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
  --prompt '{"round":5,"current_global_accuracy":0.8,"attack_goal":{"type":"untargeted_degrade","target_accuracy_drop":0.2},"controllable_client_ids":[0,1,2,3,4],"max_poison_clients":1,"client_update_stats":{}}'

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

- **`configs/base.yaml`** — FL hyperparameters (`n_clients: 20`,
  `n_compromisable: 5`, `poison_seed`, `benign_retrain_each_round`), the
  `data.noniid_bias` (FLTrust `q`), the `attack` block (`goal`,
  `max_poison_clients`, `sample_budget_in_training`, `eval_poison_clients`), and
  the `rl:` block — GRPO + LoRA + league + reward weights, including
  `reward.attacker.delta` (use-fewer-clients penalty) and `.zeta` (multi-client
  collaboration/diversity bonus), and `league_max_snapshots` (the ring-buffer cap
  on retained opponent snapshots — each costs ~115 MB of host RAM, so leaving it
  unbounded OOMs a long run).
- **`configs/attacker_agent.yaml`** — attacker goal fallback, layer-detail
  precision, poisoned-weight clamp, attacker adapter path.
- **`configs/defender_agent.yaml`** — defender defaults, defender adapter path.

Attack goals (configurable; `untargeted_degrade` is the first experiment):
`untargeted_degrade` (target accuracy drop), `slow_degrade` (per-round drop),
`targeted_label` (make ONE class be misclassified while the rest keep working).

**`targeted_label` is a separate experiment with its own commands, config,
adapters and logs — see [`TARGETED.md`](TARGETED.md).** It never shares state with
the untargeted run (only the Phase-1 honest baseline, so both start identically):

```bash
python train_targeted.py                              # train
python -m benchmark.run_targeted_benchmark --rounds 100   # evaluate
```

As shipped it is a **single-insider** attack: `fl.n_compromisable: 1` and
`attack.max_poison_clients: 1` mean client 0 is the only poisoner, and
`attack.target_label_from_client: 0` makes `data/target_label.py` derive the attacked
class at runtime from client 0's own non-IID shard (its most-represented class) and print
it at startup, since a non-IID split decides that class by RNG rather than by config.

**Target generalization (untargeted_degrade).** Rather than overfitting a single
target, training randomizes `target_accuracy_drop` each round from
`attack.target_choices` (default `[0.05, 0.10, 0.20, 0.30]`) when
`attack.sample_target_in_training: true` — the same domain-randomization idea as
the per-round poison budget, so the policy becomes **target-aware** and generalizes
to any requested drop. The sampled target is placed in the attacker's prompt AND
used by its reward every round (sampled once per round, so all `G` GRPO rollouts in
a group share it). The arms-race success gate is likewise **relative**: an attack
"passes" when its committed drop reaches `rl.win_fraction` (default 0.6) of that
round's target, so phase-switching tracks the sampled target instead of one absolute
floor (`rl.attacker_min_drop` is only the fallback when no target is known). Set the
flag to `false` to train against the single fixed `attack.goal.target_accuracy_drop`.
Evaluation never samples: pick the target at the benchmark with `--goal` (see below).

Both the reward and this gate measure the committed drop against the round's
**clean counterfactual** (the accuracy the aggregate reaches with no poison), not
against the previous round's post-attack accuracy — see
[`SYSTEM.md`](SYSTEM.md#verifiable-rewards-rlrewardspy). Without that, the
memoryless environment (`benign_retrain_each_round: false`) made a repeated,
equally damaging attack score ≈0 from the second round on, so
`rl.success_streak` consecutive wins were unreachable and the arms-race handoff
never fired.

Benchmark a trained attacker against a specific goal (fixed for the whole run):

```bash
python -m benchmark.run_benchmark --rounds 200 --goal 'untargeted_degrade=0.1'

python -m benchmark.run_benchmark --rounds 200 --goal 'untargeted_degrade=0.1' --max-poison-clients 3
# forms: untargeted_degrade=<drop> | slow_degrade=<drop> | targeted_label=<label>
```

## Freezing the defender: attacker vs the classical defenses

```bash
python main.py --env linux --freeze defender
```

This answers a different question from the arms race: *can the LLM attacker beat
conventional Byzantine-robust FL defenses?* Three things change, and nothing else
does:

1. **The defender LLM is deactivated.** No defender adapter is created, loaded,
   optimized or saved — it is not on the GPU at all.
2. **No learner switching.** The attacker is the only learner for the whole run;
   `first_learner` / `K_a` / `K_d` and the opponent league are irrelevant and off.
3. **The defense is an ensemble of the existing algorithms, all of them at once.**
   Every member scores the SAME client updates against the SAME global model and
   votes; a client is rejected once `defense.vote` members agree, and the survivors
   are FedAvg-averaged exactly as they are when the defender LLM answers.

| member | what it detects |
|---|---|
| `fltrust` | **direction** — cosine against a server update trained on a small clean root set |
| `multikrum` | **distance** — how far the update sits from the bulk of the others |
| `dnc` | **spectral** — projection on the top principal direction of the centred updates |
| `defl` | **per-layer** — robust median/MAD outlier vote over each layer's gradient-norm statistic |

Configure it under `defense:` in the config (members, vote rule, and each
algorithm's own knobs). `vote: majority` (default, ⌈n/2⌉) rather than `any`
because Multi-Krum and DnC drop a fixed quota **every** round even when nobody is
malicious, so the union would evict honest clients from the average every round.
Each verdict carries the vote count as its confidence, which keeps the attacker's
stealth reward continuous — fooling 3 of 4 members scores better than fooling
none, so GRPO still gets a gradient against a hard accept/reject defense.

The phase structure is kept with the learner pinned: a phase still ends on a
sustained win or `rl.max_phase_rounds`, and that boundary is what checkpoints the
attacker and triggers the honest FL interlude that advances the shared model.

Everything else — reward, win gate, round logs, `--debug`, `--rounds`, resume — is
unchanged. Artifacts are **namespaced** (`frozen_defender` becomes the run name,
or is appended to `--run-name`), so this run writes to
`logs/frozen_defender/` + `checkpoints/frozen_defender/` and can never resume from
or overwrite the arms-race run's adapters. Point
`rl.frozen_defender_adapter_paths.attacker` elsewhere to change that.

Cheap checks that need neither a GPU nor an LLM/Ollama:

```bash
# End-to-end: fixed attack actions vs the real ensemble, on CPU
python main.py --baseline --freeze defender --rounds 3 --debug

# The same defense as one entry in the evaluation panel
python -m benchmark.run_benchmark --rounds 200 --defenses fedavg,ensemble,fltrust,multikrum
```

## Project structure

```
core/         Shared types (ModelUpdate, DetectionVerdict, RoundLog) + aggregator interface
model/        Tiny MLP (~970 params) — the schema both LLMs operate over
data/         MNIST loading & partitioning
clients/      Honest client local training
server/       Central server + FedAvg aggregation
detector/     features.py — per-client per-layer statistical feature extractor
agents/       attacker_agent.py / defender_agent.py (pure prompt+parse), attack_ops.py (operator DSL), llm_client.py
rl/           env, rewards, turns, defenders (LLM | algorithmic ensemble), inference (dry-run),
              policy (Unsloth+LoRA), grpo, schedule, baseline
benchmark/    Defense panel (fedavg, oracle, llm_defender, fltrust, defl, dnc, multikrum,
              ensemble) + the attacker-vs-defenses harness and reports
metrics/      Ground-truth confusion/TPR/FPR/ASR/APR (research evaluation + reward source)
storage/      Phase-1 checkpoint + RL progress
configs/      YAML configuration
logs/         system.log, debug.json (--debug), round_data/rounds.jsonl,
              metrics/rounds.jsonl + summary.json, visualizations/
```

See [`SYSTEM.md`](SYSTEM.md) for the full architecture (round loop, reward
definitions, feature spec, GRPO schedule, checkpoint layout) and
[`DATA_PARTION.md`](DATA_PARTION.md) for the data partitioning.


benchmark
python -m benchmark.run_benchmark --rounds 200

screen -r fl