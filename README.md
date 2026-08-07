# Zero-Touch Federated Learning — LLM-Direct Adversarial RL

A research testbed where two LLMs **directly generate** the attack and the
defense in a federated-learning arms race, and are **reinforcement-trained**
against each other with verifiable rewards. Runs on **MNIST or CIFAR-10** —
`--dataset` switches the federation while the **same LLM keeps fine-tuning from
its last checkpoint** (see [Datasets](#datasets-mnist--cifar-10)).

## Overview

Two phases:

1. **Phase 1 (`training_rounds`):** All clients train honestly via FedAvg. The
   global model, each client's weights, and the baseline accuracy are
   checkpointed.
2. **Phase 2 (`simulation_rounds`):** The attacker is a **partial insider** — it
   controls only the first `n_compromisable` clients (default 5 of 20) and
   **chooses which of them to poison** each round while always using the round's
   exact poison-client budget (and coordinating them when it uses more than one).
   - **Attacker LLM** — input: round number, its `controllable_client_ids`, this
     round's exact `max_poison_clients` quota, per-layer **statistics** of each pool
     client's benign weights, current global accuracy, and a configurable attack
     goal. Output: a **client selection + a per-client attack plan** — for each
     chosen client, an ordered list of primitive weight operators (scale,
     sign_flip, mask, add_gaussian_noise, clip, add_constant, permute,
     scale_neurons, blend_random, quantize). A deterministic interpreter applies
     each plan to that client's benign weights to produce the poisoned weights
     sent to the server.
   - **Defense** — see [Defense: the defender LLM is currently
     disabled](#defense-the-defender-llm-is-currently-disabled). By default the
     server defends with a published **algorithm** (FLTrust / DeFL / DnC /
     Multi-Krum) chosen by the [training
     curriculum](#training-curriculum-defense-x-poisoner-count), and only the
     attacker trains. The
     alternative (`defense.mode: llm`) is the **defender LLM** — input:
     per-client, per-layer statistical feature vectors; output: a direct
     **benign/malicious classification** per client, after which the server
     FedAvg-aggregates the clients it did not flag.
   - Because we know the ground-truth poisoned set, the agents get an exact
     **verifiable reward** and are trained online with **GRPO**.
   - Usable under-filled selections are expanded to the exact quota with remaining
     pool IDs. A client counts as poisoned only if its plan **actually changed its
     weights**; no-ops (unparseable output, empty plans, ops all skipped as
     invalid, `scale factor=1.0`) send honest weights, so they are charged as
     *wasted* clients rather than entering the ground truth. The attacker's damage
     is scored against the round's **clean counterfactual** — the accuracy the
     aggregate reaches with no poison at all — so hitting the goal scores the same
     every round it is hit.

There are no hardcoded attack plugins and no episodic-memory feedback loop — the
attacker learns everything via RL.

## Datasets (MNIST / CIFAR-10)

```bash
python main.py --env linux --dataset mnist      # 28x28 digits,  MnistNet   (~970 params)
python main.py --env linux --dataset cifar10    # 32x32 colour,  Cifar10Net (~33.8k params)
```

Same flag on the benchmark: `python -m benchmark.run_benchmark --dataset cifar10`.
Omit it and `data.dataset` from the config is used. (`ciffar10`, `cifar-10` and
`CIFAR10` are all accepted spellings of `cifar10`.)

**What is per dataset, and what is shared:**

| Scope | What | Why |
|---|---|---|
| **Per dataset** | model architecture, raw data dir, Phase-1 checkpoint, live FL state, round/phase counters (`checkpoints/<dataset>/`), and all logs (`logs/<dataset>/`) | None of it is portable. A CIFAR-10 conv `state_dict` cannot load into the MNIST MLP, and round numbering indexes one dataset's FL state. |
| **Shared** | the LLM: `checkpoints/attacker_adapter/` (and `defender_adapter/`) | **This is the point of the switch.** Whichever dataset you run, training *continues* from the last adapter checkpoint — it never restarts from the base model. |

The policy can be shared because it never sees weights: it reads dimensionless
per-layer statistics and emits architecture-agnostic operator plans. Since one
adapter now sees two regimes, the regime is part of the observation — a
`"dataset"` field is included in the attacker's and defender's prompts.

**Per-dataset hyperparameters** live in `configs/base.yaml` → `datasets.<name>`,
deep-merged over the shared `fl` / `data` / `attack` / `defense` / `rl` blocks with
the per-dataset value winning. What CIFAR-10 ships, and why:

- `fl.training_rounds: 80`, `fl.lr: 0.01` — a conv net under plain SGD barely moves
  at MNIST's `0.002`. Tune from there once you see the clean baseline your Phase 1
  actually reaches.
- `attack.target_choices` trimmed to `[0.05, 0.10, 0.15, 0.20]` — a CIFAR-10
  baseline sits far below MNIST's, so an absolute `target_accuracy_drop` is a much
  larger share of the available headroom.
- `rl.max_seq_len: 16384` — the attacker prompt carries `delta_details` for every
  client in the pool, and `Cifar10Net` has 8 parameter tensors to `MnistNet`'s 4, so
  the same pool costs about twice the tokens (~3.2k at the training pool of 5, but
  ~9k+ if the benchmark widens it to all 20 clients). Past `max_seq_len`,
  generations truncate and the attack JSON stops parsing — which reads as a weak
  attacker rather than as a context error. `benchmark/harness.py` logs the real
  token count on round 1 regardless.

**Adding a third dataset**: one `DatasetSpec` in
[`data/datasets.py`](data/datasets.py), one entry in
[`model/__init__.py`](model/__init__.py)`:_BUILDERS`, and (optionally) a
`datasets:` override block. Nothing in the round loop, the operator DSL, the
rewards or the defenses is dataset-aware — the two exceptions, FLTrust (it
fine-tunes its own model on the root set) and the LLM defender (it names the task
in its prompt), take a `dataset` argument.

## Defense: the defender LLM is currently disabled

`configs/base.yaml` ships with **`defense.mode: algorithmic`**. The defender LLM
is switched off and the server defends with the published algorithms already
implemented for the benchmark, **one per round** (which one is set by the
[curriculum](#training-curriculum-defense-x-poisoner-count)):

| Algorithm | What it does |
|---|---|
| `fltrust` | Cosine trust vs a clean root update, norm-rescaled, trust-weighted (NDSS'21) |
| `defl` | Per-layer FGNV + MOUD-Vote + CLP gating with Beta trust (AAAI-23) |
| `dnc` | Spectral (top-singular-direction) outlier filter (NDSS'21) |
| `multikrum` | Distance-based selection of the `n - f` most central updates (NeurIPS'17) |

Details that matter:

- **One algorithm defends the whole round.** It is fixed in `env.begin_round()`
  and used for the round's clean counterfactual, every one of the `G` scored
  GRPO rollouts, and the commit — so the group's rewards stay comparable and the
  advantage is "which plan beat *this* defense", not "which plan got the softer
  defense".
- **The algorithm also produces the aggregate.** FLTrust re-weights and rescales,
  DeFL Beta-weights and CLP-gates; reducing them to a drop-list for FedAvg would
  throw the defense away. So `env.defend()` returns the verdicts *and* the new
  global, and the env commits that state (`env.commit_state`).
- **Scoring never advances the defense's memory.** DeFL's Beta counts and
  `S(t-1)` (and DnC's subsampling RNG) are snapshotted and rolled back for every
  uncommitted call, so all `G` rollouts face an identical defense.
- **Only the attacker trains.** There is no defender policy, so `rl/schedule.py`
  runs attacker-only phases: no defender optimizer, no defender adapter in VRAM,
  no league snapshots or curriculum for it, and **the defender checkpoint on disk
  is never overwritten**. A phase still ends on a sustained win or the cap, which
  is what schedules the honest FL interlude between phases.
- **Rotation is stateful per algorithm.** A rotating algorithm's memory (DeFL)
  only advances on the rounds it is actually selected. The curriculum's
  10-round blocks give it a contiguous stretch to build that memory in; with the
  curriculum off, `defense.selection: round_robin` gives even (but interleaved)
  coverage, and listing a single algorithm pins one for a controlled run.
- Every round log records which defense faced it
  (`attack_metadata.defense` in `logs/<dataset>/round_data/rounds.jsonl`). Compare rounds
  **within** a defense — accuracy drops are not comparable across them.

To restore the original two-sided LLM arms race, set `defense.mode: llm`. Nothing
else changes; the defender adapter resumes from where it left off.

## Training curriculum: defense x poisoner count

Two things about a Phase-2 round used to be independent coin flips: which
algorithm defends it (`defense.selection: random`) and how many clients the
attacker must poison (`attack.sample_budget_in_training`, uniform in
`[1, max_poison_clients]`). That is 4 x 5 = **20 regimes visited in random
order** — the policy almost never got two consecutive rounds in the same one, and
particular pairs ("FLTrust with 5 poisoners") could go hundreds of rounds unseen
while others repeated back to back. Both knobs also move the reward *scale*, so
shuffling them per round injected variance into exactly the signal GRPO
normalizes within a group.

`curriculum.enabled: true` (the shipped default, see `rl/curriculum.py`) replaces
both draws with one fixed, repeating sweep:

```
fltrust    1 poisoner x10 rounds,  2 x10,  3 x10,  4 x10,  5 x10    (50 rounds)
defl       1 poisoner x10 rounds,  2 x10,  3 x10,  4 x10,  5 x10    (50 rounds)
dnc        ... the same five blocks ...                             (50 rounds)
multikrum  ... the same five blocks ...                             (50 rounds)
-> wrap back to (fltrust, 1 poisoner) and repeat forever
```

Every `(defense, poisoner-count)` pair gets exactly `rounds_per_block`
**consecutive** rounds per 200-round cycle — equal opportunity by construction,
not in expectation.

- **Only GRPO rounds advance it.** The honest between-phase FL interlude does not
  call `env.begin_round()`, so a run with many phase switches still covers every
  block.
- **The cursor is checkpointed** alongside `rounds_done` in
  `checkpoints/<dataset>/rl_progress.json`, so a restart continues mid-block
  instead of replaying the first block forever. Editing the sweep takes effect on
  resume (the shape comes from the config, only the cursor from the checkpoint).
- **The attack goal stays fixed.** The curriculum varies *who* and *how many*,
  never *how hard*: every round asks for `attack.goal.target_accuracy_drop`
  (0.10). Per-round target sampling is force-disabled while it is active.
- **`defense.assumed_byzantine` does not track it.** DnC / Multi-Krum keep one
  fixed assumed adversary budget across the 1..5-poisoner blocks — a defense told
  the round's true poisoner count would not be a defense.
- **Evaluation never uses it.** `benchmark/run_benchmark.py` pins one exact quota
  and scores each defense in its own column.
- Each round log carries its block (`attack_metadata.curriculum`: algorithm,
  `n_poisoners`, cycle, block index, round-in-block), so results can be grouped by
  regime without re-deriving the sweep from round numbers.

Set `curriculum.enabled: false` to go back to the independent random draws.

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
  `checkpoints/<dataset>/rl_progress.json` are reloaded and training continues where
  it left off — not just the trained adapters and the round count (`rounds_done`),
  but also the FL round number (`round_index`, so round labels and
  `logs/<dataset>/round_data` keep advancing instead of restarting from the first
  Phase-2 round) and the arms-race
  schedule (`controller`: which agent is learning, phase index, win streak). The
  in-memory opponent league is the one piece not persisted (it restarts empty). Old
  progress files that only hold `rounds_done` still load (the rest falls back safely).
  The **adapter** is not dataset-scoped, so switching `--dataset` resumes the same
  policy against a different federation; only the FL state and counters restart.
- **Switching the base model invalidates old adapters.** A LoRA adapter is
  dimensioned for the exact base it was trained on, so adapters from a previous
  base (e.g. the earlier `Qwen2.5-1.5B-Instruct` run) **cannot** load onto
  `Qwen2.5-3B-Instruct`. If `checkpoints/attacker_adapter/` or
  `checkpoints/defender_adapter/` exist from an old base, delete them (and every
  `checkpoints/<dataset>/rl_progress.json` + `fl_state.pt`) and retrain from
  scratch. The Phase-1 FL checkpoints (`checkpoints/<dataset>/global_model.pt`,
  `client_updates.pt`, `baseline.json`) are LLM-agnostic and can stay.
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

# Pick the dataset. The SAME attacker adapter is fine-tuned by both — each keeps
# its own Phase-1 checkpoint, FL state, round counters and logs.
python main.py --env linux --dataset mnist
python main.py --env linux --dataset cifar10

# Quick smoke run: few rounds. --rounds is an ABSOLUTE budget overriding
# fl.simulation_rounds, so on a resumed run the rounds already done count toward it.
python main.py --env linux --rounds 8

# Run against a different config file
python main.py --env linux --config configs/my_experiment.yaml

# Verbose DEBUG run (GPU) — print EVERYTHING for one short run, to the console AND
# logs/<dataset>/debug.json: the exact attacker/defender LLM prompts + raw outputs, each
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

# Visualize results (both default to the mnist run; pass --dataset for the other)
python visualize_rounds.py
python visualize_rounds.py --dataset cifar10

# MONITOR LLM training
python monitor.py                 # health report + logs/mnist/monitor/health.png
python monitor.py --dataset cifar10
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
cd <repo>/logs/<dataset>/visualizations && python -m http.server 8084
# On your machine:
ssh -i <key> -L 8084:<server>:8084 <user>@<server>
# Open http://localhost:8084/report.html
```

## Configuration

- **`configs/base.yaml` → `datasets:`** — per-dataset overrides of the shared
  `fl` / `data` / `attack` / `defense` / `rl` / `curriculum` blocks, applied once
  `--dataset` is resolved (the per-dataset value wins). See
  [Datasets](#datasets-mnist--cifar-10).
- **`configs/base.yaml`** — FL hyperparameters (`n_clients: 20`,
  `n_compromisable: 5`, `poison_seed`, `benign_retrain_each_round`), the
  `data.dataset` default and `data.noniid_bias` (FLTrust `q`), the `attack` block (`goal`,
  `max_poison_clients`, `sample_budget_in_training`, `eval_poison_clients`), and
  the `rl:` block — GRPO + LoRA + league + reward weights, including the
  `.zeta` multi-client collaboration/diversity bonus and `league_max_snapshots`
  (the ring-buffer cap
  on retained opponent snapshots — each costs ~115 MB of host RAM, so leaving it
  unbounded OOMs a long run).
- **`configs/attacker_agent.yaml`** — attacker goal fallback, layer-detail
  precision, poisoned-weight clamp, attacker adapter path.
- **`configs/base.yaml` → `defense:`** — who defends. `mode` (`algorithmic`, the
  shipped default, vs `llm`), the `algorithms` rotation pool, `selection`
  (`random` / `round_robin` — inert while the curriculum is on), the
  algorithm-draw `seed`, `assumed_byzantine` (DnC / Multi-Krum's assumed adversary
  budget), and each algorithm's own knobs.
  See [Defense](#defense-the-defender-llm-is-currently-disabled).
- **`configs/base.yaml` → `curriculum:`** — the fixed `(defense algorithm x
  poisoner count)` training sweep: `enabled`, `rounds_per_block` (10),
  `poisoner_counts` (`[1,2,3,4,5]`) and `algorithms` (null = `defense.algorithms`,
  in order). Overrides `defense.selection` and
  `attack.sample_budget_in_training`. See [Training
  curriculum](#training-curriculum-defense-x-poisoner-count).
- **`configs/defender_agent.yaml`** — defender-LLM defaults + adapter path. Only
  read when `defense.mode: llm`.

Attack goals (configurable; `untargeted_degrade` is the first experiment):
`untargeted_degrade` (target accuracy drop), `slow_degrade` (per-round drop),
`targeted_label` (per-class — scaffolded).

**Target generalization (untargeted_degrade).** Rather than overfitting a single
target, training randomizes `target_accuracy_drop` each round from
`attack.target_choices` (default `[0.05, 0.10, 0.20, 0.30]`) when
`attack.sample_target_in_training: true` — the same domain-randomization idea as
an optionally sampled exact poison quota, so the policy becomes **target-aware** and generalizes
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

# Evaluate the same trained attacker on the other dataset. Results land in
# logs/<dataset>/benchmark/, so the two never overwrite each other.
python -m benchmark.run_benchmark --rounds 200 --dataset cifar10
```

## Project structure

```
core/         Shared types (ModelUpdate, DetectionVerdict, RoundLog) + aggregator interface,
              run_config.py (dataset resolution + per-dataset checkpoint/log layout)
model/        build_model(dataset) -> the FL model both LLMs operate over:
              mnist_net.py (MLP, ~970 params) / cifar_net.py (CNN, ~33.8k params)
data/         datasets.py (the dataset registry) + loaders.py (download & partitioning)
clients/      Honest client local training
server/       Central server + FedAvg aggregation + algo_defender.py (the round-rotating
              algorithmic defense that currently replaces the defender LLM)
detector/     features.py — per-client per-layer statistical feature extractor
agents/       attacker_agent.py / defender_agent.py (pure prompt+parse), attack_ops.py (operator DSL), llm_client.py
rl/           env, rewards, turns, inference (dry-run), policy (Unsloth+LoRA), grpo, schedule, baseline
metrics/      Ground-truth confusion/TPR/FPR/ASR/APR (research evaluation + reward source)
storage/      Phase-1 checkpoint + RL progress, scoped per dataset
configs/      YAML configuration
checkpoints/  <dataset>/ (FL state per dataset) + attacker_adapter/, defender_adapter/
              (the LoRA policies — SHARED across datasets)
logs/         <dataset>/{system.log, debug.json (--debug), round_data/rounds.jsonl,
              metrics/rounds.jsonl + summary.json, visualizations/, benchmark/}
```

See [`SYSTEM.md`](SYSTEM.md) for the full architecture (round loop, reward
definitions, feature spec, GRPO schedule, checkpoint layout) and
[`DATA_PARTION.md`](DATA_PARTION.md) for the data partitioning.


benchmark
python -m benchmark.run_benchmark --rounds 200
