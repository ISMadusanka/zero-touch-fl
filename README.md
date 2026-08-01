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
     chosen client, an ordered list of primitive operators (scale_delta, scale,
     sign_flip, mask, add_gaussian_noise, clip, add_constant, permute,
     scale_neurons, blend_random, quantize). A deterministic interpreter applies
     each plan to that client's benign weights to produce the poisoned weights
     sent to the server.
   - **`scale_delta` is the one that works.** Every other operator edits the
     client's absolute weights `W`; `scale_delta` edits its *update*, rebuilding
     `W' = G + factor·(W − G)`. Because this network is a positively-homogeneous
     ReLU MLP, scaling `W` barely changes its predictions — measured here,
     `scale factor=10` submits an update **9× the norm of the entire model** and
     costs **under 2%** accuracy, i.e. maximum detectability for no damage.
     `scale_delta` pushes along the direction the round is actually learning in and
     buys roughly **100× more damage per unit of update norm** (22% drop at
     `rel=1.6`). It is also the only operator the LLM can aim: the update it submits
     has `rel_update` exactly `|factor|` × the honest `rel_update` printed in its own
     prompt.
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
   - The attacker's **evasion (stealth) reward is gated on damage**: it is scaled by
     `clip(drop/target, 0, 1)`, so slipping past the defense pays only in proportion
     to what slipping past bought. Ungated, evasion is a flat payment for not being
     flagged — and the easiest way to not be flagged is a perturbation small enough
     to hide inside the honest client spread, which does nothing. That made "poison
     imperceptibly, achieve nothing" the reward's global optimum, and a long run
     converges straight to it: detection rate ≈ 0 **and** attack success ≈ 0.

There are no hardcoded attack plugins, no hardcoded detector rules, and no
episodic-memory feedback loop — the LLMs learn everything via RL.

## Single-agent training (`--freeze`)

`python main.py --env linux` runs the full arms race: both LLMs, alternating
phases, freeze-and-switch. `--freeze <agent>` instead trains **one** agent for the
whole run, with **no switching at all**.

**`--freeze defender` — attacker-only GRPO against classical defenses.** The
defender LLM is deactivated; it is not prompted, not sampled, and not trained. The
server defends with the implemented robust-FL algorithms instead:

| Algorithm | Paper | Detects by |
|-----------|-------|-----------|
| FLTrust | Cao et al., NDSS'21 | cosine trust vs a server update on a small clean root set |
| Multi-Krum | Blanchard et al., NeurIPS'17 | distance to the bulk of the other updates |
| DnC | Shejwalkar & Houmansadr, NDSS'21 | spectral (top-singular-direction) outlier score |
| DeFL | Yan et al., AAAI-23 | per-layer FGNV outlier vote + critical-learning-period gating |

By default **one algorithm judges each round** (`defense.mode: single`,
`defense.selection: rotate` — round-robin, so every defense stays in the
curriculum). Flagged clients are dropped from FedAvg. Setting
`defense.mode: union` restores the old behaviour, where all four judge and *any*
single flag drops the client. Everything else is exactly as in the normal mode:
same attacker prompt, same GRPO step, same reward terms, same win criteria
(`rl/switch.py`), same ASR/TPR/FPR metrics. The only difference is who produced
the verdicts.

**Why one at a time.** Unioning four aggregators is much more aggressive than it
looks. Multi-Krum and DnC drop a fixed count every round by construction, DeFL
always flags at least one, and FLTrust zeroes the trust of any client pointing away
from its root update — which, at `noniid_bias: 0.5`, is most of the honest
federation. On this repo's own Phase-1 checkpoint a **clean** round with no attacker
at all flagged `fltrust 12/20 · multikrum 1/20 · dnc 1/20 · defl 2/20` → **union
14/20**, leaving FedAvg six clients and dropping clean accuracy 0.782 → 0.757. That
breaks attacker training twice over: there is no gap left for any attack to fit
through, and the accuracy swing caused by *which honest clients happened to be
dropped* (sd ≈ 0.012) dwarfs the swing caused by the attack (≈ 0.0003), so the
`drop` reward is mostly defense noise. The result is a policy that converges to
"evade everything, achieve nothing" — detection rate ≈ 0 **and** attack success
≈ 0. Judging one at a time keeps a clean round within 0.01 of undefended
(0.782/0.774/0.774/0.783) and leaves a real frontier to learn against.

Three details make the comparison honest:

- **The round's defense is fixed for the whole round.** It is chosen in
  `DefenseEnsemble.begin_round()` before the clean counterfactual is measured, and
  held through all `G` rollout scorings and the commit — otherwise `drop` would
  subtract two accuracies filtered by different defenses.
- **The clean counterfactual runs through the same defense.** Multi-Krum and DnC
  drop a fixed number of clients every round by construction and DeFL always flags
  at least one, so some honest clients are excluded even on a clean round. The
  round's reference accuracy is therefore measured *with that defense running and no
  poison*, which keeps `drop` the attack's marginal damage rather than free credit
  for honest clients the defense itself discarded.
- **These are detectors here, not aggregators.** Only the verdicts are used; the
  surviving clients go through plain FedAvg. This matters most for FLTrust, whose
  own aggregation rescales every delta to `‖g0‖` and is nearly immune to magnitude
  attacks — as a detector feeding FedAvg it is far weaker, and an attack it fails to
  flag lands in full. The benchmark is the opposite: there each defense evolves its
  own model under its own aggregation rule.

Configure the panel under `defense:` in `configs/base.yaml` (`mode`, `selection`,
which algorithms, the assumed adversary budget `f`/`m`, FLTrust's root-set size,
DeFL/DnC thresholds). A plain `python main.py` run ignores that block entirely.

**`--freeze attacker`** is the mirror image: the defender LLM trains against the
frozen attacker adapter (no algorithmic defenses involved).

**Switching back.** A `--freeze` run advances the round counters and writes only the
learner's adapter — the frozen agent's checkpoint and the saved arms-race schedule
(`checkpoints/rl_progress.json` → `controller`) are left untouched. Re-running plain
`python main.py --env linux` therefore resumes the alternating, defender-LLM arms
race in the phase it stopped in, with no algorithmic defenses in the loop.

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

# ATTACKER-ONLY training: no arms-race switching, and the defender LLM is OFF —
# the server defends with the implemented algorithms instead (FLTrust + Multi-Krum
# + DnC + DeFL, all at once; a client is dropped from FedAvg if ANY of them flags
# it). Attacker rewards / win criteria / ASR are unchanged. See "Single-agent
# training" below — a later plain `python main.py --env linux` picks the
# alternating, defender-LLM arms race back up exactly where it left off.
python main.py --env linux --freeze defender

# The mirror image: train the defender LLM against the frozen attacker adapter.
python main.py --env linux --freeze attacker

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

# ...and the same loop with the algorithmic defenses instead of the defender LLM
# (the cheap way to sanity-check that path before starting a GPU run)
python main.py --env linux --dry-run --freeze defender --rounds 4

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
  `data.noniid_bias` (FLTrust `q`), the `attack` block (`goal`, `target_ladder`,
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
`targeted_label` (per-class — scaffolded).

**Budget-conditioned target ladder (untargeted_degrade).** Rather than a single
fixed target, each round's requested `target_accuracy_drop` is a deterministic
function of that round's sampled poison budget, declared in `configs/base.yaml`
under `attack.target_ladder` (1 → 0.02, 2 → 0.04, 3 → 0.06, 4 → 0.08, 5 → 0.12,
the top rung deliberately super-linear because spending the whole pool demands
more). The mapping is resolved by the single function
`rl/rewards.py::target_for_budget`, which the attacker's prompt, its reward, and
the arms-race win gate all read, so they cannot disagree about a round's target.
The env refuses to start if `attack.target_ladder` does not cover every budget in
`[1, attack.max_poison_clients]`. The reason the target scales with budget at all
is that FedAvg dilutes one poisoned client's leverage by `1/n_clients`, so a
target that ignores the round's budget is unreachable at budget 1 and trivial at
budget 5. The arms-race success gate is likewise **relative**: an attack "passes"
when its committed drop reaches `rl.win_fraction` (default 0.6) of that round's
target, so phase-switching tracks the budget-conditioned target instead of one
absolute floor (`rl.attacker_min_drop` is only the fallback when no target is
known).

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

## Project structure

```
core/         Shared types (ModelUpdate, DetectionVerdict, RoundLog) + aggregator interface
model/        Tiny MLP (~970 params) — the schema both LLMs operate over
data/         MNIST loading & partitioning
clients/      Honest client local training
server/       Central server + FedAvg aggregation + defense_ensemble.py (algorithmic defenses, --freeze defender)
detector/     features.py — per-client per-layer statistical feature extractor
agents/       attacker_agent.py / defender_agent.py (pure prompt+parse), attack_ops.py (operator DSL), llm_client.py
rl/           env, rewards, turns, inference (dry-run), policy (Unsloth+LoRA), grpo, schedule, baseline
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