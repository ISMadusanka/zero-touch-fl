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
   controls only the first `n_compromisable` clients (10 of 20 as shipped). With
   `attack.fixed_poison_clients` set (the shipped default, 10) **that whole set is
   poisoned every round** and the LLM decides only *how*; set it to `null` and the
   attacker instead **chooses which of its pool to poison** each round, always
   filling the round's exact poison-client budget. Either way it coordinates the
   clients it uses.
   - **Attacker LLM** — input: round number, its `poison_client_ids` (or
     `controllable_client_ids` + this round's exact `max_poison_clients` quota in
     selection mode), per-layer **statistics** of each pool client's benign weights,
     current global accuracy, and a configurable attack goal. The statistics are
     packed positionally (a `stats_key` legend plus one array per layer) and the whole
     prompt is held under `rl.max_context_fill` of the context window — see
     [Attacker prompt budget](#attacker-prompt-budget). Output: a **per-client attack
     plan** (plus the client selection in selection mode) — for each
     chosen client, an ordered list of primitive weight operators (scale,
     sign_flip, mask, add_gaussian_noise, clip, add_constant, permute,
     scale_neurons, blend_random, quantize). A deterministic interpreter applies
     each plan to that client's benign weights to produce the poisoned weights
     sent to the server.
   - **Defense** — see [Defense: the defender LLM](#defense-the-defender-llm). By
     default the server defends with the **defender LLM** — input: per-client,
     per-layer statistical feature vectors, packed positionally and held inside
     `rl.defender_max_prompt_fill` of the context window (see
     [Defender prompt budget](#defender-prompt-budget)); output: a direct
     **benign/malicious classification** per client, after which the server
     FedAvg-aggregates the clients it did not flag. Both sides train, alternating
     on a three-win streak. The alternative (`defense.mode: algorithmic`) replaces
     it with a published algorithm drawn per round (FLTrust / DeFL / DnC /
     Multi-Krum) and trains the attacker only.
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

## Defense: the defender LLM

`configs/base.yaml` ships with **`defense.mode: llm`** — the two-sided arms race.
A trainable defender LLM classifies every client from the per-client statistics in
`detector/features.py`, and the server FedAvgs the updates it did not flag. Both
policies train, and the learner **alternates on a success streak** rather than a
clock (`rl/switch.py`):

- The **attacker** trains against the frozen defender until `rl.success_streak`
  (3) consecutive attacks **pass** — every poisoned client evades detection *and*
  the round loses at least `rl.win_fraction × target_accuracy_drop` of accuracy.
  It is then snapshotted and frozen.
- The **defender** trains against that frozen attacker until it wins 3 rounds in a
  row — TPR ≥ `rl.defender_min_tpr` with FPR ≤ `rl.defender_max_fpr`. Then it
  freezes and the attacker starts again, one rung higher.
- `rl.max_phase_rounds` (200) ends a phase that never gets its streak, so a
  hopeless matchup cannot stall the run. Every phase boundary — win or cap — runs
  the honest FL interlude and may hand the next learner an earlier league snapshot
  of its opponent (`rl.curriculum_on_cap`).

Each side's prompt is budgeted separately; see
[Defender prompt budget](#defender-prompt-budget).

### The algorithmic alternative

Setting **`defense.mode: algorithmic`** switches the defender LLM off. The server
then defends with the published algorithms already implemented for the benchmark,
**one per round** (chosen by the
[training curriculum](#training-curriculum-fair-defense--poisoner-coverage)):

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
- **Rotation is stateful per algorithm.** A rotating algorithm's memory (DeFL's
  Beta counts, DnC's subsampling RNG) only advances on the rounds it is actually
  selected. The curriculum gives each algorithm a contiguous block of rounds, so
  that memory now advances the way it would if the algorithm were deployed alone.
- Every round log records which defense faced it and which curriculum block it
  belongs to (`attack_metadata.defense` / `attack_metadata.curriculum` in
  `logs/round_data/rounds.jsonl`). Compare rounds **within** a defense —
  accuracy drops are not comparable across them.

These algorithms are **not dead under `mode: llm`** — they are what the trained
defender is measured against. `python -m benchmark.run_benchmark` scores the
`llm_defender` column beside every one of them (plus `fedavg` as the no-defense
reference and `oracle` as the upper bound) on identical rounds, against the same
trained attacker:

```bash
python -m benchmark.run_benchmark --rounds 200 --defenses fedavg,oracle,llm_defender,fltrust,defl,dnc,multikrum
```

The `llm_defender` column needs a trained defender adapter; without one it is
dropped with a warning and the rest of the panel still runs. See
[`benchmark/README.md`](benchmark/README.md).

## Training curriculum: fair defense × poisoner coverage

Phase-2 rounds used to draw their two hardest knobs **independently at random**:
the defense algorithm (`defense.selection: random`) and the exact poison quota
(`attack.sample_budget_in_training`, uniform in `[1, max_poison_clients]`). That
gives every (algorithm, #poisoners) pair the same share of rounds *in
expectation* only — over 200 rounds with 4 algorithms × 5 quotas each cell is
`Binomial(200, 1/20)`: mean 10, sd ≈ 3.1, so cells routinely differ 2–3× and the
attacker's gradient budget lands wherever the RNG pointed. The pair also
re-rolled every round, so the policy never got a contiguous stretch against one
defense at one attack strength — the regime it has to learn to exploit.

`curriculum:` in `configs/base.yaml` (on by default) replaces both draws with a
deterministic sweep — outer loop the algorithm, inner loop the poisoner count:

```
fltrust   × 1 poisoner  × 10 rounds → × 2 × 10 → × 3 × 10 → × 4 × 10 → × 5 × 10
defl      × 1 poisoner  × 10 rounds → … (the same five blocks)
dnc       × …
multikrum × …
…then the cycle repeats from fltrust.
```

One cycle is `4 × 5 × 10 = 200` rounds and **every algorithm gets exactly 50 of
them, 10 at each attack strength**. Details:

- **The block is fixed for the whole round**, so it applies to the clean
  counterfactual, all `G` scored rollouts and the commit, exactly like the drawn
  algorithm did.
- **The FL interlude between phases does not consume a slot** — it is not a GRPO
  training round, so a block always gets its full `rounds_per_block` attacker
  rounds. The arms-race phase machinery (`rl/switch.py`) runs on top of the
  sweep without perturbing it.
- **Resume-safe.** The whole position is one integer, checkpointed with the rest
  of the Phase-2 resume state in `checkpoints/rl_progress.json`, so a restart
  continues mid-block instead of rewinding to `fltrust` × 1 every time.
- **The target drop is held fixed too** (`attack.sample_target_in_training:
  false`, `goal.target_accuracy_drop: 0.10`). A block holds the defense and the
  poisoner count fixed precisely so its 10 rounds are comparable; a target
  re-drawn each round changes what "success" means *inside* the block, since the
  win gate is `win_fraction × the round's target` and the reward is normalized by
  it. Turning target sampling back on logs a warning.
- Knobs: `rounds_per_block` (10), `poisoner_counts` (`null` =
  `1..attack.max_poison_clients`), `algorithms` (`null` = `defense.algorithms`,
  in listed order — set a list to narrow or reorder the outer loop). Counts above
  `fl.n_compromisable` are dropped with a warning rather than clamped, since
  clamping would hand the largest quota several blocks per cycle.
- **`attack.fixed_poison_clients` collapses the poisoner axis.** As shipped the
  same 10 clients are poisoned every round, so there is no attack strength left to
  sweep: `poisoner_counts` becomes `[10]` (logged at startup) and a cycle is just
  the 4 algorithms × `rounds_per_block` = 40 rounds. Set
  `attack.fixed_poison_clients: null` to get the count sweep back.
- `curriculum.enabled: false` (or removing the block) restores the random draws;
  `defense.selection` and `attack.sample_budget_in_training` are ignored while it
  is on, and the startup log says so.

Each round log records its block in `attack_metadata.curriculum`
(`algorithm`, `n_poisoners`, `cycle`, `block`, `position`, `block_round`), so a
run can be sliced per (defense, #poisoners) cell without re-deriving it from the
round number.

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
  (31 GB) has ample headroom for either. Generation is short: the attacker emits
  one plan per poisoned client, and the defender emits one verdict per
  client (20 clients). `rl.max_new_tokens` defaults to 1536 to fit ten per-client
  attack plans and the defender's full verdict list without truncation.

## Attacker prompt budget

The attacker's prompt carries per-layer update statistics for **every** client in
its pool, so it grows with the pool — and the pool is now 10 clients. Two things
keep that from crowding the model's context:

- **A positional encoding.** Each layer's statistics are one array of numbers,
  with the names sent once as `stats_key` / `whole_key` and the parameter shapes
  once as `layers`. The same numbers cost about a third of the old
  named-key-per-layer-per-client JSON.
- **A hard fill cap.** `rl.max_context_fill` (0.5) bounds
  `prompt_tokens + rl.max_new_tokens` as a fraction of `rl.max_seq_len`. A 3B
  instruct model's output quality falls off well before its context is actually
  full, so half the window is the ceiling, not the target — at the shipped
  settings (16384 window, 10 clients) a round measures ~30%.

The cap is **enforced by compaction, not truncation**: if a prompt would exceed
it, the observation is re-emitted at the next level down (4 significant figures →
3 → core per-layer statistics → whole-model statistics only), and a client is
never dropped, so what the policy sees is always a complete observation. The level
in use is logged whenever it changes. Token counts come from the real tokenizer
during training and benchmarking (`LLMPolicy.count_prompt_tokens`) and from a
character heuristic on the CPU/dry-run path. Set `max_context_fill: 1.0` to
disable the cap.

## Defender prompt budget

The defender gets **one LLM call per round** and must label **every** client in it,
so that single pass is the whole defense — a prompt that pushes the output schema
out of the model's attention produces an unparseable verdict list, which scores as
"flagged nothing". It is budgeted on **two axes**, both enforced:

```
prompt_tokens                                    <= rl.defender_max_prompt_fill  x rl.max_seq_len   # 30%
prompt_tokens + rl.defender_max_new_tokens       <= rl.defender_max_context_fill x rl.max_seq_len   # 60%
```

One total cap cannot express that, because it trades prompt room against
generation room: a small `max_new_tokens` would silently let the observation grow
to fill the entire call budget. `rl.defender_min_prompt_fill` (20%) is the other
edge of the band — a **reporting-only floor**, never a rejection. Under it the log
says there is room for a richer observation.

Three things make the single call decidable:

- **A positional encoding.** Each layer's statistics are one array of numbers, with
  the names sent once as `layer_key` / `whole_key` and the layer names once as
  `layers`. That is under half the old named-key-per-layer-per-client JSON.
- **A `cohort` reference block**, shaped like a client's own row but holding
  `[medians, mads]` — the across-client median and median absolute deviation of
  every statistic. `rel_norm` arrives cohort-relative already, but `cos_to_mean`,
  `max_pairwise_cos`, `dnc_score` and the per-layer cosines are absolute, so
  without it the model has to invent a scale for "typical". Deviation is then
  `|value - median| / (mad + 1e-6)`, read position by position.
- **A `ranked` block** pre-sorting client ids by the statistics that carry the most
  signal, most suspicious first. Presentation only — the same numbers are in
  `clients` — but sorting 20 rows by a positional column is exactly what a small
  model does unreliably in one pass, and getting it wrong costs the whole round.

The savings from the encoding are spent back on those two blocks rather than
banked; that is the point of the floor.

Over the cap the observation is **compacted, not truncated** (4 significant figures
→ 3 → core per-layer statistics → rankings dropped → whole-model statistics only),
and **a client is never dropped** — a missing client is an automatic wrong verdict.
`rl.defender_max_new_tokens` (1024) is the defender's own generation cap, separate
from the attacker's `rl.max_new_tokens`, and the benchmark's `llm_defender` column
uses it too so the evaluated configuration is the trained one.

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
  --prompt '{"round":5,"current_global_accuracy":0.8,"attack_goal":{"type":"untargeted_degrade","target_accuracy_drop":0.2},"poison_client_ids":[0,1,2],"n_poison_clients":3,"layers":{},"stats_key":[],"whole_key":[],"client_update_stats":{}}'
# (--role reads attack.fixed_poison_clients from --config to serve the same system
#  prompt the adapter was trained with; without it, the payload uses
#  "controllable_client_ids" + "max_poison_clients" instead.)

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
  `n_compromisable: 10`, `poison_seed`, `benign_retrain_each_round`), the
  `data.noniid_bias` (FLTrust `q`), the `attack` block (`goal`,
  `fixed_poison_clients`, `max_poison_clients`, `sample_budget_in_training`,
  `eval_poison_clients`), and
  the `rl:` block — GRPO + LoRA + league + reward weights, the
  `max_seq_len` / `max_new_tokens` / `max_context_fill` prompt budget, including the
  `.zeta` multi-client collaboration/diversity bonus and `league_max_snapshots`
  (the ring-buffer cap
  on retained opponent snapshots — each costs ~115 MB of host RAM, so leaving it
  unbounded OOMs a long run).
- **`configs/attacker_agent.yaml`** — attacker goal fallback, layer-detail
  precision (significant figures), poisoned-weight clamp, attacker adapter path.
  `fixed_poison_set` and the `rl.*` prompt-budget knobs are injected from
  `configs/base.yaml` at startup, not set here.
- **`configs/base.yaml` → `defense:`** — who defends. `mode` (`llm`, the shipped
  default, vs `algorithmic`), the `algorithms` rotation pool, `selection`
  (`random` / `round_robin`, ignored under `mode: llm` and while the curriculum is
  on), the algorithm-draw `seed`, `assumed_byzantine` (DnC / Multi-Krum's assumed
  adversary budget), and each algorithm's own knobs.
  See [Defense](#defense-the-defender-llm).
- **`configs/base.yaml` → `rl.defender_*`** — the defender's own prompt/response
  budget: `defender_max_prompt_fill` (30%), `defender_min_prompt_fill` (20%),
  `defender_max_context_fill` (60%) and `defender_max_new_tokens` (1024). Each
  falls back to the shared un-prefixed key when unset. See
  [Defender prompt budget](#defender-prompt-budget).
- **`configs/base.yaml` → `curriculum:`** — the order training rounds face each
  (defense, #poisoners) pair: `enabled`, `rounds_per_block`, `poisoner_counts`,
  `algorithms`. See
  [Training curriculum](#training-curriculum-fair-defense--poisoner-coverage).
- **`configs/defender_agent.yaml`** — defender-LLM defaults + adapter path. Only
  read when `defense.mode: llm`.

Attack goals (configurable; `untargeted_degrade` is the first experiment):
`untargeted_degrade` (target accuracy drop), `slow_degrade` (per-round drop),
`targeted_label` (per-class — scaffolded).

**The training target is fixed at `0.10`.** `attack.goal.target_accuracy_drop:
0.10` with `attack.sample_target_in_training: false` (the shipped default), so
every training round asks the attacker for the same 10-point drop. It is placed
in the attacker's prompt AND used by its reward, and the arms-race success gate
is **relative** to it: an attack "passes" when its committed drop reaches
`rl.win_fraction` (default 0.6) of the round's target, i.e. 0.06 here
(`rl.attacker_min_drop` is only the fallback when no target is known). Holding it
fixed is what makes a curriculum block's 10 rounds — and one block against the
next — comparable.

> **Move it together with `rl.reward.attacker.alpha`.** The damage term is
> `alpha * drop/target`, so the target is simultaneously the ambition *and* the
> divisor of damage's weight in the reward — raising it alone weakens the very
> thing it asks for. The shipped pair is `alpha 5.0 / target 0.10`; keep
> `alpha/target = 50` and a point of accuracy stays worth what it was, including
> against the absolute `min_reward_spread` / `advantage_std_floor` noise floors.
> What the target genuinely controls is where `drop_term`'s linear region ends —
> the point past which extra damage stops paying. At `0.02` that ceiling was the
> binding constraint: a 2pp cut collected nearly all the available damage reward
> and a 10pp cut was worth only ~1.4× as much.

*Optional target randomization.* Setting
`attack.sample_target_in_training: true` draws `target_accuracy_drop` per round
from `attack.target_choices` (`[0.05, 0.10, 0.15, 0.20]`) instead, making the
policy **target-aware** across drops (sampled once per round, so all `G` GRPO
rollouts in a group share it). It is off because it fights the curriculum: with
the target moving round-to-round, block-to-block differences are partly just
which targets got drawn. Turning it back on while the curriculum is enabled logs
a warning. Evaluation never samples: pick the target at the benchmark with
`--goal` (see below).

Both the reward and this gate measure the committed drop against the round's
**clean counterfactual** (the accuracy the aggregate reaches with no poison), not
against the previous round's post-attack accuracy — see
[`SYSTEM.md`](SYSTEM.md#verifiable-rewards-rlrewardspy). Without that, a
memoryless environment (`benign_retrain_each_round: false`) made a repeated,
equally damaging attack score ≈0 from the second round on, so
`rl.success_streak` consecutive wins were unreachable and the arms-race handoff
never fired.

A round whose defense produced no clean aggregate has **no** counterfactual to
measure. Those rounds are marked `clean_measured: false` in the round log, are
excluded from the damage statistics, and apply **no** gradient — the schedule passes
`skip_update=True` to `grpo_step`. Previously the reference silently fell back to the
current global's accuracy, which made `drop` identically `+0.0000` and indistinguishable
from a measured "the attack achieved nothing".

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
server/       Central server + FedAvg aggregation + algo_defender.py (the round-rotating
              algorithmic defense used under defense.mode: algorithmic)
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
