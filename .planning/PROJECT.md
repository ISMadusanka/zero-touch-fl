# zero-touch-fl — Learnable Attacker Frontier

## What This Is

A research testbed for a Stackelberg adversarial RL arms race in federated learning: a Qwen2.5-3B attacker LLM selects which of its compromisable clients to poison and emits per-client attack plans from an 11-operator DSL, while a defender (either a second LLM or a panel of four published robust-FL algorithms) decides which clients to filter. GRPO trains one side at a time against a frozen opponent, with verifiable rewards computed from the committed poisoned set.

This milestone fixes a structural failure in `--freeze defender` mode — the attacker evades detection reliably but cannot reach its accuracy-degradation goal, so attack success rate sits near zero and the policy receives no gradient — and replaces binary attack-success accounting with a graded, comparable metric suitable for a per-defense results table.

## Core Value

The attacker must face an **attack goal it can actually reach**, so that GRPO receives a non-degenerate gradient and per-defense attack success rate becomes a meaningful, comparable number.

## Requirements

### Validated

<!-- Inferred from the existing codebase (see .planning/codebase/). These work and are relied upon. -->

- ✓ Phase-1 honest FedAvg training over MNIST with IID and non-IID (FLTrust bias-q) partitioning — existing
- ✓ Phase-2 adversarial round protocol: `begin_round` → attacker action → `build_updates` → feature extraction → verdicts → `evaluate_updates` → `commit` — existing
- ✓ Partial-insider threat model: attacker controls only clients `[0, n_compromisable)`, honest majority guaranteed — existing
- ✓ Attacker chooses *which* of its pool to poison, up to a per-round budget, with a reward penalty for using more clients than needed — existing
- ✓ 11-operator attack DSL with delta-space `scale_delta` as the primary aimable operator — existing
- ✓ GRPO training with group-relative advantages, explicit KL penalty, and zero-advantage guards — existing
- ✓ Verifiable rewards computed from the committed poisoned set, with an oracle-free defender observation — existing
- ✓ Per-round clean counterfactual (`clean_reference_accuracy`) measured under the same active defense — existing
- ✓ Four algorithmic defenses: FLTrust, Multi-Krum, DnC, DeFL — existing
- ✓ `defense.mode: single` with per-round `rotate`/`random`/`fixed` selection, held constant across counterfactual, rollout scoring, and commit — existing
- ✓ Success-gated iterated best-response schedule with league snapshots and curriculum-on-cap — existing
- ✓ Benchmark harness holding one attack fixed across defenses, each evolving its own global model — existing
- ✓ Checkpoint/resume of adapters, FL state, and schedule state — existing

### Active

- [ ] Attack goal target becomes a deterministic function of the round's poison budget (1→2%, 2→4%, 3→6%, 4→8%, 5→12%), replacing `target_choices` sampling
- [ ] Attacker damage term is normalized by an online running max of achieved drop per (defense × budget), so the reward spans a usable range instead of hugging zero
- [ ] The running max is checkpointed as training state, floored to prevent division by ~0, and survives resume
- [ ] Graded attack success rate: `clip(drop / target, 0, 1)` per round, averaged over rounds, replacing binary threshold accounting
- [ ] Benchmark measures drop against a per-round, per-defense clean counterfactual instead of the fixed Phase-1 baseline
- [ ] Adaptive defense sampling: the next judging algorithm is drawn with weight inversely proportional to the attacker's measured success against it, with a floor so no defense starves
- [ ] Per-defense attack success rate is tracked continuously throughout training, not only at the end, to surface catastrophic forgetting
- [ ] Metric renaming: `evasion_rate` for evasion, `attack_success_rate` for graded goal attainment, `goal_success_rate` retained as the binary reference
- [ ] Training-side evidence that the attacker learns: non-zero group advantages, `success_streak` firing, phases switching on wins rather than the `max_phase_rounds` cap
- [ ] Reproducible benchmark run producing a per-defense table of graded ASR, detection rate, FPR, and accuracy

### Out of Scope

- Raising `n_compromisable` or the eval poison budget to make large targets reachable — the strict-minority partial-insider threat model is the premise of the work; the targets move, not the threat model
- Offline calibration sweep for the reachable ceiling — superseded by the online running max, which self-calibrates without a separate experiment
- Telling the attacker which defense judges the round — a defense-conditioned policy learns faster but weakens the threat model; the attacker stays defense-blind
- Per-class evaluation for `targeted_label` goals — a known gap, but this milestone is scoped to `untargeted_degrade`
- Datasets beyond MNIST — the failure being fixed is structural (FedAvg dilution vs. goal magnitude) and reproduces on MNIST
- Changing the defender LLM path (`python main.py` with no `--freeze`) — this milestone targets `--freeze defender` only

## Context

**The failure, and why it is arithmetic rather than a bug.** FedAvg is an unweighted mean over accepted clients (`server/aggregation.py`). With one poisoned client of twenty and a `scale_delta` factor *f*, the poisoned aggregate differs from the clean aggregate by exactly `(f−1)/20 · D_p`. Attack leverage is diluted by `1/n_clients`. Reaching `target_accuracy_drop: 0.20` on a converged MNIST model requires *f* in the hundreds, and `rel_update` scales exactly linearly with `|f|` (`agents/attack_ops.py`), so that client submits an update hundreds of times honest size. Every defense in the panel is norm- or direction-based, so it flags the client, FedAvg drops it, and the drop collapses to zero. Small *f* evades but does nothing; large *f* damages but is filtered. The reachable frontier at budget 1 sits far below 0.20.

**How that becomes a training stall.** `stealth_gate` multiplies the stealth term by `clip(drop/target, 0, 1)` (`rl/rewards.py`). At a drop of ~0.002 against a target of 0.20 the gate is ~0.01, so stealth pays ~0.005 instead of 0.5 while `drop_term` contributes ~0.01. Every rollout in a GRPO group scores approximately zero regardless of what it did; group standard deviation collapses, `group_advantages` returns all zeros with `zero_frac = 1.0`, and `skip_zero_advantage: true` skips the gradient step entirely. The gate was introduced to eliminate the opposite failure — "evade imperceptibly, achieve nothing" was the global optimum worth β=0.5 — and it succeeded, but when the target is unreachable it flattens the entire reward surface rather than just the degenerate optimum.

**Why the schedule never advances.** `attacker_succeeded` requires `drop ≥ win_fraction × target` *and* full evasion (`rl/switch.py`). Even the easiest sampled target of 0.05 demands a drop of 0.03. The `success_streak` gate never trips, so every attacker phase terminates on the `max_phase_rounds: 4000` cap, and `curriculum_on_cap` then hands the next learner an earlier snapshot on the assumption the phase stalled through weakness rather than through an unreachable objective.

**Two findings from the same investigation.** First, the benchmark grades drop against a fixed Phase-1 baseline (`benchmark/metrics.py`) while live training correctly uses the per-round clean counterfactual (`rl/env.py`). Since each defense evolves its own global model and FLTrust drops honest clients every round under a non-IID split, a defense's accuracy sags on its own and that self-inflicted sag is credited to the attacker — skewing graded ASR differently per defense and undermining the cross-defense comparison the benchmark exists to make. Second, single-defense judging already exists: `defense.mode: single` with `selection: rotate` is the shipped default and rotates round-robin every round. The problem is granularity, not absence — the attacker faces a different opponent each round, so consecutive-round learning signal is mixed across four algorithms and it can never specialize.

**Accepted risk.** Adaptive sampling concentrates rounds on the algorithm the attacker is currently worst against, which means long stretches without seeing the others. Combined with a defense-blind prompt the policy cannot partition itself by opponent, so aggregate ASR can climb while per-defense ASR silently regresses. This is a knowing trade to preserve the threat model; the mitigation is continuous per-defense tracking plus a sampling-weight floor.

**Why a moving denominator is safe here.** A high-water-mark denominator is non-stationary and would destabilize a value-based method. GRPO scores its G rollouts within a single round sharing one defense and one budget, so the denominator is constant inside every group and group-relative advantages are invariant to it. Only the absolute reward scale drifts, and nothing downstream consumes it.

**Working state.** Branch `attacker-only-learn` carries uncommitted work across `rl/rewards.py`, `rl/env.py`, `rl/turns.py`, `rl/baseline.py`, `rl/inference.py`, `server/defense_ensemble.py`, `agents/attacker_agent.py`, `agents/attack_ops.py`, `benchmark/harness.py`, `main.py`, `configs/base.yaml`, and four test files. Training runs on a remote Linux CUDA box; a vLLM generation backend handles rollouts and scoring with online LoRA sync and an HF fallback.

## Constraints

- **Threat model**: Attacker controls a strict minority (`n_compromisable: 5` of `n_clients: 20`) — the partial-insider premise is fixed; goals adapt to leverage, not the reverse
- **Tech stack**: Python 3.9+, PyTorch 2.0+, Unsloth ≥2026.6.9, Transformers ≥4.45, PEFT ≥0.13 — the Unsloth floor avoids RoPE cos/sin broadcast crashes
- **Hardware**: CUDA 13+ GPU with 12GB+ VRAM for bf16 LoRA; frozen Qwen2.5-3B base with per-agent LoRA adapters only
- **Aggregation**: Unweighted FedAvg mean — the `1/n_clients` dilution factor is the physical constraint that makes goal calibration necessary
- **Defense integrity**: Algorithms never receive the ground-truth poisoned set; they are real detectors, not oracles
- **Round consistency**: The clean counterfactual, all G rollout scorings, and the commit must be measured under the same defense, or `drop` compares accuracies filtered differently
- **Rollout isolation**: Scoring must not advance defense history (DeFL's critical-learning-period test and Beta trust counts accumulate) — `verdicts(commit=False)` snapshots and restores state
- **Reproducibility**: Seeded from `fl.poison_seed`; adaptive sampling and the running max must both be seeded and checkpointed so runs replay
- **Long runs**: `simulation_rounds: 2000000` — new per-round state must be bounded and must not leak

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Budget-conditioned target ladder 2/4/6/8/12% | Makes the goal a deterministic function of actual attacker leverage instead of an arbitrary sampled constant, so the objective is reachable at every budget | — Pending |
| Super-linear 12% at full budget (not 10%) | Spending the entire pool should demand a super-linear payoff; complements the existing `zeta` collaboration bonus for coordinated multi-client attacks | — Pending |
| Retire `target_choices` / `sample_target_in_training` | Superseded by the ladder — target now derives from budget, which is already sampled | — Pending |
| Normalize damage by online running max | Self-calibrating per (defense × budget) with no separate sweep; safe under GRPO because the denominator is constant within a group | — Pending |
| Grade drop against per-round clean counterfactual in the benchmark | A fixed baseline credits each defense's self-inflicted accuracy sag to the attacker, breaking cross-defense comparison | — Pending |
| Adaptive defense sampling with a weight floor | Concentrates training on the hardest algorithm while preventing any defense from starving out of the curriculum | — Pending |
| Attacker stays defense-blind | Preserves the realistic threat model; accepted cost is catastrophic-forgetting risk, mitigated by continuous per-defense tracking | — Pending |
| Rename `attack_success_rate` → graded goal attainment; evasion becomes `evasion_rate` | The current name means evasion, which is not what the results table needs to report | — Pending |
| Prove learning before producing results | A benchmark table from a policy that never received gradient is not defensible | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-01 after initialization*
