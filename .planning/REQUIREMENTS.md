# Requirements: zero-touch-fl — Learnable Attacker Frontier

**Defined:** 2026-08-01
**Core Value:** The attacker must face an attack goal it can actually reach, so that GRPO receives a non-degenerate gradient and per-defense attack success rate becomes a meaningful, comparable number.

## v1 Requirements

Requirements for this milestone. Each maps to a roadmap phase.

### Attack Goal Ladder

- [ ] **GOAL-01**: `target_for_budget(budget)` exists in `rl/rewards.py` as a sibling of `goal_target()` and returns the ladder value: 1→0.02, 2→0.04, 3→0.06, 4→0.08, 5→0.12
- [ ] **GOAL-02**: The ladder is declared in `configs/base.yaml` and can be re-tuned without a code change
- [ ] **GOAL-03**: `FLArmsRaceEnv._round_goal()` derives the round's target from the round's sampled poison budget instead of sampling from `target_choices`
- [ ] **GOAL-04**: `attack.target_choices` and `attack.sample_target_in_training` are removed, and no code path can still read them
- [ ] **GOAL-05**: The reward, the win gate in `rl/switch.py`, and the attacker prompt all resolve the round's target through the same single source of truth, verified by a test that they cannot disagree
- [ ] **GOAL-06**: A pre-flight measurement confirms the 2% bottom rung exceeds the per-round clean-counterfactual measurement noise for every defense; if it does not, the rung is raised before training

### Damage Normalization

- [ ] **NORM-01**: `rl/damage_norm.py` provides a torch-free `RunningMaxTable` keyed by (defense, budget) using an EMA high-water mark — rising quickly on a new best, decaying slowly so a single outlier washes out
- [ ] **NORM-02**: The denominator is floored so a cold-start cell can never cause division by approximately zero
- [ ] **NORM-03**: The attacker's damage term is normalized by the active cell's denominator
- [ ] **NORM-04**: The table is updated only after a round commits, never during rollout scoring, matching the existing rollout-isolation rule for defense state
- [ ] **NORM-05**: The table is checkpointed and restored on resume so a resumed run does not reset its denominators
- [ ] **NORM-06**: Per-cell denominator history is logged so a ratchet or collapse event is diagnosable after the fact rather than mysterious

### Graded Attack Success Rate

- [ ] **ASR-01**: A round's graded attack success is `clip(drop / target, 0, 1)`, where `target` comes from the ladder
- [ ] **ASR-02**: Graded ASR is aggregated as a macro-average of per-(defense × budget) cell means, so non-uniform cell visitation cannot skew the headline number
- [ ] **ASR-03**: The graded ASR signal is computed from the ladder target and the measured drop only, with no dependency on the normalized reward, enforced by a test that corrupts the running-max table and asserts the ASR signal is unchanged
- [ ] **ASR-04**: The live training tracker reports graded ASR alongside its existing metrics
- [ ] **ASR-05**: Per-defense graded ASR is tracked continuously throughout training, not only at the end, so regression on a defense that stops being sampled is visible while it happens
- [ ] **ASR-06**: `attack_success_rate` means graded goal attainment, evasion is reported as `evasion_rate`, and the binary `goal_success_rate` is retained for reference; every consumer is updated (`benchmark/report.py`, `visualize_rounds.py`, `metrics/tracker.py`, `metrics/types.py`, tests, READMEs)

### Benchmark

- [ ] **BENCH-01**: The benchmark computes a per-round, per-defense clean counterfactual and grades drop against it instead of the fixed Phase-1 baseline
- [ ] **BENCH-02**: The counterfactual reuses the proven `state_dict()`/`load_state_dict()` isolation pattern so it cannot advance any defense's cross-round state
- [ ] **BENCH-03**: The benchmark resolves the attack target through the ladder rather than its own `target_drop` argument
- [ ] **BENCH-04**: A benchmark run gives every defense equal rounds, independent of training-time adaptive sampling weights
- [ ] **BENCH-05**: The per-defense results table reports graded ASR, evasion rate, detection rate/TPR, FPR, precision, F1, final accuracy, and mean accuracy
- [ ] **BENCH-06**: Multi-Krum and DnC are checked for internal randomized subsampling that would make two identical clean-counterfactual evaluations disagree; if found, it is seeded

### Adaptive Defense Scheduling

- [ ] **SCHED-01**: `rl/defense_bandit.py` provides a torch-free `DefenseSuccessBandit` weighting each defense by PFSP-style inverse measured success
- [ ] **SCHED-02**: The bandit reads only the ladder-anchored graded ASR signal, never the normalized reward
- [ ] **SCHED-03**: A hard visitation cadence guarantees every defense judges at least once per K rounds regardless of sampler weights, protecting defenses that carry cross-round internal state
- [ ] **SCHED-04**: `DefenseEnsemble` gains `selection: "adaptive"` and a `set_weights()` entry point while staying unaware of drop, target, and success
- [ ] **SCHED-05**: The bandit is read before `env.begin_round()` and written after commit, resolving the ordering constraint as a one-round-stale deferred update
- [ ] **SCHED-06**: Bandit state is seeded from `fl.poison_seed`, checkpointed, and restored on resume so runs replay

### Validation Evidence

- [ ] **EVID-01**: Group advantages are demonstrably non-zero across a training run — the zero-advantage fraction is logged and stays low
- [ ] **EVID-02**: The `success_streak` gate fires and attacker phases switch on wins rather than terminating on `max_phase_rounds`
- [ ] **EVID-03**: Graded ASR climbs over the course of training
- [ ] **EVID-04**: Per-defense graded ASR does not regress on defenses the sampler de-prioritizes, confirming the cadence prevents catastrophic forgetting
- [ ] **EVID-05**: A reproducible benchmark run produces the per-defense results table from a trained policy

## v2 Requirements

Deferred. Tracked but not in this roadmap.

### Experimental Depth

- **V2-01**: Adaptive-versus-uniform training ablation quantifying what the curriculum actually bought
- **V2-02**: Full offline calibration sweep mapping the reachable drop frontier per (defense × budget × operator)
- **V2-03**: Multi-seed runs with confidence intervals on every reported number
- **V2-04**: Per-class evaluation enabling the `targeted_label` attack goal
- **V2-05**: Datasets beyond MNIST

## Out of Scope

| Feature | Reason |
|---------|--------|
| Raising `n_compromisable` or the eval poison budget | The strict-minority partial-insider threat model is the premise of the work; the targets move, not the threat model |
| Telling the attacker which defense judges the round | A defense-conditioned policy learns faster but weakens the threat model; the attacker stays defense-blind and the cadence mitigates forgetting |
| Offline calibration sweep as the reachable-max source | Superseded by the online EMA high-water mark, which self-calibrates without a separate experiment |
| Changes to the defender LLM path (`main.py` with no `--freeze`) | This milestone targets `--freeze defender` only |
| Backdoor/trigger-based ASR definition | Does not apply to an `untargeted_degrade` goal; reusing it would mislead |
| Unioning all four defenses | Already established as an anti-pattern in this codebase — a clean round flags 14/20 honest clients |
| New runtime dependencies | Research confirmed all four additions are inline; every dependency is a reproducibility liability in a research testbed |

## Open Decisions

Resolved during phase planning, not blocking:

- **Cold-start seed for NORM-02**: whether an empty cell's denominator starts at a small epsilon or at that budget's own ladder target. The latter makes early rewards target-relative and matches pre-change semantics; the former is more neutral.
- **EMA rise and decay rates for NORM-01**: needs tuning guidance; the rise must be fast enough to track genuine improvement and the decay slow enough not to chase noise.
- **PFSP exponent and success-EMA decay for SCHED-01**: whether these are config-exposed or fixed.
- **Cadence period K for SCHED-03**: must be short enough to keep DeFL's cross-round state fresh and long enough not to defeat the point of adaptive weighting.

## Traceability

Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| GOAL-01 | TBD | Pending |
| GOAL-02 | TBD | Pending |
| GOAL-03 | TBD | Pending |
| GOAL-04 | TBD | Pending |
| GOAL-05 | TBD | Pending |
| GOAL-06 | TBD | Pending |
| NORM-01 | TBD | Pending |
| NORM-02 | TBD | Pending |
| NORM-03 | TBD | Pending |
| NORM-04 | TBD | Pending |
| NORM-05 | TBD | Pending |
| NORM-06 | TBD | Pending |
| ASR-01 | TBD | Pending |
| ASR-02 | TBD | Pending |
| ASR-03 | TBD | Pending |
| ASR-04 | TBD | Pending |
| ASR-05 | TBD | Pending |
| ASR-06 | TBD | Pending |
| BENCH-01 | TBD | Pending |
| BENCH-02 | TBD | Pending |
| BENCH-03 | TBD | Pending |
| BENCH-04 | TBD | Pending |
| BENCH-05 | TBD | Pending |
| BENCH-06 | TBD | Pending |
| SCHED-01 | TBD | Pending |
| SCHED-02 | TBD | Pending |
| SCHED-03 | TBD | Pending |
| SCHED-04 | TBD | Pending |
| SCHED-05 | TBD | Pending |
| SCHED-06 | TBD | Pending |
| EVID-01 | TBD | Pending |
| EVID-02 | TBD | Pending |
| EVID-03 | TBD | Pending |
| EVID-04 | TBD | Pending |
| EVID-05 | TBD | Pending |

**Coverage:**
- v1 requirements: 35 total
- Mapped to phases: 0 ⚠️
- Unmapped: 35 ⚠️

---
*Requirements defined: 2026-08-01*
*Last updated: 2026-08-01 after initial definition*
