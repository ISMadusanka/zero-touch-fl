# Roadmap: zero-touch-fl — Learnable Attacker Frontier

## Overview

This is a brownfield milestone on a working testbed. The attacker currently evades detection but cannot reach its accuracy-degradation goal, so every GRPO rollout scores near zero, group advantages collapse, and the gradient step is skipped. The fix runs in a forced order: first make the goal reachable by tying it to the round's poison budget (nothing downstream means anything while the target is unreachable), then restore the reward's dynamic range with a bounded per-(defense × budget) denominator, then define a graded, ladder-anchored attack success rate that is provably independent of that denominator, then make the benchmark grade fairly against per-round per-defense counterfactuals, then — last, because it is the riskiest addition and the most capable of silently corrupting results — concentrate training on whichever defense the attacker is worst against. The final phase proves the attacker learns before the results table is produced from it. Every phase modifies existing modules against an existing test suite; two new torch-free controller modules (`rl/damage_norm.py`, `rl/defense_bandit.py`) are the only additions, and there are no new runtime dependencies.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Budget-Conditioned Target Ladder** - Make the attack goal a deterministic function of the round's poison budget, with one authoritative source
- [ ] **Phase 2: Bounded Damage Normalization** - Give the reward usable dynamic range per (defense × budget) without letting one outlier poison the denominator
- [ ] **Phase 3: Graded Attack Success Rate** - Replace binary success accounting with a ladder-anchored graded metric, firewalled from the reward normalizer
- [ ] **Phase 4: Fair Cross-Defense Benchmark** - Grade each defense against its own per-round clean counterfactual and report the full confusion picture
- [ ] **Phase 5: Adaptive Defense Scheduling** - Concentrate rounds on the hardest defense with a hard visitation cadence so nothing goes stale
- [ ] **Phase 6: Learning Evidence and Reproducible Results** - Prove the attacker learns, then produce the per-defense table from the trained policy

## Phase Details

### Phase 1: Budget-Conditioned Target Ladder
**Goal**: The attacker faces an accuracy-drop target that scales with the leverage it actually has this round, resolved everywhere through one function, and the bottom rung is confirmed measurable before anything is built on it
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: GOAL-01, GOAL-02, GOAL-03, GOAL-04, GOAL-05, GOAL-06
**Success Criteria** (what must be TRUE):
  1. A training round at poison budget *b* receives the ladder target for *b* (1→0.02, 2→0.04, 3→0.06, 4→0.08, 5→0.12), and editing the ladder in `configs/base.yaml` changes the round's target with no code edit.
  2. A test asserts the reward path, the win gate in `rl/switch.py`, and the attacker prompt all resolve the identical target for the same round; a tree-wide search finds no remaining reader of `attack.target_choices` or `attack.sample_target_in_training`.
  3. A recorded pre-flight measurement reports the standard deviation of `clean_reference_accuracy` across repeated clean rounds for each of the four defenses in `single` mode, and states per defense whether the 0.02 rung clears that noise by the agreed margin — if any defense fails, the bottom rung is raised in config before Phase 2 begins.
  4. The existing test suite passes, with the only edits being tests that asserted the retired `target_choices` sampling behavior.
**Plans**: TBD

### Phase 2: Bounded Damage Normalization
**Goal**: The attacker's damage term spans a usable range in every (defense × budget) cell so GRPO sees non-degenerate rewards, and the denominator cannot ratchet permanently on a single lucky round
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: NORM-01, NORM-02, NORM-03, NORM-04, NORM-05, NORM-06
**Success Criteria** (what must be TRUE):
  1. `rl/damage_norm.py` provides `RunningMaxTable`, importable without torch; an injected-outlier unit test shows the cell's denominator rises by at most the configured bound in one update and decays back as ordinary drops accumulate.
  2. A never-visited (defense, budget) cell returns a floored denominator and the reward computed from it is finite — a cold-start test asserts no division by approximately zero and no `inf`/`nan` reaches the advantage computation.
  3. Rollout scoring leaves the table untouched: a test scores G rollouts and asserts `state_dict()` is unchanged, then commits one round and asserts exactly one cell moved.
  4. Checkpoint and resume preserve every cell — a resumed run's first round normalizes against the same denominator the pre-save run's last round used.
  5. Per-cell denominator history (round index, raw drop, resulting denominator) appears in the run log, so a later jump or collapse is attributable to a specific round without re-running.
**Plans**: TBD

### Phase 3: Graded Attack Success Rate
**Goal**: Attack success becomes a bounded, ladder-anchored number that is comparable across defenses and across time, reported continuously during training, and structurally incapable of reading the reward normalizer
**Mode:** mvp
**Depends on**: Phase 1, Phase 2
**Requirements**: ASR-01, ASR-02, ASR-03, ASR-04, ASR-05, ASR-06
**Success Criteria** (what must be TRUE):
  1. Each committed round emits `clip(drop / ladder_target, 0, 1)`, and the headline ASR is the macro-average of per-(defense × budget) cell means; the pooled micro-average is logged beside it so a divergence between the two is visible.
  2. A test corrupts the running-max table with an extreme injected value and asserts every graded ASR number is unchanged — the firewall between training fuel and success signal holds before any consumer of that signal exists.
  3. The live tracker reports graded ASR and per-defense graded ASR at every reporting interval throughout a run, so a per-defense regression is visible while it happens rather than at the end.
  4. `attack_success_rate` means graded goal attainment, `evasion_rate` means evasion, and `goal_success_rate` remains the binary reference; `benchmark/report.py`, `visualize_rounds.py`, `metrics/tracker.py`, `metrics/types.py`, the tests, and the READMEs all read the renamed fields, with no consumer left on the old meaning.
  5. The full existing test suite passes after the rename.
**Plans**: TBD

### Phase 4: Fair Cross-Defense Benchmark
**Goal**: The benchmark grades every defense against its own per-round clean counterfactual under equal round allocation, so the per-defense table measures the attacker rather than each defense's self-inflicted accuracy sag
**Mode:** mvp
**Depends on**: Phase 1, Phase 3
**Requirements**: BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05, BENCH-06
**Success Criteria** (what must be TRUE):
  1. For every defense and every round, the harness computes a clean no-poison counterfactual on that defense's own evolving model and grades drop against it; the fixed Phase-1 baseline is no longer used as the degradation reference.
  2. A test asserts the counterfactual leaves cross-round defense state untouched for all four defenses including DeFL — `state_dict()` taken before equals `state_dict()` taken after.
  3. Two consecutive clean-counterfactual evaluations of the same round agree exactly under Multi-Krum and under DnC; where internal randomized subsampling was found, it is seeded per round and the determinism test proves it.
  4. The harness resolves its target through the ladder with no `target_drop` argument remaining, and a completed run shows every defense judged an equal number of rounds.
  5. A benchmark run emits the per-defense table with graded ASR, evasion rate, detection rate/TPR, FPR, precision, F1, final accuracy, and mean accuracy.
**Plans**: TBD

### Phase 5: Adaptive Defense Scheduling
**Goal**: Training concentrates on the defense the attacker is currently worst against, driven solely by the graded ASR signal, with a hard cadence that keeps every defense's policy coverage and internal state fresh
**Mode:** mvp
**Depends on**: Phase 2, Phase 3
**Requirements**: SCHED-01, SCHED-02, SCHED-03, SCHED-04, SCHED-05, SCHED-06
**Success Criteria** (what must be TRUE):
  1. `rl/defense_bandit.py` provides a torch-free `DefenseSuccessBandit` whose PFSP-style weights move inversely with measured graded success — a unit test shows a consistently-beaten defense is driven toward the floor weight and never reaches zero.
  2. An integration test corrupts the running-max table and asserts the resulting sampling weights are unchanged, confirming the bandit's only input is the ladder-anchored ASR signal.
  3. Across a simulated run under deliberately skewed weights, every defense is the active judge at least once in every window of K rounds; the test asserts this as a worst-case bound, not an expectation.
  4. `DefenseEnsemble` accepts `selection: "adaptive"` plus `set_weights()`, and a test asserts it references no drop, target, or success quantity; weights are read before `begin_round()` and recorded after commit, so the active defense is fixed for the whole round exactly as it is today.
  5. Two runs from the same `fl.poison_seed` produce an identical sequence of active defenses, and a run interrupted and resumed from checkpoint continues that same sequence.
**Plans**: TBD

### Phase 6: Learning Evidence and Reproducible Results
**Goal**: Demonstrate from run telemetry that the attacker actually receives gradient and improves, and only then produce the per-defense results table from the trained policy
**Mode:** mvp
**Depends on**: Phase 4, Phase 5
**Requirements**: EVID-01, EVID-02, EVID-03, EVID-04, EVID-05
**Success Criteria** (what must be TRUE):
  1. The logged zero-advantage fraction stays low across a training run instead of sitting at 1.0, and mean group advantage is non-zero — the gradient stall that motivated this milestone is measurably gone.
  2. The `success_streak` gate fires and the phase-transition log attributes attacker phase switches to wins rather than to the `max_phase_rounds` cap.
  3. Graded ASR trends upward over the run, reconstructable from logged per-round `(drop, target, defense, budget)` tuples.
  4. Per-defense graded ASR trend lines over the same run show no defense declining while the aggregate rises, confirming the cadence prevented catastrophic forgetting.
  5. A benchmark run from the trained checkpoint under uniform per-defense sampling — not the adaptively-sampled training stream — produces the committed per-defense results table, and re-running from the same seed and checkpoint reproduces it.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Budget-Conditioned Target Ladder | 0/TBD | Not started | - |
| 2. Bounded Damage Normalization | 0/TBD | Not started | - |
| 3. Graded Attack Success Rate | 0/TBD | Not started | - |
| 4. Fair Cross-Defense Benchmark | 0/TBD | Not started | - |
| 5. Adaptive Defense Scheduling | 0/TBD | Not started | - |
| 6. Learning Evidence and Reproducible Results | 0/TBD | Not started | - |

## Ordering Constraints

These are load-bearing, not preferences:

- **Phase 1 is unconditional.** Every downstream metric is meaningless while the goal is unreachable, and GOAL-06 is a gate: if the 2% rung does not clear per-round counterfactual noise, the ladder is wrong and everything after it inherits the error.
- **Phase 2 precedes Phase 3** so ASR-03's firewall test has a real `RunningMaxTable` to corrupt.
- **Phase 2 and Phase 3 both precede Phase 5.** The adaptive sampler must read only the ladder-anchored graded ASR, never the running-max-normalized reward — a normalization artifact reaching the sampler would steer the entire curriculum and compound for the rest of the run. NORM work and SCHED work are deliberately in separate phases.
- **Phase 5 is the last machinery phase** by risk management, not only by data dependency: it is the hardest to debug and the most capable of silently corrupting results, so it must sit on machinery that is already trusted.
- **Phase 6 splits evidence from results.** Training-side proof that the attacker learns is a prerequisite for the benchmark table being trustworthy — learning first, then results.
