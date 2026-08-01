# Project Research Summary

**Project:** zero-touch-fl — Learnable Attacker Frontier
**Domain:** Adversarial reinforcement learning for federated-learning poisoning attacks and robust-FL defenses
**Researched:** 2026-08-01
**Confidence:** MEDIUM-HIGH

<!-- Written by the orchestrator via the #222 self-heal path: the synthesizer produced this
     content but fabricated a write restriction instead of persisting it. Content is the
     synthesizer's, reformatted to this template. -->

## Executive Summary

This milestone fixes a structural training failure: the attacker's accuracy-degradation goal is unreachable under the partial-insider threat constraint, so every GRPO rollout in a group scores approximately zero, the advantage spread collapses, and the gradient step is skipped. The four-part solution is to (1) make the goal a deterministic function of actual attacker leverage via a budget-indexed target ladder, (2) normalize the reward's damage term by an online per-(defense × budget) running maximum to restore usable dynamic range, (3) replace binary success accounting with a graded, ladder-anchored metric suitable for cross-defense comparison, and (4) adaptively concentrate training on whichever defense the attacker is currently worst against.

The stack requires **zero new dependencies**. All four additions are inline implementations plus two new torch-free controller modules that mirror the codebase's existing `PhaseController` / `League` pattern. No library implements max-based reward normalization; every existing reward-normalization tool (Welford running mean/std, PopArt, `VecNormalize`) targets variance reduction around a moving center, which is the wrong shape for a monotone ceiling.

**The critical risk is the coupling between capabilities 2 and 4.** If the adaptive sampler's "measured success" statistic is sourced from the running-max-normalized training reward rather than a separate ladder-anchored ASR signal, a single normalization artifact will silently steer the entire curriculum and amplify itself over the remainder of the run — a compounding, silent-wrong-results failure. This is a hard architectural constraint on phase boundaries, not a nice-to-have. Separately, the research endorses the per-round clean counterfactual baseline as literature-backed, but flags the budget-indexed ladder and the graded ratio-to-goal ASR as genuinely novel contributions requiring first-principles defense in the paper rather than appeals to prior art.

## Key Findings

### Recommended Stack

Zero new dependencies. Every addition is bespoke control logic that standard RL/curriculum libraries either do not address at all or would address with disproportionate machinery. This is a research testbed where each added dependency is a reproducibility liability, so inline implementation is the correct call, not a compromise.

**Core additions:**
- `rl/rewards.py::target_for_budget()` — pure function, sibling of the existing `goal_target()`; the single source of truth for the budget→target ladder
- `rl/damage_norm.py::RunningMaxTable` — torch-free controller, floored dict keyed by `(defense, budget)`; ~10 lines of core logic
- `rl/defense_bandit.py::DefenseSuccessBandit` — torch-free controller, PFSP-style inverse-success weighting; ~20-30 lines
- Checkpoint extension — plain JSON aux state alongside existing adapter saves, extending `storage/checkpoint.py` rather than introducing a second mechanism

**Explicitly rejected:** PopArt (rescales a learned critic's output layer — a different problem than dividing a scalar by a running max); Welford / `VecNormalize` (variance reduction around a moving center, wrong shape); Exp3 / UCB adversarial bandits and OpenSpiel/PSRO (overkill for four fixed, named opponents); Syllabus-RL (overkill for a 4-way categorical sampler).

### Expected Features

**Must have (table stakes for a credible results table):**
- Detection rate / TPR, FPR, precision, F1 alongside any attack-success number — reviewers expect the full confusion picture
- Final accuracy, mean accuracy, and clean-accuracy cost per defense
- A methodologically sound degradation reference — the literature specifically criticizes fixed pre-attack baselines
- Budget reported and swept as an independent variable

**Should have (differentiating):**
- Graded per-round attack success replacing binary threshold accounting
- Per-defense clean counterfactual so cross-defense comparison is fair
- Adaptive opponent scheduling with anti-starvation guarantees
- Continuous per-defense ASR tracking throughout training, not only at the end

**Anti-features (deliberately not built):**
- Naive pure self-play with no historical opponent retention — documented to cause cycling and catastrophic forgetting; the existing league snapshots already avoid this
- Reusing the backdoor ASR definition for an untargeted goal — it does not apply and would mislead
- Unioning all four defenses (already established as an anti-pattern in this codebase)

### Architecture Approach

The four additions integrate cleanly with existing component boundaries and require no redesign. The codebase already answers the "where does curriculum state live" question: `PhaseController` and `League` are separate, torch-free, checkpointed controllers driven by `rl/schedule.py`, not folded into the environment. The two new controllers should be siblings of that pattern.

**Major components:**
1. `target_for_budget()` in `rl/rewards.py` — one source of truth; only `FLArmsRaceEnv._round_goal()` changes call site. The reward, the win gate in `rl/switch.py`, and the attacker prompt already consume whatever `goal` dict flows through. The benchmark is the one place currently bypassing this and must start reading it.
2. `RunningMaxTable` — injected into `env` the way `defense` already is. The key is computed inside `env.begin_round()`, the only place that knows both the active defense name and the round budget simultaneously, and exposed via `RoundContext`. **Updated only after commit, never during rollout scoring** — mirroring the codebase's existing rollout-isolation rule for defense state.
3. `DefenseSuccessBandit` — read (`weights()`) before `env.begin_round()`, written (`record()`) after commit, both inside the single `_step_round` function every schedule variant already funnels through. `DefenseEnsemble` gains `set_weights()` and a `selection="adaptive"` option but stays unaware of drop, target, and success.

**The ordering constraint is resolved as a standard one-round-stale deferred bandit.** Adaptive sampling needs success measured after commit, but the defense must be chosen before the clean counterfactual is computed. The read-before / write-after split inside `_step_round` handles this without special machinery.

**Benchmark counterfactual cost is bounded.** Adding a per-round per-defense clean counterfactual means one extra aggregate-and-evaluate per defense per round, reusing the proven `state_dict()` / `load_state_dict()` isolation pattern. The expensive resource — LLM generation — is unchanged, and the model under evaluation is a small MLP, so the added cost is negligible.

### Critical Pitfalls

1. **Sampler/normalizer feedback loop (CRITICAL)** — A ratchet artifact in the running max makes one cell look artificially weak, the sampler reads false weakness, starves the other defenses, and the error amplifies for the remainder of the run. **Prevention:** drive the sampler *only* from the ladder-anchored graded ASR, never from the normalized reward. Name the two signals distinctly. Add an integration test that corrupts the running max and asserts sampler weights are unaffected.
2. **Running-max ratchet** — A single lucky outlier permanently raises the ceiling, deflating all subsequent reward; a raw max never decreases. **Prevention:** use a bounded tracker (EMA, windowed, percentile, or clipped growth per update) rather than a raw max; log the full denominator history so a ratchet event is diagnosable rather than mysterious.
3. **Stateful-defense staleness** — DeFL's critical-learning-period test and Beta trust counts accumulate across rounds. Adaptive sampling doesn't only stale the *policy's* skill against a neglected defense; it stales that defense's *own internal state*, producing anomalous verdicts on re-activation. **Prevention:** prefer a hard visitation cadence (every defense active at least once per K rounds) over a purely probabilistic weight floor.
4. **Adaptive overfit / catastrophic forgetting** — Per-defense ASR silently declines while the aggregate improves. **Prevention:** produce the paper table from a *uniform* evaluation sweep, never from the adaptively-sampled training stream; run an adaptive-vs-uniform ablation.
5. **Aggregation ambiguity** — Average-of-ratios ≠ ratio-of-averages, and both budget sampling and adaptive defense sampling make cell visitation non-uniform and time-varying, creating a real Simpson's-paradox trap. **Prevention:** pick an explicit convention (macro-average of per-cell means recommended) and report both during development.
6. **Ladder miscalibration at the bottom rung** — Budget 1 → 2% sits close to a measurement noise floor this codebase already documented (counterfactual swing sd ≈ 0.012). Single-round drop estimates at that rung are the least trustworthy on the ladder.

**Challenge to a PROJECT.md assumption.** The pitfalls research confirmed that within-group GRPO advantages are invariant to the moving denominator, as PROJECT.md claims — but noted the claim understates the risk. Because `drop_term` is piecewise and saturating, a growing denominator **reshapes** the reward landscape across training rather than merely rescaling it: the same physical drop migrates through different regions of the shaping curve as the denominator grows. The invariance argument holds within a group; it does not make the denominator's growth inert across a run.

## Implications for Roadmap

### Phase 1: Budget-conditioned target ladder
**Rationale:** Foundational — every other addition reads the target. Nothing else is measurable until the goal is reachable.
**Delivers:** `target_for_budget()` as single source of truth; ladder in config; `target_choices` / `sample_target_in_training` retired; win gate and attacker prompt following automatically.
**Avoids:** The flat-reward stall that motivates this whole milestone.
**Pre-flight check:** confirm single-mode defense accuracy standard deviation is small relative to the 2% bottom rung before trusting it.

### Phase 2: Running-max damage normalization
**Rationale:** Restores gradient dynamic range; must land before the sampler so the sampler is never tempted to read it.
**Delivers:** `rl/damage_norm.py::RunningMaxTable`, checkpointed and resumable, floored against cold-start division; denominator history logged.
**Uses:** Inline implementation, no dependency.
**Avoids:** Ratchet pitfall — requires an explicit decision between raw max, percentile, EMA, or clipped growth.

### Phase 3: Graded ASR — live tracker and benchmark, plus per-defense clean counterfactual
**Rationale:** Produces the stable, ladder-anchored ASR signal that Phase 4 depends on, and fixes the fixed-baseline methodological flaw at the same time.
**Delivers:** `clip(drop/target, 0, 1)` per round averaged under an explicit aggregation convention; benchmark grading against per-round per-defense counterfactual; metric renaming (`evasion_rate` / `attack_success_rate` / `goal_success_rate`).
**Implements:** The ASR interface the bandit will consume.
**Avoids:** Aggregation ambiguity; fixed-baseline bias that inflates ASR differently per defense.

### Phase 4: Adaptive defense sampling
**Rationale:** Last, and strictly dependent on Phase 3's graded signal. It is the riskiest addition to debug if the earlier machinery is not already trustworthy.
**Delivers:** `rl/defense_bandit.py::DefenseSuccessBandit` with PFSP-style inverse-success weighting, hard visitation cadence, checkpointed state; continuous per-defense ASR tracking to surface forgetting.
**Avoids:** Sampler/normalizer coupling; stateful-defense staleness; catastrophic forgetting.

### Phase 5: Training evidence and benchmark results
**Rationale:** "Done" requires proving the attacker learns before the table is trustworthy.
**Delivers:** Non-zero group advantages, `success_streak` firing, phases switching on wins rather than the cap, graded ASR climbing; then a reproducible uniform-sweep benchmark table.
**Avoids:** Adaptive overfit — the table comes from uniform evaluation, not the training stream.

### Phase Ordering Rationale

- The ladder is a hard prerequisite: no downstream metric means anything while the target is unreachable.
- Phase 2 and Phase 3 are data-independent of each other and could parallelize, but Phase 2 before Phase 4 is mandatory, and Phase 3 before Phase 4 is mandatory.
- Phase 4 last is a risk-management choice, not a data dependency: it is the hardest to debug and the most capable of silently corrupting results.
- Phase 5 separates "the attacker learns" from "here are the numbers", matching the stated definition of done.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** the tracker-shape decision (raw max vs. percentile vs. EMA vs. clipped) is unresolved and consequential.
- **Phase 4:** PFSP exponent and EMA decay rate need tuning guidance; stateful-defense recovery semantics (reset vs. resync vs. cadence) are unconfirmed.

Phases with standard patterns (research can be skipped):
- **Phase 1:** a pure lookup function with one call site.
- **Phase 3:** arithmetic plus a reuse of an already-proven isolation pattern.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Zero-dependency conclusion well supported; bespoke training-loop techniques lack canonical docs |
| Features | MEDIUM | Counterfactual baseline MEDIUM-HIGH and citable; novel ladder and graded ASR defensible but without precedent |
| Architecture | HIGH | Derived from direct reading of the source files, not inference; seam points explicit |
| Pitfalls | MEDIUM | General RL/statistics claims HIGH; per-codebase interaction claims MEDIUM, from code inspection without external validation |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Tracker shape** (Phase 2 blocker): raw running max vs. bounded alternative — decide before implementation.
- **Cold-start floor** (Phase 2): epsilon, or the budget's own ladder target as the initial denominator.
- **Aggregation convention** (Phase 3): macro-average of per-cell means recommended; must be fixed explicitly and stated in the paper.
- **Stateful-defense semantics** (Phase 4): exact per-round state for DeFL, FLTrust Beta counts; recovery strategy on re-activation unconfirmed.
- **Defense determinism** (Phase 3): whether Multi-Krum or DnC use internal randomized subsampling that would make two identical clean-counterfactual evaluations disagree — flagged as a check to perform, not verified.
- **Bottom-rung noise floor** (Phase 1): confirm the 2% target exceeds counterfactual measurement noise.
- **Sanity sweep**: the offline calibration sweep was descoped in favor of the online running max, which puts more load-bearing risk on the tracker. A lightweight one-time per-defense sanity sweep is recommended even though the full sweep stays out of scope.

## Sources

### Primary (HIGH confidence)
- Direct source inspection: `rl/rewards.py`, `rl/env.py`, `rl/switch.py`, `rl/schedule.py`, `server/defense_ensemble.py`, `server/aggregation.py`, `benchmark/metrics.py`, `benchmark/harness.py`, `agents/attack_ops.py`, `storage/checkpoint.py`
- `.planning/codebase/` — ARCHITECTURE.md, STACK.md, CONCERNS.md, STRUCTURE.md, CONVENTIONS.md

### Secondary (MEDIUM confidence)
- Shejwalkar, Houmansadr, Kairouz, Ramage — "Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Production Federated Learning" (arXiv:2108.10241) — formalizes `I_θ = A_θ − A_θ*` against a no-attack counterfactual; directly endorses the counterfactual baseline fix
- BackFed (arXiv:2507.04903) — per-defense trained-from-scratch no-attack counterfactual; temporal graded metrics (`ASR_t`, `h-ASR`, Lifespan) but not per-round ratio-to-target
- AlphaStar league training — Prioritized Fictitious Self-Play; opponent *i* sampled with weight ∝ (1 − p̂ᵢ)^p — the direct precedent for inverse-success defense sampling
- Bagdasaryan et al. 2020 — canonical backdoor ASR definition (established, but not applicable to the untargeted goal here)
- HuggingFace Trainer `trainer_state.json` / torchtune `recipe_state.pt` — confirms the separate-auxiliary-state checkpoint pattern
- Henderson et al. — seed-variance literature underpinning the multi-seed recommendation

### Tertiary (LOW confidence — verify before citing with specific numbers)
- FLPoison (`vio1etus/FLPoison`) and the 2025 SoK benchmarking paper (arXiv:2502.03801) — only README and abstract were accessible; the claim that it lacks a graded ASR is based on partial reading
- RL-FL attacker papers (arXiv:2303.03320; NeurIPS'22 model-based RL attack) and Meta-Stackelberg papers (arXiv:2410.17431, arXiv:2306.13800) — PDF full-text extraction unavailable in the research environment; claims corroborated only via search snippets
- DreamerV3 percentile-based return normalization — web-synthesized, not primary-source read

**Convergent negative result:** no paper was found reporting an area-under-the-accuracy-degradation-curve metric for untargeted FL poisoning, and no prior work defines the attacker's target as a function of budget. The latter is the single item most in need of explicit first-principles defense in the paper.

---
*Research completed: 2026-08-01*
*Ready for roadmap: yes*
