# Pitfalls Research

**Domain:** Adversarial RL for federated-learning poisoning (attacker-goal calibration, reward normalization, adaptive curriculum, graded metrics)
**Researched:** 2026-08-01
**Confidence:** MEDIUM overall — general RL/statistics claims (PFSP, seed variance, Simpson's paradox, running-max instability) are HIGH-confidence textbook/literature results; every claim about how they interact with *this* codebase's specific mechanics (GRPO group scoping, stateful algorithmic defenses, `drop_term`'s piecewise shape) is derived by direct code inspection (`rl/rewards.py`, `rl/switch.py`, `rl/env.py`, `configs/base.yaml`) rather than a citable external source, because no prior work combines these four mechanisms — flagged inline as MEDIUM/LOW.

This file does **not** restate anything already catalogued in `.planning/codebase/CONCERNS.md` or `ARCHITECTURE.md`'s "Anti-Patterns" (naive weight-space operators, Sybil cloning, unioned defenses, ground-truth defender input). It is scoped to the four NEW capabilities in this milestone.

---

## Critical Pitfalls

### Pitfall 1: The running-max denominator ratchets on a single outlier and never comes back down

**What goes wrong:**
The high-water-mark tracker for damage-drop per `(defense × budget)` cell is monotonic non-decreasing by construction. One rollout that hits an unusually large `drop` — because of eval-set stochasticity, a lucky non-IID client draw in that round's honest set, or a numerically extreme `scale_delta` factor sampled early while the policy is still near-random — permanently raises the ceiling for that cell. Every subsequent round in that cell is graded against a bar that may not reflect any reproducible policy capability. This is exactly the failure this milestone exists to fix (drop_term saturating flat near zero), reintroduced at the cell level instead of globally, and it is *silent*: nothing crashes, the reward simply erodes over weeks of training and the run looks like "the attacker plateaued" rather than "the denominator got poisoned."

**Why it happens:**
A running max has no mechanism to distinguish "the policy got genuinely better" from "one sample was a statistical fluke." Cold start makes this worse: the first few observations in a cell, made before the LoRA has learned anything, have the highest relative chance of being non-representative outliers (including artifacts from `max_weight_abs` clamping producing a large-but-meaningless accuracy swing), yet — because the tracker is empty — those first observations are the ones most likely to *become* the ceiling.

**How to avoid:**
- Do not track a true all-time max. Track a bounded, decaying statistic instead: either (a) an EMA of a high percentile (e.g. DreamerV3's approach — EMA of the 90th/95th percentile of recent drops, which resists a single outlier by design), or (b) a windowed max over the last *N* committed rounds in that cell (bounded memory, old outliers age out), or (c) a clipped ratchet — the tracker may rise by at most `k×` per update (e.g. ≤25% per commit) regardless of the observed value, so one freak round cannot double the bar in a single step.
- Apply a floor **and** a ceiling relative to the ladder target for that cell, e.g. `running_max ∈ [0.5·target, 3·target]`. This bounds worst-case ratchet damage even if the tracker briefly ingests a bad sample, and keeps the reward loosely anchored to something a human can sanity-check.
- Add an explicit warmup: for the first *M* committed rounds in a cell, normalize by `target` (the ladder value) instead of the running statistic, and only switch over once the tracker has enough samples to be a meaningful estimate. This removes the cold-start division-by-near-zero failure mode entirely rather than patching it with an epsilon floor.
- Seed and checkpoint the tracker (already planned per `PROJECT.md`), but also log its full history (not just the current value) so a post-hoc audit can identify exactly which round caused a jump.

**Warning signs:**
- Log line to watch: per-cell `running_max` value plotted over training rounds. A step-function jump followed by a long flat plateau below `1.0×running_max` (i.e., `drop_term`'s `x = drop/running_max` never again approaching 1) is the signature.
- `mean_advantage` / `zero_frac` (already emitted by `group_advantages` in `rl/rewards.py`) trending toward the zero-advantage collapse specifically in rounds tagged with the cell that jumped, while other cells look healthy.
- Compare the round index of each `running_max` jump against `n_malformed` and raw weight-clamp counters for that round — a jump co-occurring with a clamp event or a high malformed count is a near-certain artifact, not a real capability gain.

**Phase to address:** The phase implementing the reward-normalization mechanism (Capability 2). Ship the bounded/decaying tracker and the audit log in the same phase as the running max itself — do not ship a naive `max()` first and "harden later"; by the time the ratchet is visible in results, weeks of training history are already contaminated.

---

### Pitfall 2: The project's invariance argument for a moving denominator is true locally but does not cover the actual risk

**What goes wrong (evaluating the project's stated assumption):**
`PROJECT.md` states: *"GRPO scores its G rollouts within a single round sharing one defense and one budget, so the denominator is constant inside every group and group-relative advantages are invariant to it. Only the absolute reward scale drifts, and nothing downstream consumes it."*

The first half is **correct and worth confirming**: `group_advantages()` (`rl/rewards.py`) computes `(r_i − mean)/std` over the G rollouts of one round, and since `begin_round()` fixes the round's defense and budget before any rollout is sampled, the running-max denominator really is a single constant for that entire group. If `drop_term` were a *linear* function of `drop/denom`, scaling the whole group by a positive constant would leave `(r_i − mean)/std` exactly unchanged, and the claim would be complete.

But `drop_term` is **not linear** — it is piecewise: `x ≤ 1` is linear (floored at −0.5), `x > 1` is a saturating asymptote (`1 + 0.5·over/(over+1)`, capped near 1.5). A nonlinear reshaping does not commute with a changing denominator the way a rescaling does. As the running max grows across training, the *same absolute drop* moves from the saturating-overshoot region (early training, small denom, most attacks look like "overshoot") into the linear region (later training, large denom, most attacks look like "undershoot"), and the local slope `d(reward)/d(drop)` in these two regions is different. That means the *shape* of the reward landscape the policy is being optimized against changes over the run, not just its scale — the claim "only the absolute reward scale drifts" understates what actually drifts. This does not break any single GRPO gradient step (each step's math is self-consistent, using only that round's fixed denom), but it does mean the effective difficulty of earning `reward ≈ 1.0` in a cell is non-stationary across the *whole run* in a way that is coupled to the ratchet in Pitfall 1: if the denominator ratchets to an outlier-inflated value, "reward ≈ 1.0" may become permanently out of reach for a policy that is, by every other measure (the ladder-based ASR), performing well.

**Verdict:** The within-group GRPO math is sound (confirmed). The broader claim that nothing depends on absolute reward scale needs to be an enforced *design invariant*, not an assumption — see Pitfall 3, where the most likely violation lives.

**How to avoid:**
- Keep the ladder-target-based ASR (`clip(drop/target, 0, 1)`) as the *only* signal anything outside the GRPO gradient step is allowed to read — success-gating (`rl/switch.py` already does this correctly, using `goal_target()`, not the running max), adaptive sampling weights (Pitfall 3), dashboards, and the paper table. Treat the running-max-normalized reward as write-only training fuel with no other consumers, and enforce that with a code comment/test, not just a design note.
- Consider normalizing `drop_term`'s *input* (`x = drop/denom`) rather than letting the denominator drift under an already-nonlinear function, or re-derive the saturation constants so they scale with denom growth (e.g., parameterize `_OVERSHOOT_HALF` relative to the current denom rather than as a fixed constant), so the curve's *shape* stays comparable even as its *anchor* moves.

**Warning signs:** A unit test that samples synthetic `drop` values, computes reward under two different (fixed) denom values from the same cell's history, and asserts the *ordering* of rollouts by reward is preserved — this is the actual invariant GRPO needs (relative ranking within a group), and it should hold trivially; what to watch for is any code path that compares raw reward *magnitudes* across rounds or cells (e.g., a logged "average attacker reward this epoch" used as a headline metric) — that number is not comparable across cells or across time by the design's own logic, and treating it as if it were is the failure mode to catch in review.

**Phase to address:** Same phase as Pitfall 1 (reward normalization). Add the cross-cell/cross-time non-comparability as an explicit code comment on the reward function and a dedicated test asserting no other module reads the raw normalized reward.

---

### Pitfall 3: Wiring adaptive-sampling weights to the same statistic the ratchet can corrupt creates a compounding feedback loop

**What goes wrong:**
Capability 4 (adaptive defense sampling) needs a "measured success" statistic per defense to invert into a sampling weight. Capability 2 (reward normalization) produces a per-`(defense×budget)` running-max-normalized reward. These are two different numbers that happen to live in the same codebase at the same time, and it is easy — implementation-convenient, even — to reuse the training reward as the sampler's success signal, since it is already computed every round. If that happens: a ratchet event in Pitfall 1 (one outlier inflates the running max for, say, FLTrust at budget 3) makes the attacker's *reward* against FLTrust crater even though its actual goal-attainment (ladder ASR) against FLTrust is fine. The sampler reads "reward against FLTrust dropped" as "attacker is newly weak against FLTrust" and reallocates rounds toward it — starving the other three defenses — compounding the very staleness problem `PROJECT.md`'s "accepted risk" paragraph already worries about, but now triggered by a normalization artifact instead of genuine relative difficulty. This is the single most dangerous *silent-wrong-results* path in the whole milestone because it links a metric explicitly designed to be temporarily unstable (Pitfall 1/2) to a curriculum controller that then acts on that instability for the rest of the run.

**Why it happens:**
Both signals are per-`(defense × budget)`, update on the same cadence (per committed round), and are trivially available at the same call site — there is no natural code boundary forcing a developer to notice they are semantically different (one is "self-relative training fuel," the other is "goal-relative comparable success").

**How to avoid:**
- Drive the sampler exclusively from the ladder-based graded ASR (`clip(drop/target, 0, 1)`, already comparable across cells and across time), never from the raw or running-max-normalized reward. Name the two quantities distinctly in code (`reward_signal` vs. `asr_signal`) so a diff that accidentally passes the wrong one into `sampling_weight()` is visually obvious in review.
- Smooth the ASR feeding the sampler with an EMA or trailing window (e.g., last 50–200 committed rounds against that defense) rather than the instantaneous last-round value — single-round ASR is noisy (Pitfall 8 discusses why), and an unsmoothed statistic driving inverse-proportional weights will make the "hardest defense" label bounce round to round, thrashing the curriculum before the attacker gets a sustained block of experience against any one opponent.
- Add an integration test: run a synthetic scenario where one cell's running max is deliberately corrupted with an injected outlier, and assert the sampling weights are unaffected.

**Warning signs:** Log and periodically diff `reward_signal` vs. `asr_signal` per cell; they should be monotonically related but not equal, and a rank-correlation check between "which defense has lowest reward this window" and "which defense has lowest ASR this window" that diverges is the tripwire.

**Phase to address:** The phase that wires Capability 4 (adaptive sampling) must depend on Capability 2 exposing a clearly separate, stable, ladder-anchored ASR output — sequence the phases so the sampler is built against that interface, not against the reward function directly.

---

### Pitfall 4: Adaptive sampling starves stateful defenses of continuity, not just of exposure

**What goes wrong:**
The standard self-play concern about adaptive/prioritized opponent sampling is that a rarely-sampled opponent's *skill coverage in the policy* goes stale (classic catastrophic forgetting). This project has a second, less obvious version of the same problem: the algorithmic defenses themselves are **stateful across committed rounds** — `PROJECT.md`'s "Rollout isolation" constraint explicitly names "DeFL's critical-learning-period test and Beta trust counts accumulate" across commits. If adaptive sampling concentrates dozens or hundreds of consecutive committed rounds on the currently-hardest defense, every *other* defense's internal history (trust scores, critical-learning-period window, whatever per-round state DeFL/FLTrust accumulate) goes stale relative to a global model and client population that kept evolving in the meantime. When the sampler eventually swings back to a long-dormant defense, that defense may verdict anomalously — not because it is a good detector of the current attack, but because its internal state reflects an old FL trajectory. This can manifest as either a false "attacker got much better against X" (X's stale internals over- or under-flag, not the policy) or a false negative signal in the exact opposite direction, and either way it feeds back into Pitfall 3's sampler with a corrupted read.

**Why it happens:**
"The judging defense is a stateless plug-in" is a reasonable simplifying assumption for defense *selection*, but the codebase already documents that at least one defense (DeFL) is explicitly not stateless. Adaptive sampling was designed against the abstraction ("pick which algorithm judges this round") without re-examining whether that abstraction still holds when visitation becomes highly non-uniform instead of round-robin.

**How to avoid:**
- Audit each of the four defenses for what per-round state persists across commits (trust counters, learning-period windows, moving baselines) and, for each, decide explicitly: does it need periodic "warm" exposure to stay meaningful, or should its state be time-decayed / reset when it has been dormant for more than *K* rounds so it doesn't apply ancient context to a new situation?
- Prefer a **hard visitation floor** over a purely probabilistic one: guarantee every defense is the active judge at least once every *K* committed rounds (e.g., forced round-robin insertion every 20th round regardless of sampling weight), rather than relying on a floor *probability* per round, which only bounds staleness in expectation — with a low floor probability and enough rounds, an unlucky streak of non-selection is still plausible. A hard cadence gives a provable staleness bound instead of a statistical one.
- Where a defense's state has drifted stale, consider explicitly re-synchronizing it (e.g., replaying its state update over the skipped rounds' honest baseline, if the algorithm supports it) rather than letting it verdict cold.

**Warning signs:** Track "rounds since last active" per defense; alert if any exceeds, say, 3× the expected round-robin interval. Track each defense's raw verdict statistics (mean confidence, flag rate) immediately after a long dormancy versus its own historical baseline — a discontinuity right at re-activation, uncorrelated with any attacker change, is the signature of stale internal state rather than a real detection shift.

**Phase to address:** Capability 4's phase, before enabling adaptive sampling for real training — this is exactly the kind of check that is cheap to do once (a one-time audit of `server/defense_ensemble.py`'s four algorithms for hidden cross-round state) and expensive to discover after the fact in a corrupted multi-week run.

---

### Pitfall 5: Concentrating training on the hardest defense can make aggregate ASR look better while every individual defense's number gets worse

**What goes wrong:**
This is the standard self-play "overfit to the hardest opponent" failure, given a specific twist here: because the LoRA policy is a single defense-blind network, time spent training hard against defense X necessarily reshapes weights shared with every other defense. If the sampler pours most rounds into the currently-worst defense, the *headline* number a naive dashboard shows — "mean ASR over recently active rounds" — improves because most recent rounds were against a target the policy is actively getting better at, even while the policy's competence against the neglected defenses regresses in the background (Pitfall 4's forgetting). A results table built from "ASR over the training window" rather than "ASR from a final uniform sweep across all four defenses" will look strictly better than reality and will not reproduce if the paper's reviewers re-run a uniform evaluation.

**How to avoid:**
- Never compute the paper-table numbers from the adaptively-sampled training stream. Always run a separate, uniform-sampling (or fixed round-robin) evaluation sweep at each checkpoint used for reporting, exactly as the existing benchmark harness already does ("holding one attack fixed across defenses, each evolving its own global model").
- Track per-defense ASR *trend lines* throughout training (already an Active requirement in `PROJECT.md`) specifically so a reviewer can see whether an easier defense's line is flat, rising, or — the failure to catch — declining while the aggregate rises.
- Run one ablation: adaptive sampling vs. uniform/round-robin sampling, both evaluated with the same final uniform sweep. If adaptive sampling's *macro-average* across defenses (each defense weighted equally) is not clearly better than uniform sampling's, the adaptive mechanism's value proposition for the paper is unproven, regardless of how good its own internal (skewed) numbers look.

**Warning signs:** Divergence between a "micro" aggregate ASR (pooling all committed rounds, implicitly weighted by how often the sampler visited each cell) and a "macro" aggregate ASR (mean of per-defense ASR, each defense equal weight) — see Pitfall 6 for why these can differ sharply, and treat a growing gap between them over training as the tripwire for this pitfall specifically.

**Phase to address:** Benchmark/results phase (whichever phase produces the final per-defense table) — bake the uniform-sweep requirement into that phase's acceptance criteria, not as an afterthought.

---

### Pitfall 6: Averaging the graded ASR the wrong way silently changes the headline number

**What goes wrong:**
Two different, both individually defensible, numbers can be called "attack success rate" for a given defense: (a) the mean, over rounds, of `clip(drop_i/target_i, 0, 1)` — average of ratios — and (b) `clip(mean(drop_i)/mean(target_i), 0, 1)` — ratio of averages. Because `target_i` varies round to round (the ladder depends on the round's randomly sampled budget), and because adaptive sampling makes the *number of rounds observed per `(defense, budget)` cell* non-uniform and time-varying, these two aggregation choices can diverge meaningfully — this is a direct instance of the classic "average of ratios ≠ ratio of averages" trap (a Simpson's-paradox-family issue), and it is easy to compute either one without realizing a choice was made. A results table that silently mixes both conventions across different defenses (e.g., because one code path pools per-round then averages, another averages per-cell then pools cells) is not comparable even to itself.

**How to avoid:**
- Pick one convention and state it explicitly next to every ASR number in code, logs, and the paper: recommended default is a **macro-average of per-`(defense, budget)`-cell means** (average within each of the 5 budget rungs first, then average the 5 rung-level numbers) — this prevents whichever rungs happen to get sampled more (because adaptive sampling skews defense visitation, and `sample_budget_in_training` skews budget uniformly but round *counts* per cell still vary) from dominating the aggregate purely by frequency.
- Report both the macro-average and the pooled (micro) average side by side at least during development, specifically to catch cases where they disagree — a large gap is itself a diagnostic signal (it means performance is very uneven across cells, which is useful to know, not just a nuisance to average away).
- Keep this convention identical between the live-training telemetry and the final benchmark table; do not let the benchmark harness (which evaluates uniformly) implicitly use a different averaging convention than the training-time tracker (which observes a skewed cell distribution) without noting the difference.

**Warning signs:** Compute both aggregates in the same log line every time ASR is reported; a persistent gap greater than a few points between them, or a gap that grows over training as adaptive sampling skews the cell-visitation distribution further, is the signal to investigate before trusting either number for the paper.

**Phase to address:** Metrics/benchmark phase — decide and document the convention before any numbers are generated for the results table, not after.

---

### Pitfall 7: Clipping the graded ASR at 1.0 hides exactly the overshoot signal the reward function is designed to reward

**What goes wrong:**
`stealth_gate`/the proposed ASR metric use `clip(drop/target, 0, 1)`, which is the right design for a bounded, probability-like success rate — but it means an attack that clears its target by 3× and one that clears it by 1% look identical in the reported ASR (both `1.0`). `drop_term`'s reward function, by contrast, deliberately keeps rewarding overshoot (up to +0.5 asymptotically) specifically because "how much margin the attack has" is informative during training. If only the clipped ASR is reported in the paper table, a reader cannot tell "the attacker barely, unreliably clears the bar" from "the attacker massively exceeds the bar every time" — two very different claims about how robust the finding is, and a reviewer question ("how close to the threshold are these numbers, really?") that the clipped metric cannot answer.

**How to avoid:**
- Report the clipped ASR as the headline (for comparability and boundedness) but also report the unclipped mean `drop_i/target_i` (or at minimum, the distribution — median and IQR — of the ratio) as a supplementary statistic, so overshoot margin is visible without corrupting the primary bounded metric.
- When budget=5's target (12%, deliberately super-linear) is involved, this distinction matters more than at the other rungs, since that rung is explicitly designed to demand a qualitatively different (coordinated multi-client) strategy — collapsing "barely made it" and "dominated it" into the same `1.0` at exactly the rung meant to test collaboration quality throws away the most interesting data point in the ladder.

**Warning signs:** If every reported ASR in a cell sits at exactly `1.0` with no variance, that is itself suspicious — it likely means the metric is saturated and the unclipped ratio (or the running-max-normalized reward) should be inspected to see how much headroom actually exists.

**Phase to address:** Metrics phase, alongside Pitfall 6.

---

### Pitfall 8: The 2% floor rung sits close to the counterfactual's own measurement noise

**What goes wrong:**
`ARCHITECTURE.md`'s own prior investigation found that, in *union* mode, the accuracy swing from *which* honest clients a defense happens to drop has `sd ≈ 0.012` — larger than the entire attack effect it was trying to measure (`≈0.0003`). Single-mode judging reduces this (fewer honest clients get dropped per round), but does not necessarily eliminate it: the clean counterfactual re-runs FedAvg + evaluation for a hypothetical no-poison round, under the *same* active defense, and that defense still makes real (if less aggressive) filtering decisions on the honest-only set, decisions which can vary round to round with normal training-noise variation in the client updates. At the new ladder's smallest rung (budget 1 → target 2%), a counterfactual noise floor of even a few tenths of a percent is a meaningful fraction of the signal being measured, and at that rung, single-round `drop` estimates are the least trustworthy of the whole ladder — which is also, not coincidentally, the rung every training run passes through most often if `sample_budget_in_training` samples budgets uniformly, since it's one of five equally likely outcomes.

**How to avoid:**
- Do not report or gate on a single round's `drop` at the 2% rung; require averaging over enough rounds (or an explicit confidence interval) before treating a single low-budget round's ASR as meaningful, especially for any success-streak-style gating logic that reads absolute drop values.
- Verify whether any defense algorithm has non-deterministic internals (e.g., randomized subsampling in a spectral-decomposition step) that would make two evaluations of "the same round's" clean baseline disagree purely from RNG, independent of any real attack signal — if so, seed that RNG deterministically per round for both the counterfactual pass and the real pass, so `drop` measures only "was there poison," not "did the algorithm's internal randomness also change."
- For the final benchmark table, run multiple `fl.poison_seed` values (and, if feasible, multiple non-IID partition draws) and report the resulting spread, not a single run's point estimate — this is the standard deep-RL-evaluation fix (Henderson et al.'s "Deep RL That Matters" and follow-on seed-count analyses found 5 seeds routinely insufficient to distinguish methods; high-variance settings may need on the order of 25) applied here to a setting that already has an internally-documented noise source at the exact same order of magnitude as the smallest target.

**Warning signs:** Compute the standard deviation of `clean_reference_accuracy` across multiple clean (`no-poison`) rounds under a fixed defense and fixed global model — if that spread is not comfortably smaller (e.g. 3-5x) than the 2% target, the smallest rung cannot produce a trustworthy single-round ASR, full stop, and this should be measured *before* trusting any budget-1 result in the paper table.

**Phase to address:** Benchmark/results phase, but the diagnostic measurement (clean-round accuracy variance under single-mode defense) should happen as an early validation step in whichever phase implements the ladder, since it determines whether the 2% rung is usable at all.

---

### Pitfall 9: The ladder is calibrated once, globally, but the reachable ceiling is defense-specific

**What goes wrong:**
The target ladder (2/4/6/8/12%) is a function of `round_budget` only — it is the same five numbers regardless of which of the four algorithmic defenses is currently judging. But the four defenses have different detection strictness and therefore different *reachable* damage ceilings at a given budget (this is precisely why the milestone introduces a *defense-specific* running max for the reward, per Capability 2). This means the ladder can simultaneously be well-calibrated for one defense (e.g., DnC, if it's comparatively permissive) and badly miscalibrated for another (e.g., FLTrust, if it's comparatively strict) — some `(defense, budget)` cells may have a ladder target the attacker can clear immediately and trivially (leaving the drop_term saturating in the flat overshoot region most rounds, weak gradient), while others may have a target that stays out of reach for a long time even as the running-max reward inside that cell looks "healthy" relative to its own history (Pitfall 2's monitoring-interpretation trap: reward ≈ its own cell's ceiling is not the same claim as "ASR is near 1"). A training log that only shows reward-per-cell will look uniformly fine; only the ladder-relative ASR per `(defense, budget)` cell reveals the miscalibration.

**How to avoid:**
- Before trusting a full training run, do a cheap calibration pass: for each defense, sweep a range of `scale_delta` factors at each budget level against a frozen (non-learning) or lightly-tuned policy and record the maximum achievable drop under that defense's actual filtering. Compare this per-defense ceiling against the shared ladder target at that budget. Any cell where the ceiling is far below the target (structurally unreachable) or far above it (trivial, no learning pressure) should be flagged before RL training starts, not discovered after weeks of a run.
- Since `PROJECT.md` explicitly retires "offline calibration sweep... superseded by the online running max, which self-calibrates without a separate experiment" as Out of Scope, the running max is the *only* mechanism catching this at runtime — which makes Pitfall 1's ratchet robustness doubly important, since it is now load-bearing for calibration validity, not just reward smoothing. Consider at minimum a lightweight sanity check (not a full offline sweep) run once at the start of training per defense, to seed the running-max trackers with a reasonable initial estimate rather than starting them cold at zero/floor for every cell.

**Warning signs:** Per-`(defense, budget)` ladder-relative ASR that is persistently near `0` for many hundreds of rounds in one specific cell while the schedule (`rl/switch.py`) never sees `attacker_succeeded()` return true for that cell's rounds — distinguishable from Pitfall 1 by checking whether the *reward* (running-max-relative) is also stuck near zero (true miscalibration/unreachable) or is healthy (Pitfall 2's monitoring trap — reward saturated against its own history, ASR still low against the fixed ladder, meaning the cell is *hard but the policy has adapted to its own ceiling*, not necessarily broken).

**Phase to address:** The phase implementing the ladder (Capability 1), as a pre-flight validation step, and continuously monitored throughout the Capability 2/4 phases since the running max is the only runtime safety net once the offline sweep is deliberately skipped.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|-----------------|------------------|
| Reusing the raw/normalized attacker reward as the adaptive-sampler's success signal (Pitfall 3) instead of a dedicated ladder-based ASR channel | Less code, one fewer signal to compute and thread through | Compounding feedback loop between reward ratchet and curriculum skew — corrupts both simultaneously | Never — split the two signals from the start |
| True running max instead of a bounded/decaying tracker (Pitfall 1) | Simplest possible implementation, "self-calibrating" as designed | Permanent ratchet from a single outlier; the exact failure this milestone exists to fix, recurring at cell granularity | Only acceptable with a hard ceiling relative to the ladder target and a logged audit trail from day one |
| Skipping the offline per-defense calibration sweep (already an explicit Out of Scope decision) | Saves a separate experiment; matches "self-calibrating" design philosophy | No independent check that the shared ladder is reachable under the strictest defense; the running max becomes silently load-bearing for calibration correctness | Acceptable only if a lightweight one-time sanity sweep still seeds the trackers and a monitoring dashboard exists (Pitfall 9) |
| Reporting ASR from the adaptively-sampled training stream instead of a uniform re-evaluation sweep (Pitfall 5) | No extra evaluation runs needed | Headline numbers don't reproduce under a fair (uniform) re-run; undermines the cross-defense comparison the benchmark exists to make | Never for anything destined for the paper table; fine for a quick in-training sanity check only |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|-----------------|
| Unbounded running-max history kept only as a scalar (no audit trail) | Cannot diagnose *why* a cell's reward collapsed weeks later | Log the full per-cell tracker history (value + round + raw drop that caused each update), not just the current scalar | As soon as a ratchet event happens and needs post-hoc investigation — by definition, after the fact |
| Recomputing sampler weights from an unsmoothed, single-round success statistic every round | Sampling distribution changes every round, attacker never gets a sustained block of experience against one defense | EMA or windowed statistic (Pitfall 3) with a rate limit on how fast weights can shift | Visible once the "which defense is hardest" label starts flipping faster than the success-streak window (`success_streak`) can register a win against any of them |

## "Looks Done But Isn't" Checklist

- [ ] **Running-max normalization:** Often missing a ceiling/decay — verify the tracker cannot ratchet unboundedly from one outlier round; check it against a synthetic injected-outlier test, not just normal training data.
- [ ] **Adaptive defense sampling:** Often missing a *hard* visitation cadence — verify a probabilistic floor alone actually bounds worst-case staleness for the training length planned, not just in expectation (Pitfall 4).
- [ ] **Graded ASR reporting:** Often missing an explicit aggregation-convention statement — verify the paper table and the training telemetry use the *same* averaging rule (macro vs. micro, Pitfall 6) and that it's written down somewhere a reader can check.
- [ ] **Budget-conditioned ladder:** Often missing a per-defense reachability check — verify at least a lightweight sanity sweep confirms every `(defense, budget)` cell's target is neither trivial nor structurally unreachable before trusting a long run (Pitfall 9).
- [ ] **Per-round counterfactual:** Often missing a noise-floor characterization — verify the standard deviation of clean-round accuracy under each single-mode defense is measured and is comfortably smaller than the smallest ladder target (2%) before trusting budget-1 results (Pitfall 8).
- [ ] **Catastrophic-forgetting mitigation:** Often missing an actual regression alarm — verify "continuous per-defense tracking" (already planned) has a concrete threshold/alert, not just a dashboard nobody is required to look at.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|----------------|------------------|
| Running-max ratchet corrupted by an outlier (Pitfall 1) | LOW if audit trail exists, HIGH if not | With a logged history: identify the offending round, cap/reset the tracker to a percentile of its own history excluding the outlier, resume. Without a history: must re-derive a reasonable ceiling from a fresh calibration sweep or accept the current run's numbers are unreliable for that cell. |
| Sampler corrupted by reward-signal coupling (Pitfall 3) | MEDIUM | Re-point the sampler at the ladder ASR, recompute weights retroactively from logged per-round drop/target values (if logged, which they should be regardless of reward normalization), and re-run only the affected phase rather than the whole training run. |
| Discovered a mis-calibrated ladder cell mid-run (Pitfall 9) | MEDIUM–HIGH | If unreachable: treat as a data point (report the ceiling as a finding, not a bug — "budget 5 against FLTrust tops out at X% under the partial-insider constraint" is itself a legitimate result) rather than re-tuning the ladder after the fact, which would invalidate comparability with earlier training. If trivial: no correction needed, just note the rung provided little training signal for that defense. |
| Micro/macro ASR aggregation mismatch discovered late (Pitfall 6) | LOW | Recompute from logged per-round `(drop, target, defense, budget)` tuples — if these are logged (they should be, independent of which aggregate is reported), both conventions can be reconstructed after the fact with no re-training needed. |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-------------------|----------------|
| Running-max ratchet (P1) | Reward-normalization phase (Capability 2) | Synthetic outlier-injection test; per-cell tracker history logged and inspected before any long run |
| Nonlinear reshaping under a moving denominator (P2) | Reward-normalization phase (Capability 2) | Unit test asserting only rank order (not magnitude) of rewards within a group matters; code-level enforcement that raw reward is never read outside the GRPO step |
| Sampler/ratchet feedback loop (P3) | Adaptive-sampling phase (Capability 4), sequenced after Capability 2 exposes a stable ASR interface | Integration test injecting a corrupted running max and asserting sampling weights are unaffected |
| Stateful-defense staleness (P4) | Adaptive-sampling phase (Capability 4) | Audit of each defense's cross-round state; hard visitation-cadence test; re-activation discontinuity check |
| Overfit-to-hardest-defense (P5) | Benchmark/results phase | Uniform-sweep-only reporting rule; adaptive-vs-uniform-sampling ablation with macro-averaged comparison |
| Ratio-aggregation ambiguity (P6) | Metrics phase | Written, single aggregation convention; both macro and micro logged side by side during development |
| Clipped-ASR overshoot blindness (P7) | Metrics phase | Unclipped ratio (or distribution) reported alongside clipped ASR, especially for the budget-5 rung |
| Counterfactual noise floor at the 2% rung (P8) | Ladder phase (Capability 1) pre-flight + Benchmark phase | Measured clean-round accuracy stddev under each single-mode defense, compared against the 2% target before trusting budget-1 numbers; multi-seed benchmark runs |
| Defense-specific ladder miscalibration (P9) | Ladder phase (Capability 1) | Lightweight per-defense calibration sanity sweep before RL training; continuous ladder-relative-ASR monitoring per cell throughout training |

## Sources

- Direct code inspection: `rl/rewards.py` (`drop_term`, `stealth_gate`, `attacker_reward`, `group_advantages`), `rl/switch.py` (`SwitchConfig`, `attacker_succeeded`, `PhaseController`), `rl/env.py` (round-budget sampling), `configs/base.yaml` — HIGH confidence, primary source for all codebase-specific claims.
- `.planning/PROJECT.md` — the "Accepted risk" and "Why a moving denominator is safe here" paragraphs, directly evaluated in Pitfall 2 and the catastrophic-forgetting discussion (Pitfall 4/5).
- `.planning/codebase/ARCHITECTURE.md` — "Unioning all defense algorithms" anti-pattern, cited for its measured `sd ≈ 0.012` counterfactual noise figure (Pitfall 8).
- AlphaStar (DeepMind) league training / Prioritized Fictitious Self-Play — opponent sampling weighted toward the opponent currently beating the agent, mitigated via a persistent league + experience replay (not full opponent retirement). MEDIUM confidence (web search synthesis, not a direct paper read).
- DreamerV3-style return normalization (EMA of high/low percentiles) as a robust alternative to raw running max/std — MEDIUM confidence (web search synthesis).
- Henderson et al., "Deep Reinforcement Learning That Matters," and follow-on seed-count power analyses — 5 seeds is typically insufficient, high-variance settings may need ~25 — HIGH confidence, well-established result, cited for Pitfall 8's multi-seed recommendation.
- Simpson's paradox / "average of ratios ≠ ratio of averages" — HIGH confidence, standard statistics result, applied to Pitfall 6's aggregation-convention concern.
- General curriculum-RL literature on unreachable/infeasible task rungs stalling learning identically to flat sparse reward, and adaptive revert-on-failure curricula as the standard mitigation — MEDIUM confidence (web search synthesis), applied to Pitfall 9.

---
*Pitfalls research for: adversarial RL / federated-learning poisoning testbed — subsequent milestone (attack-goal calibration, reward normalization, adaptive curriculum, graded metrics)*
*Researched: 2026-08-01*
