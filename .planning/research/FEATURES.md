# Feature Research

**Domain:** Adversarial RL for federated-learning poisoning attacks and robust-FL defenses (attack-success accounting, benchmark reporting, curriculum/self-play design)
**Researched:** 2026-08-01
**Confidence:** MEDIUM overall (cross-checked across 3+ independent sources per major claim; individual paper internals were read via search-engine summaries, not full-text PDF extraction — see Sources for per-claim caveats)

## Direct Answers to the Five Research Questions

### Q1 — How does the literature define/compute/report "attack success"?

**Backdoor/targeted ASR (dominant definition, MEDIUM confidence, cross-checked across Bagdasaryan et al. 2020 "How To Backdoor Federated Learning" (AISTATS'20), and multiple 2023-2025 backdoor-FL papers found via search):**
ASR = fraction of trigger-embedded (or attacker-target-labeled) test samples that the global model classifies as the attacker's target label, evaluated after some number of rounds. This is overwhelmingly the modal definition — every backdoor-FL paper surveyed uses this form. It requires a trigger/target-label pair and a held-out trigger test set; it does **not** apply to your `untargeted_degrade` goal.

**Untargeted/model-degradation success (MEDIUM confidence, cross-checked across Fang et al. 2020 "Local Model Poisoning Attacks to Byzantine-Robust FL" (USENIX Security'20) and Shejwalkar & Houmansadr 2021 "Manipulating the Byzantine" (NDSS'21), and Shejwalkar/Houmansadr/Kairouz/Ramage 2022 "Back to the Drawing Board" (arXiv:2108.10241)):**
There is **no single standard metric name** analogous to backdoor-ASR. The literature reports raw outcomes instead of a normalized "success rate":
- **Absolute accuracy drop / error-rate increase** — the dominant form. Fang et al. and Shejwalkar & Houmansadr report the increase in test error (or decrease in accuracy) the attack causes under each defense, as a raw percentage-point number in a table, not a 0-1 "rate."
- **Attack impact** `I_θ = A_θ − A_θ*` — the one formalized metric found. Defined in "Back to the Drawing Board" as the gap between the maximum accuracy the global model would reach with **no attack** and the maximum accuracy it reaches **under attack**, both measured as the running max over training rounds. This is the closest published analogue to your `drop = clean_ref − post_accuracy`, and it is explicitly framed as methodologically superior to a fixed baseline (see Q2).
- **Rounds-to-threshold / convergence delay** — used in some papers as a secondary untargeted-attack metric (how many extra rounds are needed to reach a target accuracy), but far less common than the raw-drop number.
- **No published "area under the accuracy-degradation curve" metric** was found for untargeted FL poisoning specifically. AUC-style curves appear in adjacent areas (e.g., adversarial-robustness certification curves) but not as a standard FL-poisoning table column.

**Graded/continuous success formulations (LOW-MEDIUM confidence — this is the thinnest part of the literature):**
The most concrete published example is **BackFed** (arXiv:2507.04903, 2025 standardized backdoor-FL benchmark), which reports three temporally-graded metrics instead of one binary number: `ASR_t` (mean ASR over the last *t* rounds of the attack window, default *t*=30), `h-ASR` (peak ASR across all rounds), and **Lifespan** (rounds the ASR stays above a 50% threshold after the attack stops). These are graded *over time*, not graded *per-round as a fraction of a target* — no paper found defines a `clip(drop/target, 0, 1)`-style per-round partial-credit score. **This specific formulation — continuous per-round graded credit toward an explicit numeric target — is not attested in the surveyed FL-poisoning literature.** It is closer in spirit to reward-shaping practice from general RL (potential-based/ratio-to-goal shaping) than to any published FL-benchmark metric. Treat your graded ASR as a **novel accounting method for this domain**, defensible by analogy to BackFed's temporal-graded stance and to standard RL reward-shaping literature, but you should state explicitly in the paper that no direct precedent exists and justify it from first principles (which PROJECT.md already does via the stealth-gate collapse argument).

### Q2 — What baseline is standard for accuracy degradation, and is the fixed baseline criticized?

Three baselines appear in the literature; usage is genuinely split, but the trend among the most rigorous/recent work is running-max **or** per-defense counterfactual, not a fixed baseline:

| Baseline | Who uses it | Status |
|---|---|---|
| Fixed pre-attack baseline (e.g., Phase-1 checkpoint accuracy) | Common in early/simpler papers that run one defense at a time and don't compare across defenses | **Explicitly criticized.** "Back to the Drawing Board" defines attack impact against the running max no-attack accuracy specifically because different runs/configs converge to different ceilings; a fixed number silently miscredits attacks when the ceiling itself moves. |
| Per-round (or running-max) no-attack counterfactual, same defense | "Back to the Drawing Board" (`I_θ = A_θ − A_θ*`, both running maxima); BackFed (per-defense counterfactual: "evaluate ACC of the global model trained from scratch **with the defense strategy** in a no-attack scenario," explicitly to "isolate the impact of defenses") | **Endorsed by the two most methodologically self-aware sources found.** This is the same design your codebase already uses in training (`clean_reference_accuracy`) and the same fix PROJECT.md proposes for the benchmark. |
| No-defense control (FedAvg with no filtering, no attack) | Used as a third reference column in some robust-aggregation papers to show the defense's own accuracy cost | Reported *alongside* the other two, not as the drop denominator — it answers "does the defense hurt clean training," a different question from "how much did the attack do." |

**Direct implication for this project:** the milestone's planned fix (grade drop against the per-round, per-defense clean counterfactual rather than the fixed Phase-1 baseline) is not an ad hoc workaround — it matches the position taken by the field's own methodology-critique paper (Shejwalkar et al. 2022) and by the most recent standardized benchmark (BackFed 2025). This is citable, not merely justified by internal arithmetic.

### Q3 — What metrics accompany ASR in a standard comparison table?

Cross-checked across FLTrust (arXiv:2012.13995), the DnC paper ("Manipulating the Byzantine," NDSS'21), FLDetector, and BackFed (arXiv:2507.04903): a reviewer of a robust-FL results table expects, at minimum:

- **Main/global task accuracy** under attack (and, ideally, under no attack) — always present
- **TPR (detection rate) and FPR** for detection-style defenses (FLTrust, DnC, Multi-Krum, DeFL are all detection/filtering defenses in this codebase) — near-universal in papers that propose or compare filtering defenses; example reported numbers found: FLTrust ≈92.8% TPR / ≈6.4% FPR vs. an average of 67.3% TPR / 33.4% FPR for prior baselines (MEDIUM confidence, single-source snippet, treat magnitude as illustrative not exact)
- **Precision/F1** appear specifically in papers framing detection as classification (BackFed's anomaly-detection table reports Precision, Recall, ASR together) — expected when you present detection as a binary classifier, less universal than TPR/FPR alone
- **Clean-accuracy cost** — accuracy of the defense with zero attackers present, to show the defense doesn't sabotage honest training on its own (this is exactly the "clean round with no attacker" check your own ARCHITECTURE.md anti-pattern section already performs — it is standard practice, not a repo-specific sanity check)
- **Communication rounds / convergence** — reported in some papers as a secondary axis (does the attack/defense change how many rounds are needed), less consistently than the accuracy and TPR/FPR columns
- **Attack success metric itself** (ASR for backdoor, accuracy-drop/impact for untargeted) as the headline number

A reviewer used to this literature will look for the *combination* of (final/mean accuracy) × (TPR/FPR or precision/F1) × (attack success metric) × (clean-cost column), swept across at least one axis (attacker budget and/or defense algorithm). A table reporting graded ASR + detection rate + FPR + accuracy per defense, as this milestone's target output specifies, matches this expectation directly.

### Q4 — How is attacker poison budget reported/swept, and is scaling the attack goal WITH budget established?

**Reporting/sweeping budget:** standard and near-universal. Papers report the number or fraction of compromised clients (commonly swept from ~10% up through 30-50%, sometimes higher in less realistic setups) as an **independent variable**, and observe the resulting accuracy drop or ASR as the **dependent variable** at each budget level. Examples: Fang et al. and Shejwalkar & Houmansadr both sweep the malicious-client fraction across multiple values in their tables; BackFed instead **holds it constant** (10%) across its main experiments and only varies it in an appendix sweep. "Back to the Drawing Board" goes further and argues the realistic operating range for production cross-device FL is far smaller than most papers use (≤0.1% for data poisoning, ≤0.01% for model poisoning, versus the 25-50% compromise fractions common in academic papers), explicitly criticizing inflated attacker fractions as unrealistic — directly relevant to this project's `n_compromisable=5/20=25%` choice, which sits at the high end of what "Back to the Drawing Board" calls out, though a 25% strict-minority partial-insider setting is still common and accepted in the cross-silo/simulation-testbed literature this project belongs to (as opposed to production cross-device FL, which is what that paper is specifically about).

**Scaling the attack GOAL with budget — this is genuinely novel, not established practice.** No paper found defines the attacker's *target* degradation as a deterministic function of its budget (i.e., "at budget k, the target is X%"). The universal pattern is the reverse: budget is swept, and the *resulting* drop is reported as an outcome, with no explicit target being set at all — these are single-shot/best-effort attacks, not goal-conditioned RL agents with a numeric target to hit. The concept of a goal-conditioned attacker with a budget-indexed target ladder belongs to the RL/curriculum-design side of the literature (reward shaping, curriculum learning), not the FL-poisoning empirical-benchmark side. **State this plainly in the paper**: the budget→target ladder is a training/curriculum design choice introduced to give GRPO a reachable objective, not a replication of a benchmark convention — its justification must rest on the RL training-dynamics argument (unreachable target ⇒ collapsed reward surface ⇒ no gradient) already documented in PROJECT.md, not on an appeal to prior art. This is the single biggest "needs defending in the paper" item this research surfaced.

### Q5 — Curriculum/opponent-scheduling in adversarial self-play RL: table stakes vs. differentiator vs. anti-feature

Cross-checked against AlphaStar league training (DeepMind, Vinyals et al. 2019, "Grandmaster level in StarCraft II"), Policy Space Response Oracles / PSRO surveys (2024), and general self-play surveys (arXiv:2408.01072):

- **Table stakes:** uniform or round-robin opponent sampling; retaining a pool of historical opponent snapshots rather than pure naive self-play. Pure self-play against only the current opponent is a documented anti-pattern (see below) — some form of historical retention is baseline good practice, and this project already has it (league snapshots, ring buffer).
- **Differentiating:** **Prioritized Fictitious Self-Play (PFSP)** — AlphaStar's mechanism samples opponents with probability weighted toward those the learner currently loses to most, specifically because uniform sampling "wastes many games against players that are defeated almost 100% of the time." This is the direct precedent for this project's planned adaptive defense sampling (weight inversely proportional to measured attacker success against each defense, with a floor). The analogy is close enough to cite directly: PFSP's floor/anti-starvation logic (never fully dropping an opponent from the pool) maps onto this project's "floor so no defense starves." Also differentiating: continuous per-opponent performance tracking throughout training (not just at phase end) to catch regression against opponents not currently being sampled — this is exactly the mitigation AlphaStar-style league training uses against the catastrophic-forgetting risk described below.
- **Anti-features:**
  - **Naive/pure self-play with no historical retention** — well-documented to cause cycling in non-transitive games (strategy A beats B, B beats C, C beats A) and catastrophic forgetting of previously-defeated opponents. This project's league snapshots already avoid this.
  - **Full opponent-identity conditioning without a realism cost** — in a real threat model the attacker does not get told which defense algorithm judges each round; conditioning the attacker's policy on defense identity would train faster but produces a threat model reviewers can reasonably call unrealistic. Note the literature does have a **different, legitimate** thread here — "adaptive attacks" (e.g., Fang et al.'s adaptive variant, Xie et al.'s knowledge-aware attacks) explicitly assume the attacker knows the defense and is evaluated as a *worst-case* bound, which is a valid but different research question from a defense-blind deployed-attacker threat model. This project's choice to stay defense-blind is defensible as the more realistic threat model, but should be explicitly distinguished in the paper from the "adaptive, defense-aware attacker" literature so a reviewer doesn't read the defense-blind design as an oversight.
  - **Concentrating training exclusively on the single hardest opponent (no floor)** — the known failure mode PFSP was invented to avoid the opposite of (starving weak opponents); doing the opposite (only ever training against the hardest opponent) risks overfitting to one opponent and regressing against the others — exactly the catastrophic-forgetting risk this project's Key Decisions table already flags as an accepted risk requiring continuous per-defense tracking as mitigation. The literature supports treating that mitigation as necessary, not optional.

## Feature Landscape

### Table Stakes (Reviewers/Readers Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Backdoor-style ASR reserved for `targeted_label` goals only, never conflated with untargeted drop | Universal convention (Bagdasaryan et al. 2020 and all downstream backdoor-FL work) — a reviewer will assume "ASR" means trigger→target-label unless told otherwise | LOW | This project already scopes `targeted_label` out of the current milestone (PROJECT.md); naming must not reuse "ASR" ambiguously across the two goal types when that work resumes |
| Per-defense TPR/FPR reported alongside attack-success metric | Standard in every detection-defense comparison found (FLTrust, DnC, BackFed) | LOW (already computed in `metrics/compute.py`) | Existing capability; just needs to land in the final benchmark table |
| Clean-accuracy-cost column (defense accuracy with zero attackers) | Standard sanity check distinguishing "defense hurts honest training" from "defense stops attacks" | LOW | Already implemented as the anti-pattern check documented in ARCHITECTURE.md; formalize it as a benchmark table column |
| Attacker budget reported and swept as an axis of the results table | Universal across Fang 2020, Shejwalkar & Houmansadr 2021/2022, BackFed 2025 | LOW-MEDIUM | This project already sweeps budget 1-5; needs to appear explicitly as a table axis, not just an internal training variable |
| Per-round / per-defense clean counterfactual as the drop denominator (not a fixed baseline) | Endorsed by the two most methodology-focused sources found (Shejwalkar et al. 2022, BackFed 2025) | MEDIUM (already exists in training via `clean_reference_accuracy`; benchmark side is the active gap) | Directly matches this milestone's planned fix — cite these two sources in the paper to preempt the "why not a fixed baseline" reviewer question |
| Historical-opponent retention in adversarial self-play (vs. naive current-opponent-only self-play) | Documented failure mode (cycling, forgetting) if omitted | MEDIUM | Already implemented (league snapshot ring buffer) — validated by literature, not a novel choice |
| Single-defense judging per round rather than unioning all defenses' flags | Ensemble-of-detectors combination rules in the literature (e.g., voting, weighted union) exist specifically to avoid the over-flagging failure mode this project found empirically (14/20 honest clients flagged under naive OR-union) | LOW (already implemented) | Confirms the ARCHITECTURE.md anti-pattern finding is not a repo-specific quirk; naive OR-combination of independent detectors is a known bad combination rule in ensemble-detection generally |

### Differentiators (What Makes This Work Novel vs. the Literature)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Budget-indexed attack-goal ladder (target is a deterministic function of poison budget) | Solves a training-dynamics problem (unreachable target ⇒ zero-variance reward ⇒ no GRPO gradient) that fixed-target papers never encounter because they don't train a goal-conditioned RL policy against a numeric target — no prior FL-poisoning paper does this | MEDIUM | **Not established practice — must be justified from RL training dynamics, not benchmark convention (see Q4).** This is the strongest "defend this in the paper" item |
| Graded/continuous attack success (`clip(drop/target, 0, 1)` per round, averaged) replacing binary threshold accounting | Produces a non-degenerate, comparable per-defense number instead of a brittle pass/fail; closest published relative is BackFed's temporally-graded `ASR_t`/`h-ASR`/Lifespan, but no paper grades per-round against an explicit numeric target this way | LOW-MEDIUM | Cite BackFed as the nearest precedent for "graded rather than binary" thinking in this specific literature, while being explicit that the exact ratio-to-goal formulation is new |
| Online running-max normalization of the damage term, self-calibrating per (defense × budget) | Removes the need for an offline calibration sweep; keeps GRPO's group-relative advantages well-scaled without destabilizing training (denominator constant within a GRPO group) | MEDIUM-HIGH | No direct FL-poisoning precedent found; the closest general analogue is adaptive/automatic reward normalization in RL generally, not something documented in this specific application. Flag as a from-scratch design justified by the GRPO-groups-share-one-denominator argument already in PROJECT.md |
| Adaptive per-defense sampling weighted inversely by measured attacker success, with an anti-starvation floor | Direct structural analogue to AlphaStar's PFSP; concentrates training where the attacker is weakest while guaranteeing continued exposure to every algorithm | MEDIUM | Cite PFSP/AlphaStar league training explicitly — this is the strongest positive precedent found in the whole research pass, and it substantially de-risks defending this design choice |
| Continuous per-defense ASR tracking throughout training (not only at phase end) to surface catastrophic forgetting | Mitigates the known self-play/league-training risk that concentrating on the hardest opponent silently regresses performance against opponents no longer being sampled | LOW-MEDIUM (already partially exists via `metrics/tracker.py`) | Directly supported by self-play/PSRO literature's documented catastrophic-forgetting risk; frame as adopting a known best practice from game-AI league training into the FL-poisoning setting, which is itself a novel transfer |
| Attacker stays defense-blind (no defense-identity conditioning) while defense sampling adapts underneath it | Preserves a realistic "deployed attacker doesn't know which detector is watching" threat model while still letting the *training curriculum* (not the attacker's observation) adapt | LOW (policy decision, not implementation) | Must be explicitly distinguished in the paper from the "adaptive, defense-aware attacker" literature (Fang's adaptive attack, etc.), which studies a different, defense-aware threat model as a worst-case bound — don't let a reviewer conflate the two |
| Super-linear budget→target scaling at max budget (12% at full budget, not a linear extrapolation of 2%/4%/6%/8%) | Rewards genuinely coordinated multi-client attacks over merely using more clients at the same per-client leverage | LOW | No literature precedent found either way (this is a reward-curve shape choice, not an accounting convention); complements the existing diversity/coordination bonus (`zeta`) which literature (DBA, CollaPois — coordinated-but-distinct multi-client attacks) does support as a real attack pattern worth rewarding |

### Anti-Features (Seem Reasonable, Actually Problematic Here)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| Raise `n_compromisable` (or the eval poison budget) to make large fixed targets reachable | Simplest fix to "the attacker can't reach 20% drop" | Changes the threat model rather than the training objective; "Back to the Drawing Board" explicitly criticizes inflated compromise fractions as unrealistic for the class of FL system this testbed models — moving the goalposts here trades scientific validity for an easy fix | Keep the strict-minority threat model fixed; make the *target* a function of the *fixed* budget instead (this milestone's actual approach) — already correctly identified as Out of Scope in PROJECT.md |
| Offline calibration sweep to find the reachable ceiling per (defense × budget) before training | Seems more principled than an online running max | Requires a full separate experiment per config, doesn't adapt if training dynamics shift the reachable ceiling mid-run, and adds an extra artifact to keep in sync | Online running-max normalization, self-calibrating and checkpointed — already the chosen approach; correctly flagged Out of Scope as superseded |
| Give the attacker the defense's identity each round (defense-conditioned policy) | Would let the attacker specialize per-defense faster, improving apparent training efficiency | Weakens the threat model into the "adaptive, fully-informed attacker" category studied elsewhere in the literature as a worst-case bound, not a realistic deployed threat; conflates two different research questions | Keep the attacker defense-blind; let only the *curriculum* (which defense judges this round) adapt, not the attacker's observation — already correctly Out of Scope |
| Union-of-all-defenses verdict (flag a client if ANY of FLTrust/Multi-Krum/DnC/DeFL flags it) | Intuitively "more defenses = more robust" | Empirically over-flags honest clients (14/20 in this codebase's own measurement), collapsing the room for any attack to fit through and burying the true attack-damage signal (≈0.0003) under honest-client-selection noise (sd≈0.012) — a documented anti-pattern in this project and consistent with how ensemble-detector combination is normally handled (weighted/majority rules, not naive OR) | Single-defense-per-round judging (rotate/random/fixed), already implemented |
| Sybil-style identical attack plans across multiple compromised clients | Naive way to "scale up" an attack by using more clients | Directly detected by cosine-similarity/sign-agreement style features; DBA/CollaPois-style literature on coordinated multi-client attacks specifically succeeds by being coordinated-but-distinct, not identical | Distinct per-client roles under one coordinated objective (already implemented; rewarded by the `zeta` diversity term) |
| Fixed pre-attack (Phase-1) baseline as the drop denominator for a multi-defense benchmark table | Simple, one number, easy to explain | Credits each defense's own self-inflicted accuracy sag (e.g., FLTrust dropping honest clients under non-IID splits every round) to the attacker, skewing cross-defense comparison — this is the literature-endorsed criticism from Q2, not just an internal finding | Per-round, per-defense clean counterfactual as the denominator — already the planned fix |
| Pure naive self-play (attacker/defender only ever see each other's current, latest policy) | Simplest possible adversarial-training loop | Documented to cause cycling and catastrophic forgetting in non-transitive games; can look like "training" while never converging to a robust strategy | League/snapshot-based training with a retained opponent pool (already implemented) |

## Feature Dependencies

```
Budget-indexed attack-goal ladder
    └──requires──> Fixed, non-negotiable poison budget schedule (n_compromisable stays constant)
                       └──requires──> Partial-insider threat model unchanged

Graded attack success rate (clip(drop/target, 0, 1))
    └──requires──> Budget-indexed attack-goal ladder (needs a per-round target to grade against)
    └──requires──> Per-round, per-defense clean counterfactual (needs a correct "drop" numerator)

Online running-max damage normalization
    └──requires──> Graded attack success rate is NOT itself normalized by the running max — normalization applies to the raw damage/reward term, keep these two independent
    └──enhances──> GRPO gradient signal (restores usable reward variance)

Adaptive per-defense sampling (PFSP-style)
    └──requires──> Continuous per-defense ASR tracking (the weighting signal)
    └──enhances──> Reachable-target ladder (concentrates training where the ladder is hardest to hit)
    └──conflicts partially with──> Attacker stays defense-blind (creates the accepted catastrophic-forgetting risk; mitigated, not eliminated, by continuous tracking)

Continuous per-defense ASR tracking
    └──requires──> Metrics/tracker.py already tracks TPR/FPR/ASR per round — extend to per-defense breakdown, not a new subsystem

Benchmark per-defense results table (graded ASR + TPR + FPR + accuracy)
    └──requires──> Per-round, per-defense clean counterfactual in the benchmark harness (not just training)
    └──requires──> Metric renaming (evasion_rate vs attack_success_rate vs goal_success_rate) to avoid reporting the wrong number in the table
```

### Dependency Notes

- **Graded ASR requires the budget-indexed ladder:** grading `drop/target` is meaningless without a well-defined, reachable `target` at every budget — the ladder must land first, or the graded metric inherits the old unreachable-target problem.
- **Graded ASR requires the corrected benchmark baseline:** if the benchmark still grades against the fixed Phase-1 baseline while the ladder and reward use the per-round counterfactual, training and reporting will silently disagree about what "drop" means — this is exactly the inconsistency PROJECT.md's Context section already diagnoses between `benchmark/metrics.py` and `rl/env.py`.
- **Running-max normalization is independent of graded ASR:** the running max rescales the *reward term* the policy is trained on; the graded ASR is a *reporting metric*. Keep them as two separate quantities even though both involve dividing by a form of "how much drop was achieved relative to a reference" — conflating them risks the reported ASR silently drifting as the running max updates, which would break run-to-run comparability of the results table.
- **Adaptive defense sampling conflicts partially with staying defense-blind:** the curriculum (which defense is chosen next) is allowed to adapt to measured performance; the attacker's *observation* must not reveal which defense was chosen. These are different objects (environment scheduling vs. agent input) and must be kept structurally separate in the implementation, not just conceptually — this is the same discipline the DnC/FLTrust literature applies to defenders (features only, never ground truth).

## MVP Definition

### Launch With (v1 — this milestone)

- [ ] Budget-indexed target ladder — table stakes for making the graded metric meaningful at all
- [ ] Per-round, per-defense clean counterfactual in both training and benchmark — literature-endorsed, closes the internal inconsistency PROJECT.md diagnoses
- [ ] Graded attack success rate (`clip(drop/target, 0, 1)`, averaged per round) — the paper's headline metric; cite BackFed as the nearest precedent for graded-over-binary thinking
- [ ] Online running-max damage normalization, checkpointed and floored — required for GRPO to receive a non-degenerate gradient at all
- [ ] Continuous per-defense metric tracking (TPR/FPR/graded ASR) throughout training — needed to detect the catastrophic-forgetting risk this design knowingly accepts
- [ ] Metric renaming (`evasion_rate` / `attack_success_rate` / `goal_success_rate`) — table-stakes hygiene; a results table with an ambiguously-named "attack_success_rate" column will not survive review

### Add After Validation (v1.x)

- [ ] Adaptive/PFSP-style defense sampling with an anti-starvation floor — add once the ladder + graded ASR are proven to produce a non-zero gradient under the current round-robin schedule; layering curriculum change on top of an unproven reward change conflates two variables
- [ ] Per-defense results table in the benchmark harness, fully reproducible end-to-end — natural final step once all upstream pieces are validated to be learning

### Future Consideration (v2+)

- [ ] `targeted_label` goal support with backdoor-style ASR (per-class evaluation) — explicitly out of scope this milestone; requires the standard backdoor-ASR definition (Q1) rather than the graded degrade-goal metric, so it is a genuinely separate metric track, not an extension of this milestone's graded ASR
- [ ] Defense-aware/adaptive-attacker variant as a deliberate worst-case-bound study, run as a *separate* experiment from the defense-blind main results — valuable for a "how much does realism cost the defender" discussion, but must not be merged into the primary threat model

## Feature Prioritization Matrix

| Feature | Research Value (defensibility in a paper) | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Budget-indexed target ladder | HIGH (needs explicit defense — no prior art) | MEDIUM | P1 |
| Per-round/per-defense counterfactual in benchmark | HIGH (literature-endorsed) | MEDIUM | P1 |
| Graded ASR reporting | HIGH (closest thing to a novel-metric contribution) | LOW-MEDIUM | P1 |
| Running-max normalization | MEDIUM (defensible via GRPO-group argument, not literature) | MEDIUM-HIGH | P1 |
| Continuous per-defense tracking | MEDIUM (supports the accepted-risk mitigation story) | LOW | P1 |
| Metric renaming | LOW novelty, HIGH hygiene value | LOW | P1 |
| Adaptive/PFSP defense sampling | HIGH (strong precedent, good story) | MEDIUM | P2 |
| Final per-defense benchmark table | HIGH (the paper's actual results) | LOW (mostly composition of P1 items) | P2 |
| Defense-aware attacker ablation | MEDIUM (nice-to-have contrast) | HIGH | P3 |
| `targeted_label` backdoor-ASR track | MEDIUM (different metric family entirely) | HIGH | P3 |

**Priority key:** P1: needed for this milestone's stated core value (reachable goal + comparable ASR). P2: strengthens the paper but depends on P1 landing first. P3: separate research threads, explicitly out of scope now.

## Competitor/Prior-Art Feature Analysis

| Feature | Fang 2020 / Shejwalkar 2021 (classical attacks) | BackFed 2025 (standardized backdoor benchmark) | AlphaStar league training (game AI) | This Project's Approach |
|---------|--------------------------------------------------|--------------------------------------------------|--------------------------------------|--------------------------|
| Attack-goal structure | Best-effort optimization, no explicit numeric target | Fixed trigger/target-label pair per run, no drop target | N/A (win/loss, not a numeric target) | Explicit numeric target, ladder-indexed by budget — **the novel piece** |
| Success accounting | Raw accuracy drop / error increase, single number | Temporally-graded (`ASR_t`, `h-ASR`, Lifespan) | Win rate | Per-round graded ratio-to-target, averaged — closer to BackFed's temporal-grading spirit than to Fang/Shejwalkar's single-number reporting |
| Baseline for drop | Mixed: some fixed, some running-max | Per-defense, trained-from-scratch, no-attack counterfactual | N/A | Per-round, per-defense clean counterfactual — matches BackFed's approach |
| Budget handling | Independent variable, swept | Held constant (10%) in main results | N/A | Fixed strict-minority budget; **goal** (not budget) is the swept/scaled quantity — inverted relative to the classical-attack literature |
| Opponent/defense curriculum | N/A (not an RL setting) | N/A | PFSP: performance-weighted sampling with anti-starvation floor | Adaptive defense sampling directly modeled on PFSP |
| Threat-model realism stance | "Back to the Drawing Board" pushes toward small, realistic attacker fractions | Not a focus | N/A | Partial-insider minority (25%) — more realistic than un-bounded academic setups, less strict than "Back to the Drawing Board"'s production cross-device numbers; acceptable for a cross-silo-style testbed but should be justified as such in the paper |

## Sources

- Bagdasaryan, Veit, Hua, Estrin, Shmatikov. "How To Backdoor Federated Learning." AISTATS 2020. arXiv:1807.00459 — backdoor ASR definition, model-replacement attack, single-shot ~100% backdoor accuracy claim. [MEDIUM confidence, cross-checked via multiple search snippets, not full-text-verified]
- Fang, Cao, Jia, Gong. "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning." USENIX Security 2020. — untargeted attack framing, error-rate reporting, budget-as-swept-variable convention. [MEDIUM confidence]
- Shejwalkar, Houmansadr. "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning" (DnC). NDSS 2021. https://people.cs.umass.edu/~amir/papers/NDSS21-model-poisoning.pdf — DnC defense design, 1.5-60x reported attack improvements over prior SOTA. [MEDIUM confidence]
- Shejwalkar, Houmansadr, Kairouz, Ramage. "Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Production Federated Learning." arXiv:2108.10241, IEEE S&P 2022. — `I_θ = A_θ − A_θ*` attack-impact definition, per-round running-max no-attack counterfactual, criticism of unrealistic attacker-fraction assumptions (≤0.1%/0.01% for production cross-device FL vs. 25-50% common in academic papers), criticism of small client populations in prior benchmarks. [MEDIUM confidence — extracted via ar5iv HTML fetch, not independently cross-checked against a second full-text source]
- "SoK: Benchmarking Poisoning Attacks and Defenses in Federated Learning." arXiv:2502.03801 (2025). — identified as directly relevant but full text could not be extracted (PDF rendering unavailable in this environment); listed for follow-up, not used as a cited source of specific claims. [LOW confidence / unverified]
- "BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning." arXiv:2507.04903 (2025). — `ASR_t`/`h-ASR`/Lifespan graded-over-time metrics, per-defense trained-from-scratch no-attack counterfactual, Precision/Recall/ASR table for anomaly-detection defenses, held-constant 10% attacker fraction in main results. [MEDIUM confidence, extracted via ar5iv HTML fetch]
- Wang, Sreenivasan, Rajput, Vishwakarma, Agarwal, Sohn, Lee, Papailiopoulos. "Attack of the Tails: Yes, You Really Can Backdoor Federated Learning." NeurIPS 2020. arXiv:2007.05084 — edge-case backdoors, budget-parameter (δ) framing under norm-based defenses. [LOW-MEDIUM confidence]
- FLTrust: Cao, Fang, Liu, Gong. "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping." NDSS 2021. arXiv:2012.13995 — TPR/FPR reporting convention (≈92.8%/6.4% reported figures found via search, treat as illustrative). [LOW confidence, single-source snippet]
- Li, Wu, Zhu, Zheng. "Learning to Backdoor Federated Learning." arXiv:2303.03320, and Li, Sun, Zheng. "Learning to Attack Federated Learning: A Model-Based Reinforcement Learning Attack Framework." NeurIPS 2022. — direct prior art for RL-trained FL attackers (analogous framing to this project's GRPO attacker), located via search but full text not extracted; flagged for follow-up full reading before citing specifics. [LOW confidence / unverified — cite cautiously, verify claims before use in the paper]
- "Meta Stackelberg Game: Robust Federated Learning against Adaptive and Mixed Poisoning Attacks." arXiv:2410.17431 (2024), and "A First Order Meta Stackelberg Method for Robust Federated Learning." arXiv:2306.13800 — direct prior art for framing FL attacker/defender training as a Stackelberg game with RL, matching this project's own framing; full text not extracted, located via search abstracts only. [LOW confidence / unverified — read full text before citing]
- Vinyals et al. "Grandmaster level in StarCraft II using multi-agent reinforcement learning" (AlphaStar). Nature 2019 / DeepMind technical report. — Prioritized Fictitious Self-Play (PFSP) definition, league training with main agents/exploiters, anti-starvation weighting rationale. [MEDIUM confidence, cross-checked across multiple secondary summaries, not the primary paper text directly]
- "A Survey on Self-Play Methods in Reinforcement Learning." arXiv:2408.01072 (2024). — non-transitivity, cycling, and catastrophic-forgetting failure modes of naive self-play; population-based mitigations (PSRO, SP-PSRO). [MEDIUM confidence]
- Lanctot et al. "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" (PSRO). NeurIPS 2017, and "Policy Space Response Oracles: A Survey" (2024). — meta-strategy solvers as the general mechanism generalizing round-robin/uniform opponent sampling. [LOW-MEDIUM confidence, survey-level snippets only]
- DBA (Xie et al., "DBA: Distributed Backdoor Attacks against Federated Learning," ICLR 2020) and CollaPois-style collaborative-backdoor literature — support for coordinated-but-distinct multi-client attack patterns as more effective/stealthy than identical Sybil clones, consistent with this project's diversity/`zeta` reward term. [LOW confidence, located via search summary only, not full text]

**Coverage caveat:** Several PDFs (the SoK benchmarking paper, the two RL-attacker papers, and the Meta-Stackelberg papers) could not be rendered for full-text extraction in this environment (no `pdftoppm`/poppler-utils available, and WebFetch's PDF-to-text pass returned only structural/binary content for these specific files). Their existence, titles, and headline claims are corroborated by multiple independent search-engine result snippets (MEDIUM confidence at the claim-existence level), but exact quotes, equations, and table numbers from those specific papers should be independently verified from the primary source before being quoted directly in a paper draft.

---
*Feature research for: adversarial RL for federated-learning poisoning attacks and robust-FL defenses*
*Researched: 2026-08-01*
