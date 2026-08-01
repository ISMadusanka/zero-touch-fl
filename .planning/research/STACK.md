# Stack Research

**Domain:** Adversarial RL for federated-learning poisoning (attacker-goal calibration, reward normalization, adaptive opponent sampling) — additions to an existing GRPO/LoRA testbed
**Researched:** 2026-08-01
**Confidence:** MEDIUM (see per-item confidence below; no item is HIGH because these are bespoke-training-loop techniques, not packaged APIs with canonical docs)

**Scope note:** This file covers ONLY the four additions in the milestone (budget→target ladder, running-max damage normalization, graded ASR, adaptive defense sampling) plus the checkpointing needed to support them. It does not re-litigate PyTorch/Unsloth/Transformers/PEFT/GRPO/vLLM, which are already chosen and documented in `.planning/codebase/STACK.md`.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Plain Python (no new package) | stdlib | Budget→target ladder | A `{budget: target}` lookup table (or `2*budget/100` for budgets 1-4 plus a hand-set 0.12 at budget 5) is a pure function with five entries. No library exists for "attack-goal ladders" — it's domain-specific policy, not a general algorithm. |
| Plain Python dict + `max()` (no new package) | stdlib | Running-max damage normalizer, keyed by `(defense, budget)` | This is a monotonic high-water-mark ratchet, not a mean/std normalizer. No RL library (RLlib, CleanRL, TRL, Stable-Baselines3) ships a "running max" reward normalizer under any name — every framework's built-in normalizer (Welford running mean/std, PopArt, `VecNormalize`) targets **variance reduction for stationary-ish targets**, not a **monotone denominator**. Implementing this as a dependency would mean depending on a library for the one line `running_max[key] = max(running_max.get(key, floor), observed_drop)`. |
| PFSP-style inverse-performance weighting (no new package) | — | Adaptive defense sampling | AlphaStar's Prioritized Fictitious Self-Play samples opponent *i* with weight `∝ (1 - p̂ᵢ)^p` where `p̂ᵢ` is the agent's measured win rate against *i* — i.e., weight is inversely proportional to success, tunable exponent controls how sharply it concentrates. This is the exact shape of "weight inversely proportional to the attacker's measured success against each defense, with a floor." With only 4 fixed opponents (FLTrust, Multi-Krum, DnC, DeFL) this is a ~20-line categorical sampler, not a job for a bandit library. |
| Extend existing `storage/checkpoint.py` state dict | — | Persist running-max table + defense-sampling EMA stats across resume | The project already checkpoints non-model RL state (schedule state, league snapshots) alongside LoRA adapters via a custom save/load path. The new state (a `dict[(defense,budget)] -> float` and a `dict[defense] -> ema_success]`) is plain JSON-serializable data — add it to the same checkpoint schema rather than introducing `accelerate.save_state()` or a new serialization library. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| — (none) | — | — | No supporting library is recommended for any of the four additions. See "What NOT to Use" for the libraries considered and rejected. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| JSON (stdlib `json`) for the new auxiliary state, not `torch.save`/pickle | Serialize the running-max table and defense-sampling stats | The existing checkpoint already mixes tensor state (adapters, via PEFT's own `save_pretrained`) with non-tensor training state. Prefer a small human-readable JSON file for the *new* scalar state (running max per cell, per-defense EMA, success-rate history) written next to the existing checkpoint artifact — this mirrors the pattern HF `Trainer` uses (`trainer_state.json` alongside `pytorch_model.bin`) and torchtune's `recipe_state.pt` (auxiliary state saved separately from adapter weights, keys removed to avoid duplicating model tensors). Being plain JSON also makes the running-max table diffable in git and inspectable without loading PyTorch, which matters for a "prove the reward isn't degenerate" milestone. |

## Installation

```bash
# No new dependencies for these four additions.
# Everything is implemented inline in rl/rewards.py, rl/turns.py or rl/schedule.py (or wherever
# the defense-selection logic lives) and storage/checkpoint.py.
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|--------------------------|
| Inline running max, keyed by (defense, budget), floored | PopArt (mean/std adaptive normalization of a critic's output layer) | Only if you introduce a learned value/critic head whose *scale* drifts across tasks of very different reward magnitude (PopArt's actual purpose — see van Hasselt et al. 2016, "Learning values across many orders of magnitude"). This project's GRPO has no critic and normalizes advantages *within* a group already; PopArt solves a different problem (cross-task value-scale drift for a shared network head) and would add a dependency and a second, conflicting normalization scheme on top of GRPO's own group z-score. |
| Inline running max | Welford's online algorithm for running mean/variance | Use Welford (or Stable-Baselines3's `VecNormalize`, which implements it) when you need a *denominator that shrinks noise* (an std-based normalizer) for a reward or observation stream whose scale is roughly stationary. Wrong tool here: the design need is explicitly a monotone ceiling ("online running max"), not a variance estimate, and PROJECT.md's own rationale (`Key Decisions`) already selected the max because it self-calibrates the reachable ceiling per cell — a std-based normalizer would not represent "best achieved drop." |
| PFSP-style inverse-performance weight, implemented inline | OpenSpiel's PSRO (Policy Space Response Oracles) | PSRO is the right tool when the opponent *population itself* must grow (new best-response policies are trained and added to a meta-game solved via a meta-strategy solver). Here the opponent set is fixed and small (4 named algorithmic defenses); there is no meta-game to solve, only a sampling *weight* to update. Bringing in OpenSpiel (a large library with its own game-abstraction API) to compute a 4-way softmax-like weight is disproportionate and would require adapting the codebase's defense objects to OpenSpiel's `Game`/`Policy` interfaces. |
| PFSP-style inverse-performance weight | Exp3 / UCB adversarial or stochastic bandits | Use an adversarial bandit (Exp3) or UCB when the reward-generating process for each arm is either adversarial (opponent actively trying to fool the sampler) or when you need formal regret guarantees and are tuning exploration against unknown, possibly non-stationary arm distributions. Here the "arms" (defenses) are not adversarial — they are fixed deterministic/near-deterministic detectors — and PROJECT.md wants an interpretable floor-guaranteed weight, not a regret-minimizing exploration schedule with its own hyperparameters (γ, learning rate) that don't map cleanly onto "starvation floor." PFSP's simpler `(1-success)^p` + floor is what AlphaStar shipped in production for exactly this shape of problem and is far easier to reason about and debug in a research write-up. |
| PFSP-style inverse-performance weight | Syllabus-RL (`Syllabus-RL` on PyPI) with its Prioritized Level Replay (PLR) curriculum module | Syllabus is a real, current (2024-2025) library providing a portable curriculum-learning API including PLR (bandit-based level/task sampling) with multi-process synchronization. It's a good fit if the project later wants to swap between several curriculum strategies with a common API, or genuinely needs distributed-multiprocess curriculum sync. For *this* milestone — one categorical distribution over 4 known defenses, updated from a scalar success signal already computed every round — adopting Syllabus adds an external dependency, its own concept vocabulary (Curriculum, TaskSampler), and multiprocess plumbing this single-process training loop doesn't need. Revisit if the roadmap later adds many more opponent types or needs PLR's staleness-aware replay. |
| Plain accuracy-drop / TPR-FPR metrics reported per round (existing) + inline graded ASR | FLPoison (`vio1etus/FLPoison`, companion to "SoK: Benchmarking Poisoning Attacks and Defenses in Federated Learning," arXiv:2502.03801, 2025) | FLPoison is the current (2025) standard FL-poisoning benchmark harness (15 attacks, 17 defenses, unified evaluation) and is worth citing/comparing against in the benchmark writeup, but neither its README nor the SoK paper's abstract defines a continuous, target-normalized attack-success metric (`clip(drop/target, 0, 1)`) — its reported metrics are accuracy-drop and detection TPR/FPR, the same binary-flavored accounting this milestone is moving away from. There is nothing to import; the graded-ASR formula in PROJECT.md is itself the novel contribution here. Treat FLPoison as related work / a citation, not a dependency. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|--------------|
| PopArt / pop-art-style output-layer rescaling for the damage-normalization term | It solves a different problem (adaptively rescaling a *learned critic's* output layer to preserve gradients while the target's mean/std drifts). This project has no critic and needs a plain scalar divisor, not a network-preserving rescale. Using it would add TensorFlow/PyTorch critic-coupling complexity for no benefit. | Inline running max with a floor, per (defense, budget) cell |
| Welford / `VecNormalize`-style running std for the damage denominator | Std-based normalizers assume you want to suppress variance around a moving center; PROJECT.md's own rationale explicitly wants the denominator to represent the *best drop achieved so far* (a ceiling), which is categorically a max, not a std. Using std here would let the denominator shrink when recent rounds do worse, inflating normalized reward in exactly the wrong direction. | Inline running max |
| Exp3 / UCB adversarial-bandit machinery for defense selection | Designed for regret-bounded exploration against unknown or adversarial arm-reward processes; brings hyperparameters (learning rate, exploration bonus) with no natural mapping to "starvation floor," and its exploration behavior is harder to inspect/explain in a paper than a closed-form priority weight. | PFSP-style `(1 - success_rate)^p` weighting with a floor, computed inline |
| OpenSpiel / full PSRO for opponent sampling | Built for growing a policy population via best-response oracles over a meta-game; massive overkill and a heavyweight, differently-abstracted dependency for weighting 4 fixed, named algorithms. | Inline PFSP-style weighting |
| Introducing `accelerate.save_state()` or a new serialization scheme for the auxiliary RL state | The project already has a working custom checkpoint path (`storage/checkpoint.py`) that saves adapters + schedule state + league snapshots together; adding a second, framework-owned checkpoint mechanism (Accelerate's) would create two sources of truth for "is training resumable" and doesn't buy anything for a single-process LoRA trainer. | Extend the existing checkpoint dict with the new JSON-serializable keys |
| Pickling the running-max table / defense-sampling stats via `torch.save` | Opaque binary, not diffable, and these are plain Python floats/dicts with no tensors — no reason to route them through PyTorch's pickle-based serializer. | Plain `json.dump`/`json.load` alongside the existing checkpoint file |

## Stack Patterns by Variant

**If the running-max cell for a given (defense, budget) has zero observations yet (cold start, e.g., right after a checkpoint-schema migration or a newly-added budget tier):**
- Floor the denominator to a small positive epsilon (the project's existing constraint already calls for "floored to prevent division by ~0") and optionally floor it to the *target* for that budget (so early reward isn't absurdly inflated by a near-zero max) rather than a raw epsilon like `1e-6`.
- Because GRPO's advantages are computed *within* a round/group sharing one (defense, budget) cell (per PROJECT.md's own invariant), a cold-start cell only distorts the *absolute* reward scale for that round, not the group-relative advantage sign — consistent with the existing "why a moving denominator is safe here" reasoning.

**If defense-sampling weights must be seeded/reproducible (the project's existing reproducibility constraint):**
- Draw the categorical sample from a `numpy.random.Generator` seeded from the same `fl.poison_seed` already used for non-IID partitioning and budget sampling, not from global `random`/`numpy.random` state, so the sampling stream is independently replayable and doesn't get perturbed by unrelated calls elsewhere in the round loop.

**If per-defense success EMA (feeding the PFSP-style weight) needs a decay rate:**
- Use a simple exponential moving average (`ema = decay*ema + (1-decay)*new_success`) rather than an unbounded running mean — unlike the damage-normalization max (which must be monotone by design), the *success rate per defense* should be allowed to drift down (catastrophic forgetting must be visible), so this one auxiliary stat is intentionally the opposite shape (leaky, not ratcheting) from the damage normalizer. Keep these two clearly separate in code and naming (`running_max_drop` vs `defense_success_ema`) since they use opposite update rules for related-sounding purposes — this is the single most likely source of a subtle bug if implemented in a rush.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|------------------|-------|
| N/A (no new packages) | Existing stack (PyTorch 2.0+, Unsloth ≥2026.6.9, Transformers ≥4.45, PEFT ≥0.13) | All four additions are pure-Python control logic and data bookkeeping around the existing GRPO/env/reward code; they add no import-time or version constraints. |

## Sources

- [PFSP / AlphaStar league training — "Grandmaster level in StarCraft II using multi-agent reinforcement learning"](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf) — priority weight formula `(1-p̂ᵢ)^p`, confidence MEDIUM (cross-checked across DeepMind primary paper + independent survey/blog summaries)
- ["A Survey on Self-play Methods in Reinforcement Learning"](https://arxiv.org/pdf/2408.01072) — corroborates PFSP formula and self-play opponent-sampling taxonomy, confidence MEDIUM
- [PopArt — "Preserving Outputs Precisely while Adaptively Rescaling Targets" (DeepMind blog)](https://deepmind.google/discover/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/) and [van Hasselt et al., "Learning values across many orders of magnitude"](https://arxiv.org/pdf/1602.07714) — confirms PopArt normalizes a learned critic's mean/std, not a max, confidence MEDIUM
- [Welford's online algorithm summaries](https://www.embeddedrelated.com/showarticle/785.php), [rolling-variance](https://github.com/RichieHakim/rolling-variance) — confirms Welford targets running mean/variance (used by Stable-Baselines3-style `VecNormalize`), not max, confidence LOW (general web sources, not benchmarked against this project's exact use case)
- [OpenSpiel / PSRO — "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning"](https://arxiv.org/pdf/1711.00832) and [Self-Play PSRO](https://arxiv.org/pdf/2207.06541) — confirms PSRO's meta-game/best-response-oracle scope, confidence MEDIUM
- [Syllabus — "Syllabus: Portable Curricula for Reinforcement Learning Agents"](https://www.researchgate.net/publication/385921100_Syllabus_Portable_Curricula_for_Reinforcement_Learning_Agents), [Syllabus-RL on PyPI](https://pypi.org/project/Syllabus-RL/), [Prioritized Level Replay docs](https://ryannavillus.github.io/Syllabus/modules/syllabus.curricula.plr.html) — confidence MEDIUM on capability description, LOW on exact current PyPI version (not confirmed)
- [FLPoison — "SoK: Benchmarking Poisoning Attacks and Defenses in Federated Learning" (arXiv:2502.03801)](https://arxiv.org/abs/2502.03801) and [GitHub vio1etus/FLPoison](https://github.com/vio1etus/FLPoison) — confirms this is the current standard FL-poisoning benchmark and that it does not define a graded/continuous ASR metric in its public README/abstract, confidence LOW (could not access full paper body, only README + abstract)
- [PyTorch/PEFT checkpoint patterns — HF `transformers` PEFT docs](https://huggingface.co/docs/transformers/main/en/peft), [torchtune checkpointing internals](https://docs.pytorch.org/torchtune/0.6/_modules/torchtune/training/checkpointing/_checkpointer.html), [HF forum: saving optimizer/model state with 2 LoRAs](https://discuss.huggingface.co/t/what-is-the-best-way-to-save-the-state-of-a-model-and-optimizer-when-the-model-has-2-loras/169458) — confirms the standard pattern of saving adapter weights and auxiliary/recipe state as separate artifacts, confidence MEDIUM
- [TRL GRPOTrainer advantage/reward-scaling discussion](https://www.stephendiehl.com/posts/grpotrainer/), [GDPO: Group reward-Decoupled Normalization Policy Optimization](https://nvlabs.github.io/GDPO/) — background confirming GRPO-family group normalization is a separate, within-group mechanism from any persistent cross-round reward normalizer, confidence LOW (secondary sources, not TRL's own changelog)

---
*Stack research for: adversarial RL attack-goal calibration, reward normalization, and adaptive opponent sampling additions to zero-touch-fl*
*Researched: 2026-08-01*
