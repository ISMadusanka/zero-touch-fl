# Architecture Research

**Domain:** Adversarial RL for federated-learning poisoning — integrating 4 additions into an existing round protocol
**Researched:** 2026-08-01
**Confidence:** HIGH (derived directly from reading the current codebase: `rl/env.py`, `rl/rewards.py`, `rl/switch.py`, `rl/schedule.py`, `rl/turns.py`, `server/defense_ensemble.py`, `agents/attacker_agent.py`, `benchmark/harness.py`, `benchmark/metrics.py`, `storage/checkpoint.py`); the delayed-feedback bandit / curriculum-sampling pattern used to resolve the ordering constraint is standard RL engineering practice (see Sources).

This is **not** a greenfield architecture. It is an integration map for four additions onto the existing round protocol:

```
env.begin_round() -> defense.begin_round() -> clean_reference_accuracy()
  -> attacker samples G rollouts -> build_updates() -> features()
  -> defense.verdicts(commit=False) per rollout -> reward per rollout
  -> GRPO step -> turn.commit() -> set_committed_poison() -> env.commit()
```

Every recommendation below plugs into an existing seam rather than adding a new layer.

## Standard Architecture

### System Overview (round-scoped, showing the 4 additions in place)

```
┌───────────────────────────────────────────────────────────────────────────┐
│ rl/schedule.py :: _step_round()  — the ONE per-round driver (used by all  │
│ three schedule variants: best_response, fixed, single_learner)            │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [0] ADAPTIVE SAMPLING — pre-round, uses STALE (last-commit) stats        │
│      state["defense_bandit"].weights(env.defense.names)                   │
│           │                                                                │
│           ▼                                                                │
│      env.defense.set_weights(w)          (new: DefenseEnsemble method)    │
│                                                                             │
│  [1] ctx = env.begin_round()                                              │
│        │                                                                   │
│        ├─ self.round_budget = self._round_budget()                        │
│        ├─ self.round_goal = TARGET LADDER(round_budget)  ◄── addition #1  │
│        │     rl/rewards.py::target_for_budget()  (SAME fn used by         │
│        │     switch.py, attacker_agent prompt, benchmark)                 │
│        ├─ active = self.defense.begin_round()   (adaptive|rotate|random)  │
│        │     ── samples using weights set in [0] ──                       │
│        ├─ key = (active, round_budget)                                    │
│        ├─ ctx.damage_norm = damage_norm_table.get(key)     ◄── addition #2│
│        └─ clean_acc = clean_reference_accuracy()  (unchanged)             │
│                                                                             │
│  [2] G rollouts scored — turn.reward() called G times, commit=False       │
│        drop = ctx.clean_accuracy - post_acc                               │
│        damage = drop_term(drop, target) / ctx.damage_norm  (reads only;   │
│                 table is NOT mutated here — mirrors "rollout isolation")  │
│                                                                             │
│  [3] GRPO step, then turn.commit() (commit=True) → info, drop             │
│                                                                             │
│  [4] AFTER commit — the only WRITE points:                                │
│        graded = stealth_gate(drop, target)      (reused, not new math)    │
│        damage_norm_table.update(key, drop)               ◄── addition #2 │
│        metrics_tracker / RoundLog: record graded ASR      ◄── addition #3 │
│        if adaptive: defense_bandit.record(active, graded) ◄── addition #4│
│                                                                             │
└───────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities (existing vs. new)

| Component | Responsibility | Status |
|-----------|----------------|--------|
| `rl/rewards.py::goal_target(goal)` | Single source of truth for "what drop does this round's goal ask for" | Existing — extend, don't replace |
| `rl/rewards.py::target_for_budget(budget)` *(new)* | Pure function: budget → target drop (the ladder) | **New**, addition #1 |
| `FLArmsRaceEnv._round_goal()` | Builds `round_goal` dict for the round | Existing — call the ladder fn instead of `target_choices` sampling |
| `RoundContext` | Read-only per-round observation (`goal`, `clean_accuracy`, budget, pool) | Existing — gains `damage_norm` field, addition #2 |
| `rl/damage_norm.py::RunningMaxTable` *(new)* | Per-(defense × budget) running max of achieved drop; checkpointed | **New**, addition #2 |
| `rl/defense_bandit.py::DefenseSuccessBandit` *(new)* | Per-defense running graded-success estimate → sampling weights, with a floor | **New**, addition #4 |
| `server/defense_ensemble.py::DefenseEnsemble` | Picks + runs the judging algorithm(s) for the round | Existing — gains `set_weights()` + `selection="adaptive"`, stays UNAWARE of drop/target/success |
| `rl/schedule.py::_step_round` | The ONE per-round driver all three schedules funnel through | Existing — gains the read/write calls above; no new schedule variant needed |
| `metrics/tracker.py`, `metrics/compute.py` | Live per-round metric accumulation | Existing — add graded ASR alongside renamed `evasion_rate` |
| `benchmark/harness.py`, `benchmark/metrics.py` | Per-defense evaluation loop | Existing — add one cheap clean-counterfactual aggregate+eval per defense per round |
| `storage/checkpoint.py` | Save/load state files | Existing — add `damage_norm.json` / `defense_bandit.json` save/load pair, wired like `controller` already is |

## Question-by-Question Resolution

### 1. Where does the budget → target mapping live?

**One function, one place: `rl/rewards.py`.** This module already IS the documented single source of truth — `goal_target(goal)`'s own docstring says it is "shared by the attacker reward … and the schedule's relative win-gate (`rl/switch.py`) so the two never disagree." Add a sibling pure function next to it:

```python
def target_for_budget(budget: int, cfg: dict | None = None) -> float:
    """Deterministic budget -> target_accuracy_drop ladder (2/4/6/8/12%, or from cfg)."""
```

Then change exactly one call site, `FLArmsRaceEnv._round_goal()` (`rl/env.py`), to build `round_goal["target_accuracy_drop"] = target_for_budget(self.round_budget)` instead of sampling from `target_choices`. Nothing else needs to change:

- **Reward** (`attacker_reward` → `goal_target(goal)`) already reads `round_goal` via `ctx.goal` / `turn.goal`. Unaffected.
- **Win gate** (`rl/switch.py::attacker_succeeded`) already takes `goal` and calls `goal_target(goal)`. Unaffected.
- **Attacker prompt** (`agents/attacker_agent.py::build_user_prompt(goal=...)`) already serializes whatever `goal` dict it's given into `attack_goal` in the JSON payload. Unaffected — the ladder is invisible to the prompt-builder, it just sees a bigger/smaller `target_accuracy_drop` for a bigger/smaller `max_poison_clients`, which is exactly the intended signal.
- **Benchmark** (`benchmark/harness.py`) is the ONE place that currently does NOT read `ctx.goal` at all — it takes a fixed `target_drop` argument. This is the actual gap: replace that fixed parameter with a per-round call to `target_for_budget(ctx.budget)` (or read `ctx.goal["target_accuracy_drop"]` if the harness also switches env to produce a real `round_goal` — recommended, since it means the harness and live training share the identical code path with zero duplication).

**Why this satisfies "cannot disagree":** there is exactly one function definition; every consumer either calls it directly or reads the `goal` dict that `_round_goal()` populated from it. Retiring `target_choices` / `sample_target_in_training` (per PROJECT.md) is a deletion in `rl/env.py` + `configs/base.yaml`, not a new abstraction.

**Config surface:** the ladder values (2/4/6/8/12%) belong in `configs/base.yaml` under `attack:` (e.g. `attack.target_ladder: [0.02, 0.04, 0.06, 0.08, 0.12]`, indexed by `budget - 1`), loaded by `target_for_budget`'s `cfg` parameter — matching the existing convention of config-driven thresholds (`SwitchConfig.from_cfg`).

### 2. Where does the (defense × budget) running-max table live?

**Not inside `server/defense_ensemble.py` and not inside `rl/rewards.py`.** Reasoning:

- `DefenseEnsemble` is a *mechanism* (run algorithm X, return verdicts) — it has no concept of accuracy drop, goals, or reward shaping, and its module docstring is explicit that it stays that way ("nothing downstream changes" when the judge changes). Coupling it to damage normalization would violate that boundary.
- `rl/rewards.py` is a pure, stateless, unit-tested module (`tests/test_reward_reference.py` calls these functions with plain floats). Embedding mutable, checkpointed state into it breaks that testability contract.

**New module: `rl/damage_norm.py::RunningMaxTable`** — torch-free, unit-testable in isolation (same pattern as `rl/switch.py::PhaseController`/`SwitchConfig`), holding `{(defense_key, budget): running_max}` with a configurable floor. Two methods only: `get(key) -> float` (read) and `update(key, drop) -> None` (write). `state_dict()` / `load_state_dict()` for checkpointing.

**Who owns the instance, who updates it, and when:**

- **Construction & checkpoint lifecycle**: owned by the driver layer (`main.py::run_phase2`, alongside where `defense = build_ensemble(...)` is already built), NOT by `rl/schedule.py`'s `state` dict and NOT by `FLArmsRaceEnv`. It is injected into the env the same way `defense` already is (`FLArmsRaceEnv(config, ..., defense=defense, damage_norm=damage_norm_table)`) — env consumes it as a collaborator it doesn't own, exactly as it already does with `defense`.
- **Key composition**: `env.begin_round()` is the only place that knows BOTH the active defense name(s) (`self.defense.active_names`, already a property) and this round's budget (`self.round_budget`) at the same instant — so env computes the key and stashes it (`self._damage_norm_key`), then exposes the READ value on `RoundContext.damage_norm`, exactly like it already exposes `clean_accuracy` and `goal`. This keeps `AttackerTurn` and `_log_round` (`rl/schedule.py`) reading ONE value off `ctx` instead of threading it through two independent call sites that could drift (today, `_log_round` already *re-derives* the reward from scratch for logging — reusing `ctx.damage_norm` there guarantees the logged reward matches the one GRPO actually trained on).
- **Update point — commit only, never during rollout scoring.** This is a hard constraint that already exists in this codebase for exactly the same reason: `DefenseEnsemble.verdicts(commit=False)` snapshots/restores algorithm state around every scoring call so that "scoring must not advance defense history" (see its module docstring and `FLArmsRaceEnv`'s "Rollout isolation" constraint in `.planning/codebase/ARCHITECTURE.md`). The running max must obey the identical rule: if it were updated from G speculative (uncommitted, possibly off-policy/high-temperature) rollout drops, the normalizer a rollout is scored against would depend on the OTHER rollouts sampled in the same group, breaking GRPO's "constant within a group" safety argument that PROJECT.md relies on ("GRPO scores its G rollouts within a single round sharing one defense and one budget, so the denominator is constant inside every group"). Concretely: add `env.record_damage(drop)` called exactly once, from `rl/schedule.py::_step_round`, immediately after `turn.commit()` returns — never from `turn.reward()`.
- **Checkpointing**: a new small `save_damage_norm()`/`load_damage_norm()` pair in `storage/checkpoint.py` (own JSON file, `damage_norm.json`), wired into `main.py::run_phase2` the same way `fl_state_cb`/`progress_cb` closures already are, and called from `rl/schedule.py::_checkpoint()` alongside the existing adapter/FL-state/progress saves.

### 3. The ordering constraint — defense chosen before the round, success known after it

This is the crux of addition #4 and it is **not glossable**: `DefenseEnsemble.begin_round()` runs strictly before `clean_reference_accuracy()` (enforced today by a comment in `rl/env.py::begin_round` — "Pick this round's defense BEFORE the clean counterfactual is measured"), while the attacker's measured success against that defense is only known once `turn.commit()` returns, many steps later in the same function.

**Standard structure: a deferred-update / one-round-stale bandit**, split into two objects with disjoint read/write timing:

1. **`rl/defense_bandit.py::DefenseSuccessBandit`** (new, separate controller — see Q5) holds a per-defense running estimate of graded success (EMA or bounded window), and exposes:
   - `weights(names) -> dict[str, float]`: sampling weight per defense, inversely proportional to its current success estimate, floored at `min_weight` so no defense's weight ever reaches zero (the starvation floor).
   - `record(name, graded_success) -> None`: update the estimate for one defense.
2. **Round timeline, all inside `rl/schedule.py::_step_round`** (the single function every schedule variant already calls):
   - **Before `env.begin_round()`**: `env.defense.set_weights(bandit.weights(env.defense.names))`. These weights reflect every round's outcome **up to and including the previous commit** — i.e. exactly one round stale, by construction, because the bandit has literally not been told about this round yet.
   - **`env.begin_round()`**: `DefenseEnsemble.begin_round()` samples the active algorithm from those (stale) weights when `selection == "adaptive"`. This is the only read of the bandit's state for the round; nothing later in the round can change which defense judges it (matching the existing "held constant across counterfactual, rollout scoring, and commit" invariant).
   - **Round runs to completion** (clean counterfactual, G rollouts, GRPO step, commit) — the bandit is not touched.
   - **After `turn.commit()`**: compute `graded = stealth_gate(drop, goal_target(ctx.goal))` — this is *literally the same function* `rl/rewards.py` already uses inside `attacker_reward` to gate stealth, so grading a round's success needs no new formula, just reuse. Call `bandit.record(active_defense_name, graded)`.
   - **Next iteration of the loop**: step 1 re-reads the bandit, now including this round's update.

This is the standard shape for adaptive-curriculum / task-sampling under delayed feedback: **decide from the statistics as of the last completed unit of work, observe the outcome only after that unit commits, and let the one-step lag be the mechanism** rather than trying to make the decision and the outcome simultaneous (which is impossible here — the defense's identity is a precondition for computing the very accuracy the success signal is made from). It requires no additional synchronization, no rollback, and no change to the "defense held constant for the whole round" invariant; it only requires that the READ (`weights()`) and WRITE (`record()`) calls sit on opposite sides of `turn.commit()` inside `_step_round`.

**Cold start:** round 1 (or any round before the bandit has seen a given defense) has no prior estimate. `weights()` must return the floor-uniform distribution for defenses it has never recorded, matching `DefenseEnsemble`'s existing `random`/`rotate` cold-start behavior (index 0 used as-is on the first round).

**Checkpointing:** same pattern as `RunningMaxTable` — a `defense_bandit.json` save/load pair, `state_dict()`/`load_state_dict()`, wired through `rl/schedule.py::_checkpoint()`. Reproducibility (PROJECT.md constraint: "adaptive sampling and the running max must both be seeded and checkpointed so runs replay") is satisfied because `DefenseEnsemble.begin_round()` already draws from the run's own `rng` for `selection == "random"`; `adaptive` reuses that same `self._rng` for its weighted draw, so the sampling stream stays deterministic under a fixed `fl.poison_seed`.

### 4. Where does the benchmark's per-defense clean counterfactual belong, and how is the cost bounded?

**It belongs in `benchmark/harness.py`'s round loop**, not in `server/defense_ensemble.py` — the benchmark's whole design is "each defense evolves its OWN global model" (module docstring), which is architecturally distinct from the live-training path where one shared global model is judged by one algorithm at a time. Reusing `DefenseEnsemble` here would conflate two different aggregation topologies.

**Mechanism — reuse the existing snapshot/restore isolation pattern, not a new one.** Every defense object already exposes `.step(updates, poisoned_ids)` and (per the live-training path's precedent in `DefenseEnsemble.verdicts(commit=False)`) can be snapshotted/restored around a scoring call without corrupting its cross-round history (DeFL's critical-learning-period test, FLTrust's trust state, etc.). The harness needs the same trick per defense, per round:

```python
for name, d in defenses.items():
    state = d.state_dict()
    clean_res = d.step(honest_updates, set())      # no poison, same defense's OWN model/state
    clean_acc = eval_on(clean_res.new_global)        # scratch eval, not the real one
    d.load_state_dict(state)                         # roll back — this call must not count
    res = d.step(updates, poisoned_ids)              # the REAL (poisoned) step, as today
    ...
    metrics[name].record(ctx.round_num, res.verdicts, poisoned_ids, acc,
                         clean_accuracy=clean_acc, ...)   # extended signature
```

`benchmark/metrics.py::DefenseMetrics` currently takes one fixed `target_drop`/`baseline_accuracy` at construction and computes a single `goal_threshold` once. It needs to become **per-round**: `record()` gains a `clean_accuracy` (and, once addition #1 lands, a per-round `target_drop` from `target_for_budget(ctx.budget)`) so `goal_hit`/graded-ASR are computed against *this round's* counterfactual, mirroring exactly what `rl/rewards.py::attacker_reward` already does for live training (`reference_accuracy` per round, not a fixed baseline) — this is the fix PROJECT.md calls out as "a fixed baseline credits each defense's self-inflicted accuracy sag to the attacker."

**Cost bound.** The added cost is bounded and small by construction, for a reason specific to this project: the expensive resource is the LLM attacker generation (one `policy.generate(...)` call per round, shared across all defenses, unchanged by this addition), while the model under evaluation is a ~970-parameter MLP (`model/mnist_net.py`) — a full MNIST test-set forward pass is milliseconds. The addition is exactly **one extra cheap `.step()` + one extra cheap `.evaluate()` per defense per round** (doubling the per-round, per-defense cost of an already-cheap loop), with **zero extra LLM calls**. No throttling, sampling, or caching is needed; if the harness is ever run with a much larger model, the natural bound to add later is to compute the clean counterfactual only every K rounds and forward-fill between (the same tradeoff `env.clean_reference_accuracy()` already accepts by caching once per round rather than once per rollout).

### 5. Standard component boundary for bandit/curriculum state in an RL training loop

**Neither the environment nor a monolithic "schedule."** The codebase already answers this question for a sibling concern: `rl/switch.py::PhaseController` is a **separate, torch-free, unit-testable controller class** — not part of `FLArmsRaceEnv` (which "knows nothing about LLMs" or training phases) and not inlined into `rl/schedule.py` (which only *drives* it: constructs it, calls `record()`/`next_phase()` at defined lifecycle points, and persists its `state_dict()`). `League` (also in `rl/schedule.py`) follows the same shape for opponent-snapshot curriculum: a small stateful object held in the `train()`-local `state` dict, exercised at fixed points (`_post_round_bookkeeping`, `_opponent_generator`).

Both new pieces of state — `RunningMaxTable` and `DefenseSuccessBandit` — should be **new siblings of `PhaseController` and `League`**, not new responsibilities bolted onto `FLArmsRaceEnv` or `DefenseEnsemble`:

| State | Module | Instantiated by | Read at | Written at | Checkpointed via |
|---|---|---|---|---|---|
| `PhaseController` (existing) | `rl/switch.py` | `rl/schedule.py::train()` | every round (`_step_round`'s caller) | `ctrl.record()` after each committed round | `save_progress(controller=...)` |
| `League` (existing) | `rl/schedule.py` | `train()` | phase start (`_opponent_generator`) | `_post_round_bookkeeping` on a cadence | not persisted today (in-memory only) |
| `RunningMaxTable` (new) | `rl/damage_norm.py` | `main.py::run_phase2` (co-located with `defense = build_ensemble(...)`), injected into `env` | `env.begin_round()` (populates `ctx.damage_norm`) | `env.record_damage(drop)`, called from `_step_round` right after `turn.commit()` | new `damage_norm.json` pair in `storage/checkpoint.py`, wired via a new `damage_norm_cb` alongside `fl_state_cb`/`progress_cb` |
| `DefenseSuccessBandit` (new) | `rl/defense_bandit.py` | `rl/schedule.py::train()`, added to `state` dict | `_step_round`, before `env.begin_round()` (feeds `env.defense.set_weights()`) | `_step_round`, after `turn.commit()` | new `defense_bandit.json` pair, wired via `_checkpoint()` the same way `controller` is |

The generalizable rule this codebase already follows: **the environment owns FL/round mechanics (what accuracy resulted from these weights and verdicts); the defense module owns detection mechanics (which clients does this algorithm flag); everything about *how the training loop uses those outcomes to make its next decision* — phase switching, opponent curriculum, reward normalization, defense-choice curriculum — is a separate, checkpointed, torch-free controller object instantiated and driven by `rl/schedule.py`.** `RunningMaxTable` is the one partial exception worth flagging: because its *key* requires information that only `env.begin_round()` has synchronized at one instant (active defense name(s) + this round's budget), the read side is most naturally exposed through `RoundContext` rather than re-derived by the driver — but its *lifecycle* (construction, checkpointing) still belongs outside the environment, exactly like `defense` itself is built externally and merely attached.

## Build Order

Dependency graph (arrows = "must land before"):

```
#1 target ladder ──────────────┬──────────────► #3 graded ASR (live tracker)
  (rl/rewards.py,               │
   rl/env.py)                   └──────────────► #3 graded ASR (benchmark)
                                                        ▲
                                                        │ requires
                              benchmark per-defense     │
                              clean counterfactual ─────┘
                              (benchmark/harness.py)

#1 target ladder ──────────────────────────────► #2 running-max normalization
                                                   (rl/damage_norm.py)

#3 graded ASR ──────────────────────────────────► #4 adaptive defense sampling
                                                   (rl/defense_bandit.py)
```

**Recommended sequence:**

1. **Target ladder (#1) first, unconditionally.** Every other addition either reads `goal_target`/`round_goal` directly (#2, the win-gate is already wired) or reads a per-round graded-success number computed FROM the target (#3, #4). Building this first also immediately unblocks re-running the existing test suite (`tests/test_switch.py`, `tests/test_reward_reference.py`) to confirm the schedule can now reach `success_streak` at every budget — the concrete symptom PROJECT.md is fixing.
2. **Graded ASR in the live tracker (#3, live half)** next — it is a pure read of quantities the round loop already computes (`drop`, `ctx.goal`) via the existing `stealth_gate` function, so it has no new state and no ordering hazards. Do the metric rename (`evasion_rate` / `attack_success_rate` / `goal_success_rate`) here too, since it touches the same call sites.
3. **Benchmark per-defense clean counterfactual**, then **graded ASR in the benchmark (#3, benchmark half)** — these two are coupled (the benchmark can't grade a per-round drop without the per-round clean reference to subtract from), so build them together, immediately after step 1 makes `target_for_budget` available for the benchmark to consume too.
4. **Running-max normalization (#2)** — independent of #3, so it can be built in parallel with step 2/3 by a different work-stream, but should land before #4 so the reward surface is already well-calibrated before adaptive sampling starts reshaping which defense gets seen most.
5. **Adaptive defense sampling (#4) last.** It strictly needs #3's graded-success signal (there is no other source of "measured success against defense X" in the codebase), and it is the addition most likely to make debugging confusing if the reward/target machinery underneath it isn't already trustworthy — per PROJECT.md's own "Accepted risk" note, adaptive sampling concentrates rounds on the hardest defense, so any residual bug in #1–#3 will be amplified exactly where it's hardest to see (the defense being visited least).

## Anti-Patterns to Avoid

### Anti-Pattern: Updating the running-max or the bandit from rollout-scoring calls

**What people do:** Update `RunningMaxTable`/`DefenseSuccessBandit` inside `turn.reward()` (called G times per round) instead of only after `turn.commit()`.

**Why it's wrong:** This is exactly the "Rollout isolation" violation the codebase already guards against for `DefenseEnsemble` (scoring must not advance defense history) — it would make the normalizer/weights depend on off-policy, higher-temperature speculative rollouts rather than the single committed action, breaking the "constant within a GRPO group" safety property the whole damage-normalization design relies on, and making bandit weights non-reproducible across a resume that doesn't replay identical rollouts.

**Do this instead:** Read once at `env.begin_round()` (stashed on `ctx`), write once after `turn.commit()`, both inside `rl/schedule.py::_step_round`.

### Anti-Pattern: Letting `DefenseEnsemble` compute its own adaptive weights

**What people do:** Add drop/target/success awareness directly into `server/defense_ensemble.py::begin_round()` so it can "just figure out" which algorithm the attacker is beating.

**Why it's wrong:** `DefenseEnsemble` has no visibility into `drop`, `goal`, or `reference_accuracy` — those live in `rl/env.py` and `rl/rewards.py`. Threading them through would turn a pure detection-mechanism module into a reward-aware one, contradicting its own docstring ("nothing downstream changes" when the judge changes) and making it untestable without the full RL stack.

**Do this instead:** `DefenseEnsemble` only ever receives a plain `{name: weight}` dict via `set_weights()` and samples from it; all the success bookkeeping lives in `rl/defense_bandit.py`, owned by the training driver.

### Anti-Pattern: Grading the benchmark against a fixed baseline while grading live training against a per-round counterfactual

**What people do:** Ship the target ladder and running max for live training but leave `benchmark/metrics.py` computing `goal_threshold` once from a fixed `baseline_accuracy`.

**Why it's wrong:** This is the exact bug PROJECT.md documents as already present — a defense that sags on its own (e.g. FLTrust under non-IID) gets that self-inflicted sag credited to the attacker, which "skews graded ASR differently per defense and undermines the cross-defense comparison the benchmark exists to make."

**Do this instead:** `DefenseMetrics.record()` must take a per-round `clean_accuracy` (from the new per-defense clean-counterfactual step) exactly as `rl/rewards.py::attacker_reward` already takes a per-round `reference_accuracy`.

## Sources

- Primary source: direct reading of this repository's `rl/env.py`, `rl/rewards.py`, `rl/switch.py`, `rl/schedule.py`, `rl/turns.py`, `server/defense_ensemble.py`, `agents/attacker_agent.py`, `benchmark/harness.py`, `benchmark/metrics.py`, `metrics/compute.py`, `metrics/tracker.py`, `storage/checkpoint.py`, `configs/base.yaml`, and `.planning/PROJECT.md` / `.planning/codebase/ARCHITECTURE.md` (2026-08-01 snapshot).
- [Reinforcement Learning with Curriculum Sampling (RLCS)](https://www.emergentmind.com/topics/reinforcement-learning-with-curriculum-sampling-rlcs) — general pattern for adaptive, progress-driven task/curriculum sampling in RL, consistent with the deferred-update bandit structure recommended for addition #4.
- [Distributionally Robust Multi-Task Reinforcement Learning via Adaptive Task Sampling](https://arxiv.org/html/2605.14350v1) — adaptive per-task sampling weighted by measured performance, the same shape as weighting defenses by measured attacker success.
- General multi-armed-bandit practice (EXP3 / UCB with delayed feedback): the standard resolution when an arm must be committed before its reward is observable is to act on the statistics as of the last completed pull and update only after the outcome resolves — applied here as "one-round-stale" defense-selection weights.

---
*Architecture research for: adversarial RL / robust-FL poisoning testbed — integration of budget-conditioned targets, damage normalization, graded ASR, and adaptive defense sampling*
*Researched: 2026-08-01*
