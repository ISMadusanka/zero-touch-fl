# Phase 1: Budget-Conditioned Target Ladder - Context

**Gathered:** 2026-08-01
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers **one deterministic `budget → target_accuracy_drop` function, declared in config, resolving the round's attack target identically for the reward, the win gate, and the attacker prompt** — with the retired per-round target sampling removed and the 0.02 bottom rung empirically confirmed measurable before Phase 2 begins.

Concretely:
- `target_for_budget()` added to `rl/rewards.py` as a sibling of `goal_target()` (GOAL-01)
- The ladder declared in `configs/base.yaml`, re-tunable with no code edit (GOAL-02)
- `FLArmsRaceEnv._round_goal()` derives the target from the round's sampled poison budget (GOAL-03)
- `attack.target_choices` and `attack.sample_target_in_training` deleted, with no reader left anywhere (GOAL-04)
- A test proving the reward path, `rl/switch.py`'s win gate, and the attacker prompt cannot disagree about a round's target (GOAL-05)
- A recorded pre-flight measurement of `clean_reference_accuracy` noise per defense, gating whether 0.02 survives as the bottom rung (GOAL-06)

**Not this phase:** damage normalization (Phase 2), graded ASR (Phase 3), the benchmark's `target_drop` argument and per-round counterfactual (Phase 4), adaptive defense sampling (Phase 5), learning evidence (Phase 6).

</domain>

<decisions>
## Implementation Decisions

### Ladder Declaration

- **D-01:** The ladder is declared in `configs/base.yaml` as an **explicit budget→drop map** under `attack.target_ladder`, not a positional list:

  ```yaml
  attack:
    target_ladder:
      1: 0.02
      2: 0.04
      3: 0.06
      4: 0.08
      5: 0.12   # super-linear: spending the whole pool demands more
  ```

  Rationale: each rung names its own budget, so there is no implicit budget→index offset that a mis-edit could shift silently. A positional list was rejected specifically because a shifted list would still produce valid-looking targets that nothing in the reward path would flag. Implementation note: PyYAML yields integer keys here, but the loader must coerce/accept quoted string keys too so a hand-edited `"1": 0.02` does not silently miss.

- **D-02:** The ladder values for this milestone are `1→0.02, 2→0.04, 3→0.06, 4→0.08, 5→0.12` (locked upstream in REQUIREMENTS.md GOAL-01 and PROJECT.md Key Decisions — not re-opened in this discussion). The 0.12 top rung is deliberately super-linear.

### Off-Ladder Budgets

- **D-03:** A budget with no rung in the ladder is a **config error that fails fast at startup**, not a runtime condition. `FLArmsRaceEnv.__init__` validates that `attack.target_ladder` covers every budget in `[1, budget_cap]` and raises a `RuntimeError` naming the missing rungs. — **Reversibility:** reversible — the check is a few lines in one constructor with no persisted state or published contract behind it.

  Rationale: the realistic trigger is someone raising `fl.n_compromisable` / `attack.max_poison_clients` without extending the ladder. Clamping to the nearest rung was rejected because two budgets would then share one target, which silently destroys the independence of the per-(defense × budget) cells that Phase 2's normalizer and Phase 3's graded ASR are both keyed on. Extrapolating was rejected because it would invent targets never validated as reachable — the exact failure this milestone exists to fix — and the super-linear top rung gives no clean slope to extend anyway.

  Startup-time failure is the right moment: catching it before a `simulation_rounds: 2000000` run starts on the remote CUDA box costs seconds; catching it mid-run costs the run.

### Config Absence and Defaults

- **D-04:** A module-level `DEFAULT_TARGET_LADDER = {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.12}` lives in `rl/rewards.py`. When `attack.target_ladder` is present in config it **replaces the default wholesale**; when absent, the default is used. Per-rung merging was explicitly rejected — the effective ladder must be readable from one place, not mentally merged across two files.

  Rationale: this is the established convention throughout this codebase (`attack.get("target_choices", [...])` and `fl.get("n_compromisable", ...)` in `rl/env.py`, `DEFAULT_GOAL` in `agents/attacker_agent.py`). It also protects Success Criterion 4: existing test fixtures and benchmark configs that build minimal config dicts keep working untouched, so the only test edits in this phase are the ones asserting retired `target_choices` sampling behavior.

  Known cost, accepted: the ladder numbers appear in two files. If GOAL-06 forces the bottom rung up, whoever edits `configs/base.yaml` should consider whether `DEFAULT_TARGET_LADDER` should move with it.

### Function Signature

- **D-05:** `def target_for_budget(budget: int, ladder: dict | None = None) -> float`, falling back to `DEFAULT_TARGET_LADDER` when `ladder` is `None`. It stays a **pure function with no module-level mutable state**, mirroring its sibling `goal_target(goal)`.

  `FLArmsRaceEnv.__init__` loads the ladder from config once into `self.target_ladder`; `_round_goal()` passes it explicitly:

  ```python
  def _round_goal(self) -> dict:
      return {"type": "untargeted_degrade",
              "target_accuracy_drop": target_for_budget(
                  self.round_budget, self.target_ladder)}
  ```

  Rationale: GOAL-01 writes the signature as `target_for_budget(budget)`, which would require the ladder to live in module state installed at startup. That was rejected — CLAUDE.md's module design rules say "no module-level mutable state (singletons avoided)", it makes startup order-dependent, and tests would have to set and reset a global or leak state between cases. That hazard gets worse immediately: Phase 2's rollout-isolation tests assert `state_dict()` is unchanged across G rollouts, and global reward-module state would muddy exactly that kind of assertion. Taking the whole config dict was also rejected — `goal_target(goal)` deliberately takes only the goal dict, not config, and a reward primitive should not know the config file's nested shape.

  **Deviation from GOAL-01 as literally written:** the requirement text says `target_for_budget(budget)`. The optional second parameter satisfies that call form (a bare `target_for_budget(3)` works and returns the default ladder's value) while keeping the function pure. Planner should treat GOAL-01 as satisfied.

### Claude's Discretion

The user selected only the ladder-shape area, leaving these to the researcher and planner:

- Whether `target_for_budget` clamps or raises on `budget <= 0`. Note `_round_budget()` returns `rng.randint(1, budget_cap)` so budgets are always ≥ 1 in practice; this is a defensive-contract question, not a live path.
- Whether `target_for_budget` keeps `goal_target`'s `max(target, 1e-6)` floor. Recommend yes for consistency — `goal_target` will read the ladder-populated value anyway, so the floor applies downstream regardless.
- Whether a rung raised by GOAL-06 carries a provenance note (measured noise floor, date) beside it in config.
- Exact wording of the `RuntimeError` from D-03.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone Requirements and Scope
- `.planning/REQUIREMENTS.md` §Attack Goal Ladder — GOAL-01 … GOAL-06, the six requirements this phase must satisfy
- `.planning/REQUIREMENTS.md` §Out of Scope — in particular "Raising `n_compromisable` or the eval poison budget" (the threat model is fixed; targets move, not the model)
- `.planning/ROADMAP.md` §Phase 1 — the four success criteria, and §Ordering Constraints for why Phase 1 is unconditional and GOAL-06 is a gate
- `.planning/PROJECT.md` §Context — the arithmetic derivation of the failure: FedAvg's `(f−1)/20 · D_p` dilution, the `stealth_gate` flattening, and why `success_streak` never fires

### Prior Research (already completed for this milestone — read before re-researching)
- `.planning/research/ARCHITECTURE.md` §"One function, one place" — names `rl/rewards.py::target_for_budget()` as the single source of truth and walks each consumer (reward, win gate, prompt, benchmark) explaining why only `_round_goal()` changes
- `.planning/research/PITFALLS.md` — the ladder-target ASR is the only signal anything outside the GRPO gradient step may read; relevant as a constraint Phase 1 must not violate
- `.planning/research/SUMMARY.md` — milestone-level synthesis of the four additions

### Code the Phase Touches
- `rl/rewards.py:85-100` — `goal_target(goal)`, the existing documented single source of truth; `target_for_budget` is its sibling. Note its docstring already claims the "never disagree" property that GOAL-05 must now test.
- `rl/env.py:83-89` — `sample_target` / `target_choices` attributes to delete
- `rl/env.py:178-186` — `_round_goal()`, the single call site that changes
- `rl/env.py:172-176` — `_round_budget()`, the budget this ladder is keyed on
- `rl/switch.py:28` — imports and calls `goal_target(goal)`; follows automatically, no edit expected
- `agents/attacker_agent.py:35` — `DEFAULT_GOAL`; `build_user_prompt(goal=...)` serializes whatever goal dict it is handed
- `configs/base.yaml:26-45` — the `attack:` block; lines 34-38 are the deletions

### Project Conventions
- `.claude/CLAUDE.md` §Module Design — "No module-level mutable state (singletons avoided)", the rule that shaped D-05
- `.claude/CLAUDE.md` §Error Handling — contract violations raise `RuntimeError`; the pattern D-03 follows

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`rl/rewards.py::goal_target(goal)`** (lines 85-100): already the documented single source of truth — its docstring states it is "shared by the attacker reward … and the schedule's relative win-gate (`rl/switch.py`) so the two never disagree." `target_for_budget` extends this module rather than introducing a new one. `goal_target` dispatches on goal type (`slow_degrade` → `per_round_drop`, else → `target_accuracy_drop`) and floors at `1e-6`.
- **`FLArmsRaceEnv._round_budget()`** (`rl/env.py:172-176`): already produces the per-round budget the ladder keys on — `rng.randint(1, budget_cap)` when sampling, else the cap. Nothing about budget sampling changes.
- **Config-with-inline-default pattern**: `attack.get("target_choices", [...])`, `fl.get("n_compromisable", ...)` — the convention D-04 follows.

### Established Patterns
- **Config defaults live at the read site, not in a schema layer.** There is no config validation module; each consumer supplies its own default. D-04 keeps this, D-03 adds the first startup-time cross-field validation (`ladder` vs `budget_cap`) — planner should place it in `FLArmsRaceEnv.__init__` beside the existing `budget_cap` clamping rather than inventing a validation module.
- **`round_goal` is the transport.** `begin_round()` sets `self.round_goal = self._round_goal()` (`rl/env.py:204`) and it flows to the reward, the win gate, and the prompt as an ordinary dict. The ladder is invisible to every consumer — they see a bigger `target_accuracy_drop` for a bigger `max_poison_clients`, which is exactly the intended signal.
- **`rl/switch.py` is already wired.** It imports `goal_target` and applies `win_fraction` relative to whatever the round's goal asks. No edit expected — but GOAL-05's test must prove this rather than assume it.

### Integration Points
- **`rl/env.py::_round_goal()`** — the one behavioral change: sample-from-`target_choices` becomes `target_for_budget(self.round_budget, self.target_ladder)`.
- **`FLArmsRaceEnv.__init__`** — loads `self.target_ladder` from `attack.target_ladder` (D-04) and runs the coverage validation (D-03); deletes `self.sample_target` and `self.target_choices`.
- **`configs/base.yaml`** — `attack.target_ladder` added; `sample_target_in_training` and `target_choices` deleted.
- **Tests** — `tests/test_switch.py` and `tests/test_reward_reference.py` are the natural homes for GOAL-05's "cannot disagree" assertion; both already exercise the goal→target path.

### Known Hazards
- **Three declarations of `target_accuracy_drop: 0.20` exist** — `configs/base.yaml:30`, `configs/attacker_agent.yaml:12`, and `DEFAULT_GOAL` in `agents/attacker_agent.py:35`. This phase does not resolve their fate (the user did not select that area). Whatever the planner decides, the eval/`--dry-run`/`--baseline`/`infer.py` paths must not be left silently reading 0.20 while training reads the ladder.
- **Brownfield working tree.** Branch `attacker-only-learn` carries uncommitted work across 16 files including `rl/env.py`, `rl/rewards.py`, `configs/base.yaml`, and four test files. Read the working-tree state, not `HEAD`.
- **`benchmark/harness.py:29`** takes a fixed `target_drop` argument and bypasses `ctx.goal` entirely. That is BENCH-03 / Phase 4 — leave it alone here, but do not assume it follows the ladder yet.
- **README.md:321-331** documents the `target_choices` sampling behavior being retired. GOAL-04's "no remaining reader" sweep should catch documentation too.

</code_context>

<specifics>
## Specific Ideas

- The explicit-map config form was chosen over a compact list on an explicit **silent-failure** argument: a shifted positional list still produces plausible targets and nothing downstream would flag it. The user consistently preferred the form where a mistake is loud.
- Same instinct drove D-03: fail at startup rather than degrade gracefully, because a graceful degradation here corrupts the per-(defense × budget) cell structure that Phases 2 and 3 are built on, and corrupted results are worse than a stopped run.
- D-04 was chosen against strict single-declaration purity, on the grounds that it keeps Success Criterion 4 honest — the only test edits in this phase should be the retired sampling assertions, not a sweep through every fixture that builds a config dict.

</specifics>

<deferred>
## Deferred Ideas

Nothing raised in this discussion fell outside the phase boundary. The following are **in-phase gray areas the user chose not to discuss** — they are open questions for the researcher and planner, not deferred scope:

1. **Fate of the fixed `target_accuracy_drop: 0.20`** (GOAL-04 adjacent) — whether the ladder becomes the universal target source across eval, `--dry-run`, `--baseline`, and `infer.py`, or a fixed fallback survives for budget-less paths; and what `target_for_budget` means for `slow_degrade` / `targeted_label` goals, which `goal_target` handles differently. Three declarations exist (see Known Hazards above).
2. **GOAL-06 pre-flight rig** — repetitions per defense, what margin counts as "clears the noise" (2σ, 3σ, or another bar), whether the measurement script is committed as a tool or thrown away, and where the recorded result lives. `.planning/STATE.md` §Blockers already flags this as a gate rather than a task: if 0.02 does not clear the noise for every defense, the rung is raised in config before Phase 2 starts.
3. **Enforcement strength for GOAL-04 / GOAL-05** — whether "cannot disagree" and "no remaining reader" are ordinary unit assertions, a repo-scanning test that fails on any reintroduced key, or a structural guard making disagreement impossible. Success Criterion 2 asks for both a test and a tree-wide search; whether the search is automated or one-time is undecided.

Genuinely out of this milestone (already recorded in `.planning/REQUIREMENTS.md` §v2 Requirements): the offline calibration sweep of the reachable drop frontier (V2-02) and multi-seed confidence intervals (V2-03) — either would give the ladder empirical grounding beyond GOAL-06's single-rung noise check.

</deferred>

---

*Phase: 1-Budget-Conditioned Target Ladder*
*Context gathered: 2026-08-01*
