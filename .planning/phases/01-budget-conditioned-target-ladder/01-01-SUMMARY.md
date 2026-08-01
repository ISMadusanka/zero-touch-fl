---
phase: 01-budget-conditioned-target-ladder
plan: 01
subsystem: rl
tags: [reward-shaping, federated-learning, grpo, adversarial-rl, target-ladder]

# Dependency graph
requires: []
provides:
  - "rl/rewards.py::target_for_budget(budget, ladder=None) + DEFAULT_TARGET_LADDER — the single source of truth for a budget's target_accuracy_drop"
  - "FLArmsRaceEnv.target_ladder — loaded, key/value-coerced, and coverage-validated at construction"
  - "FLArmsRaceEnv._round_goal() deriving target_accuracy_drop from the round's poison budget for untargeted_degrade goals"
  - "tests/test_target_ladder.py — 7-test proof that the reward path, the win gate, and the attacker prompt cannot disagree about a round's target"
affects: [01-02, 01-03, phase-2-damage-normalization, phase-3-graded-asr]

# Actuals (#2632)
actuals:
  tokens: 4630
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure reward primitive with an explicit-argument ladder (no module-level mutable state), mirroring goal_target(goal)"
    - "Startup-time RuntimeError coverage validation beside an existing constructor-time clamp (rl/env.py's budget_cap block)"

key-files:
  created:
    - tests/test_target_ladder.py
  modified:
    - rl/rewards.py
    - rl/env.py

key-decisions:
  - "D-02/D-04/D-05 (locked in 01-CONTEXT.md) implemented verbatim: ladder values 1->0.02..5->0.12, wholesale config replacement (no per-rung merge), pure function taking the ladder as an explicit optional parameter rather than module state"
  - "D-03 RuntimeError (not the codebase's usual ValueError) implemented as specified — a deliberate, documented deviation from the surrounding constructor-check convention"
  - "GA-1: all three fixed target_accuracy_drop: 0.20 declarations (configs/base.yaml, configs/attacker_agent.yaml, agents/attacker_agent.py::DEFAULT_GOAL) left untouched; a mechanical guard (test_round_goal_never_returns_the_fixed_config_target) proves the training path never resolves to that sentinel instead"

patterns-established:
  - "Budget-conditioned target resolution: any consumer needing a round's attack target must read env.round_goal (set from _round_goal()), never sample or hardcode target_accuracy_drop"

requirements-completed: [GOAL-01, GOAL-03, GOAL-05]

coverage:
  - id: D1
    description: "target_for_budget(budget, ladder=None) + DEFAULT_TARGET_LADDER added to rl/rewards.py as a pure sibling of goal_target, with the same 1e-6 floor"
    requirement: "GOAL-01"
    verification:
      - kind: unit
        ref: "tests/test_target_ladder.py#test_off_ladder_budget_raises_in_the_pure_function"
        status: pass
    human_judgment: false
  - id: D2
    description: "FLArmsRaceEnv.__init__ loads self.target_ladder (coercing int/quoted-string keys), validates coverage over [1, budget_cap], and raises RuntimeError naming missing rungs"
    requirement: "GOAL-03"
    verification:
      - kind: unit
        ref: "tests/test_target_ladder.py#test_off_ladder_budget_raises_at_construction"
        status: pass
    human_judgment: false
  - id: D3
    description: "_round_goal() derives target_accuracy_drop from the round's poison budget for untargeted_degrade goals, and returns other goal types (e.g. slow_degrade) unchanged"
    requirement: "GOAL-03"
    verification:
      - kind: unit
        ref: "tests/test_target_ladder.py#test_non_untargeted_goal_is_returned_unchanged"
        status: pass
    human_judgment: false
  - id: D4
    description: "The reward path (goal_target), the schedule's win gate (attacker_succeeded), and the attacker prompt (build_user_prompt) all resolve the identical target at every rung, under exact float == with no tolerance, at both ends of the ladder"
    requirement: "GOAL-05"
    verification:
      - kind: unit
        ref: "tests/test_target_ladder.py#test_budget_three_resolves_one_target_end_to_end"
        status: pass
      - kind: unit
        ref: "tests/test_target_ladder.py#test_every_rung_agrees_across_reward_win_gate_and_prompt"
        status: pass
    human_judgment: false
  - id: D5
    description: "Retired self.sample_target / self.target_choices attributes are gone from rl/env.py, in code and in prose"
    verification:
      - kind: unit
        ref: "grep -c 'target_choices' rl/env.py; grep -c 'sample_target' rl/env.py (both 0)"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-08-01
status: complete
---

# Phase 1 Plan 1: Budget-Conditioned Target Ladder (Tracer + Hardening) Summary

**`target_for_budget()` resolves an attack target deterministically from a round's poison budget, and one end-to-end test proves the reward path, the schedule's win gate, and the attacker prompt cannot disagree about it at any rung.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-08-01T12:23:00Z
- **Completed:** 2026-08-01T12:36:00Z
- **Tasks:** 2
- **Files modified:** 3 (2 modified, 1 created)

## Accomplishments

- Added `DEFAULT_TARGET_LADDER` (`{1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.12}`) and `target_for_budget(budget, ladder=None)` to `rl/rewards.py` as a pure sibling of `goal_target`, keeping its `max(target, 1e-6)` floor and raising `RuntimeError` (naming the offending budget and the covered rungs) for any off-ladder key, including an explicitly empty `{}`.
- Wired `FLArmsRaceEnv.__init__` to load `self.target_ladder` from `attack.target_ladder` (wholesale replacement, int/quoted-string key coercion), validate coverage over `[1, budget_cap]` at construction (`RuntimeError` naming missing rungs), and warn when the configured goal type is not `untargeted_degrade` (ladder inactive for that goal).
- Rewrote `_round_goal()` to derive `target_accuracy_drop` from `target_for_budget(self.round_budget, self.target_ladder)` for `untargeted_degrade` goals, passing other goal types through unchanged.
- Deleted the retired `self.sample_target` / `self.target_choices` attributes and their comment block from `rl/env.py` — no remaining code or prose reference (`grep -c` for both returns `0`).
- Wrote `tests/test_target_ladder.py` (7 tests): the Task 1 tracer proving budget 3 resolves `0.06` identically through `goal_target`, `attacker_succeeded`, and `build_user_prompt`'s serialized JSON; plus six Task 2 hardening tests covering every rung at both boundaries, pairwise-distinct rungs, off-ladder budgets at both the constructor and the pure function, `slow_degrade` pass-through, and the GA-1 guard against the fixed `0.20` sentinel ever leaking through as a resolved target.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end — poison budget 3 resolves target 0.06 through reward, win gate, and prompt** - `c7bfeab` (feat, tracer)
2. **Task 2: Harden the ladder — all five rungs, both ends, off-ladder, and non-untargeted goals** - `c47341e` (test)

## Files Created/Modified

- `rl/rewards.py` - Added `DEFAULT_TARGET_LADDER` module constant and `target_for_budget(budget, ladder=None)` pure function (sibling of `goal_target`)
- `rl/env.py` - Load/validate `self.target_ladder` in `__init__`; rewrote `_round_goal()`; deleted `self.sample_target`/`self.target_choices`
- `tests/test_target_ladder.py` - New: 7 tests locking in the ladder and its three-way agreement across consumers

## Decisions Made

- Followed all five locked CONTEXT.md decisions (D-01 through D-05) verbatim, including the deliberate `RuntimeError` (not `ValueError`) exception type for D-03's coverage check, which diverges from the rest of the codebase's constructor-check convention by explicit user instruction.
- GA-1 (fixed `target_accuracy_drop: 0.20` fate): left all three declarations (`configs/base.yaml:30`, `configs/attacker_agent.yaml:12`, `agents/attacker_agent.py::DEFAULT_GOAL`) untouched, since every training/`--dry-run`/`--baseline` consumer routes through `_round_goal()` and therefore follows the ladder automatically. Added the mechanical guard test instead of touching those declarations.
- Test fixture merge semantics: implemented `_merge_cfg` as a one-level-deep merge (not fully recursive) so an override like `attack={"goal": {...}}` replaces the base `goal` dict wholesale rather than partially merging into it — this matches D-04's "present config replaces the default wholesale, no per-rung merge" semantics and was necessary to make `test_non_untargeted_goal_is_returned_unchanged` construct the intended `slow_degrade` goal without leftover `target_accuracy_drop` bleeding in from the base fixture.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test fixture merge helper produced a corrupted goal dict**
- **Found during:** Task 2 (`test_non_untargeted_goal_is_returned_unchanged`)
- **Issue:** The initial `_deep_merge` helper recursed fully into nested dicts, so overriding `attack.goal` with a `slow_degrade` dict merged it against the base `untargeted_degrade` goal instead of replacing it, leaving a stray `target_accuracy_drop: 0.2` key in the fixture's `slow_degrade` goal.
- **Fix:** Changed the helper to merge only one level below `fl`/`attack`, so a second-level key like `goal` is replaced wholesale by the override — consistent with D-04's "wholesale replacement, no per-rung merge" semantics that the plan itself specifies for `attack.target_ladder`.
- **Files modified:** `tests/test_target_ladder.py` (test-only; no production code affected)
- **Verification:** `python tests/test_target_ladder.py` — all 7 tests pass, including the corrected fixture behavior.
- **Committed in:** `c47341e` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug, test-fixture-only)
**Impact on plan:** No production code affected; the fix was necessary for the test fixture to actually exercise the scenario the plan describes. No scope creep.

## Issues Encountered

None beyond the fixture bug documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `rl/rewards.py::target_for_budget` and `DEFAULT_TARGET_LADDER` are ready for Plan 01-02 to wire `configs/base.yaml`'s `attack.target_ladder` declaration and delete the retired config keys (`attack.sample_target_in_training`, `attack.target_choices`), plus the tree-wide GOAL-04 deletion sweep.
- No blockers. All regression suites (`test_reward_reference`, `test_switch`, `test_clean_reference`, `test_freeze_mode`, `test_resume`, `test_fl_interlude`, `test_benchmark`) pass unmodified.
- GOAL-06 (the bottom-rung noise pre-flight) remains a gate for Plan 01-03, unaffected by this plan.

---
*Phase: 01-budget-conditioned-target-ladder*
*Completed: 2026-08-01*

## Self-Check: PASSED

- FOUND: tests/test_target_ladder.py
- FOUND: .planning/phases/01-budget-conditioned-target-ladder/01-01-SUMMARY.md
- FOUND: c7bfeab (Task 1 commit)
- FOUND: c47341e (Task 2 commit)
- FOUND: f43d020 (SUMMARY commit)
