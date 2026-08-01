---
phase: 01-budget-conditioned-target-ladder
plan: 02
subsystem: rl
tags: [config, reward-shaping, federated-learning, target-ladder, test-automation]

# Dependency graph
requires:
  - phase: 01-01
    provides: "rl/rewards.py::target_for_budget(budget, ladder=None) + DEFAULT_TARGET_LADDER; FLArmsRaceEnv.target_ladder load/validate; tests/test_target_ladder.py (7 tests)"
provides:
  - "attack.target_ladder declared in configs/base.yaml as an explicit budget-to-drop map (1->0.02 .. 5->0.12), re-tunable with no code edit"
  - "attack.sample_target_in_training and attack.target_choices deleted from configs/base.yaml, with their comment block"
  - "benchmark/run_benchmark.py no longer writes the phantom env.sample_target attribute"
  - "README.md documents the budget-conditioned ladder in place of the retired per-round target sampling"
  - "tests/test_target_ladder.py: 4 new tests (11 total) — config-override proof, quoted-string-key proof, shipped-config completeness proof, and a tree-wide automated scan that fails permanently on any reintroduction of the retired keys"
affects: [01-03, phase-2-damage-normalization, phase-3-graded-asr]

# Actuals (#2632)
actuals:
  tokens: 3252
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Tree-wide os.walk repo scan as a permanent regression guard for retired config keys, instead of a one-time grep"
    - "Config-driven declaration with a code-level fallback default (D-04): the ladder now lives in configs/base.yaml, DEFAULT_TARGET_LADDER stays as the fallback for config-absent construction"

key-files:
  created: []
  modified:
    - configs/base.yaml
    - benchmark/run_benchmark.py
    - README.md
    - tests/test_target_ladder.py

key-decisions:
  - "D-01/D-04 implemented verbatim: attack.target_ladder declared as an explicit budget->drop YAML map (not a positional list), replacing the code default wholesale when present"
  - "GOAL-04 enforcement is a permanent automated test (test_retired_target_sampling_keys_have_no_reader_left), not a one-time grep — verified fail-first by planting and removing a retired token in README.md"
  - "test_shipped_config_declares_a_complete_ladder asserts structure (coverage + strict monotonicity), not the five literal numbers, so a future GOAL-06 rung raise does not force an unrelated test edit"

patterns-established:
  - "Retirement-by-scan: any future config-key retirement in this repo can reuse the os.walk skip-list / extension-list pattern in tests/test_target_ladder.py rather than a manual grep sweep"

requirements-completed: [GOAL-02, GOAL-04]

coverage:
  - id: D1
    description: "attack.target_ladder declared in configs/base.yaml as an explicit budget-to-drop map covering [1, max_poison_clients]; retiring sample_target_in_training and target_choices with no code edit needed to re-tune a rung"
    requirement: "GOAL-02"
    verification:
      - kind: unit
        ref: "tests/test_target_ladder.py#test_config_ladder_overrides_the_default_with_no_code_edit"
        status: pass
      - kind: unit
        ref: "tests/test_target_ladder.py#test_quoted_string_ladder_keys_are_accepted"
        status: pass
      - kind: unit
        ref: "tests/test_target_ladder.py#test_shipped_config_declares_a_complete_ladder"
        status: pass
    human_judgment: false
  - id: D2
    description: "Retired keys (attack.sample_target_in_training, attack.target_choices) and the phantom benchmark attribute write deleted; no remaining reader anywhere outside .planning/, enforced by a permanent tree-wide scan"
    requirement: "GOAL-04"
    verification:
      - kind: unit
        ref: "tests/test_target_ladder.py#test_retired_target_sampling_keys_have_no_reader_left"
        status: pass
    human_judgment: false

duration: 17min
completed: 2026-08-01
status: complete
---

# Phase 1 Plan 2: Budget-Conditioned Target Ladder (Config + Retirement) Summary

**`attack.target_ladder` moved from a code default into `configs/base.yaml` as an explicit budget-to-drop map, and the retired per-round target-sampling keys are gone everywhere with a permanent tree-wide scan (not a one-time grep) proving they cannot silently reappear.**

## Performance

- **Duration:** 17 min
- **Started:** 2026-08-01T12:40:00Z
- **Completed:** 2026-08-01T12:57:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Declared `attack.target_ladder` in `configs/base.yaml` with the locked five rungs (`1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.12`), replacing the deleted `sample_target_in_training` / `target_choices` keys and their comment block; left `attack.goal`, `max_poison_clients`, `sample_budget_in_training`, and `eval_poison_clients` untouched, rewording only `goal.target_accuracy_drop`'s inline comment to describe its role as a fallback for paths that don't build a per-round goal.
- Removed the phantom `env.sample_target = False` attribute write from `benchmark/run_benchmark.py` (Plan 01 deleted the attribute it targeted), replacing the two-line comment with a one-liner about the eval-budget contract; left `env.sample_budget = False` / `env.budget_cap = eval_budget` untouched.
- Rewrote `README.md`'s "Target generalization" paragraph into a "Budget-conditioned target ladder" paragraph describing the ladder, `target_for_budget` as the single resolving function, the startup coverage check, and the FedAvg-dilution rationale for why the target scales with budget; added `target_ladder` to the Configuration bullet's enumerated `attack` keys.
- Added four tests to `tests/test_target_ladder.py` (11 total, up from 7): a config-override proof that the default ladder does not survive as a merge, a quoted-string-key acceptance proof, a shipped-config completeness/monotonicity proof (deliberately structural, not pinning literal values), and a tree-wide `os.walk` scan that fails permanently — naming file and line — on any reintroduction of either retired token outside `.planning/` and the scanning test itself.
- Ran the fail-first proof manually: planted `target_choices` in `README.md`, confirmed the scan failed and named the exact line, then removed it (`git checkout -- README.md`) and confirmed the suite went green again.

## Task Commits

Each task was committed atomically:

1. **Task 1: Declare the ladder in config and delete the retired sampling machinery** - `863b5c3` (feat)
2. **Task 2: Prove config drives the ladder, and make the retirement self-enforcing** - `3e325b6` (test)

## Files Created/Modified

- `configs/base.yaml` - Added `attack.target_ladder` map with block comment; deleted `sample_target_in_training` / `target_choices` and their comment header; reworded `goal.target_accuracy_drop`'s inline comment
- `benchmark/run_benchmark.py` - Deleted the phantom `env.sample_target = False` write and its comment; replaced with a one-line eval-budget note
- `README.md` - Rewrote the target-generalization paragraph around the ladder; added `target_ladder` to the Configuration key list
- `tests/test_target_ladder.py` - Added `import yaml` and four new test functions (config override, quoted keys, shipped-config completeness, tree-wide retirement scan)

## Decisions Made

- Followed D-01/D-04 verbatim: the ladder is an explicit budget-keyed YAML map (not a positional list), and a present `attack.target_ladder` replaces `DEFAULT_TARGET_LADDER` wholesale — no per-rung merge.
- GOAL-04's "no remaining reader" requirement is implemented as a permanent, self-enforcing test rather than a one-time grep, per the plan's explicit instruction — the scan walks the whole repo tree (excluding `.planning/`, `.git`, caches, and generated/data directories) and fails naming file:line on any hit.
- `test_shipped_config_declares_a_complete_ladder` intentionally asserts structure (coverage of `[1, max_poison_clients]`, strict monotonicity) rather than the five literal numbers, so GOAL-06's possible bottom-rung raise in Plan 03 does not force an edit to this test — the exact values stay pinned once, in Plan 01's `target_for_budget` behavior check.

## Deviations from Plan

None - plan executed exactly as written. Both tasks' `<verify>` and full acceptance-criteria checklists passed on the first attempt with no auto-fixes required.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `attack.target_ladder` is live in `configs/base.yaml`; GOAL-02 and GOAL-04 are satisfied.
- The retirement scan (`test_retired_target_sampling_keys_have_no_reader_left`) is a standing regression guard for the rest of this milestone — any future reintroduction of either retired token fails the suite immediately.
- All 20 test files in `tests/` pass (`tests/test_target_ladder.py` now at 11 tests; full suite unmodified elsewhere).
- Plan 01-03 (GOAL-06's bottom-rung noise pre-flight) can proceed: nothing in this plan touches `clean_reference_accuracy` or the noise-measurement surface, and `DEFAULT_TARGET_LADDER` is available if the code-level default also needs to move should the bottom rung be raised.

---
*Phase: 01-budget-conditioned-target-ladder*
*Completed: 2026-08-01*

## Self-Check: PASSED

- FOUND: configs/base.yaml (target_ladder present, retired keys absent)
- FOUND: benchmark/run_benchmark.py (phantom write removed)
- FOUND: README.md (target_ladder documented)
- FOUND: tests/test_target_ladder.py (11 tests)
- FOUND: 863b5c3 (Task 1 commit)
- FOUND: 3e325b6 (Task 2 commit)
