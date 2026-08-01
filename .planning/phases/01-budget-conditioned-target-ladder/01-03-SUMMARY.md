---
phase: 01-budget-conditioned-target-ladder
plan: 03
subsystem: rl
tags: [reward-shaping, federated-learning, target-ladder, statistics, defense-benchmarking]

# Dependency graph
requires:
  - phase: 01-01
    provides: "rl/rewards.py::target_for_budget(budget, ladder=None) + DEFAULT_TARGET_LADDER"
  - phase: 01-02
    provides: "attack.target_ladder declared in configs/base.yaml; tests/test_target_ladder.py (11 tests) incl. the tree-wide retirement scan"
provides:
  - "benchmark/noise_probe.py — committed, re-runnable clean-counterfactual noise measurement tool (summarize_noise, measure_defense, run_probe, main)"
  - "tests/test_noise_probe.py — 6 torch-free tests proving the statistics/verdict logic, including the raise-on-<2-samples guard and the inclusive at-the-margin comparison"
  - ".planning/phases/01-budget-conditioned-target-ladder/01-NOISE-BASELINE.md — the recorded per-defense GOAL-06 measurement and the unresolved rung-2 collision"
affects: [phase-2-damage-normalization, phase-3-graded-asr, phase-4-benchmark-counterfactual]

# Actuals (#2632)
actuals:
  tokens: 6752
  tasks: 2
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Torch-free pure-statistics core (summarize_noise) with all decision logic isolated from the heavy FL rig (measure_defense/run_probe), mirroring rl/rewards.py's pure-function convention"
    - "Deferred heavy imports inside main()'s function body (torch, data.mnist_loader, storage.checkpoint, rl.env, server.defense_ensemble, benchmark.phase1) so --help works without torch, matching benchmark/run_benchmark.py's established shape"
    - "math.isclose tolerance (rel_tol=1e-9) around an inclusive >= boundary comparison, to absorb floating-point noise from a sqrt-derived statistic without masking a genuine margin failure"

key-files:
  created:
    - benchmark/noise_probe.py
    - tests/test_noise_probe.py
    - .planning/phases/01-budget-conditioned-target-ladder/01-NOISE-BASELINE.md
  modified: []

key-decisions:
  - "GA-2 (01-CONTEXT.md, locked): 20 clean rounds per defense, 3-sigma margin, a committed tool (not throwaway), result recorded in .planning/ with the raw JSON pasted in — all implemented verbatim."
  - "Bug found and fixed during Task 2 (Rule 1): configs/base.yaml defaults fl.device to \"cuda\"; the probe's --device flag only reached measure_defense's env/ensemble construction, not the Phase-1 CPU fallback (run_phase1 reads config[\"fl\"][\"device\"] directly). Fixed by overriding config[\"fl\"][\"device\"] = args.device immediately after loading the config, before any FL construction."
  - "Rung-2 collision (P-02, plan line 235): the measurement demands a bottom rung of 0.05 (fltrust: sd=0.016184, 3sd=0.048552), which EXCEEDS rung 2 (0.04). Per the plan's explicit instruction for this exact collision, NO config or code was changed — configs/base.yaml, rl/rewards.py, and tests/test_target_ladder.py are all untouched. The collision is recorded in 01-NOISE-BASELINE.md and surfaced at the Task 3 checkpoint as a developer decision, not resolved by the executor."
  - "Two measurement caveats recorded rather than silently accepted: multikrum/dnc's sd=0.0 is a determinism artifact of benign_retrain_each_round: false (byte-identical samples), not evidence of a low noise floor — a vacuous pass. defl's samples are a 2-state oscillation from its accumulating internal state, not Gaussian noise, and its pass is marginal (0.019837 vs 0.02)."

patterns-established:
  - "GOAL-06-style pre-flight gates: measure first, record the raw JSON alongside a rounded-for-display table, and never let a code path silently choose a friendlier answer (fewer rounds/defenses/margin) to avoid a collision."

requirements-completed: []
# GOAL-06 remains gated: the measurement tool and the recorded artifact are both delivered and committed,
# but the requirement's "gating" behavior is not satisfied until Task 3's blocking checkpoint is resolved —
# the rung-2 collision found by the measurement has not yet been decided by the developer.

coverage:
  - id: D1
    description: "benchmark/noise_probe.py measures per-defense clean-counterfactual noise (no poisoning client, no LLM) with summarize_noise's statistics/verdict logic proven torch-free"
    requirement: "GOAL-06"
    verification:
      - kind: unit
        ref: "tests/test_noise_probe.py#test_zero_variance_samples_clear_any_positive_rung"
        status: pass
      - kind: unit
        ref: "tests/test_noise_probe.py#test_clears_uses_an_inclusive_comparison_at_exactly_the_margin"
        status: pass
      - kind: unit
        ref: "tests/test_noise_probe.py#test_fewer_than_two_samples_raises_instead_of_certifying"
        status: pass
      - kind: integration
        ref: "python -m benchmark.noise_probe --rounds 20 --device cpu --sigma-margin 3.0 --out logs/noise_probe.json (exit 0, logs/noise_probe.json has 4 defense entries, n=20 each)"
        status: pass
    human_judgment: false
  - id: D2
    description: "01-NOISE-BASELINE.md records the measurement, and the required bottom rung (0.05) collides with ladder rung 2 (0.04) — no config/code was changed, and the collision is a developer decision"
    requirement: "GOAL-06"
    verification: []
    human_judgment: true
    rationale: "The plan explicitly reserves the resolution of a rung collision for a developer decision (P-02, plan line 235) — no automated check can choose a re-spaced ladder, a dropped defense, or an accepted gap on the developer's behalf. This is exactly what Task 3's checkpoint:human-verify exists for."

duration: ~1h 40min (dominated by CPU Phase-1 training: 45 rounds x ~120s/round on a CPU-only box)
completed: 2026-08-01
status: blocked
---

# Phase 1 Plan 3: Budget-Conditioned Target Ladder (GOAL-06 Noise Pre-Flight) Summary

**A committed clean-counterfactual noise probe measured all four defenses in single mode and found fltrust's noise floor demands a bottom rung of 0.05 — which collides with ladder rung 2 (0.04) and is recorded, unresolved, as a developer decision at a blocking checkpoint.**

## Performance

- **Duration:** ~1h 40min (Phase-1 CPU fallback training dominated: 45 rounds at ~120s/round with no saved checkpoint on this box, then ~9 minutes for the 4×20 defense-measurement rounds, which replay frozen weights and are much faster)
- **Started:** 2026-08-01 (worktree session start)
- **Completed (Tasks 1-2):** 2026-08-01
- **Tasks:** 2 of 3 (Task 3 is the blocking `checkpoint:human-verify`, not yet resolved)
- **Files modified:** 3 created, 0 modified

## Accomplishments

- Built `benchmark/noise_probe.py`: `summarize_noise` (pure, torch-free — raises `ValueError` naming the count on <2 samples instead of certifying an aborted run; inclusive `>=` margin comparison tolerant of floating-point noise at the exact boundary via `math.isclose`; unrounded `sd`), `measure_defense` (pins one algorithm in `single`/`fixed` mode per defense, deep-copies config and start state so all four defenses start identically), `run_probe` (orders defenses as requested, aggregates an `all_clear` / `required_rung` verdict), and a `--help`-able CLI with deferred heavy imports.
- Wrote `tests/test_noise_probe.py` — 6 torch-free tests proving the statistics/verdict logic in isolation, importing only `summarize_noise`.
- Ran the probe for real (`python -m benchmark.noise_probe --rounds 20 --device cpu --sigma-margin 3.0 --out logs/noise_probe.json`) — no saved Phase-1 checkpoint existed, so the CPU `run_phase1` fallback trained 45 honest FedAvg rounds first (baseline accuracy 0.7822), then measured 20 clean rounds per defense.
- Recorded the full result in `.planning/phases/01-budget-conditioned-target-ladder/01-NOISE-BASELINE.md`: fltrust FAILS (sd=0.016184, 3sd=0.048552 vs the 0.02 rung, required_rung=0.05); multikrum and dnc pass trivially (sd=0.0 — a determinism artifact, not a real noise-floor result); defl passes marginally (3sd=0.019837 vs 0.02) but its samples are a 2-state oscillation, not Gaussian noise.
- Identified and recorded the collision the plan explicitly anticipated: the required rung (0.05) exceeds ladder rung 2 (0.04). Per prohibition P-02, made **no** change to `configs/base.yaml`, `rl/rewards.py`, or `tests/test_target_ladder.py` — the collision is surfaced at the Task 3 checkpoint instead of resolved unilaterally.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build the committed clean-counterfactual noise probe and its statistics tests** - `c85b3bd` (test)
   - Follow-up bug fix (Rule 1, found while executing Task 2): `ebe96f8` (fix) — `noise_probe.py` must override `config["fl"]["device"]` before the Phase-1 CPU fallback, or `run_phase1` inherits `configs/base.yaml`'s `fl.device: "cuda"` default and crashes with `Torch not compiled with CUDA enabled` on this CPU-only box.
2. **Task 2: Run the probe, record the verdict — collision found, no rung raised** - `41f8cea` (docs)

**Task 3 (checkpoint:human-verify, gate="blocking") is NOT started** — it requires a developer decision on the rung-2 collision and cannot be auto-approved (see Deviations / Next Phase Readiness below).

_Note: Task 1 is `test`-typed per the plan's `tdd="true"` framing of Task 1, but in practice both the tool and its tests were authored together and verified against the plan's literal acceptance-criteria one-liners before committing — there was no separate RED-phase failing-test commit, since `summarize_noise` did not exist as a stub to fail against first. This is a minor deviation from strict TDD sequencing; the resulting test coverage and behavior are unaffected._

## Files Created/Modified

- `benchmark/noise_probe.py` - Clean-counterfactual noise probe: `summarize_noise`, `measure_defense`, `run_probe`, `main` (CLI)
- `tests/test_noise_probe.py` - 6 torch-free tests of `summarize_noise`'s statistics/verdict logic
- `.planning/phases/01-budget-conditioned-target-ladder/01-NOISE-BASELINE.md` - Recorded GOAL-06 measurement, per-defense verdicts, two measurement caveats, and the unresolved rung-2 collision

## Decisions Made

- Followed GA-2 (01-CONTEXT.md, locked) verbatim: 20 rounds per defense, 3-sigma margin, committed tool, result recorded in `.planning/` with raw JSON pasted in.
- `required_rung` is computed via `round(threshold / 0.01, 9)` before `math.ceil`, and `clears` uses `math.isclose(rel_tol=1e-9, abs_tol=1e-12)` alongside the plain `<=` check — both guard against a mathematically-exact boundary (e.g. `sd == rung / sigma_margin`) landing a few floating-point ULPs on the wrong side of the comparison, verified against the plan's own literal acceptance-criteria snippet (`d = 0.02/3.0; s([0.8-d, 0.8+d], 0.02, 3.0)`), which fails without this tolerance.
- The device-override bug (see Deviations) was fixed inline per Rule 1 rather than deferred, since it blocked Task 2 from running at all on this CPU-only box.
- No config or code changed in response to the collision (P-02) — see Overall Verdict in `01-NOISE-BASELINE.md` for the four representative resolution options left for the developer.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `noise_probe.py` did not override `fl.device` before the Phase-1 CPU fallback**
- **Found during:** Task 2 (first real run of the probe)
- **Issue:** `configs/base.yaml` defaults `fl.device` to `"cuda"` (RL training needs a GPU). The probe's `--device` CLI flag only reached `measure_defense`'s env/ensemble construction as an explicit parameter; `run_phase1(config, ...)` reads `config["fl"]["device"]` directly from the loaded YAML dict, which still said `"cuda"`. On this CPU-only box (`torch.cuda.is_available()` is `False`), the first run crashed with `AssertionError: Torch not compiled with CUDA enabled` about 4 seconds in, before any measurement occurred.
- **Fix:** In `main()`, immediately after `yaml.safe_load`, added `config.setdefault("fl", {})["device"] = args.device` so the CPU override reaches every code path that reads `config["fl"]["device"]`, not only the paths that take an explicit `device=` parameter.
- **Files modified:** `benchmark/noise_probe.py`
- **Verification:** `python tests/test_noise_probe.py` and `python -m benchmark.noise_probe --help` both still pass after the fix; the corrected full run completed with exit 0 and produced `logs/noise_probe.json` with 4 defense entries at `n=20` each.
- **Committed in:** `ebe96f8`

---

**Total deviations:** 1 auto-fixed (1 bug). No scope creep — the fix was strictly necessary for the probe to run at all on this machine, and it is the exact CPU-only environment the plan's own precondition anticipated.

### Not a deviation, but worth flagging explicitly: the background-run relaunch

The first attempt to run the probe (before the device fix) was launched via this executor's own background-Bash mechanism and was killed 4 seconds in when the executor's turn ended — a subagent's background processes do not survive the subagent returning. After the device fix, the orchestrator relaunched the identical command from its own (persistent) session and it ran to completion (exit 0, ~1h35m wall clock). All numbers in `01-NOISE-BASELINE.md` were read from that run's `logs/noise_probe.json` and `logs/noise_probe_run.log` directly — none were transcribed from the orchestrator's summary message, per prohibition P-01 (never report a measurement from memory or a paraphrase).

## Issues Encountered

- **The rung-2 collision** (see Overall Verdict in `01-NOISE-BASELINE.md`): the measurement demands a bottom rung of `0.05`, which exceeds rung 2 (`0.04`). This is not a bug in the probe or the ladder — it is exactly the scenario `01-03-PLAN.md`'s Task 2 action block calls out by name ("If raising rung 1 would make it meet or exceed rung 2 ... Halt, record the measurement and the collision in the artifact, and surface it at the checkpoint"). No further automated resolution is possible; the developer must choose among the options recorded in `01-NOISE-BASELINE.md`'s Overall Verdict section (re-space the ladder, drop fltrust from the panel, investigate fltrust's instability upstream, or accept a documented gap for budget 1).

## User Setup Required

None - no external service configuration required. `data/mnist_raw/MNIST` was copied into this worktree from the main repo checkout (gitignored, not committed) so the probe could run without a network download; this is a worktree-local convenience, not a project setup step.

## Next Phase Readiness

- **Phase 2 (damage normalization) is BLOCKED on this plan's Task 3 checkpoint**, per `.planning/STATE.md`'s existing blocker: "GOAL-06 is a gate, not a task ... if the 0.02 bottom rung does not clear per-round clean-counterfactual noise for every defense, the rung must be raised in config before Phase 2 begins." The measurement is done and says the rung must move, but *how* it moves (re-space vs. drop fltrust vs. accept a gap) is undecided.
- All 21 test files in `tests/` remain green (`tests/test_target_ladder.py`'s tree-wide retirement scan included) — nothing in this plan touched the ladder's config or code, so Plan 01/02's guarantees are untouched.
- `benchmark/noise_probe.py` is committed and re-runnable exactly as `01-NOISE-BASELINE.md`'s "How to Re-Run" section states — once the developer decides how to resolve the collision, re-running the probe against the new ladder is the natural verification step before sealing the gate.

---
*Phase: 01-budget-conditioned-target-ladder*
*Completed: 2026-08-01 (Tasks 1-2; Task 3 checkpoint pending)*

## Self-Check: PASSED

- FOUND: benchmark/noise_probe.py
- FOUND: tests/test_noise_probe.py
- FOUND: .planning/phases/01-budget-conditioned-target-ladder/01-NOISE-BASELINE.md
- FOUND: .planning/phases/01-budget-conditioned-target-ladder/01-03-SUMMARY.md
- FOUND: c85b3bd (Task 1 commit)
- FOUND: ebe96f8 (Rule 1 bug-fix commit)
- FOUND: 41f8cea (Task 2 artifact commit)
- FOUND: 31df76b (SUMMARY commit)
