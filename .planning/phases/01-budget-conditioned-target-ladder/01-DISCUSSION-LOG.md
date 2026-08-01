# Phase 1: Budget-Conditioned Target Ladder - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-01
**Phase:** 1-Budget-Conditioned Target Ladder
**Areas discussed:** Ladder shape & off-ladder budgets

---

## Area Selection

Four gray areas were offered. The user selected one.

| Area | Description | Selected |
|------|-------------|----------|
| Ladder shape & off-ladder budgets | Config form (positional list vs explicit map) and behavior for budgets outside the ladder | ✓ |
| Fate of the fixed 0.20 target | Whether the ladder becomes universal across eval/dry-run/DEFAULT_GOAL, and what it means for non-`untargeted_degrade` goals | |
| GOAL-06 pre-flight rig | Sample size per defense, pass margin, committed vs throwaway script, where the record lives | |
| Enforcement strength for GOAL-04/05 | Unit assertions vs repo-scanning test vs structural guard | |

The three unselected areas were carried into CONTEXT.md `<deferred>` as open in-phase questions for the researcher and planner, not as deferred scope.

---

## Ladder shape & off-ladder budgets

### Q1 — How should the ladder be declared in `configs/base.yaml`?

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit budget→drop map (Recommended) | `attack.target_ladder: {1: 0.02, ...}`. Each rung names its own budget — no off-by-one possible; reads exactly as REQUIREMENTS.md states it. Cost: PyYAML int keys need a coercion guard for quoted keys. | ✓ |
| Positional list | `[0.02, 0.04, 0.06, 0.08, 0.12]`, index = budget-1. Matches the config style being retired (`target_choices`, `defense.algorithms`) and is compact. Cost: implicit offset, so a mis-edit shifts every rung silently. | |
| List of `{budget, target}` pairs | Self-documenting, avoids YAML int-key coercion, room for per-rung metadata later. Cost: verbose for a five-entry table. | |

**User's choice:** Explicit budget→drop map
**Notes:** The deciding factor was silent-failure risk — a shifted positional list still produces plausible-looking targets that nothing in the reward path would flag.

---

### Q2 — What happens when a round's budget has no rung in the ladder?

Scenario put to the user: someone raises `fl.n_compromisable` to 8 and `attack.max_poison_clients` with it, but leaves the 5-rung ladder alone.

| Option | Description | Selected |
|--------|-------------|----------|
| Fail fast at startup (Recommended) | Validate in `FLArmsRaceEnv.__init__` that the ladder covers `[1, budget_cap]`; raise `RuntimeError` naming missing rungs. A missing rung is a config mistake, and any silent substitute makes Phase 3's graded ASR incomparable across budgets. Cost: minimal-config fixtures must supply a covering ladder or rely on the code default. | ✓ |
| Clamp to nearest rung + warn | Top rung applies above the ladder; warn once. Never dies mid-run, which matters at `simulation_rounds: 2000000` on a remote box. Cost: two budgets share one target, so per-(defense × budget) cells silently stop being independent, and a warning in `logs/system.log` is easy to miss. | |
| Extrapolate from the ladder | Extend the last slope so every budget gets a distinct target. Cost: invents targets never validated as reachable — the exact failure this milestone fixes — and the deliberately super-linear 12% top rung gives no clean slope to extend. | |

**User's choice:** Fail fast at startup
**Notes:** Consistent with the Q1 instinct — prefer a loud failure over graceful degradation, because degradation here corrupts the cell structure Phases 2 and 3 are built on. Catching it before a 2M-round run starts costs seconds; catching it mid-run costs the run.

---

### Q3 — What if `attack.target_ladder` is absent from a config entirely?

Context given: older configs, benchmark configs, and the minimal dicts the tests build.

| Option | Description | Selected |
|--------|-------------|----------|
| Code default, config overrides (Recommended) | `DEFAULT_TARGET_LADDER` in `rl/rewards.py`; config replaces it wholesale. Matches the existing convention (`attack.get("target_choices", [...])`, `fl.get("n_compromisable", ...)`, `DEFAULT_GOAL`) and keeps test fixtures and benchmark configs untouched. Cost: ladder numbers live in two files. | ✓ |
| Config-only — missing key raises | One authoritative copy of the numbers, same failure path as Q2. Cost: every fixture building a config dict must carry a ladder, expanding test edits beyond Success Criterion 4's "only the retired `target_choices` assertions". | |
| Merge per-rung over the default | A config could re-tune just the bottom rung (the likely GOAL-06 outcome) without restating all five. Cost: the effective ladder is never visible in one place. | |

**User's choice:** Code default, config overrides
**Notes:** Chosen against strict single-declaration purity, explicitly to protect Success Criterion 4. Per-rung merging was rejected on the same readability grounds that drove Q1. Accepted cost recorded in CONTEXT.md: if GOAL-06 raises the bottom rung, whoever edits `configs/base.yaml` should consider moving `DEFAULT_TARGET_LADDER` with it.

---

### Q4 — How does the configured ladder reach `target_for_budget()`?

Tension surfaced: GOAL-01 writes the signature as `target_for_budget(budget)`, but a one-argument function must read the ladder from somewhere — and `.claude/CLAUDE.md` §Module Design says "no module-level mutable state (singletons avoided)".

| Option | Description | Selected |
|--------|-------------|----------|
| Optional ladder arg (Recommended) | `target_for_budget(budget, ladder=None)` falling back to `DEFAULT_TARGET_LADDER`. Pure, like its sibling `goal_target(goal)`; env loads `self.target_ladder` once and passes it. Cost: call site reads with two args rather than GOAL-01's bare form. | ✓ |
| Module-level ladder set at startup | Matches GOAL-01 literally; all callers guaranteed identical. Cost: module-level mutable state CLAUDE.md forbids, order-dependent startup, and tests must set/reset a global or leak state — a real hazard with Phase 2's rollout-isolation `state_dict()` assertions coming next. | |
| Take the whole config dict | One argument shape for all callers, no duplicated extraction. Cost: couples a pure reward primitive to the config file's nested shape; `goal_target(goal)` deliberately takes only the goal dict. | |

**User's choice:** Optional ladder arg
**Notes:** CONTEXT.md records this as a deliberate, documented deviation from GOAL-01 as literally written — `target_for_budget(3)` still works as a one-arg call and returns the default ladder's value, so the requirement is satisfied in substance. The planner should not treat GOAL-01 as unmet.

---

## Claude's Discretion

Left to the researcher and planner (the user selected only one of four areas):

- Whether `target_for_budget` clamps or raises on `budget <= 0` — a defensive-contract question, since `_round_budget()` returns `rng.randint(1, budget_cap)` and never produces a non-positive budget in practice
- Whether `target_for_budget` keeps `goal_target`'s `max(target, 1e-6)` floor (recommended: yes, for consistency)
- Whether a rung raised by GOAL-06 carries a provenance note (measured noise floor, date) beside it in config
- Exact wording of the off-ladder `RuntimeError`

## Deferred Ideas

No scope creep occurred — every thread stayed inside the phase boundary.

Recorded in CONTEXT.md `<deferred>` as **open in-phase questions**, not deferred scope: the fate of the three `target_accuracy_drop: 0.20` declarations, the GOAL-06 pre-flight rig design, and the enforcement strength for GOAL-04/05.

Noted as genuinely out of milestone (already in `.planning/REQUIREMENTS.md` §v2): the offline calibration sweep of the reachable drop frontier (V2-02) and multi-seed confidence intervals (V2-03), either of which would give the ladder empirical grounding beyond GOAL-06's single-rung noise check.
