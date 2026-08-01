# Phase 1: Budget-Conditioned Target Ladder - Pattern Map

**Mapped:** 2026-08-01
**Files analyzed:** 7 (2 new-function-in-existing-file, 3 modify, 2 test-only)
**Analogs found:** 7 / 7 (all patterns exist in-tree; no external research needed)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `rl/rewards.py` (add `target_for_budget` + `DEFAULT_TARGET_LADDER`) | utility (pure reward primitive) | transform | `rl/rewards.py::goal_target()` (same file, same role) | exact — sibling function in same module |
| `rl/env.py::FLArmsRaceEnv.__init__` | config/constructor | request-response (startup validation) | same method's existing `budget_cap` clamp block (`rl/env.py:74-89`) | exact — same constructor, adjacent lines |
| `rl/env.py::_round_goal()` | transform | request-response | itself, pre-change (`rl/env.py:178-186`) | exact — one behavioral edit |
| `configs/base.yaml` (`attack.target_ladder`) | config | — | `attack.target_choices` / `defense.algorithms` (existing map/list-shaped keys in same file) | exact — same file, same section |
| `tests/test_reward_reference.py` (GOAL-05 test addition) | test | request-response | existing tests in same file (`test_attacker_relative_win_gate`-style fixtures) | exact |
| `tests/test_switch.py` (GOAL-05 test addition) | test | request-response | `test_attacker_relative_win_gate` (`tests/test_switch.py:71-93`) | exact — already tests goal→target agreement |
| `README.md:320-332` | doc | — | none (deletion/rewrite of prose, not a code pattern) | n/a |

No files in this phase require a *cross-role* analog search outside `rl/rewards.py` and `rl/env.py` — the phase is additive-in-place, not a new module.

## Pattern Assignments

### `rl/rewards.py::target_for_budget()` (utility, pure reward primitive)

**Analog:** `rl/rewards.py::goal_target()` (same file, lines 85-100) — copy this shape exactly.

**Full existing function to mirror:**
```python
def goal_target(goal: dict) -> float:
    """The target accuracy drop this goal asks for (>0).

    Single source of truth shared by the attacker reward (which normalizes the
    drop by it) and the schedule's relative win-gate (``rl/switch.py``) so the two
    never disagree about what the round's target is. ``slow_degrade`` uses
    ``per_round_drop``; ``untargeted_degrade`` (and, for now, ``targeted_label``,
    which falls back to overall accuracy until per-class eval is wired in) uses
    ``target_accuracy_drop``.
    """
    gtype = goal.get("type", "untargeted_degrade")
    if gtype == "slow_degrade":
        target = float(goal.get("per_round_drop", 0.02))
    else:
        target = float(goal.get("target_accuracy_drop", 0.20))
    return max(target, 1e-6)
```

**What `target_for_budget` must copy from this:**
- Signature shape: takes only the domain value it needs (`goal: dict` → `budget: int, ladder: dict | None = None`), never the whole config — matches D-05's rejection of "take the whole config dict".
- The `max(target, 1e-6)` floor at the return (D-05 discretion: recommend yes for consistency, since `goal_target` will read the ladder-populated `target_accuracy_drop` downstream anyway).
- Docstring convention: one-line summary, then a paragraph naming every consumer by name ("Single source of truth shared by ... so the two never disagree") — `target_for_budget`'s docstring should say the same about `rl/env.py::_round_goal()`, `attacker_reward` (via `goal_target`), and `rl/switch.py`'s win gate.
- No module-level mutable state: `goal_target` reads only its argument; `target_for_budget` must do the same — the ladder is passed in, not read from a module global (this is exactly what D-05 rejected: installing the ladder as module state at startup).

**Where `DEFAULT_TARGET_LADDER` goes** — module-level constant declared near the top of the reward-shaping constants, mirroring:
```python
# Upper bound of the drop term. Reaching the goal exactly scores 1.0; overshoot
# is worth at most another 0.5 (see :func:`drop_term`).
_OVERSHOOT_BONUS = 0.5
_OVERSHOOT_HALF = 1.0
```
`DEFAULT_TARGET_LADDER = {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.12}` should sit as a public (no leading underscore, since D-04 treats it as a documented default consumers may reference — cf. `DEFAULT_GOAL` in `agents/attacker_agent.py:35`, which is also public/un-prefixed) module constant, placed either right before `goal_target()` (its sibling-in-spirit) or right before the new `target_for_budget()` definition — keep it directly above the function that consumes it, matching how `_OVERSHOOT_BONUS`/`_OVERSHOOT_HALF` sit directly above `drop_term()`.

**Suggested shape** (not to be copied verbatim without planner review, but matches the sibling exactly):
```python
DEFAULT_TARGET_LADDER = {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.12}


def target_for_budget(budget: int, ladder: dict | None = None) -> float:
    """The target accuracy drop for a given per-round poison budget.

    Single source of truth shared by ``FLArmsRaceEnv._round_goal()`` (which
    builds the round's goal dict from it), the attacker reward (via
    ``goal_target``, which reads whatever ``_round_goal()`` produced), and the
    schedule's relative win-gate (``rl/switch.py``) — so all three read the
    identical target for a given budget and can never disagree.

    ``ladder`` maps ``budget -> target_accuracy_drop``; falls back to
    ``DEFAULT_TARGET_LADDER`` when ``None`` (config-absent case, D-04: a
    present ``attack.target_ladder`` replaces the default wholesale, no
    per-rung merge).
    """
    table = ladder if ladder is not None else DEFAULT_TARGET_LADDER
    target = float(table[int(budget)])
    return max(target, 1e-6)
```
(KeyError on an off-ladder budget is intentionally NOT caught here — D-03 puts that failure at `FLArmsRaceEnv.__init__` startup time, not inside this pure function; planner should decide whether this function re-raises with a clearer message or lets `KeyError` surface, since D-03's validation should make this path unreachable in practice.)

---

### `rl/env.py::FLArmsRaceEnv.__init__` (constructor, startup validation)

**Analog for the config-read-with-inline-default idiom** — exact existing lines to copy the *style* from:
```python
# rl/env.py:74-89 (current, pre-change)
self.n_compromisable = max(1, min(
    int(fl.get("n_compromisable", self.n_clients)), self.n_clients))
self.budget_cap = max(1, min(
    int(attack.get("max_poison_clients", self.n_compromisable)), self.n_compromisable))
self.sample_budget = bool(attack.get("sample_budget_in_training", True))
self.sample_target = bool(attack.get("sample_target_in_training", False))
self.target_choices = [float(x) for x in attack.get(
    "target_choices", [0.05, 0.10, 0.20, 0.30])]
```
New read should follow the same `attack.get(key, default)` idiom (D-04's "established convention"):
```python
self.target_ladder = attack.get("target_ladder", DEFAULT_TARGET_LADDER)
```
Note: per D-04, `attack.get` alone is NOT quite the pattern here — D-04 says presence replaces wholesale, which `.get(key, default)` already does correctly (no merge happens with `dict.get`), so this one-liner is sufficient; no extra merge logic needed. The PyYAML string/int key coercion noted in D-01 ("loader must coerce/accept quoted string keys too") needs a small normalization step this analog does NOT have a precedent for — see below.

**Analog for startup-time contract validation that raises** — this repo has no config-validation module (confirmed: `.claude/CLAUDE.md` §Error Handling matches `server/defense_ensemble.py.__init__`, the only constructor-time validate-and-raise precedent found):
```python
# server/defense_ensemble.py:106-114
def __init__(self, defenses: dict, *, mode: str = "single",
             selection: str = "rotate", rng=None):
    if not defenses:
        raise ValueError("DefenseEnsemble needs at least one defense")
    if mode not in self.MODES:
        raise ValueError(f"defense.mode must be one of {self.MODES}, got {mode!r}")
    if selection not in self.SELECTIONS:
        raise ValueError(
            f"defense.selection must be one of {self.SELECTIONS}, got {selection!r}")
```
**Important discrepancy for the planner:** every existing constructor-time contract check in this codebase (`server/defense_ensemble.py:109,111,113`; `rl/switch.py:132,154`; `benchmark/defenses/__init__.py:50,55,67`) raises **`ValueError`**, not `RuntimeError`. The only `RuntimeError` precedent in the whole tree is a *test mock* (`tests/test_defense_ensemble.py:89: raise RuntimeError("svd did not converge")`, simulating an SVD failure — not a contract-violation example). CLAUDE.md's Error Handling section (`raise RuntimeError("svd did not converge")`) appears to have been generated by an automated CLAUDE.md tool that mistook that test mock for a real convention. D-03 explicitly says "raises a `RuntimeError`" — the planner should follow D-03 literally (it is a locked decision, not a discretion item) but should NOT expect this to match the rest of the codebase's `ValueError` convention; consider flagging this to the user if the planner wants a course correction, but D-03 is explicit and should be honored as written.

**Message-wording style to copy** (from the same `ValueError` precedents, adapted to `RuntimeError`):
```python
missing = [b for b in range(1, self.budget_cap + 1) if b not in self.target_ladder]
if missing:
    raise RuntimeError(
        f"attack.target_ladder is missing rungs for budgets {missing} "
        f"(covers {sorted(self.target_ladder)}, needs [1, {self.budget_cap}])")
```
Place this validation directly after `self.budget_cap` is computed (`rl/env.py:80-81`) and after `self.target_ladder` is loaded — mirrors D-03's "belongs beside the existing `budget_cap` clamping."

**Key-coercion note (no existing precedent in this codebase):** D-01 requires the loader to "coerce/accept quoted string keys too." No existing config reader in this tree does string/int key coercion on a YAML mapping — this will be new code, not a copied pattern. Suggested minimal shape, placed right where `self.target_ladder` is assigned:
```python
raw_ladder = attack.get("target_ladder", DEFAULT_TARGET_LADDER)
self.target_ladder = {int(k): float(v) for k, v in raw_ladder.items()}
```

**Deletions** (GOAL-04) — remove these two lines and their preceding comment block (`rl/env.py:83-89`):
```python
self.sample_target = bool(attack.get("sample_target_in_training", False))
self.target_choices = [float(x) for x in attack.get(
    "target_choices", [0.05, 0.10, 0.20, 0.30])]
```

---

### `rl/env.py::_round_goal()` (the one behavioral change)

**Current (pre-change):**
```python
def _round_goal(self) -> dict:
    """This round's attack goal. When target sampling is on (untargeted_degrade),
    draw ``target_accuracy_drop`` from ``target_choices`` so the policy becomes
    TARGET-AWARE and generalizes across targets; otherwise the fixed config goal.
    The returned dict is the CLEAN goal shown to the LLM and used by the reward."""
    if self.sample_target and self.goal.get("type") == "untargeted_degrade":
        return {"type": "untargeted_degrade",
                "target_accuracy_drop": self.rng.choice(self.target_choices)}
    return self.goal
```

**Target shape** (per CONTEXT.md D-05, already given verbatim by the user):
```python
def _round_goal(self) -> dict:
    return {"type": "untargeted_degrade",
            "target_accuracy_drop": target_for_budget(
                self.round_budget, self.target_ladder)}
```
Note `self.round_budget` must already be set before `_round_goal()` is called — confirmed in `begin_round()` (`rl/env.py:203-204`): `self.round_budget = self._round_budget()` runs immediately before `self.round_goal = self._round_goal()`, so the ordering already holds; no call-site reordering needed.

Import to add at the top of `rl/env.py` (alongside existing local imports):
```python
from rl.rewards import target_for_budget, DEFAULT_TARGET_LADDER
```
(matches the existing import-grouping style: stdlib, then local — see `rl/env.py:22-29`.)

---

### `configs/base.yaml` (`attack.target_ladder`)

**Analog — the block being replaced, exact current text (`configs/base.yaml:26-45`):**
```yaml
attack:
  goal:
    type: "untargeted_degrade"    # untargeted_degrade | slow_degrade | targeted_label
    target_accuracy_drop: 0.20    # untargeted_degrade: fallback target used at EVAL / when sampling is off
    # per_round_drop: 0.02        # slow_degrade
    # label: 7                    # targeted_label (per-class eval is a TODO; falls back to overall)
  # --- Attack-goal target sampling (generalize instead of overfitting a single target) ---
  sample_target_in_training: true # true = randomize untargeted_degrade's target_accuracy_drop each
                                  # round from `target_choices`, so the policy is TARGET-AWARE and
                                  # generalizes to any requested drop (mirrors sample_budget_in_training).
                                  # false = always use goal.target_accuracy_drop. Eval always fixes it.
  target_choices: [0.05, 0.10, 0.20, 0.30]  # per-round target pool (untargeted_degrade only)
  # --- Client-selection budget (the attacker chooses WHICH & HOW MANY to poison) ---
  max_poison_clients: 5           # TRAINING budget cap: attacker may poison up to this many of its
                                  # pool each round (must be <= n_compromisable). Reward penalizes
                                  # using more than needed (see rl.reward.attacker.delta).
  sample_budget_in_training: true # true = randomize the per-round cap in [1..max_poison_clients] so
                                  # the policy is BUDGET-AWARE (generalizes to any eval budget).
  eval_poison_clients: 1          # EVALUATION default budget (benchmark --max-poison-clients override)
```
**Style precedent for a map-shaped config key** — `defense.algorithms: ["fltrust", "multikrum", "dnc", "defl"]` (a list) and the deleted `target_choices` (also a list) are the closest existing collection-typed keys in this file; there is no existing YAML mapping key for a per-comment-annotated nested config in `attack:` to copy verbatim, but the comment style (`# key = meaning, inline units/behavior note`) is consistent throughout the file (e.g. `configs/base.yaml:9-11` for `poison_fraction`). D-01's exact YAML is already specified by the user:
```yaml
attack:
  target_ladder:
    1: 0.02
    2: 0.04
    3: 0.06
    4: 0.08
    5: 0.12   # super-linear: spending the whole pool demands more
```
Delete `sample_target_in_training` and `target_choices` (lines 33-38 in current numbering); keep `goal.target_accuracy_drop: 0.20` per Known Hazards (its fate is explicitly out of scope for this phase — do not touch `attack.goal` itself).

---

### `tests/test_switch.py` — GOAL-05 "cannot disagree" test

**Analog:** `test_attacker_relative_win_gate` (`tests/test_switch.py:71-93`) — already constructs goal dicts and asserts the win gate reads `target_accuracy_drop`/`per_round_drop` via `goal_target`. Fixture style to copy:
```python
def _cfg(**kw):
    base = dict(min_phase_rounds=3, max_phase_rounds=10, success_streak=2,
                attacker_min_drop=0.02, attacker_min_evaded=1.0,
                defender_min_tpr=0.99, defender_max_fpr=0.10)
    base.update(kw)
    return SwitchConfig(**base)
```
A new GOAL-05 test should build a goal via `target_for_budget(budget)` and assert `goal_target({"type": "untargeted_degrade", "target_accuracy_drop": target_for_budget(budget)})` equals `target_for_budget(budget)` again (round-trip identity), and/or that `attacker_succeeded`'s `min_drop` bar matches `cfg.win_fraction * target_for_budget(budget)` for each rung 1-5 — mirroring the existing `small`/`big` two-case pattern at lines 74-83, but driven by ladder rungs instead of hand-picked targets. Import addition: `from rl.rewards import target_for_budget, goal_target` alongside the existing `from rl.switch import (...)` line 12.

**D-04 preservation constraint:** none of `_cfg()`'s minimal dict construction changes — `SwitchConfig` never touches the ladder directly (only `goal_target`/`target_for_budget` do), so this fixture is untouched except for the new test function appended.

---

### `tests/test_reward_reference.py` — GOAL-05 test (alternate home)

**Analog:** the module-level `GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}` constant (line 30) and its use throughout — e.g.:
```python
GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}

def test_identical_attack_scores_identically_every_round():
    clean, post = 0.90, 0.60
    r = [attacker_reward(clean, post, GOAL, [0], _evaded([0]), 0, pool_size=5)
         for _ in range(3)]
```
A GOAL-05 test here should construct `goal = {"type": "untargeted_degrade", "target_accuracy_drop": target_for_budget(budget)}` for each ladder rung and assert `goal_target(goal) == target_for_budget(budget)` — proving the reward path (`goal_target`, consumed inside `attacker_reward`) agrees with the ladder function directly. Import addition: `from rl.rewards import target_for_budget` alongside the existing `from rl.rewards import (attacker_reward, defender_reward, drop_term, group_advantages)` at lines 26-28 (add `goal_target` too, since it is not currently imported here despite being used implicitly via `attacker_reward`).

**Test runner convention** (both test files) — no pytest; both use the same manual `_run()` harness:
```python
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} ... tests passed.")

if __name__ == "__main__":
    _run()
```
New test functions just need the `test_` prefix; no registration step.

---

## Shared Patterns

### Config-read-with-inline-default
**Source:** `rl/env.py:71-89` (multiple examples in one block)
**Apply to:** `rl/env.py::__init__`'s new `self.target_ladder = attack.get("target_ladder", DEFAULT_TARGET_LADDER)` line — same idiom, no new abstraction.

### Startup-time RuntimeError (D-03's mandated exception type — note the codebase-wide convention is actually ValueError; follow D-03 as written since it is a locked decision)
**Source:** message-wording style from `server/defense_ensemble.py:109-114`; exception TYPE mandated by CONTEXT.md D-03 as `RuntimeError` (deviates from the rest of the codebase's `ValueError` habit — flag but do not silently "correct" this, D-03 is locked)
**Apply to:** the ladder-coverage check in `FLArmsRaceEnv.__init__`, placed beside the `budget_cap` computation.

### Pure-function-with-explicit-argument (no module globals)
**Source:** `rl/rewards.py::goal_target(goal)` — CLAUDE.md's "no module-level mutable state" rule
**Apply to:** `target_for_budget(budget, ladder=None)` — the ladder is a parameter, never installed as global state; `FLArmsRaceEnv` owns the one `self.target_ladder` instance and passes it explicitly at each call site.

## No Analog Found

None — every file this phase touches has a same-file or same-directory analog already read above. This is an additive/in-place phase, not a greenfield module, so no cross-codebase pattern search beyond the files above was needed.

## Deletion Sweep (GOAL-04 — every hit for `target_choices` / `sample_target_in_training`)

Exhaustive grep results, tree-wide (code, config, tests, docs):

| File:Line | Content | Action |
|---|---|---|
| `configs/base.yaml:34` | `sample_target_in_training: true # ...` | delete |
| `configs/base.yaml:35` | comment: "round from `target_choices`, ..." | delete (part of same comment block) |
| `configs/base.yaml:38` | `target_choices: [0.05, 0.10, 0.20, 0.30]  # ...` | delete |
| `rl/env.py:84` | comment: "target_accuracy_drop from target_choices each round ..." | delete (part of removed comment block) |
| `rl/env.py:87` | `self.sample_target = bool(attack.get("sample_target_in_training", False))` | delete |
| `rl/env.py:88-89` | `self.target_choices = [float(x) for x in attack.get("target_choices", [...])]` | delete |
| `rl/env.py:180` | docstring: "draw ``target_accuracy_drop`` from ``target_choices``..." | rewrite docstring for new `_round_goal()` |
| `rl/env.py:185` | `"target_accuracy_drop": self.rng.choice(self.target_choices)}` | replaced by `target_for_budget(...)` call |
| `README.md:320-332` | full paragraph "Target generalization (untargeted_degrade)..." describing `target_choices` / `sample_target_in_training` behavior | rewrite to describe the ladder instead |
| `.planning/ROADMAP.md:31,33` | references `target_choices`/`sample_target_in_training` as things GOAL-04 retires | planning doc, no code action — already describes this phase's own goal, leave as-is (historical/requirements doc) |
| `.planning/PROJECT.md:35,89` | same — describes this phase's goal | leave as-is (requirements doc, not a "reader") |
| `.planning/research/SUMMARY.md:81`, `ARCHITECTURE.md:68,89,96`, `REQUIREMENTS.md:14-15` | describe the retirement as this phase's own deliverable | leave as-is (planning artifacts, not code readers) |
| `.planning/phases/01-.../01-DISCUSSION-LOG.md:34,63,64` | discussion transcript referencing the old keys | leave as-is (historical record) |

**Only two files need code edits for the sweep:** `configs/base.yaml` and `rl/env.py`. `README.md` needs a prose rewrite (doc, not code). All `.planning/*.md` hits are planning artifacts describing this very phase's goal — they are not "readers" in the GOAL-04 sense and should not be edited by this phase's implementation (they may be updated by `/gsd-close-phase` bookkeeping later, not by the planner's plans).

## Three `target_accuracy_drop: 0.20` declarations — consumer map

| Declaration | File:Line | Read by |
|---|---|---|
| `attack.goal.target_accuracy_drop: 0.20` | `configs/base.yaml:30` | `FLArmsRaceEnv.__init__` → `self.goal` (`rl/env.py:71-73`); used as fallback goal when `_round_goal()` doesn't override it (post-change: `_round_goal()` ALWAYS returns the ladder value now — see Known Hazard below, this becomes dead for training once GOAL-03 lands, but stays live for `--dry-run`/`--baseline` paths that construct `RoundContext` differently — planner must check whether those paths call `_round_goal()` or read `self.goal` directly) |
| `target_accuracy_drop: 0.20` in `configs/attacker_agent.yaml:12` (not yet read — file not opened this pass; CONTEXT.md cites it, planner should verify) | `configs/attacker_agent.yaml:12` | likely `AttackerAgent.__init__` config fallback — verify against `agents/attacker_agent.py:80+` (not read in this pass; recommend planner Read `agents/attacker_agent.py:80-140` before writing the plan for this file) |
| `DEFAULT_GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}` | `agents/attacker_agent.py:35` | fallback goal for `AttackerAgent` when no goal is supplied to `build_user_prompt`/constructor; used by direct-inference paths (`rl/inference.py`, possibly `--dry-run`) — this is the prompt-serialization surface, not the reward surface |

**Hazard for the planner (already flagged in CONTEXT.md Known Hazards):** after GOAL-03, `_round_goal()` unconditionally returns `target_for_budget(...)`, so training NEVER reads `configs/base.yaml:30`'s `0.20` fallback anymore — but eval/`--dry-run`/`--baseline`/`rl/inference.py` paths that don't go through `FLArmsRaceEnv._round_goal()` (if any exist) would still silently use `0.20`. The planner must trace whether `--dry-run` and `--baseline` build their own `RoundContext`/goal independent of `_round_goal()`, or route through the same env. This phase's CONTEXT.md explicitly defers resolving this (see `<deferred>` §1) — the planner should decide whether Phase 1 needs a "no silent 0.20 in training" guard test or whether this is fully deferred to a later phase.

## `clean_reference_accuracy` / GOAL-06 pre-flight measurement surface

- **Computed at:** `rl/env.py::FLArmsRaceEnv.clean_reference_accuracy()` (`rl/env.py:233-274`) — lazily cached per round in `self._clean_ref_acc`, invalidated on `restore_fl_state` and `run_benign_fl_round`. This is the "noise" GOAL-06 needs to characterize: repeated calls under IDENTICAL conditions (same global weights, same defense) should ideally return the same value, but stochastic elements (dropout-free eval should actually be deterministic — MNIST eval has no randomness in this codebase unless the defense's own algorithm has internal RNG, e.g. `Multi-Krum`/`DnC` subsampling with `dnc_niters`) may introduce variance.
- **Defense selection in `single` mode:** `server/defense_ensemble.py` — `selection: "rotate" | "random" | "fixed"` (`configs/base.yaml:71-75`), advanced by `DefenseEnsemble.begin_round()`, called from `FLArmsRaceEnv.begin_round()` (`rl/env.py:213`) BEFORE `clean_reference_accuracy()` is first computed for the round (`rl/env.py:215`) — this ordering is load-bearing (comment at `rl/env.py:208-212`) and is exactly why the pre-flight rig must pick a defense algorithm explicitly rather than let rotation pick one, if it wants repeated-round noise measurement per FIXED defense.
- **Existing script/harness host candidates:**
  - `benchmark/harness.py::run_benchmark()` — already loops `n_rounds` per defense and records `DefenseMetrics` (`benchmark/metrics.py`, not read this pass); structurally the closest existing "repeat N times per defense" loop, but it also runs the attacker LLM each round, which is unnecessary for a pure noise-measurement (GOAL-06 only needs the CLEAN counterfactual, no attack).
  - No existing `--dry-run` path was found in this pass that isolates `clean_reference_accuracy()` alone in a repeat-N loop; `main.py` was not read this pass (out of the phase's touched-file list) — recommend the planner Read `main.py`'s `--dry-run` branch before deciding whether GOAL-06's script is a new small standalone tool (e.g. `benchmark/noise_probe.py`) or a mode added to `benchmark/harness.py`. Given `benchmark/harness.py`'s docstring is explicitly about attacker-vs-defenses comparison (not noise calibration), a NEW small script is more consistent with "one function, one place" than overloading the harness.

## Metadata

**Analog search scope:** `rl/`, `agents/`, `server/`, `configs/`, `tests/`, `benchmark/`, `README.md`, `.claude/CLAUDE.md`
**Files scanned (Read in full or targeted range):** `rl/rewards.py`, `rl/env.py`, `rl/switch.py`, `configs/base.yaml`, `agents/attacker_agent.py` (partial, lines 1-80), `tests/test_reward_reference.py`, `tests/test_switch.py`, `server/defense_ensemble.py` (partial, lines 95-125), `benchmark/harness.py` (partial, lines 1-60), `README.md` (partial, lines 305-339)
**Grep sweeps:** `target_choices|sample_target_in_training` (tree-wide), `raise (RuntimeError|ValueError)` (tree-wide `*.py`)
**Pattern extraction date:** 2026-08-01
