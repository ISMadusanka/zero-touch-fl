# GOAL-06 Noise Baseline — Clean-Counterfactual Pre-Flight

**Measured:** 2026-08-01

## Command

```
python -m benchmark.noise_probe --rounds 20 --device cpu --sigma-margin 3.0 --out logs/noise_probe.json
```

Resolved parameters (from `logs/noise_probe.json`):

- seed: `0` (`fl.poison_seed`, no `--seed` override)
- device: `cpu`
- rounds per defense: `20`
- sigma margin: `3.0`
- bottom rung evaluated against: `0.02` (`attack.target_ladder[1]` in `configs/base.yaml` at the time of measurement)
- Phase-1 start state: no saved checkpoint existed (`storage.checkpoint.state_exists()` was `False`), so the probe ran the `run_phase1` CPU fallback (45 honest FedAvg rounds, 20 clients, `local_epochs: 3`) to build its start state before measuring. Observed Phase-1 baseline accuracy: `0.7822`.

## Per-Defense Results

Values below are rounded to 6 decimal places for display only. The pass/fail (`clears`) comparison in `benchmark/noise_probe.py::summarize_noise` used the unrounded floats — see the raw JSON in §Raw Data.

| defense | n | mean | sd | 3·sd | bottom rung | clears | required rung |
|---|---|---|---|---|---|---|---|
| fltrust | 20 | 0.762680 | 0.016184 | 0.048552 | 0.02 | **no** | 0.05 |
| multikrum | 20 | 0.765500 | 0.000000 | 0.000000 | 0.02 | yes | 0.00 |
| dnc | 20 | 0.783200 | 0.000000 | 0.000000 | 0.02 | yes | 0.00 |
| defl | 20 | 0.768505 | 0.006612 | 0.019837 | 0.02 | yes | 0.02 |

Per-defense verdicts, in prose:

- **fltrust** does **not** clear the noise floor: its round-to-round clean-counterfactual standard deviation (`sd = 0.016184`) is large enough that `3·sd = 0.048552` exceeds the 0.02 bottom rung by more than double. A budget-1 round's measured `drop` under fltrust would be dominated by this swing, not by the attack.
- **multikrum** clears trivially: all 20 samples are byte-identical (`sd = 0.0`). See Observation 1 below — this is a vacuous pass, not evidence of a low noise floor.
- **dnc** clears trivially for the same reason as multikrum: all 20 samples are byte-identical (`sd = 0.0`). See Observation 1.
- **defl** clears, but marginally: `3·sd = 0.019837` is just under the 0.02 rung (a difference of about 0.000163). See Observation 2 — its samples are a two-state oscillation, not Gaussian noise, so the sd computed from them is a shaky basis for a 3-sigma comparison in the first place.

## Observations (not conclusions)

1. **multikrum and dnc's "sd = 0.0" is determinism, not a low noise floor.** Both defenses returned 20 byte-identical clean-counterfactual samples (verified against the raw `samples` arrays in `logs/noise_probe.json`). With `fl.benign_retrain_each_round: false`, every round's honest updates are the same frozen Phase-1 client weights, and Multi-Krum / DnC apply a deterministic selection rule (both drop a fixed number of clients "by construction" per `server/defense_ensemble.py`'s own docstring) to those same weights every round — so the aggregate, and therefore the clean-counterfactual accuracy, is identical every time. A rung "clearing" a `threshold = 0.0` is a vacuous pass: it says nothing about whether a real, non-degenerate noise source would still be smaller than the rung.
2. **defl's samples are a two-state oscillation, not Gaussian noise.** The 20 samples (in requested order) are: `0.7831`, `0.763`, then `0.7739`/`0.7621` alternating for the remaining 18 rounds (9 pairs). This is consistent with DeFL's accumulating internal state (its critical-learning-period test and Beta trust counts) settling into a 2-cycle rather than producing i.i.d. noise around a fixed mean. A standard deviation computed from a deterministic 2-state oscillation is not really "noise" in the sense the 3-sigma gate assumes, and its pass is marginal in any case (`0.019837` vs the `0.02` rung — a margin of about 0.8%).

Both observations weaken confidence in the *passing* verdicts (multikrum, dnc, defl) as much as the failing fltrust verdict strengthens the case for a rung raise — none of the four defenses gives a clean, well-behaved noise measurement at n=20. This reinforces, rather than undercuts, treating the outcome below as a developer decision.

## Overall Verdict: Rung-Collision — Developer Decision Required

**The bottom rung (0.02) does NOT clear 3-sigma noise for fltrust.** The measurement demands a bottom rung of at least **0.05** (`ceil(3 × 0.016184 / 0.01) × 0.01`).

**This is the collision case anticipated in `01-03-PLAN.md` (the task's action block, "If raising rung 1 would make it meet or exceed rung 2..."):** the required rung (`0.05`) **exceeds rung 2** (`0.04` in `configs/base.yaml` / `DEFAULT_TARGET_LADDER`). Raising rung 1 to `0.05` would make the ladder `{1: 0.05, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.12}` — no longer strictly increasing — which would fail `tests/test_target_ladder.py::test_shipped_config_declares_a_complete_ladder` and violate the per-(defense × budget) independence the whole ladder exists to guarantee.

Per prohibition P-02 and the plan's explicit instruction for this exact collision, **no code or config change has been made**:

- `configs/base.yaml` — **unchanged**.
- `rl/rewards.py` (`DEFAULT_TARGET_LADDER`) — **unchanged**.
- `tests/test_target_ladder.py` — **unchanged**.

None of the following were done, and must not be done to resolve this measurement:

- Lowering rung 2 (or any other rung) to make room for a raised rung 1.
- Relaxing or editing `test_shipped_config_declares_a_complete_ladder`'s monotonicity assertion.
- Picking an intermediate value (e.g. 0.03) that "fits" between the current rungs 1 and 2 — the measurement says 0.05, not a number chosen to avoid the collision.
- Re-running with fewer rounds, fewer defenses, or a smaller `--sigma-margin` to obtain a more convenient answer (prohibition P-01 — a fabricated, recalled, or convenience-adjusted noise floor would silently invalidate every downstream number).

**This is surfaced at Task 3 (the blocking `checkpoint:human-verify`) as a developer decision.** Representative options for the developer to weigh (not a recommendation, and not an exhaustive list):

1. Re-space the whole ladder (e.g. `{1: 0.06, 2: 0.08, 3: 0.10, 4: 0.12, 5: 0.16}` or similar) so every rung, including a new rung 1, clears fltrust's measured noise while staying strictly increasing.
2. Drop fltrust from the `--freeze defender` algorithmic panel (`defense.algorithms` in `configs/base.yaml`), if its clean-counterfactual instability under this non-IID split makes it unsuitable as a training-time opponent regardless of the ladder.
3. Investigate whether fltrust's round-to-round swing (mean `0.762680`, range `[0.7369, 0.7868]` — see raw samples below) is itself a symptom worth fixing upstream (e.g. its trust-vector sensitivity under `noniid_bias: 0.5`), before deciding how the ladder should respond to it.
4. Accept a documented gap for budget 1 specifically (e.g. exclude budget 1 from fltrust-judged rounds, or flag it as an untrustworthy cell in the eventual per-defense results table) rather than changing the ladder at all.

## Raw Data

Full contents of `logs/noise_probe.json` (not committed — `logs/` is gitignored; pasted here so the artifact of record is self-contained):

```json
{
  "sigma_margin": 3.0,
  "rung": 0.02,
  "n_rounds": 20,
  "device": "cpu",
  "seed": 0,
  "defenses": [
    {
      "name": "fltrust",
      "n": 20,
      "mean": 0.76268,
      "sd": 0.016183868511576587,
      "min": 0.7369,
      "max": 0.7868,
      "sigma_margin": 3.0,
      "threshold": 0.04855160553472976,
      "rung": 0.02,
      "clears": false,
      "required_rung": 0.05,
      "samples": [
        0.7684, 0.7714, 0.7788, 0.7771, 0.7684, 0.7401, 0.7821, 0.756,
        0.7732, 0.7369, 0.7784, 0.738, 0.7467, 0.7588, 0.7569, 0.7714,
        0.7868, 0.7379, 0.7475, 0.7788
      ]
    },
    {
      "name": "multikrum",
      "n": 20,
      "mean": 0.7655,
      "sd": 0.0,
      "min": 0.7655,
      "max": 0.7655,
      "sigma_margin": 3.0,
      "threshold": 0.0,
      "rung": 0.02,
      "clears": true,
      "required_rung": 0.0,
      "samples": [
        0.7655, 0.7655, 0.7655, 0.7655, 0.7655, 0.7655, 0.7655, 0.7655,
        0.7655, 0.7655, 0.7655, 0.7655, 0.7655, 0.7655, 0.7655, 0.7655,
        0.7655, 0.7655, 0.7655, 0.7655
      ]
    },
    {
      "name": "dnc",
      "n": 20,
      "mean": 0.7832,
      "sd": 0.0,
      "min": 0.7832,
      "max": 0.7832,
      "sigma_margin": 3.0,
      "threshold": 0.0,
      "rung": 0.02,
      "clears": true,
      "required_rung": 0.0,
      "samples": [
        0.7832, 0.7832, 0.7832, 0.7832, 0.7832, 0.7832, 0.7832, 0.7832,
        0.7832, 0.7832, 0.7832, 0.7832, 0.7832, 0.7832, 0.7832, 0.7832,
        0.7832, 0.7832, 0.7832, 0.7832
      ]
    },
    {
      "name": "defl",
      "n": 20,
      "mean": 0.768505,
      "sd": 0.006612448487512033,
      "min": 0.7621,
      "max": 0.7831,
      "sigma_margin": 3.0,
      "threshold": 0.0198373454625361,
      "rung": 0.02,
      "clears": true,
      "required_rung": 0.02,
      "samples": [
        0.7831, 0.763, 0.7739, 0.7621, 0.7739, 0.7621, 0.7739, 0.7621,
        0.7739, 0.7621, 0.7739, 0.7621, 0.7739, 0.7621, 0.7739, 0.7621,
        0.7739, 0.7621, 0.7739, 0.7621
      ]
    }
  ],
  "all_clear": false,
  "required_rung": 0.05
}
```

Printed table (from `logs/noise_probe_run.log`, matches the JSON above):

```
| defense | n | mean | sd | 3·sd | bottom rung | clears | required rung |
|---|---|---|---|---|---|---|---|
| fltrust | 20 | 0.762680 | 0.016184 | 0.048552 | 0.020000 | no | 0.050000 |
| multikrum | 20 | 0.765500 | 0.000000 | 0.000000 | 0.020000 | yes | 0.000000 |
| dnc | 20 | 0.783200 | 0.000000 | 0.000000 | 0.020000 | yes | 0.000000 |
| defl | 20 | 0.768505 | 0.006612 | 0.019837 | 0.020000 | yes | 0.020000 |

Verdict: the 0.020000 bottom rung does NOT clear 3.0 sigma for fltrust; required bottom rung >= 0.050000.
```

## How to Re-Run

```
python -m benchmark.noise_probe --rounds 20 --device cpu --sigma-margin 3.0 --out logs/noise_probe.json
```

Re-run this measurement if any of the following change:

- `attack.max_poison_clients`, `fl.n_compromisable`, or `fl.benign_retrain_each_round` in `configs/base.yaml`.
- Any `defense:` hyperparameter (`root_size`, `dnc_*`, `defl_*`, `assumed_malicious`, etc.).
- Phase 4's BENCH-06 determinism work on Multi-Krum / DnC subsampling — Observation 1 above notes that the current `sd = 0.0` for both is a product of `benign_retrain_each_round: false` producing byte-identical rounds; any change to subsampling behavior (or to `benign_retrain_each_round`) invalidates that reading.
- The ladder itself, once the collision above is resolved by a developer decision — the new bottom rung (whatever it becomes) should be re-measured against this same noise floor.
