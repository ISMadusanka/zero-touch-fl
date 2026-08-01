# Codebase Concerns

**Analysis Date:** 2026-08-01

## Tech Debt

**League memory management:**
- Issue: The opponent league snapshots adapters into an in-memory ring buffer. Each snapshot is ~115 MB (full LoRA state dict for both adapters at lora_r=16). While a hard cap was added (`league_max_snapshots: 10`, default ~2.3 GB total), very long runs with higher snapshot frequency could still accumulate significant memory.
- Files: `rl/schedule.py` (lines 53–98)
- Impact: Long runs (100k+ rounds) with frequent snapshots could trigger OOM despite the ring buffer. Older runs pre-cap (before commit 96bef31) showed unbounded growth (~22 GB by round 10k, ~223 GB by round 100k).
- Fix approach: Monitor `league_max_snapshots` setting in config; confirm ring-buffer eviction works under load. Add memory usage logging in League.snapshot() to surface growth. Consider adaptive snapshot frequency if memory pressure is detected.

**LLM client error handling:**
- Issue: OpenAI and Ollama backends catch exceptions in `call()` and `complete()` and return empty dict/string rather than propagating errors. This silently degrades the round.
- Files: `agents/llm_client.py` (lines 76–78, 96–98)
- Impact: A transient API failure (rate limit, timeout, network blip) will produce an unparseable LLM response, which is silently treated as a valid no-op attack/verdict, advancing the round with no poison/all flagged. The error is logged but no retry is attempted. Over a 2M-round run, this could accumulate unnoticed.
- Fix approach: Replace silent failures with explicit retry-with-backoff in the OpenAI/Ollama clients, or bubble the error up so the training loop can decide (pause, skip round, checkpoint and exit). Log the raw exception details, not just a summary string.

**Attack plan parsing robustness:**
- Issue: The attacker's JSON output parsing in `agents/attack_ops.extract_json()` and the subsequent operator application in `agents/attack_ops.apply_plan()` is extremely forgiving—bad fields, invalid targets, malformed ops are silently skipped. There is no upper bound on how many ops can be invalid before the entire plan is treated as a no-op.
- Files: `agents/attack_ops.py` (lines 58–80, apply_plan body), `agents/attacker_agent.py` (select_and_apply)
- Impact: A partially-broken LLM output (e.g., correct format but random field names or incorrect layer names) is silently downgraded to "no attack attempted on this client". The attacker learns that malformed output costs no reward penalty—convergence to minimal attacks is possible if the LLM drifts.
- Fix approach: Count and log malformed/skipped operators per client; if >50% of operations in a plan are skipped, either flag it as a failed round or apply a reward penalty explicitly tied to plan validity (e.g., `gamma` in the reward config, `agents/attack_ops.py` already tracks `n_malformed`).

**State consistency on resume:**
- Issue: Phase-2 resume loads the saved FL state (global model + per-client benign weights) in `main.py` (lines 270–277), but if the save is missing (e.g., checkpoint from before `save_fl_state` was added), the round resumes from the Phase-1 baseline instead. The warning logs but does not halt—downstream the attacker/defender may train against a different snapshot of client weights than they expect.
- Files: `main.py` (lines 270–277), `storage/checkpoint.py` (save_fl_state/load_fl_state)
- Impact: A resumed run that loads an old checkpoint without saved FL state will drift: the attacker adapts to client-weight dynamics from rounds 0–N, then on resume faces rounds N+1+ with fresh (Phase-1 baseline) client weights, breaking distributional consistency and reward reliability.
- Fix approach: Make `load_fl_state()` return an explicit sentinel on missing state, then halt with a clear error if resume is requested but state is unavailable. Or, auto-retrain from Phase 1 on mismatch.

**Defense ensemble error accumulation:**
- Issue: When an algorithmic defense (FLTrust / Multi-Krum / DnC / DeFL) raises an exception, `DefenseEnsemble.verdicts()` catches it, logs it once per 500 rounds (line 104), increments an error counter, and skips that defense. If *all* defenses fail in `union` mode, the method returns empty verdicts (all clients benign). In `single` mode, if the active defense fails, no backup is used—the round gets an empty verdict.
- Files: `server/defense_ensemble.py` (lines 184–212)
- Impact: A sustained failure in one algorithm (e.g., DnC's spectral decomposition hits a numerical edge case) causes it to be silently skipped in union mode, or to crash the round in single mode if it is the only active defense. Silent skips in union mode gradually weaken the ensemble without alerting the operator.
- Fix approach: Log the first N failures verbosely with full tracebacks; thereafter log once per K rounds. In single mode, fall back to a safe default (e.g., flag nothing, or flag the top-K by magnitude) instead of crashing. Track persistent errors and alert if an algorithm fails N times in a row (e.g., >10).

## Known Bugs

**Incomplete per-class evaluation for targeted attacks:**
- Symptoms: The attack config in `configs/base.yaml` (line 32) includes a `targeted_label` goal type, with a comment "per-class eval is a TODO; falls back to overall". The system currently only evaluates overall accuracy, not per-class accuracy.
- Files: `configs/base.yaml` (line 32), `rl/env.py` (evaluate_updates)
- Trigger: Setting `attack.goal.type: "targeted_label"` and `attack.goal.label: <N>` will configure the attack but the metric recorded will be overall accuracy, not the target class's accuracy. Reward scoring will be incorrect.
- Workaround: Do not use `targeted_label` goal type; stick to `untargeted_degrade` or implement per-class accuracy in `FLArmsRaceEnv.evaluate_updates()` and the reward computation.

**Zero-advantage collapse when policy explores poorly:**
- Symptoms: If a GRPO group (G=4 rollouts) all sample the same action and earn the same reward, the within-group advantage is exactly zero. The loss then becomes `kl_beta * KL` alone, pulling the policy *toward the base model* (un-learning). The guards (`resample_on_zero_advantage`, `skip_zero_advantage`) attempt to fix this, but if they fail, the policy can un-learn over multiple rounds.
- Files: `rl/grpo.py` (lines 59–68), `rl/schedule.py` (GRPO config in train())
- Trigger: Low temperature sampling (near-greedy) + a reward landscape with many identical-value regions. Attacker mode is most vulnerable because the reward surface (drop vs evasion tradeoff) is thin.
- Workaround: Ensure `resample_on_zero_advantage: true` and `resample_temperature: 1.3` or higher. Monitor per-phase `mean_advantage` in the round logs; if it stays near zero for >10 rounds, pause and inspect the policy.

## Security Considerations

**Checkpoint and credential exposure:**
- Risk: Saved checkpoints (model weights, adapters, LoRA dicts) are stored unencrypted in `checkpoints/`. If the repo is ever leaked or backed up unsecurely, the fine-tuned Qwen2.5 adapters are exposed. These adapters themselves are not secrets, but the training data (rewards, round logs) is embedded in their learned weights.
- Files: `storage/checkpoint.py`, `rl/policy.py` (save/load), `checkpoints/` directory
- Current mitigation: `.gitignore` excludes `checkpoints/` from version control; only developers with local access can see the trained models.
- Recommendations: Add encryption for checkpoints if the system is moved to shared cloud storage. Document that adapters should be treated as sensitive artifacts (not committed, not shared publicly). Add a checkpoint integrity check (SHA256 hash on load) to detect tampering.

**Attack plan validation and injection:**
- Risk: The attacker LLM outputs an attack plan as a JSON list of operators. While the parser is forgiving (bad fields silently skip), there is no schema validation or upper bounds on the number of operations or parameter magnitudes. A malicious or broken LLM could emit plans with extremely large scaling factors or per-client operation counts that trigger numerical instability.
- Files: `agents/attack_ops.py` (apply_plan, lines 150+)
- Current mitigation: NaN/Inf scrubbing (line ~270) and weight clamping to `±max_weight_abs` catch most overflows downstream.
- Recommendations: Add pre-application validation: check operator counts per client (e.g., >100 ops per client is suspicious), check parameter ranges (e.g., scale factors outside [0.01, 100]), and log rejected plans. Fail loudly on invalid plans rather than silently downgrading them.

## Performance Bottlenecks

**Per-round feature extraction:**
- Problem: Every round, `detector/features.py` computes per-layer statistical features (rel_update, RMS, energy_frac, etc.) for all N clients. This is O(N × L) where L is the number of layers. At default N=20, L=4, this is negligible (~5ms), but at N=100+ it could become measurable.
- Files: `detector/features.py`, `rl/env.py` (env.features called once per round)
- Cause: Naive iteration over clients and layers with multiple nested torch ops (norm, mean, std).
- Improvement path: Vectorize the computation (batch all client updates into a single tensor, compute stats in parallel). Profile with N=100+ to confirm impact. For now, acceptable on modern hardware.

**Model evaluation on test set:**
- Problem: Every round (Phase 2), the server evaluates the global model on the full MNIST test set (~10k images). At 2M simulation rounds, this is 2M × 10k = 20B forward passes. Even on GPU, this is significant compute.
- Files: `server/fed_server.py` (evaluate), `rl/env.py` (evaluate_updates called per rollout and per commit)
- Cause: Evaluation is run G times per round (for rollout scoring) + 1 for the committed round, = O(G+1) per round. Each evaluation is a full forward pass over the test set.
- Improvement path: Sample the test set (evaluate on 1k images instead of 10k), or cache the test-set forward passes and update incrementally. Low priority unless eval time dominates total round time.

**Unsloth generation with KV cache:**
- Problem: With `use_fast_generate: true`, Unsloth's fused paged-KV generation kernel is attempted, but if it fails (version mismatch, incompatible hardware), it falls back to standard Transformers generation. The fallback is slow (no paged KV cache).
- Files: `rl/policy.py` (lines 52–58, and the generate call in generate method)
- Cause: The fallback is silent; users may not notice they are on the slow path.
- Improvement path: Log a warning if the fast path fails. Pre-validate the fast-generate setup at policy initialization and warn early if it will fail. Profile to quantify slowdown (likely <5% per-token time, acceptable for research).

## Fragile Areas

**Attacker LLM selection logic:**
- Files: `agents/attacker_agent.py` (select_and_apply, lines ~150–200)
- Why fragile: The attacker selects which pool clients to poison by returning a JSON list of client IDs, but the parsing does not validate that IDs are unique, within range, or ≤budget before truncation. The truncation to the budget happens *after* dedup, so large lists of out-of-range IDs pass through silently. The system then counts "n_malformed" but still applies the plan, potentially wasting reward.
- Safe modification: Add pre-application validation: `assert all(0 <= id < len(pool) for id in selected_ids)`, and if any ID is invalid, fail the entire plan and log it. Update the reward penalty (`gamma` in config) to reflect invalid selections, not just invalid operators.
- Test coverage: `tests/test_attacker_select.py` covers normal cases and out-of-budget truncation, but not out-of-range IDs or non-unique IDs.

**Defense ensemble mode transitions:**
- Files: `server/defense_ensemble.py` (verdicts, begin_round)
- Why fragile: The ensemble can switch between `mode="single"` and `mode="union"` at runtime via config reload, but the active algorithm in single mode is determined in `begin_round()` and held for the rest of the round. If `defense.selection` is `"random"` and the RNG is not properly seeded with the same seed across resume, different algorithms may be active at the same commit step in a resumed run, breaking reproducibility.
- Safe modification: Ensure the RNG is seeded deterministically from `fl.poison_seed` at initialization. Log which algorithm is active per round. Add a test that verifies single-mode selection is reproducible across resume.
- Test coverage: `tests/test_defense_ensemble.py` covers union and single modes, but does not test resume with random selection.

**FL state synchronization on resume:**
- Files: `main.py` (lines 262–277, env.restore_fl_state), `rl/env.py` (restore_fl_state), `rl/schedule.py` (train loop where env is used)
- Why fragile: The global model and per-client benign weights are stored separately (in `env.global_weights` and `env.client_weights`). On resume, the FL state is restored, but if any phase of the training loop (attacker/defender GRPO) modifies the global model without going through the checkpoint callback, the restored state and the running state diverge. The checkpoint callback is only called at `save_every` intervals, so a crash between checkpoints loses up to `save_every` rounds of training (default 25).
- Safe modification: After each round.commit(), call `fl_state_cb()` to save the updated global model synchronously, not just at checkpoint intervals. Or, use a file-backed mmap for the model so changes are persisted immediately.
- Test coverage: `tests/test_resume.py` covers basic resume, but does not simulate a crash mid-phase and verify state recovery.

## Scaling Limits

**Simulation_rounds configuration:**
- Current capacity: Default is 2,000,000 rounds. At ~2 seconds per round (attacker LLM + defender LLM + eval), that is ~46 days of wall-clock time on a single GPU.
- Limit: Per-round memory grows with `league_max_snapshots` (each snapshot is ~115 MB). At the default cap (10), total is ~2.3 GB per adapter pair, acceptable. Storage for logs grows linearly: at default ~1 KB per round (JSONL), 2M rounds = ~2 GB.
- Scaling path: If increasing `simulation_rounds` beyond 2M, monitor memory (league snapshots) and disk (round logs). Consider archiving old logs (>1 week old) to external storage.

**Num clients (N):**
- Current capacity: Default N=20; tested up to N=20.
- Limit: Feature extraction is O(N × L); evaluation is O(N) (average update norm). At N=100, feature computation and FedAvg aggregation will be noticeably slower. The attacker's context window grows (client stats are in the prompt), potentially hitting the LLM's token limit.
- Scaling path: Vectorize feature extraction (batch ops). Sample features (report stats for a random subset of clients) instead of all. Use a smaller test set or validation-set sampling for evaluation.

**Model size:**
- Current capacity: MnistNet is ~970 parameters. Qwen2.5-3B is fine-tuned with LoRA.
- Limit: Qwen2.5-3B + 2× LoRA adapters (lora_r=16) requires ~12 GB VRAM in bf16 mode (default). If `load_in_4bit: true`, it shrinks to ~6 GB. Larger models (Qwen-7B, Qwen-72B) will require more.
- Scaling path: Use QLoRA (`load_in_4bit: true`) for bigger models. Or use a smaller base model (Qwen-1B, Llama-3.2-1B). Distributed training (multi-GPU) is not currently supported.

**Checkpoint file size:**
- Current capacity: Global model state dict (~4 KB for MnistNet), client updates (N × ~4 KB), FL state (global + per-client, ~few KB total). Per-checkpoint total: ~100 KB. At `save_every=25` and 2M rounds, that is ~8M checkpoints = ~800 GB. In practice, old checkpoints are not archived, so only the latest `checkpoint_dir` is kept (~1–10 MB depending on job age).
- Scaling path: Archive checkpoints older than N days. Use differential checkpointing (only save deltas from the last checkpoint). For long runs, implement checkpoint garbage collection.

## Dependencies at Risk

**Unsloth dependency:**
- Risk: Unsloth is a third-party optimization library for LLM fine-tuning. It patches transformers at import time to accelerate Attention and projection layers. If the installed Transformers version diverges from Unsloth's supported matrix, generation can fail or be silently slow.
- Impact: `rl/policy.py` imports unsloth early and sets `UNSLOTH_DISABLE_FAST_GENERATION=1` to fall back to standard Transformers if the fast path breaks, but the fallback is not logged loudly.
- Migration plan: Monitor Unsloth release notes for compatibility breaks with newer Transformers. If Unsloth becomes unmaintained, replace with native Transformers + PEFT (LoRA only, no fused ops). The RL loop will still train; it will just be slower.

**OpenAI API deprecation:**
- Risk: The system supports OpenAI as an LLM backend. If the OpenAI API changes (authentication, response format, rate limits), the system could break. The repo has no pinned version for the `openai` package.
- Impact: A major API upgrade could require prompt changes or response parsing updates. The code catches all exceptions and returns empty, masking the actual error.
- Migration plan: Pin the `openai` package version in `requirements.txt`. Add an integration test that periodically calls the OpenAI backend with a real (or mocked) API key. Document the minimum supported OpenAI API version.

**Ollama dependency:**
- Risk: Ollama is an inference backend (offline or remote). If the Ollama server is unavailable or returns malformed responses, the training loop gets empty text and treats it as a no-op.
- Impact: Unlike OpenAI (which has monitored uptime), Ollama is a local service that may crash or hang. There is no timeout or heartbeat check.
- Migration plan: Add a health-check endpoint to Ollama (or fallback to OpenAI if Ollama is down). Add request timeouts and retry-with-backoff. Log Ollama connectivity issues prominently.

## Missing Critical Features

**Per-class targeted attack evaluation:**
- Problem: The attack config supports `goal.type = "targeted_label"`, but the reward and evaluation are hard-coded to overall accuracy. Per-class accuracy metrics are not computed.
- Blocks: Targeted label attacks cannot be trained or evaluated properly. Users who set this goal will get incorrect rewards.
- Priority: Medium (only relevant if targeted attacks are intended; untargeted_degrade works).

**Replay buffer / experience replay for RL:**
- Problem: GRPO samples G fresh completions per round and updates immediately. There is no replay buffer to reuse high-reward trajectories or off-policy learning. This limits sample efficiency.
- Blocks: Long training runs are sample-inefficient (each sample is used once). Scaling to larger/harder tasks will require more data.
- Priority: Low (GRPO is designed for single-iteration updates; a replay buffer would require architectural changes).

**Multi-GPU distributed training:**
- Problem: The policy is trained on a single GPU. No distributed training (DDP, FSDP) is implemented.
- Blocks: Training on larger models or faster training on multi-GPU hardware is not possible.
- Priority: Low (single GPU is acceptable for a 3B model and 2M-round training; would become relevant for 7B+ models).

## Test Coverage Gaps

**LLM client error paths:**
- What's not tested: The error handling in `agents/llm_client.py` (both OpenAI and Ollama clients catching and returning empty on failure) is not tested. No test verifies that a network failure or API error is handled gracefully.
- Files: `agents/llm_client.py`, `agents/llm_client.py`
- Risk: A transient API failure could silently proceed as a no-op round, unnoticed.
- Priority: Medium. Add tests that mock the OpenAI/Ollama backends to return errors and verify the round completes (no crash) with expected empty output.

**Attack plan malformed operator counting:**
- What's not tested: The `n_malformed` counter in apply_plan is incremented but not verified by any test. A plan with 50% invalid operations should be flagged but is not.
- Files: `agents/attack_ops.py`, `agents/attacker_agent.py`
- Risk: An LLM that drifts toward invalid operators escapes unnoticed.
- Priority: Medium. Add a test that generates a plan with mixed valid/invalid operators and verifies the malformed count matches expected.

**Resume from missing FL state:**
- What's not tested: The warning in `main.py` line 274 is logged if FL state is not found, but there is no test that verifies the run completes (or fails loudly) in that case.
- Files: `main.py`, `storage/checkpoint.py`
- Risk: A user might resume a run on an older checkpoint and not realize the FL state is missing until training quality degrades.
- Priority: Medium. Add a test that deletes the FL state file and resumes; verify the warning is logged and the run behaves correctly.

**Defense ensemble algorithm failure recovery:**
- What's not tested: If an algorithm in the ensemble raises an exception, the error counting and fallback behavior (union vs. single mode) are not directly tested.
- Files: `server/defense_ensemble.py`
- Risk: A persistent algorithm failure might silently degrade ensemble quality.
- Priority: Low. Add tests that mock algorithm failures and verify verdicts are still produced (or an error is raised loudly).

**Checkpoint save/load atomicity:**
- What's not tested: If the process crashes mid-save (e.g., torch.save() is interrupted), the checkpoint file may be corrupted. On the next load, it will fail. There is no atomic write (write-to-temp-then-rename) or corruption detection.
- Files: `storage/checkpoint.py`
- Risk: A crash during checkpoint save could render the run unresumable.
- Priority: Low (rare in practice, but high impact). Use atomic writes and SHA256 hash checks on load.

---

*Concerns audit: 2026-08-01*
