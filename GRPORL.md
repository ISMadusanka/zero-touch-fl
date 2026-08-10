# GRPO & the RL Loop — How It Works in This Project

A from-scratch, plain-English guide to the reinforcement-learning engine behind
the attacker/defender arms race: what GRPO is, the maths (with worked numbers),
how one RL round is implemented in code, and how the policy (the LLM) actually
produces and learns from its answers.

> Companion doc: [HOWATTACKDEFEND.md](HOWATTACKDEFEND.md) explains the *game*
> (what the attacker/defender do and how they adapt). This doc explains the
> *machinery* (GRPO, the round loop, the policy functions).

## Contents
1. [The big picture](#1-the-big-picture)
2. [What "policy" and "reward" mean here](#2-what-policy-and-reward-mean-here)
3. [What GRPO is (and why we use it)](#3-what-grpo-is-and-why-we-use-it)
4. [The GRPO maths, with a worked example](#4-the-grpo-maths-with-a-worked-example)
5. [The KL penalty, explained simply](#5-the-kl-penalty-explained-simply)
6. [How one RL round is implemented](#6-how-one-rl-round-is-implemented)
7. [The policy functions (`rl/policy.py`)](#7-the-policy-functions-rlpolicypy)
8. [The reward functions (`rl/rewards.py`)](#8-the-reward-functions-rlrewardspy)
9. [A full annotated round trace](#9-a-full-annotated-round-trace)
10. [Design choices & FAQ](#10-design-choices--faq)
11. [Glossary](#11-glossary)

---

## 1. The big picture

We have **two LLMs** (an attacker and a defender) that we want to *improve* by
trial and error, with **no human-labelled training data**. Instead of labels, we
have a **scorer** that can grade any answer using ground truth:

- attacker answer → apply the poison → measure accuracy drop + whether it evaded → **reward**
- defender answer → compare its flags to the known poisoned set → **reward**

This is **RL with verifiable rewards (RLVR)**: the reward is computed by a
deterministic, trustworthy function (not a learned "reward model"), so it can't
be gamed by fooling a judge. The algorithm we use to turn those rewards into
model improvements is **GRPO**.

The whole engine lives in `rl/`:

| File | Role |
|---|---|
| `rl/policy.py` | The LLM itself: one frozen base + two trainable LoRA adapters. Generates answers, computes log-probs. |
| `rl/rewards.py` | The scorers: `attacker_reward`, `defender_reward`, and `group_advantages`. |
| `rl/turns.py` | Wraps one FL round for one learner (builds the prompt, scores a candidate, commits the chosen one). |
| `rl/grpo.py` | **One GRPO update**: sample → score → advantage → gradient step. |
| `rl/schedule.py` | The training loop that decides who learns and for how long. |
| `rl/env.py` | The federated-learning world: clients, FedAvg, accuracy oracle. |

---

## 2. What "policy" and "reward" mean here

**Policy** = the thing that chooses actions. Here the policy *is* the LLM (with a
specific LoRA adapter active). Given a **state** (the prompt describing the round)
it outputs an **action** (a block of text — an attack plan or a set of verdicts).
It's a *probabilistic* policy: it doesn't output one fixed answer, it samples from
a probability distribution over possible answers.

**Reward** = a single number grading how good that action was (higher = better).

**Policy gradient** (the core RL idea) = *nudge the policy's parameters so that
high-reward actions become more probable and low-reward actions become less
probable.* GRPO is one specific, efficient recipe for doing that nudge.

---

## 3. What GRPO is (and why we use it)

**GRPO = Group-Relative Policy Optimization.**

The hard part of policy gradients is knowing whether a reward is "good." If a plan
scores 0.5, is that good or bad? You need a **baseline** to compare against.
Classic methods (PPO/A2C) train a *second* neural network (a "critic" / value
function) just to predict that baseline. That's extra model, extra memory, extra
things to get wrong.

**GRPO's trick: skip the critic. Use the group as the baseline.** For each
situation, sample **G** answers, and judge each one *relative to the average of
the other G−1*. An answer that beats its own batch gets reinforced; one that
trails the batch gets discouraged. The batch average *is* the baseline.

> Simple analogy: a teacher grading on a curve. They don't need an absolute
> definition of "a good essay" — they compare the essays in the batch to each
> other and reward the ones above the class average.

Why it fits this project (see also the docstring in `rl/grpo.py`):
- **Verifiable rewards** make the per-answer scores trustworthy.
- **No critic** means less GPU memory and fewer moving parts — good for running
  two adapters on one base model.
- The reward is **outcome-based** (did the attack land? did the flag match truth?),
  exactly the regime GRPO was designed for.

---

## 4. The GRPO maths, with a worked example

One GRPO update (`grpo_step` in [rl/grpo.py:26](rl/grpo.py)) does five things:

### Step 1 — Sample G answers
Ask the policy for **G** completions for the same prompt (`G=4` by default).
```
completions = policy.generate(adapter, system, user, n=G, temperature=1.0, ...)
```
Temperature > 0 so the 4 answers actually differ (that's where the learning
signal comes from).

### Step 2 — Score each
```
rewards = [reward(c) for c in completions]      # e.g. [0.0, 0.5, 0.1, 1.0]
```

### Step 3 — Turn rewards into *advantages* (the group-relative part)
`group_advantages` ([rl/rewards.py](rl/rewards.py)) z-scores the rewards:

```
spread = max(rewards) − min(rewards)
mean   = average(rewards)
std    = population standard deviation(rewards)
Aᵢ     = (rᵢ − mean) / max(std, advantage_std_floor)
```
Two guards separate "the plans really differed" from "the measurement wobbled":

* If `spread < min_reward_spread` (default **0.02**) the group is **degenerate**:
  all-zero advantages and `zero_advantage_fraction = 1.0` (a dead, no-gradient
  round — see §5/§10). The bar is a noise floor, not exact equality: accuracy is
  measured on 10k test examples, so it is quantized to 1e-4, which at the smallest
  `target_accuracy_drop` the config can ask for (0.05) is ~2e-3 of reward per
  *single flipped test example* — ~1e-3 at the shipped training target of 0.10.
  The old test was `std < 1e-6` — a thousand times below that noise — so
  two behaviourally identical rollouts were routinely z-scored up to `A = ±1.2` and
  trained on at full strength. A real 1% accuracy gap scores 0.2 at that target, so
  genuine differences clear the bar easily.
* `advantage_std_floor` (default **0.05**) floors the denominator. Plain z-scoring
  is scale-free — a group spread over 0.02 and one spread over 1.0 both come out at
  `A = ±1.2` — so the update size carried no information about *how much* better the
  winning rollout was. Above this std it reduces to standard GRPO z-scoring.

Set both to `0` in `configs/base.yaml` for textbook GRPO.

**Worked example** with `rewards = [0.0, 0.5, 0.1, 1.0]`:

| Plan | reward | reward − mean (mean=0.4) | advantage `Aᵢ` (std≈0.394) | meaning |
|---|---|---|---|---|
| A | 0.0 | −0.4 | **−1.02** | well below average → push down |
| B | 0.5 | +0.1 | **+0.25** | a bit above → push up a little |
| C | 0.1 | −0.3 | **−0.76** | below average → push down |
| D | 1.0 | +0.6 | **+1.52** | well above → push up hard |

The advantage is just "how many standard deviations above/below the batch average
this answer scored." Positive = make it more likely; negative = less likely.

### Step 4 — Build the loss (policy gradient + KL)
For each completion the code computes the per-token log-probabilities under the
current policy (`lp`, **with gradient**) and under the frozen base model (`ref`,
**no gradient**), then:

```
pg     = −(Aᵢ · mean_over_tokens(lp))          # policy-gradient term
kl     = mean_over_tokens( exp(ref−lp) − (ref−lp) − 1 )   # KL penalty (k3 estimator)
lossᵢ  = (pg + kl_beta · kl) / G
```
Summed over the G completions. (`kl_beta = 0.02` by default.)

What `pg = −(Aᵢ · mean logπ)` does when minimized:
- If `Aᵢ > 0` → minimizing the loss **increases** `logπ` → the model becomes
  **more likely** to produce that answer.
- If `Aᵢ < 0` → it **decreases** `logπ` → **less likely**.
- The size of the nudge scales with `Aᵢ`, so plan D (A=+1.52) moves the weights
  ~6× harder than plan B (A=+0.25), and plan A/C get pushed the other way.

`mean_over_tokens(lp)` means the term is **length-normalized** (averaged over the
answer's tokens), so long answers aren't unfairly favored.

### Step 5 — One gradient step
```
loss.backward(); clip_grad_norm_(..., grad_clip=1.0); optimizer.step()
```
That's it — **one RL round = one such update on one adapter.**

---

## 5. The KL penalty, explained simply

The `kl_beta · kl` term is a **leash**. `kl` measures how far the trained model's
word-probabilities have drifted from the **original base model's**. The penalty
grows as it drifts, so it keeps the policy from wandering into degenerate text
(gibberish, repetition, broken JSON) just to chase reward.

- `ref − lp` = how much *more* likely the base model thought each token was,
  compared to the current policy.
- `exp(ref−lp) − (ref−lp) − 1` is the **k3 estimator** — a cheap, low-variance,
  always-non-negative estimate of the KL divergence per token.

**Important interaction with zero-advantage:** when all G rewards tie, every
`Aᵢ = 0`, so `pg = 0` and the loss is *only* the KL term. Minimizing pure KL drags
the adapter **back toward the untrained base** — i.e. it *un-learns*. That's the
attacker-collapse mechanism we fixed: `grpo_step` now **skips the step entirely**
on a fully-degenerate group (`skip_zero_advantage`) and can **re-roll once** at a
higher temperature first (`resample_on_zero_advantage`). See §10.

---

## 6. How one RL round is implemented

A round is orchestrated by `_step_round` in [rl/schedule.py](rl/schedule.py).
Here is the end-to-end flow:

```
1. ctx = env.begin_round()              # consume one CURRICULUM slot -> this round's
                                        #   defense algorithm + exact poison quota
                                        #   (rl/curriculum.py; both held for a 10-round
                                        #   block), build the honest client updates
2. turn = AttackerTurn(...) or DefenderTurn(...)   # turns.py — freeze the opponent,
                                        #   build the learner's prompt
3. stats = grpo_step(policy, learner, optimizer, turn, ...)   # grpo.py:26
      ├─ generate G candidate answers              (policy.generate)
      ├─ reward each   → turn.reward(c)            (turns.py → rewards.py)
      ├─ advantages    → group_advantages(rewards) (rewards.py:119)
      └─ one gradient step on the learner's adapter
4. best = argmax(stats["rewards"])      # pick the highest-scoring candidate
5. info = turn.commit(best)             # turns.py → env.commit (env.py:165):
                                        #   FedAvg the un-flagged clients, update the
                                        #   global model, measure new accuracy
6. success = committed_success(...)     # switch.py — did the learner WIN this round?
7. _log_round(...)                      # write logs/round_data/round_NNN.json
```

Key points:

- **Scoring vs committing are different.** Steps 3's `turn.reward` runs on a
  *throwaway* copy of the model (`env.evaluate_updates`, [rl/env.py:160](rl/env.py)) —
  it does **not** change the real model. Only step 5 (`commit`) changes it. That's
  why an attacker round shows G+1 model evaluations in the log (G rehearsals + 1
  commit) and a defender round shows just 1 (scoring a verdict is pure counting).
- **The opponent is frozen.** Inside a turn the *other* agent only generates
  (never receives gradients), which is what makes the learner's reward a clean
  "best response" score. The freeze is enforced simply by passing only the
  learner's adapter+optimizer to `grpo_step`.
- **One round = one gradient update = (usually) one committed FL round.** The
  `Round N` you see in logs is this RL round (offset by the 20 Phase-1 honest
  rounds). See [HOWATTACKDEFEND.md](HOWATTACKDEFEND.md) §7.5 for decoding the log line.
- **Who learns each round** is decided by the schedule (`best_response` by
  default): train one agent until it wins, freeze it, switch. See `rl/schedule.py`
  and [HOWATTACKDEFEND.md](HOWATTACKDEFEND.md) §6.

---

## 7. The policy functions (`rl/policy.py`)

`LLMPolicy` ([rl/policy.py:35](rl/policy.py)) is the only GPU-heavy module. Its
design is **"two checkpoints on one brain"**:

- **One** frozen **Qwen2.5-3B-Instruct** base (bf16 LoRA by default; 4-bit
  QLoRA optional via `rl.load_in_4bit`), loaded once.
- **Two LoRA adapters** over it — `"attacker"` and `"defender"`. A LoRA adapter is
  a small set of trainable low-rank matrices added to the frozen base; it's like a
  lightweight "personality patch." Two adapters = two independently-trained
  policies that share one big base model (huge memory saving vs two full models).

### The four things GRPO needs from the policy

**(a) `set_adapter(name)` / `disable_adapter()`** — [policy.py:119](rl/policy.py)
Switch which "personality" is active. `set_adapter("attacker")` makes the attacker
the live policy; `disable_adapter()` exposes the **bare base model**, which is
used as the KL reference.

**(b) `generate(adapter, system, user, n, temperature, max_new_tokens)`** — [policy.py:219](rl/policy.py)
Sample `n` answers (no gradient). Two paths:
- `_fast_generate` ([policy.py:241](rl/policy.py)) — the default: standard
  Transformers generation **with a KV cache** (fast). It flips the model to
  `eval()` during generation (so gradient-checkpointing doesn't disable the cache)
  then back to `train()` for the backward pass.
- `_manual_generate` ([policy.py](rl/policy.py)) — a fallback no-cache
  decoder (slower, O(L²)) used automatically if the fast path errors on this
  Unsloth/Transformers combo. In both paths `temperature=0` → greedy (argmax);
  `>0` → **untruncated** sampling from the tempered softmax.

> **Both paths must sample the exact distribution the loss differentiates.**
> Sampling shape comes from `_sampling_config`, a *fresh* `GenerationConfig` — never
> merged with `model.generation_config`, because Qwen2.5-Instruct ships chat-tuned
> defaults (`top_k: 20`, `top_p: 0.8`, `repetition_penalty: 1.05`) that HF applies to
> anything the caller does not override. Rollouts used to be drawn from a truncated,
> history-dependent distribution while the loss scored the untruncated softmax; a
> repetition penalty is especially wrong here because the attacker's output is
> repetitive JSON. Every warper is now explicitly disabled, so **temperature is the
> only shaping** — and the log-prob passes below are told what it was.

Each `generate` also records, for the rollouts it just produced:
`last_generation_ids` (the exact sampled completion tokens, cut at the first EOS)
and `last_generation_completed`. Both are overwritten by the next `generate` —
including the frozen opponent's during reward scoring — so `grpo_step` captures
them *before* calling `turn.reward()`.

**(c) `policy_token_logprobs(adapter, system, user, completion, completion_ids=, temperature=)`**
Re-run the chosen adapter over `prompt + completion` **with gradients** and return
the log-probability of each completion token. This is the differentiable `lp` in
the GRPO loss — the term we actually backprop through.

**(d) `reference_token_logprobs(system, user, completion, completion_ids=, temperature=)`**
Same thing but under `disable_adapter()` (the base model) and **no gradient** —
this is the `ref` used by the KL penalty. Same ids and temperature as (c), so the
KL compares the two distributions token-for-token.

> Why recompute log-probs separately from generation? Because generation is done
> with `no_grad` for speed; to get a gradient we must re-run a forward pass over
> the produced tokens **with** grad enabled. Since the answer was just sampled from
> this same policy, the importance ratio is 1 — which is why GRPO here needs **no
> PPO-style clipping** (see §10). That identity is only true if we score *what was
> actually sampled*, hence two requirements:
>
> * **`completion_ids`, not the decoded text.** GRPO passes the sampled token ids
>   straight back in. Re-tokenizing the decoded string is only approximately
>   inverse — BPE round-tripping is not guaranteed to reproduce the ids, and these
>   completions are JSON full of digits, decimals and whitespace runs, exactly where
>   re-merge differs. Any mismatch means differentiating a sequence the policy never
>   produced. (Text re-tokenization survives as a fallback for the frozen inference
>   backends, which cannot hand back ids.)
> * **`temperature` = the sampling temperature.** The policy being optimized is
>   `softmax(logits / T)`. Scoring at a fixed `T = 1` was harmless only while
>   `rl.temperature` was exactly 1.0 — but the zero-advantage re-roll draws at
>   `resample_temperature` (1.3), and lowering `rl.temperature` would have silently
>   biased every update with no error anywhere.

### Supporting helpers
- `adapter_parameters(name)` ([policy.py:124](rl/policy.py)) — the trainable LoRA
  tensors for one adapter, handed to its `AdamW` optimizer.
- `get_adapter_state` / `set_adapter_state` ([policy.py:134](rl/policy.py)) — CPU
  copy / restore of an adapter's weights. Used to **snapshot** a frozen agent into
  the opponent league and to swap in past snapshots (curriculum).
- `save_adapter` / `load_adapter` ([policy.py:145](rl/policy.py)) — persist/restore
  an adapter to disk (`adapter_model.safetensors` + `adapter_config.json`) for
  checkpointing and resume.
- `PolicyGenerator` ([policy.py:342](rl/policy.py)) — a tiny wrapper that binds the
  policy to **one fixed adapter**, exposing the `generate(system, user, n, temp)`
  interface the turns use for the **frozen opponent**.

---

## 8. The reward functions (`rl/rewards.py`)

These are the deterministic graders. (Full game-level intuition is in
[HOWATTACKDEFEND.md](HOWATTACKDEFEND.md) §2–§3; here's the mechanical summary.)

**`attacker_reward`** ([rl/rewards.py:21](rl/rewards.py)):
```
reward = alpha · clip(drop / target, −0.5, 1.5)   # did accuracy fall?  (alpha=1.0)
       + beta  · stealth                          # did it evade?        (beta=0.5)
       − gamma · malformed_fraction               # was the plan valid?  (gamma=1.0)
```
`drop = prev_accuracy − post_accuracy`. `stealth` is a **continuous** evasion
signal: the mean over poisoned clients of `1 − P(malicious)`, where `P(malicious)`
is derived from the defender's **confidence** (`_soft_malicious_prob`). Continuous
(not 0/1) on purpose — it gives GRPO a usable spread even when the hard flags tie.

**`defender_reward`** ([rl/rewards.py](rl/rewards.py), default `soft_f1`):
a confidence-weighted F1 of "flagged the poisoned clients, spared the honest ones."
Range [0, 1].

**`_soft_malicious_prob`** ([rl/rewards.py](rl/rewards.py)) is what makes both of
those continuous. It prefers a verdict's explicitly calibrated
`p_malicious` and only falls back to reconstructing the probability from
`(is_suspicious, confidence)` when none was supplied. The distinction matters:
`confidence` means *certainty in the verdict just given* (the LLM defender's
contract), whereas every algorithmic defense naturally produces a *suspicion
score* whose decision boundary is not at 0.5 — FLTrust drops at `trust <= 0`,
Multi-Krum/DnC drop a fixed count per round. Feeding such a score through the
reconstruction ran it **backwards** over every un-flagged client, so a poisoned
client the defense had *nearly dropped* scored as maximally stealthy: the attacker
was being trained to sit on the detection boundary. Multi-Krum/DnC scores are
additionally unbounded (and `+inf` for a non-finite client), so clipping them
collapsed `stealth` to a binary and destroyed the group spread the continuous
reward exists to create.

Moving the score into `p_malicious` did not fix that on its own — a raw score in the
right *field* is still a raw score, and the first attempt at this shipped exactly
that: FLTrust reported `1 − ReLU(cos)`, DeFL `votes/L`, Multi-Krum/DnC the cohort
rank. All three fail in ways that are invisible in the logs:

| defense | reported | what went wrong |
|---|---|---|
| FLTrust | `1 − ReLU(cos)` | Cosines are ~0.05 on a model this small (the ~970-parameter image model those runs used; 681 now), so every **accepted** client reported `p ≈ 0.95`. Stealth was ~0.03 whether the attack evaded detection or not — 3% of its configured 0.5 weight. |
| DeFL | `votes/L` | The flag test is `votes >= threshold` with an **adaptive** threshold; on a two-layer model it settles at `votes >= 1`, so a **flagged** client reported `p = 1/2 = 0.5`. One recorded round paid `att_reward = 0.440` — the highest in the sample — for an attack that was fully detected and did no damage. |
| Multi-Krum / DnC | cohort rank | Bounded and monotone, but the ranks are a fixed 0..1 spread every round: the mean is always ~0.5, and a client's `p` moved when *other* clients moved. It carried no information about whether this client was detected. |

`p_malicious` therefore carries a **contract**, stated in `core.types.DetectionVerdict`:

> `p_malicious >= 0.5` if and only if `is_suspicious`

enforced at the producers by `benchmark.defenses.base.boundary_calibrated_p`, which
maps a defense's own score to `0.5 * (1 + tanh(m / s))` where `m` is the signed
distance past *that defense's own boundary* and `s` is the median `|m|` in the round.
The **sign** is absolute, so the hard flag and the soft score can never disagree; the
**magnitude** is round-relative, so the map neither saturates on scores that live at
1e-3 nor collapses on scores that live at 1e9. `tests/test_p_malicious_calibration.py`
asserts the contract, the reward-level consequence (evading must out-score being
caught), and that the accepted side keeps a usable gradient, for all four defenses.

**`group_advantages`** ([rl/rewards.py](rl/rewards.py)): the z-scoring from §4,
plus the degeneracy gate and the `zero_advantage_fraction` signal.

---

## 9. A full annotated round trace

A real attacker-learning round from the logs, annotated:

```
[rl.env]  Round 22: poisoned_ids=[0] (global_acc=0.5621)      ← begin_round: client 0 is poisoner
[aggregation] averaging 2/5 updates → acc 0.6317              ┐
[aggregation] averaging 5/5 updates → acc 0.7259              │ 4 REHEARSALS: each of the G=4
[aggregation] averaging 5/5 updates → acc 0.8052              │ candidate plans is scored on a
[aggregation] averaging 3/5 updates → acc 0.7806             │ throwaway model copy (no commit)
[aggregation] averaging 3/5 updates → acc 0.5596              ┘ ("X/5" = clients left after the
[metrics]  Round 22 tp=0 fn=1 fp=2 ...                            defender's flags)
[rl.schedule] Round 22 [learn=attacker ph=0.1]: acc 0.5621->0.5596 | mean_r=-0.962 zero_adv=0.00 step
                                                              ↑ the 5th aggregation was the COMMIT
                                                                of the best candidate; 0.5596 is the
                                                                real new accuracy
```
- `mean_r = -0.962` is the **group average** of the 4 rewards (the training signal).
- `att_reward` on that line is the reward of the **one committed** plan (different
  number — group-mean vs the chosen one).
- `zero_adv = 0.00` → the 4 plans had a healthy reward spread → a real gradient was
  applied (`step`, not `SKIP`).

Mapped to code: `begin_round` ([env.py:106](rl/env.py)) → `grpo_step`
([grpo.py:26](rl/grpo.py)) does the 4 `turn.reward` scorings ([turns.py](rl/turns.py)
→ `env.evaluate_updates`, [env.py:160](rl/env.py)) → pick argmax → `turn.commit`
→ `env.commit` ([env.py:165](rl/env.py)) → `_log_round` writes the JSON.

---

## 10. Design choices & FAQ

**Why no PPO clipping?** PPO clips the importance ratio because it reuses one batch
of samples for several gradient steps, during which the policy drifts from the one
that generated them. GRPO here is **single-iteration**: we sample, do *one* step,
then resample next round. The sampling policy == the policy being updated, so the
ratio is 1 by construction and there's nothing to clip (see `rl/grpo.py` docstring).

**Why is the policy-gradient term length-normalized (`mean` over tokens)?** So a
long answer isn't rewarded/penalized just for having more tokens; each answer
contributes on equal footing regardless of length.

**What stops the model from collapsing when rewards tie (zero-advantage)?** Three
guards in `grpo_step`:
1. `resample_on_zero_advantage` — re-roll the G answers once at a higher temperature
   (`resample_temperature=1.3`) to try to recover a spread;
2. `skip_zero_advantage` — if still flat, **skip the optimizer step** so the
   KL-only gradient can't drag the model back toward base;
3. the `zero_advantage_fraction` is logged so the monitor can warn you.

**Why two LoRA adapters instead of two models?** Memory and simplicity — one frozen
3B base (bf16 or 4-bit), plus two small adapter weight-sets. Switching "who is playing" is
just `set_adapter(name)`; the KL reference is the same base via `disable_adapter()`.

**Where does `G` come from / can I change it?** `rl.G` in `configs/base.yaml`
(default 4). Bigger G = more stable advantage estimate but more compute per round.
`G=1` is not valid — with one sample there's no group to compare against (no
advantage, no gradient).

**Is the reward learned (could it be gamed)?** No. Rewards are computed from ground
truth (`poisoned_ids`) and the measured test accuracy — a fixed, verifiable
function. This is RLVR; there's no learned reward model to exploit.

---

## 11. Glossary

| Term | Plain meaning |
|---|---|
| **Policy** | The LLM (with an adapter active) that maps a prompt → a probability distribution over answers. |
| **Action** | One sampled answer (an attack plan or a verdict set). |
| **Reward** | A number grading one action (computed from ground truth). |
| **Rollout / completion** | One generated answer. GRPO samples `G` of them per round. |
| **Advantage** | How much better/worse an answer scored than its group's average (z-scored). |
| **Policy gradient** | The weight update that makes high-advantage answers more likely. |
| **GRPO** | Policy gradient that uses the group average as the baseline (no critic network). |
| **KL penalty** | A leash keeping the trained model close to the original base model. |
| **k3 estimator** | The cheap, low-variance formula used to estimate that KL per token. |
| **Zero-advantage** | All G rewards tie → no spread → no gradient → a wasted/​harmful round. |
| **LoRA / QLoRA** | Small trainable low-rank weights added to a frozen (4-bit) base model. |
| **RLVR** | Reinforcement Learning with Verifiable Rewards (reward from ground truth, not a learned judge). |
| **Temperature** | How random the LLM's sampling is (0 = always the top choice; >0 = varied). |
| **RL round** | One full cycle: sample G → score → advantage → one gradient step → commit. |
