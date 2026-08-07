# How the Attacker and Defender Learn (GRPO, G=4)

A plain-English walkthrough of how the two LLMs in this project learn during the
adversarial arms race — with simple worked examples — plus how **stochastic
defender scoring** keeps the attacker's learning alive.

---

## 1. The learning recipe (same for both agents)

Both the attacker and the defender are trained with **GRPO** (Group-Relative
Policy Optimization). Every learning round follows the same 3-step recipe:

1. **Try 4.** The LLM writes **G = 4** different answers for the *same* situation.
2. **Score + average.** Each answer gets a reward. Compute the **average** of the 4.
3. **Nudge.** Push the model *toward* answers that scored **above average**, and
   *away* from those **below average**. The further from average, the bigger the nudge.

The clever bit: the **average of the batch is the yardstick**. We don't need to
know what a "good" score is in absolute terms — we only need to know which of the
4 beat the others. That is what "Group-Relative" means: each answer is judged
relative to its 4-sample group, so no separate value/critic network is needed.

> The exact maths: with rewards `r₁..r₄`, advantage `Aᵢ = (rᵢ − mean) / spread`.
> Positive advantage → make that answer more likely; negative → less likely.
> (See `group_advantages` in `rl/rewards.py` and the update in `rl/grpo.py`.)

---

## 2. Attacker example

**Situation:** client 2 is the poisoner this round. The attacker LLM proposes 4
attack plans (the "rehearsals" you see in the logs):

| # | Attack plan | What happens | Reward | vs avg (0.4) | Model does |
|---|---|---|---|---|---|
| A | scale weights ×10 | too obvious → defender flags it → poison removed | **0.0** | −0.4 | ⬇️ discourage |
| B | add tiny noise | subtle, slips by, small accuracy drop | **0.5** | +0.1 | ⬆️ mild push |
| C | flip all signs | huge outlier → flagged | **0.1** | −0.3 | ⬇️ discourage |
| D | scale ×2 + tiny noise | slips by **and** drops accuracy a lot | **1.0** | +0.6 | ⬆️⬆️ strong push |

Average = (0.0 + 0.5 + 0.1 + 1.0) / 4 = **0.4**.

So **D** (best, +0.6 above average) gets a big push, **B** a small push, and
**A** / **C** get pushed down. Next time in a similar spot, the attacker is more
likely to write "scale ×2 + tiny noise"-style stealthy attacks. Repeat for
thousands of rounds → it drifts toward attacks that **damage *and* evade**.

> **What the attacker is rewarded for:** causing an accuracy drop **and** staying
> stealthy (not being confidently flagged). See `attacker_reward` in `rl/rewards.py`.

---

## 3. Defender example

**Same round, client 2 is poisoned.** The defender LLM proposes 4 verdict-sets:

| # | Verdict | Outcome | Reward | vs avg (0.6) | Model does |
|---|---|---|---|---|---|
| A | flag {2}, high confidence | caught the poisoner, no false alarms | **1.0** | +0.4 | ⬆️⬆️ strong push |
| B | flag {} (nobody) | missed the poisoner | **0.0** | −0.6 | ⬇️⬇️ strong discourage |
| C | flag {1, 2} | caught 2, but wrongly accused honest client 1 | **0.6** | 0.0 | ➡️ no change |
| D | flag {2}, low confidence | caught it, but unsure | **0.8** | +0.2 | ⬆️ mild push |

Average = (1.0 + 0.0 + 0.6 + 0.8) / 4 = **0.6**.

So **A** (flag *exactly* the poisoner, confidently) gets the big push; **B**
(missed it) gets strongly discouraged; **C** (over-flagged an innocent) is only
average, so no push; **D** is mildly rewarded. The defender drifts toward
**"flag the real outlier confidently, don't miss it, don't falsely accuse
anyone."**

> **What the defender is rewarded for:** a confidence-weighted F1 score — catch
> the bad client (recall), spare the good ones (precision), and be confident when
> right. See `defender_reward` (`soft_f1`) in `rl/rewards.py`. Ground truth is
> used **only** to compute the reward; the defender never sees the answer in its
> prompt — it must infer it from the statistical clues (`detector/features.py`).

### Why a defender round shows fewer log lines

Scoring an **attacker** attempt needs a full model evaluation (apply poison →
aggregate → test), so each attacker-learning round shows **G + 1 = 5**
`FedAvg`/accuracy lines (4 rehearsals + 1 commit). Scoring a **defender** attempt
is just *counting* (compare flags to ground truth) — no model evaluation per
attempt — so a defender-learning round shows just **1** `FedAvg`/accuracy line
(the commit). Defender rounds are faster for that reason.

---

## 4. The one failure to watch for: zero-advantage

The whole thing only works if the 4 scores **differ**. If all 4 attempts score
the *same* — e.g. the attacker is so outmatched that every plan gets caught for
**0.0**:

| A | B | C | D | avg |
|---|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

…then every answer is "exactly average," every nudge is **zero**, and the model
learns **nothing** that round. This is the *zero-advantage collapse* that killed
the attacker in the original run (frozen reward ≈ 0, zero-advantage fraction
rising to ~0.65, accuracy-drop and stealth declining).

This is also why **"1 attempt per round" cannot work** with GRPO: with a single
number there is no average to compare against, so there is never a nudge.

Two safeguards address it:

- **Stochastic defender scoring** (Section 5) — makes the 4 attempts get
  *different* reactions, restoring a spread.
- **Skip-the-step guard** — if a round still comes out totally flat, skip the
  update instead of letting it drag the model backwards (`skip_zero_advantage`
  and `resample_on_zero_advantage` in `rl/grpo.py`).

---

## 5. Stochastic defender scoring

This is the fix that keeps the attacker's learning signal alive against a strong
defender.

### First: what "temperature" means for an LLM

When an LLM produces an answer it picks from a list of options with
probabilities. **Temperature** controls how adventurous that pick is:

- **Temperature = 0** → always pick the single most likely option → **same answer
  every time** (deterministic, "greedy").
- **Temperature = 0.7** → sometimes pick a less-likely option → **the answer
  varies** each time, even for the identical question.

A "stochastic defender" just means running the defender LLM at temperature **> 0**
so its verdicts wobble a bit instead of being identical.

### The problem it fixes

The attacker is scored by running each of its 4 plans past the **frozen**
defender and seeing how it reacts (the stealth part of the reward depends on how
*confident* the defender is that the poison is malicious).

The **old** code ran that frozen defender at temperature **0** (greedy). Against
a strong defender it gave the *exact same* confident verdict to all 4 plans:

| Plan | Defender's verdict (greedy, temp = 0) | stealth | reward |
|---|---|---|---|
| A | "malicious, confidence 0.9" | 0.05 | 0.05 |
| B | "malicious, confidence 0.9" | 0.05 | 0.05 |
| C | "malicious, confidence 0.9" | 0.05 | 0.05 |
| D | "malicious, confidence 0.9" | 0.05 | 0.05 |

All four scores **identical** → average = each score → every "above/below
average" is **zero** → **no nudge, no learning.**

### The fix: vary the defender *while scoring*

Now, when scoring the 4 candidates, the frozen defender runs at
`scoring_opponent_temperature: 0.7`, so it reacts a little differently to each
plan:

| Plan | Defender's verdict (stochastic, temp = 0.7) | stealth | reward |
|---|---|---|---|
| A | "malicious, confidence 0.9" | 0.05 | 0.05 |
| B | "malicious, confidence 0.5" (hesitant) | 0.25 | 0.25 |
| C | "looks benign" (**missed it!**) | 1.00 | 0.60 |
| D | "malicious, confidence 0.7" | 0.15 | 0.15 |

Now the scores **differ** → there is an average (≈ 0.26) and a spread → the
attacker gets a signal: *"plan C made the defender hesitate / slip — do more like
C."* It learns which attacks the defender finds **less convincing**, even when it
still mostly catches them. That is the gradient toward stealthier attacks that
the greedy defender never revealed.

In code, this is the split in `rl/turns.py`: scoring calls
`_apply(text, self.scoring_opp_temp)` (0.7), so each of the 4 candidates gets a
freshly sampled, varied defender verdict.

### The safety detail: commit stays greedy

**Scoring** uses the wobbly defender (to create a learning signal), but the
**commit** — the one plan that actually updates the model, and the round that
decides "did the attack *win*?" — uses the **greedy, temperature-0 defender**:

- `reward()` → stochastic defender (temp 0.7) → for **learning**
- `commit()` → greedy defender (temp 0.0) → for the **real outcome + win/loss check**

So the attacker can't be falsely credited a "win" just because the random
defender happened to fumble during scoring. Wins are always measured against the
real, deterministic defender. (Config: `scoring_opponent_temperature: 0.7` vs
`opponent_temperature: 0.0`.)

### Tradeoff

This adds a bit of noise to the attacker's reward (a plan's score depends partly
on which defender sample it drew). With 4 attempts averaged and the greedy commit
as the source of truth, that is a good trade — a little noise in exchange for
escaping the no-learning trap. To reduce noise, lower
`scoring_opponent_temperature` (e.g. 0.4); for more exploration, raise it.

### Side benefit (symmetry)

It works both ways: when the **defender** is learning, its frozen attacker is
also sampled at this temperature, so the defender sees a *variety* of attacks
each round instead of one fixed one — which stops it from overfitting to a single
attack.

---

## 6. What each agent adapts to (and based on what)

Both agents adapt by GRPO nudging **only their own LoRA adapter weights**. They
never edit each other; they react to a *frozen* opponent. What differs is **what
each one sees** and **what feedback drives the change**.

### The attacker

- **What it sees each round** (`agents/attacker_agent.py` `build_user_prompt`):
  the round number, the **current global accuracy**, the **attack goal** (during
  training, "drop accuracy by 0.10" — fixed, see `attack.goal`), which client IDs
  it controls, how many of them it must poison this round, and **per-layer
  statistics of the benign weights it is about to poison** (shape, mean, std,
  min, max, L2 norm, abs-mean).
- **What it is blind to:** the defender's clues, the defender's verdicts, and
  *why* it got caught. It does **not** see how the defender thinks.
- **What it adapts on:** the **reward only** — "did my plan cause an accuracy
  drop, and did it stay stealthy (not confidently flagged)?" It learns *indirectly*
  which kinds of plans slip past, by comparing the 4 attempts' rewards (Section 2).
  Because it's blind to the defender, stochastic scoring (Section 5) is what gives
  it any visible difference between attempts.
- **Who it adapts *to*:** in `best_response` mode, the **frozen current defender**.
  It keeps adapting until an attack *passes* (a sustained win), then it's frozen.

### The defender

- **What it sees each round** (`detector/features.py` `compute_client_features`):
  for every client, **statistical clues** about its weight update — how large it
  is vs the others (`rel_norm`), whether it points the same direction as the
  majority (`cos_to_median`, `cos_to_mean`), whether its weight signs agree
  (`sign_agreement`), and outlier/collusion signals (`max_pairwise_cos`,
  `dnc_score`).
- **What it is blind to:** the ground-truth answer. It is never told which client
  is actually poisoned — it must infer it from the clues. (Ground truth is used
  *only* to compute the reward.)
- **What it adapts on:** the **soft-F1 reward** — "did I flag the truly poisoned
  client (recall), avoid flagging honest ones (precision), and was I confident
  when right?"
- **Who it adapts *to*:** in `best_response` mode, the **frozen current attacker**.
  It keeps adapting until it reliably catches that attacker, then it's frozen.

### The ratchet (how they co-improve)

```
attacker adapts → beats frozen defender → FREEZE attacker
                       defender adapts → catches frozen attacker → FREEZE defender
attacker adapts again → beats the NEW frozen defender → ...
```

Neither sees the other's internals — each only ever responds to the other's
*frozen behaviour*, measured through rewards. That's why the loop is stable: only
one side moves at a time, and it has a fixed target to climb against.

### How the handoff is decided (when to stop and switch)

The rule in one line:

> **An agent keeps training until it *beats* its frozen opponent a few times in a
> row. Then it stops, freezes itself, and lets the opponent train against that
> stronger version.**

The trigger is *"I just won, convincingly"* — not a timer or a fixed round count.
The logic lives in [rl/switch.py](rl/switch.py).

**Step 1 — what counts as a "win" this round.** Every committed round we check
whether the learner won:

- **Attacker wins** (`attacker_succeeded`) when: its poison **evaded** the defender
  (the poisoned client was *not* flagged) **and** the model **lost accuracy** by at
  least `attacker_min_drop` (default 0.02). → *"my attack got through AND did damage."*
- **Defender wins** (`defender_succeeded`) when: it **caught** the poison
  (TPR ≥ `defender_min_tpr`, default 0.99) **and** didn't **over-flag** honest
  clients (FPR ≤ `defender_max_fpr`, default 0.10). → *"flagged the bad one, spared
  the good ones."*

> The win is judged on the **committed** round against the **real, greedy** opponent
> (not the randomized scoring one from §5), so a win is genuine, not luck.

**Step 2 — the switch decision.** A `PhaseController` tracks a **streak** of
consecutive winning rounds (any non-winning round resets it to 0). It switches when
**both**:

1. the phase has lasted at least `min_phase_rounds` (default 8) — don't hand off on
   a lucky early fluke; and
2. the streak has reached `success_streak` (default 3) — the win repeats, it's not
   a one-off.

Then it **freezes the learner, saves the checkpoint, and switches to the opponent.**

**Worked example** (attacker phase; needs 3 wins in a row, min 8 rounds):

| Round | Won? | Streak | Switch? |
|---|---|---|---|
| 1–5 | mixed | resets to 0 on each loss | no (also < 8 rounds) |
| 6 | ✅ | 1 | no |
| 7 | ❌ | 0 | no (streak broke) |
| 8 | ✅ | 1 | no |
| 9 | ✅ | 2 | no |
| 10 | ✅ | **3** | ✅ **yes** — 3-in-a-row *and* past round 8 → freeze attacker, defender's turn |

**Step 3 — the safety valve.** If an agent simply *can't* win it would train forever,
so there's a ceiling: if a phase reaches `max_phase_rounds` (default 200) **without**
a sustained win, it **switches anyway**. After such a "stuck" phase, the next learner
can be given an **earlier, weaker snapshot** of its opponent to practise against
(`curriculum_on_cap`).

So there are exactly **two reasons an agent stops and hands off:**

| Reason | Meaning | What happens next |
|---|---|---|
| **`success`** (good) | "I beat my frozen opponent 3 rounds running." | Freeze me at this strong version; opponent now trains to beat it. |
| **`cap`** (give-up) | "I trained the max rounds and still can't win." | Switch anyway; next learner may get an easier snapshot so we don't stall. |

In the training log this shows up as:
```
Phase 4 [attacker] ended (success) after 23 rounds — froze attacker
Phase 5 [defender] ended (cap) after 1000 rounds — froze defender
```

**The knobs** (in `configs/base.yaml`):

| Setting | Default | Controls |
|---|---|---|
| `success_streak` | 3 | wins-in-a-row needed to hand off |
| `min_phase_rounds` | 8 | earliest a phase may switch (anti-fluke floor) |
| `max_phase_rounds` | 200 | force-switch ceiling (anti-stall) |
| `attacker_min_drop` | 0.02 | accuracy drop that counts as an attacker win |
| `attacker_min_evaded` | 1.0 | fraction of poisoned clients that must evade |
| `defender_min_tpr` / `defender_max_fpr` | 0.99 / 0.10 | what counts as a defender win |

---

## 7. How to tell if the models are improving

There are two questions, and they use different data:

* **(A) Is each agent learning during its own turns?** → `monitor.py` health report + graphs.
* **(B) Is the *arms race* actually ratcheting up?** → the phase/win dynamics in the training log.

### 7.1 Run the monitor

```bash
python monitor.py                       # reads logs/round_data/, prints report + writes logs/monitor/health.png
python monitor.py --window 50           # bigger 'recent' window for the rolling stats
python monitor.py --log-dir logs/round_data --out logs/monitor/health.png
```

It reads the per-round JSON files (written every round), so you can run it **while
training is still going**.

### 7.2 Per-agent learning health (the report)

The report splits rounds by which agent was learning and trends each metric over
**early third → late third** of that agent's rounds. What "good" looks like:

| Metric (per agent) | Source | Improving looks like |
|---|---|---|
| **GRPO mean-reward** | `train.mean_reward` | late **>** early, slope **> 0** (the learner is winning more) |
| **zero_adv** (recent) | `train.zero_advantage_fraction` | **low** (≪ 0.7). High = attempts tie = no gradient |
| **reward_var** (recent) | variance of mean-reward | **> ~1e-4** (it's still exploring, not frozen) |
| Attacker: **induced acc-drop** | `baseline − test_accuracy` | trending **up** (landing more damage) |
| Attacker: **stealth** | defender confidence on poisoned | trending **up** (evading more) |
| Attacker: **malformed** | `n_malformed` | **low** (< 0.3 — plans still parse) |
| Defender: **TPR** | recomputed from verdicts | **high** (~1.0 — catching the poison) |
| Defender: **FPR** | recomputed from verdicts | trending **down** (fewer false alarms) |
| Defender: **flag_rate** | `(tp+fp)/n` | **between** ~0.05 and ~0.95 (not flag-all / flag-none) |

The report prints a `✓` or `⚠` for each collapse check (reward not improving,
high zero-advantage, do-nothing collapse, malformed degeneration, flag-all/none,
stuck variance). All `✓` = healthy.

### 7.3 The four graphs (`logs/monitor/health.png`)

1. **GRPO mean reward per agent — want UP.** Both lines should *trend* up over
   their own phases. (They won't rise at the same time — see 7.4.)
2. **Defender TPR (up) vs FPR (down).** Healthy defender: TPR high, FPR low.
3. **Zero-advantage fraction — want LOW.** Rising toward 1.0 = the learner is
   losing its gradient (the old attacker-collapse signature).
4. **Attacker induced acc-drop & stealth — want UP.**

### 7.4 Arms-race health — the most important view (and it's new)

Because we now train to *wins* and alternate, the clearest "are they improving?"
signal is in the **training log** (`rl.schedule` lines), not just the per-agent
trends. Look for three things:

- **Phases end in wins and alternate.** You'll see lines like
  `Phase 4 [attacker] ended (success) after 22 rounds — froze attacker`, then a
  defender phase, then attacker again. **`ended (success)` on both sides,
  alternating** = a live ratchet. Repeated **`ended (cap)`** for the *same* agent
  with no `WIN`s = that side is **stuck** (can't beat the frozen opponent).
- **Accuracy sawtooths.** The shared global accuracy should **fall during attacker
  phases** (attacker landing damage) and **recover during defender phases**. A
  flat-high line the whole run = attacker never lands anything (defender
  dominates); flat-low = defender can't recover.
- **Phases get *longer* over time** = genuine co-improvement. If each new attacker
  phase needs **more** rounds to find a win (the defender got harder to beat), and
  each new defender phase needs **more** rounds to catch the attacker (the attacker
  got sneakier), both sides are truly improving. If both keep winning in the
  *minimum* rounds every time, they're trading trivial blows, not climbing.

> The original health report already hints at this: *"healthy training OSCILLATES —
> when one agent trains, its reward rises while the opponent's falls, then they
> swap. Flatlining of BOTH rewards (with low variance) is the real 'stuck' signal."*

### 7.5 Decoding a single training log line

```
Round 23 [learn=attacker ph=4.7 WIN]: acc 0.5596->0.7493 | att_reward=-0.500 def_reward=0.500 | mean_r=-0.359 zero_adv=0.00 step
```

| Field | Meaning |
|---|---|
| `learn=attacker` | who is learning this round |
| `ph=4.7` | phase **4**, round **7** within the phase |
| `WIN` / `...` | did the learner win this committed round (per the success-gate) |
| `acc 0.5596->0.7493` | global accuracy **before → after** the committed move (the true marginal) |
| `mean_r` | group-average reward over the 4 attempts (the training signal) |
| `zero_adv` | 1.0 = all 4 tied (dead round); 0.0 = healthy spread |
| `step` / `SKIP` | `SKIP` = a zero-advantage round was skipped (the guard working) |

A healthy attacker phase shows `WIN` appearing more often as the phase goes on,
`zero_adv` near 0, and mostly `step` (few `SKIP`s).

### 7.6 Other data sources

- **`logs/metrics/` (`metrics.tracker`)** — per-round `tp/fn/fp/tn`, `tpr`, `fpr`,
  `apr` (accuracy-preservation ratio = current/baseline), and `attack_success`
  (was any poisoned client missed). Good for a ground-truth view of detection.
- **`logs/round_data/round_*.json`** — the raw record per round. Beyond what the
  monitor charts, each file now also stores `attack_metadata.phase_index`,
  `phase_round`, `learner_success`, and `train.stepped` / `train.resampled` — useful
  if you want to chart phase outcomes or count skipped rounds yourself.

### 7.7 Quick "is it healthy?" checklist

✅ Healthy:
- both agents' GRPO mean-reward trend up **within their own phases**;
- `zero_adv` stays low, `reward_var` stays non-zero;
- phases **alternate** and mostly end in `success`;
- global accuracy **oscillates** (down in attacker phases, up in defender phases);
- phase lengths **grow** over successive iterations.

🚩 Red flags:
- `zero_adv` climbing toward 1.0, or many `SKIP` lines → a learner lost its signal;
- one agent's phases always `ended (cap)` with no `WIN`s → it can't beat the
  frozen opponent (for the attacker, this is the "needs a foothold" case — raise
  `poison_fraction`, see `configs/base.yaml`);
- accuracy flat for the whole run → no real fight happening;
- attacker `malformed` rate climbing or `flag_rate` pinned at ~0 or ~1 → output
  degeneration / degenerate classifier.

> A caveat on the monitor's `induced acc-drop`: it is measured against the fixed
> Phase-1 baseline (`baseline − test_accuracy`), so during long alternating phases
> it drifts and its whole-run *slope* is less meaningful than it was under the old
> fast-alternation schedule. For the true per-round effect, read the
> `acc X->Y` (before→after) in the training log; for the arms-race picture, use 7.4.

---

## One-sentence summary

The attacker learns *"which poison does the most damage while slipping past the
defender,"* and the defender learns *"which clients to flag (and how
confidently) to catch the poison without falsely accusing honest clients"* —
both by generating **4 attempts**, scoring each, and reinforcing the ones that
beat the group average; **stochastic defender scoring** keeps those 4 attempts
distinguishable so the attacker always has something to learn from.
