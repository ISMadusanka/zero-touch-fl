# How the Attack Works and How the Defender Learns (GRPO, G=4)

A plain-English walkthrough of the two halves of this system:

* the **attack** — label flipping on a detection-adaptive ladder. It is *not* an
  LLM, has no policy, and learns nothing. It reacts.
* the **defender** — the one trainable agent, an LLM fine-tuned with GRPO.

They are still coupled, and the coupling is the whole point: **the defender's own
verdicts decide how strong the next round's attack is.**

---

## 1. The attack: label flipping on a ladder

A fixed set of insider clients (`attack.poison_client_ids`, default `[0]`) trains
each round on **its own data with some of the labels flipped**. The flip is
symmetric — `y → 9 − y`, so `0↔9, 1↔8, … 4↔5` — the standard label-flipping
attack in the FL-poisoning literature.

Nothing edits the weights. The poisoned client runs the *same* local training, on
the *same* examples, for the *same* number of epochs, at the *same* learning rate
as it would have honestly. The only difference on the wire is the labels it
optimized against.

> **Why the poison is in the DATA, not the weights.** A weight-space edit (scale
> ×2, flip the signs, add noise) is a post-hoc transform of a vector the client
> already computed, so its magnitude is a free parameter — an attacker can dial it
> to whatever a detector tolerates, and the resulting update need not resemble
> anything a real client would ever send. A label flip has exactly **one** knob:
> how many of the client's own examples are mislabelled. Whatever comes out is a
> genuine SGD trajectory of a real (if wrong) objective, so it lands inside the
> honest update distribution by construction. That is what makes catching it a
> meaningful test of a detector rather than a test of outlier arithmetic.

### The ladder

How many labels get flipped is not fixed. It walks:

```
start:  flip 100% of the poisoned client's data for this round
CAUGHT      -> step down one notch (10% by default):  100% -> 90% -> 80% -> ...
NOT CAUGHT  -> HOLD. Send the same level again, unchanged.
CAUGHT at the floor (50%) -> RESET to 100% and start descending again.
```

On a client holding **1000** examples in a round that is exactly:

| Round | Sent | Caught? | Next |
|---|---|---|---|
| 1 | 1000 labels flipped (100%) | ✅ caught | 900 |
| 2 | 900 (90%) | ✅ caught | 800 |
| 3 | 800 (80%) | ❌ missed | **800** (hold) |
| 4 | 800 (80%) | ❌ missed | **800** (hold) |
| 5 | 800 (80%) | ✅ caught | 700 |
| … | … | … | … |
| 9 | 500 (50%, the floor) | ✅ caught | **1000** (reset) |

So the attack **probes for the largest poison the defender still misses**, backs
off whenever it is caught, and sits on whatever level is working for as long as it
keeps working.

**Why there is a floor.** Below about half the client's data, the honest gradient
dominates its own update: the round stops being an attack and becomes noise, and
neither "caught" nor "missed" carries information any more. `floor_fraction: 0.5`
stops the descent there.

**Why it resets instead of parking.** If the ladder stopped at the floor forever,
the defender would spend the rest of the run being trained on one static, barely
detectable setting. The reset forces it to keep re-proving itself across the whole
range of attack strengths, round after round.

The code is `agents/label_flip_attacker.py` (the ladder) and `data/label_flip.py`
(the flip itself). The ladder advances **exactly once per committed round** —
never per scored rollout, or the attack schedule would depend on `rl.G`, a
sampling hyperparameter, instead of on whether the defense actually caught
anything.

### What "caught" means

With several poisoned clients, `attack.schedule.caught_rule` decides how many the
defense must flag for the round to count as caught:

| Rule | Steps down when… | Effect |
|---|---|---|
| `all` (default) | **every** poisoned client was flagged | one surviving insider = a successful round, so the ladder holds |
| `majority` | more than half were flagged | tolerates the odd unlucky flag |
| `any` | at least one was flagged | harshest; fastest descent |

Ground truth is what actually shipped flipped labels, not the configured set: a
client whose level rounded to zero flips sent an honest update, and holding the
defense responsible for not flagging it would step the ladder on a detection that
could not have happened.

---

## 2. The learning recipe (the defender)

The defender is trained with **GRPO** (Group-Relative Policy Optimization). Every
round follows the same 3-step recipe:

1. **Try 4.** The LLM writes **G = 4** different verdict-sets for the *same*
   cohort of client updates.
2. **Score + average.** Each verdict-set gets a reward. Compute the **average**.
3. **Nudge.** Push the model *toward* the answers that scored **above average**,
   and *away* from those **below average**. The further from average, the bigger
   the nudge.

The clever bit: the **average of the batch is the yardstick**. We never need to
know what a "good" score is in absolute terms — only which of the 4 beat the
others. That is what "Group-Relative" means, and it is why no separate
value/critic network is needed.

> The exact maths: with rewards `r₁..r₄`, advantage `Aᵢ = (rᵢ − mean) / spread`.
> Positive advantage → make that answer more likely; negative → less likely.
> (See `group_advantages` in `rl/rewards.py` and the update in `rl/grpo.py`.)

### Worked example

**Situation:** client 0 flipped 80% of its labels this round.

| # | Verdict | Outcome | Reward | vs avg (0.6) | Model does |
|---|---|---|---|---|---|
| A | flag {0}, high confidence | caught the poisoner, no false alarms | **1.0** | +0.4 | ⬆️⬆️ strong push |
| B | flag {} (nobody) | missed the poisoner | **0.0** | −0.6 | ⬇️⬇️ strong discourage |
| C | flag {0, 1} | caught 0, but wrongly accused honest client 1 | **0.6** | 0.0 | ➡️ no change |
| D | flag {0}, low confidence | caught it, but unsure | **0.8** | +0.2 | ⬆️ mild push |

Average = (1.0 + 0.0 + 0.6 + 0.8) / 4 = **0.6**.

**A** gets the big push; **B** is strongly discouraged; **C** (over-flagged an
innocent) is only average, so no push; **D** is mildly rewarded. The defender
drifts toward *"flag the real outlier confidently, don't miss it, don't falsely
accuse anyone."*

Then the committed round's verdicts go to the ladder. If it flagged client 0, the
next round's attack drops to 70%.

> **What the defender is rewarded for:** a confidence-weighted F1 — catch the bad
> client (recall), spare the good ones (precision), and be confident when right.
> See `defender_reward` (`soft_f1`) in `rl/rewards.py`. Ground truth is used
> **only** to compute the reward; the defender never sees the answer in its prompt
> — it must infer it from the statistical clues (`detector/features.py`).

---

## 3. The one failure to watch for: zero-advantage

The whole thing only works if the 4 scores **differ**. If all 4 verdict-sets score
the *same* — e.g. all four flag nobody:

| A | B | C | D | avg |
|---|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

…then every answer is "exactly average," every nudge is **zero**, and the model
learns **nothing** that round. Worse, the loss then reduces to the KL term alone,
which pulls the adapter *back toward the base model* — active un-learning.

This is also why **"1 attempt per round" cannot work** with GRPO: with a single
number there is no average to compare against, so there is never a nudge.

Two safeguards address it (`rl/grpo.py`):

- **`resample_on_zero_advantage`** — re-draw the whole group once at a higher
  temperature to try to recover a spread.
- **`skip_zero_advantage`** — if it is still flat, skip the optimizer step
  entirely rather than stepping on a pure KL-to-base signal.

"Flat" is decided by `min_reward_spread` (0.02), not by exact equality: the reward
is a soft F1 over per-client confidences, so two behaviourally identical
verdict-sets routinely differ by a hair. Treating that as signal meant z-scoring
sampling noise up to full-magnitude advantages — GRPO confidently reinforcing a
coin flip.

**Where the spread comes from.** All G rollouts classify the *same* cohort of
updates, so the only source of variation is the defender's own sampling
temperature (`rl.temperature`, default 1.0). At temperature 0 every rollout is
identical, the advantages collapse, and no step is ever taken.

---

## 4. What the defender adapts to, and based on what

- **What it sees each round** (`detector/features.py` `compute_client_features`):
  for every client, **statistical clues** about its weight update — how large it is
  vs the others (`rel_norm`), whether it points the same direction as the majority
  (`cos_to_median`, `cos_to_mean`), whether its weight signs agree
  (`sign_agreement`), and outlier/collusion signals (`max_pairwise_cos`,
  `dnc_score`).
- **What it is blind to:** the ground-truth answer, and the ladder's current level.
  It is never told which client is poisoned or how hard — it must infer both from
  the clues.
- **What it adapts on:** the **soft-F1 reward** — "did I flag the truly poisoned
  client (recall), avoid flagging honest ones (precision), and was I confident when
  right?"
- **Who it adapts *to*:** an attack whose strength **its own past performance set**.

### The coupling

```
defender catches the attack  ->  ladder backs off a notch  ->  the poison gets
                                 subtler and harder to see
defender misses it           ->  ladder holds              ->  the same level is
                                 sent again until it is caught
defender catches it at 50%   ->  ladder resets to 100%     ->  full-strength poison
                                 returns; the climb repeats
```

This is a real adversarial curriculum, and it is generated by the defender's own
failures. A fixed poison rate cannot do that: it is either always easy (and the
defender learns nothing after the first few rounds) or always impossible (and the
defender never gets a gradient).

It also gives you a **free diagnostic**: the ladder only moves when the defender
lands detections. A run whose flip fraction sits at 100% for hundreds of rounds is
a defender that is catching nothing, however healthy its reward curve looks.

---

## 5. Phases (when the defender is checkpointed)

There is only one learner, so nothing alternates. What survives from the old
two-agent schedule is the **phase** structure, and it still earns its place: the
opponent is not static, so freezing the defender when it sustains a win captures
it at each attack level it has actually mastered.

**Step 1 — what counts as a "win" this round.** `defender_succeeded`
(`rl/switch.py`): it **caught** the poison (TPR ≥ `defender_min_tpr`, default 0.99)
**and** did not **over-flag** honest clients (FPR ≤ `defender_max_fpr`, default
0.10). On a round with no effective poison (a level that rounded to zero flips) it
wins by **staying quiet** — TPR is undefined there, and scoring a flawless round 0
would train it to invent detections.

**Step 2 — the switch decision.** A `PhaseController` tracks a **streak** of
consecutive winning rounds (any non-winning round resets it to 0). The phase ends
when **both**:

1. it has lasted at least `min_phase_rounds` (default 3) — don't freeze on a lucky
   early fluke; and
2. the streak has reached `success_streak` (default 3) — the win repeats.

Then the adapter is **snapshotted and saved**, and a fresh phase begins.

**Worked example** (needs 3 wins in a row, min 3 rounds):

| Round | Won? | Streak | Phase ends? |
|---|---|---|---|
| 1 | ✅ | 1 | no |
| 2 | ❌ | 0 | no (streak broke) |
| 3 | ✅ | 1 | no |
| 4 | ✅ | 2 | no |
| 5 | ✅ | **3** | ✅ **yes** — freeze, checkpoint, next phase |

**Step 3 — the safety valve.** If the defender simply *can't* win it would train
forever, so `max_phase_rounds` (default 200) ends the phase anyway.

In the training log:
```
Phase 4 [defender] ended (success) after 7 rounds — froze defender
Phase 5 [defender] ended (cap) after 200 rounds — froze defender
```

**The knobs** (`configs/base.yaml`, `rl:`):

| Setting | Default | Controls |
|---|---|---|
| `success_streak` | 3 | wins-in-a-row needed to freeze |
| `min_phase_rounds` | 3 | earliest a phase may end (anti-fluke floor) |
| `max_phase_rounds` | 200 | force-end ceiling (anti-stall) |
| `defender_min_tpr` / `defender_max_fpr` | 0.99 / 0.10 | what counts as a win |

And the attack's own knobs (`attack.schedule`):

| Setting | Default | Controls |
|---|---|---|
| `start_fraction` | 1.0 | the top of the ladder — where each cycle begins |
| `step_fraction` | 0.1 | how far one detection backs the attack off |
| `floor_fraction` | 0.5 | the bottom; caught here resets instead of descending |
| `caught_rule` | `all` | how many poisoned clients must be flagged to step down |

---

## 6. How to tell if it is working

Two questions, two data sources:

* **(A) Is the defender learning?** → `monitor.py` health report + graphs.
* **(B) Is the attack still an attack?** → the ladder walk and the induced drop.

### 6.1 Run the monitor

```bash
python monitor.py                       # reads logs/round_data/, prints report + writes logs/monitor/health.png
python monitor.py --window 50           # bigger 'recent' window for the rolling stats
python monitor.py --log-dir logs/round_data --out logs/monitor/health.png
```

It reads the per-round log (appended every round), so you can run it **while
training is still going**.

### 6.2 What "good" looks like

| Metric | Source | Improving looks like |
|---|---|---|
| **GRPO mean-reward** | `train.mean_reward` | late **>** early, slope **> 0** |
| **zero_adv** (recent) | `train.zero_advantage_fraction` | **low** (≪ 0.7). High = attempts tie = no gradient |
| **reward_var** (recent) | variance of mean-reward | **> ~1e-4** (still exploring, not frozen) |
| **TPR** | recomputed from verdicts | **high** (~1.0 — catching the poison) |
| **FPR** | recomputed from verdicts | trending **down** (fewer false alarms) |
| **flag_rate** | `(tp+fp)/n` | **between** ~0.05 and ~0.95 (not flag-all / flag-none) |
| **flip fraction** | `attack_metadata.flip_fraction` | a **saw-tooth**: walks down, hits 50%, resets |
| **induced acc-drop** | `attack_metadata.induced_drop` | materially **> 0** at high ladder levels |

The report prints a `✓` or `⚠` for each collapse check. All `✓` = healthy.

### 6.3 The four graphs (`logs/monitor/health.png`)

1. **Defender GRPO mean reward — want UP.**
2. **TPR (up) vs FPR (down).**
3. **Zero-advantage fraction — want LOW.** Rising toward 1.0 = the learner is
   losing its gradient.
4. **The ladder (raw) vs induced acc-drop.** The flip fraction is plotted
   **unsmoothed** — the saw-tooth *is* the signal, and a rolling mean would hide
   exactly the case you need to see (a ladder that stopped moving).

### 6.4 Reading the two together

> The defender's reward should rise **while** the flip fraction walks down.

- A rising reward at a flip fraction pinned to **100%** means it is winning the
  easiest version of the problem — it has not been asked anything harder yet.
- A **falling** reward right after a reset is expected: the ladder just handed it
  full-strength poison again after it had adapted to 50%.
- Phases getting **longer** over successive cycles is genuine improvement: it takes
  more rounds to reach a sustained win because the level it has to beat is lower.

### 6.5 Decoding a single training log line

```
Round 23 [ph=4/7 def=llm WIN]: flip=80%(160 labels) acc 0.9012->0.8455
  (clean_ref=0.9040 drop=+0.0585 eff=+0.59) | def_reward=0.812 ladder=step_down->70%
  | grpo_loss=0.0041 mean_r=0.74 spread=0.21 zero_adv=0.00(first=0.00) step
```

| Field | Meaning |
|---|---|
| `ph=4/7` | phase **4**, round **7** within the phase |
| `WIN` / `...` | did the defender win this committed round |
| `flip=80%(160 labels)` | the ladder level this round was sent at |
| `acc X->Y` | global accuracy at round start → after the committed aggregate |
| `clean_ref` | what the SAME clients on their REAL labels would have scored |
| `drop` | `clean_ref − post`: what the flipped labels actually cost |
| `eff` | that drop on the goal's scale (1.0 = hit `target_accuracy_drop`) |
| `ladder=…->70%` | the transition, and next round's level |
| `mean_r` / `spread` | group-average reward and its span over the 4 attempts |
| `zero_adv` | 1.0 = all 4 tied (dead round); 0.0 = healthy spread |
| `step` / `SKIP-degenerate` | was a gradient actually applied |

### 6.6 Other data sources

- **`logs/metrics/`** — per-round `tp/fn/fp/tn`, `tpr`, `fpr`, `apr`
  (accuracy-preservation ratio) and `attack_success` (did the round clear the damage
  bar).
- **`logs/round_data/rounds.jsonl`** — the raw record per round, one JSON object per
  line. Beyond what the monitor charts it carries `attack_metadata.ladder` (the full
  transition), `flip_plan` (per-client flip counts), `phase_index`, `phase_round`,
  `learner_success` and `train.stepped` / `train.resampled`.
- **`logs/debug.json`** (`python main.py --debug`) — every prompt, every raw
  completion, every per-client update, for a few fully-detailed rounds.

### 6.7 Quick "is it healthy?" checklist

✅ Healthy:
- GRPO mean-reward trends up; `zero_adv` stays low; `reward_var` stays non-zero;
- the **flip fraction saw-tooths** — it walks down and resets, repeatedly;
- `induced_drop` is materially positive at the high ladder levels;
- phases mostly end in `success`, and get longer over successive cycles.

🚩 Red flags:
- `zero_adv` climbing toward 1.0, or many `SKIP-degenerate` lines → the defender
  lost its signal. Check `rl.temperature` is > 0.
- **the ladder never steps** → the defender is catching (almost) nothing, so it is
  being trained against one static attack level forever.
- **`induced_drop` ≈ 0 even at 100% flipped** → the poison is not reaching the
  aggregate, so a high TPR means the defender learned to detect a formality. Check
  the poisoned client's share of the federation and `fl.client_round_fraction`.
- `flag_rate` pinned at ~0 or ~1 → degenerate classifier (flag-none / flag-all).

---

## One-sentence summary

The attack flips labels on a ladder that backs off whenever it is caught and
resets once it bottoms out, and the defender learns *"which clients to flag, and
how confidently, to catch the poison without falsely accusing honest ones"* by
generating **4 verdict-sets**, scoring each, and reinforcing the ones that beat the
group average — so the difficulty of what it faces next is set by how well it did
just now.
