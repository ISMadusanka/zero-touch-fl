# Image / Figure Brief for the Report

Each entry is a self-contained "prompt": what the image should show, its purpose,
suggested style, and where it belongs. Items marked **[DONE — TikZ]** already
exist as vector figures in `Chapters/ch_3.tex`; the rest are gaps worth filling,
either as more TikZ diagrams (same style, no external files) or as
separately-generated raster images if you'd rather design them in a tool like
Canva/PowerPoint/draw.io.

---

## Chapter 1 — Introduction

**1.1 System-in-context overview** (optional, sets the scene)
- **Prompt:** A simple, uncluttered diagram of a standard federated learning
  loop — a central server icon connected to ~5–6 client icons in a ring or
  star layout, arrows showing "global model out / update in." Highlight 2 of
  the clients in red/orange with a small "compromised" icon to signal the
  attack surface this report is about, before any of the LLM machinery is
  introduced.
- **Style:** Clean, minimal, conceptual (not technical) — this is the "explain
  it to a non-specialist" figure for the opening pages.
- **Placement:** Chapter 1, §1.1 Background and Motivation.

---

## Chapter 2 — Literature Review

**2.1 Defense taxonomy**
- **Prompt:** A tree or mind-map diagram with "Byzantine-robust FL defenses"
  as the root, branching into five categories: Coordinate/Distance-based
  (median, trimmed mean, Krum/Multi-Krum), Spectral (DnC), Trust-bootstrapped
  (FLTrust), Similarity-based (FoolsGold), Layer-profile-based (DeFL). Each
  leaf node names the method; a one-line mechanism tag under each leaf (e.g.
  "cosine trust vs. clean root set" under FLTrust).
- **Style:** Simple hierarchical diagram, consistent box style with Chapter 3's
  figures if you want visual continuity (rounded boxes, light fill).
- **Placement:** Chapter 2, end of §2.3 Byzantine-Robust and Statistical
  Defenses, as a visual summary before §2.4.

---

## Chapter 3 — Methodology

**3.1 Paradigm timeline** — **[DONE — TikZ, `fig:paradigm-timeline`]**
Four-stage flow: Stage 0 (hardcoded plugins + rule detector + episodic memory)
→ Stage 1 (LLM policies + GRPO + fixed clock) → Stage 2 (stochastic scoring +
success-gated schedule) → Stage 3 (benchmark harness), with the scoped-out
neuron-level attack shown as a dashed side branch.

**3.2 Old (Stage 0) architecture** — **[DONE — TikZ, `fig:old-architecture`]**
Plugin attacker → poisoned/honest updates → hardcoded detector → aggregator
(FLTrust branch + method dispatch), closed by the episodic-memory feedback
loop looping back into both the attacker and the detector.

**3.3 Current training architecture — GAP, not yet drawn**
- **Prompt:** The Stage 2/3 training-time loop, one round: "Honest local
  updates (N=20)" → "Attacker LLM (select ≤ b clients + DSL plan)" →
  "Poisoned ∪ honest updates" → "Feature extractor" → "Defender LLM (verdict +
  confidence)" → "FedAvg over non-flagged clients" → "Updated global model" →
  "Verifiable rewards ($R_{att}$, $R_{def}$)" → "GRPO update (learner only,
  opponent frozen)" → looped back to the top. Same box/arrow visual language as
  Figures 3.1–3.2 for consistency (this is effectively the "successor" diagram
  to Figure 3.2 — worth placing right after it or at the end of §3.2.3 so a
  reader can flip between "before" and "after" on facing content).
- **Style:** TikZ box-and-arrow, matches existing figures. I can generate this
  one directly if you want it added.
- **Placement:** Chapter 3, end of §3.2.3 (Stage 2) or start of §3.4 (RL
  Methodology overview).

**3.4 Attacker action contract — GAP**
- **Prompt:** A narrower, "inside one LLM call" diagram: left side lists the
  attacker's observation inputs (round #, global accuracy, attack goal,
  controllable client IDs, budget, per-client update stats) feeding into a
  single "Attacker LLM" box; right side shows its structured JSON output
  (client selection + ordered operator plan) feeding into a "Deterministic
  interpreter" box, which outputs "Poisoned weights." Optionally annotate the
  interpreter box with 2–3 of the 10 operator names as examples.
- **Style:** Simple two-column input/output diagram, not a loop.
- **Placement:** Chapter 3, §3.3 Attacker Agent Design, near the Observation/
  Action Contract subsections.

**3.5 Reward composition — GAP**
- **Prompt:** A single horizontal "stacked bar" or "term breakdown" diagram
  showing the five components of the attacker's reward as labelled blocks
  summing left to right: `+ accuracy-drop credit`, `+ stealth`,
  `− malformed-plan penalty`, `− extra-client penalty`, `+ diversity bonus`,
  equalling `R_att`. Use `+`/`−` coloring (green for reward-increasing terms,
  red/orange for penalties) so the incentive direction is visually obvious at
  a glance.
- **Style:** Flat, infographic-style bar, not a flowchart — this one reads
  better as a compact designed graphic than as TikZ boxes.
- **Placement:** Chapter 3, §3.5.3 Reward Formulation.

**3.6 Benchmark harness** — **[DONE — TikZ, `fig:benchmark-harness`]**
One committed attack plan fanning out to all seven defenses (FedAvg, Oracle,
FLTrust, Multi-Krum, DnC, DeFL, LLM Defender), each keeping its own global
model, converging into per-defense metrics.

---

## Chapter 4 — Results, Discussion and Conclusion

**4.1 Headline accuracy comparison (bar chart)**
- **Prompt:** A horizontal bar chart, one bar per defense (FedAvg, Oracle,
  LLM Defender, FLTrust, DeFL, DnC, Multi-Krum), bar length = final test
  accuracy from Table 4.1 (0.092, 0.784, 0.092, 0.772, 0.338, 0.787, 0.784).
  Draw a vertical dashed reference line at the 0.782 clean baseline so it's
  immediately visible which defenses sit at/above baseline vs. which
  collapsed. Sort bars ascending or group by outcome (collapsed / partial /
  neutralized) rather than alphabetically, so the pattern reads at a glance.
- **Style:** Plain data bar chart (matplotlib-style), not a diagram — real
  chart from the real numbers, no illustration needed.
- **Placement:** Chapter 4, §4.2/4.3 Measured Results, right after the results
  table.

**4.2 Detection rate vs. realized damage (scatter/quadrant chart)**
- **Prompt:** A scatter plot, x-axis = detection rate (`detect%`), y-axis =
  mean accuracy drop, one point per defense, labelled directly on the plot.
  This is the chart that visually makes Chapter 4's central finding
  obvious: DeFL sits at high x (91%) but also high y (large drop) — an
  outlier in the top-right, away from the expected "high detection → low
  drop" trend that FLTrust/DnC/Multi-Krum/Oracle follow along the bottom.
  Consider shading/annotating the "expected" region (bottom-right: high
  detection, low drop) vs. where DeFL and the LLM Defender actually land.
- **Style:** Clean analytical scatter chart, minimal gridlines, direct point
  labels (avoid a legend if there's room to label points inline).
- **Placement:** Chapter 4, §4.6 Detection Quality Is Not Realized Protection
  — this chart *is* that section's argument in visual form.

**4.3 Accuracy trajectory over rounds (optional, if round-by-round data is
exported)**
- **Prompt:** A line chart, one line per defense, x-axis = round number
  (1–50), y-axis = test accuracy, showing how FedAvg/LLM Defender crash early
  and stay flat near chance level while FLTrust/DnC/Multi-Krum stay near
  baseline and DeFL degrades partway through the run. Only worth producing if
  you have the per-round `history.json`/CSV the benchmark harness writes out;
  otherwise the two charts above already carry the chapter's argument.
- **Style:** Multi-line time series, legend by defense name.
- **Placement:** Chapter 4, §4.3 Measured Results, as a supplement to the
  summary table.

---

## Notes on how to produce these

- Figures marked **[DONE — TikZ]** need nothing further — they compile as
  part of the `.tex` source once you add the `tikz` package (see the comment
  at the top of `Chapters/ch_3.tex`).
- The two "GAP" diagrams in Chapter 3 (3.3 current architecture, 3.4 attacker
  contract) are the same visual style as the two already done — I can
  generate these as TikZ too if you want them added directly to the chapter.
- The Chapter 4 charts (4.1, 4.2, 4.3) are **real data plots**, not
  diagrams — best produced from the actual benchmark numbers with a plotting
  library (e.g. Python/matplotlib) rather than hand-drawn, so the chart is
  guaranteed to match the table exactly. I can generate the plotting code (or
  the charts themselves) once you confirm which of these you want.
- 2.1 (defense taxonomy) and 3.5 (reward composition) are the two best
  candidates for an actual designed/illustrated graphic rather than a TikZ
  box diagram, if you want a more polished, less "textbook" look for the
  report.
