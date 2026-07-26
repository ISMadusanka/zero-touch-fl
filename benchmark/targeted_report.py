"""Render TARGETED-benchmark results.

The untargeted report (``benchmark/report.py``) answers "how much accuracy did the
attack cost?". That question is the wrong one here: a targeted attack that works
perfectly barely moves overall accuracy — on 10 balanced classes, destroying one
costs only ~10 points — so a report that shows only overall accuracy makes a
complete success look like a near-miss.

So this report leads with the two numbers the targeted experiment is actually
about, per defense:

  * the TARGET class's recall (should collapse when the attack works), and
  * every other class's recall (should stay where the clean model had it).

The per-class matrix underneath shows the same thing class by class, against a
``clean`` reference row, so "only the target broke" is visible directly rather
than inferred.
"""
import csv
import json
import os

# Headline table: (summary-key, header, format).
_COLS = [
    ("defense", "defense", "{}"),
    ("detection_rate", "detect%", "{:.1%}"),
    ("fpr", "FPR", "{:.1%}"),
    ("f1", "F1", "{:.2f}"),
    ("target_recall_final", "TGT_final", "{:.3f}"),
    ("target_recall_mean", "TGT_mean", "{:.3f}"),
    ("others_recall_mean", "others_mean", "{:.3f}"),
    ("mean_collateral", "collat", "{:.3f}"),
    ("final_accuracy", "overall", "{:.3f}"),
    ("attack_success_rate", "atk_thru", "{:.1%}"),
    ("targeted_success_rate", "tgt_succ", "{:.1%}"),
]


def _cell(value, fmt: str) -> str:
    return "n/a" if value is None else fmt.format(value)


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def line(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out = [line(rows[0]), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows[1:]]
    return "\n".join(out)


def format_summary_table(summaries: list[dict]) -> str:
    rows = [[h for _, h, _ in _COLS]]
    for s in summaries:
        rows.append([_cell(s.get(key), fmt) for key, _, fmt in _COLS])
    return _table(rows)


def format_per_class_table(summaries: list[dict], label: int, key: str = "per_class_final") -> str:
    """Per-class recall matrix: one row per defense, one column per class.

    The target class's column is marked with ``*`` so the collapse is easy to find,
    and a ``clean`` row gives the unpoisoned reference every column is judged
    against. ``key`` selects the final-round (default) or mean-over-rounds view.
    """
    with_classes = [s for s in summaries if s.get(key)]
    if not with_classes:
        return "(no per-class data)"
    n = len(with_classes[0][key])
    header = ["defense"] + [f"*{c}" if c == label else str(c) for c in range(n)]
    header += ["TARGET", "others"]
    rows = [header]

    clean = with_classes[0].get("per_class_clean")
    if clean:
        others = [v for i, v in enumerate(clean) if i != label]
        rows.append(
            ["clean"] + [f"{v:.3f}" for v in clean]
            + [f"{clean[label]:.3f}", f"{(sum(others) / len(others)) if others else 0:.3f}"]
        )

    for s in with_classes:
        vals = s[key]
        others = [v for i, v in enumerate(vals) if i != label]
        rows.append(
            [str(s.get("defense", "?"))] + [f"{v:.3f}" for v in vals]
            + [f"{vals[label]:.3f}", f"{(sum(others) / len(others)) if others else 0:.3f}"]
        )
    return _table(rows)


def _verdict_line(summaries: list[dict], label: int) -> str:
    """One plain-English sentence about the undefended (fedavg) outcome.

    ``fedavg`` is the no-defense world, so it isolates what the ATTACK itself did,
    with no defense masking or amplifying it.
    """
    base = next((s for s in summaries if s.get("defense") == "fedavg"), None)
    if not base or base.get("target_recall_final") is None:
        return ""
    tgt_clean = base.get("target_recall_clean", 0.0)
    tgt = base["target_recall_final"]
    oth_clean = base.get("others_recall_clean", 0.0)
    oth = base.get("others_recall_final", 0.0)
    drop = tgt_clean - tgt
    collat = oth_clean - oth
    ok = drop >= 0.5 * max(tgt_clean, 1e-9) and collat <= 0.05
    verdict = "TARGETED ATTACK WORKED" if ok else "targeted attack did NOT fully land"
    return (f"\n{verdict} (undefended / fedavg): class {label} recall "
            f"{tgt_clean:.3f} -> {tgt:.3f} (lost {drop:+.3f}); "
            f"other classes {oth_clean:.3f} -> {oth:.3f} (lost {collat:+.3f})\n")


def render(summaries: list[dict], n_rounds: int, baseline_accuracy: float,
           label: int, out_dir: str | None = None, goal: dict | None = None,
           n_poisoners: int | None = None) -> str:
    title = (f"TARGETED POISONING BENCHMARK — {n_rounds} attack rounds  "
             f"(goal: misclassify label {label}; clean overall acc = {baseline_accuracy:.3f}"
             f"{f'; poisoned clients per round = {n_poisoners}' if n_poisoners is not None else ''})")
    bar = "=" * len(title)
    text = f"{bar}\n{title}\n{bar}\n{format_summary_table(summaries)}\n"
    text += _verdict_line(summaries, label)
    text += (f"\nPER-CLASS RECALL — FINAL model  (* = attack target, class {label})\n"
             f"{format_per_class_table(summaries, label, 'per_class_final')}\n")
    text += (f"\nPER-CLASS RECALL — MEAN over all {n_rounds} rounds\n"
             f"{format_per_class_table(summaries, label, 'per_class_mean')}\n")
    text += (
        "\nLegend:\n"
        f"  TGT_*     recall on the ATTACKED class ({label}) — LOWER means the attack worked\n"
        "  others_*  mean recall on the other 9 classes — should stay at the 'clean' row\n"
        "  collat    mean recall the other classes lost per round (0 = perfectly targeted)\n"
        "  overall   plain top-1 accuracy; a perfect targeted attack only costs ~1/n_classes\n"
        "  detect%   fraction of poisoned-client rounds the defense flagged (recall / TPR)\n"
        "  atk_thru  fraction of rounds a poisoned client EVADED detection (fn>0)\n"
        "  tgt_succ  fraction of rounds the TARGETED goal was met: enough of the target\n"
        "            class destroyed AND collateral within the goal's tolerance\n"
        "A defense is doing its job when TGT_final stays near the 'clean' row.\n"
    )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "targeted_benchmark.json"), "w") as f:
            json.dump({"n_rounds": n_rounds, "baseline_accuracy": baseline_accuracy,
                       "target_label": label, "n_poisoners": n_poisoners,
                       "goal": goal, "results": summaries}, f, indent=2)
        if summaries:
            # Flatten list-valued per-class fields into per-class columns so the CSV
            # opens cleanly in a spreadsheet.
            flat = []
            for s in summaries:
                row = {k: v for k, v in s.items() if not isinstance(v, list)}
                for field in ("per_class_clean", "per_class_final", "per_class_mean"):
                    for i, v in enumerate(s.get(field) or []):
                        row[f"{field}_{i}"] = v
                flat.append(row)
            fields = list(dict.fromkeys(k for row in flat for k in row))
            with open(os.path.join(out_dir, "targeted_benchmark.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(flat)
        text += f"\n[saved] {os.path.join(out_dir, 'targeted_benchmark.json')} + .csv\n"
    return text
