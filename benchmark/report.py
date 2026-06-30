"""Render benchmark results: a console table + JSON/CSV dump."""
import csv
import json
import os

# (summary-key, column header, format string)
_COLS = [
    ("defense", "defense", "{}"),
    ("detection_rate", "detect%", "{:.1%}"),
    ("fpr", "FPR", "{:.1%}"),
    ("precision", "prec", "{:.2f}"),
    ("f1", "F1", "{:.2f}"),
    ("final_accuracy", "final_acc", "{:.3f}"),
    ("mean_accuracy", "mean_acc", "{:.3f}"),
    ("mean_acc_drop", "acc_drop", "{:+.3f}"),
    ("attack_success_rate", "atk_thru", "{:.1%}"),
]


def format_table(summaries: list[dict]) -> str:
    rows = [[h for _, h, _ in _COLS]]
    for s in summaries:
        rows.append([fmt.format(s[key]) for key, _, fmt in _COLS])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def line(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out = [line(rows[0]), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows[1:]]
    return "\n".join(out)


def render(summaries: list[dict], n_rounds: int, baseline_accuracy: float,
           out_dir: str | None = None) -> str:
    title = (f"DEFENSE BENCHMARK — {n_rounds} attack rounds  "
             f"(clean baseline acc = {baseline_accuracy:.3f})")
    bar = "=" * len(title)
    text = f"{bar}\n{title}\n{bar}\n{format_table(summaries)}\n"
    text += (
        "\nLegend:\n"
        "  detect%   fraction of poisoned-client rounds the defense flagged (recall / TPR)\n"
        "  FPR       honest clients wrongly flagged\n"
        "  acc_drop  mean test-accuracy lost vs the clean baseline (lower is better)\n"
        "  atk_thru  fraction of rounds a poisoned client slipped through (attack success)\n"
    )
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
            json.dump({"n_rounds": n_rounds, "baseline_accuracy": baseline_accuracy,
                       "results": summaries}, f, indent=2)
        if summaries:
            with open(os.path.join(out_dir, "benchmark.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
                w.writeheader()
                w.writerows(summaries)
        text += f"\n[saved] {os.path.join(out_dir, 'benchmark.json')} + benchmark.csv\n"
    return text
