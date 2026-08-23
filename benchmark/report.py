"""Render benchmark results: console tables + JSON/CSV dump.

Two shapes, because the benchmark has two:

* :func:`render` — one attack against a panel of defenses (rows = defenses). The
  original report, unchanged.
* :func:`render_matrix` — a panel of attacks against a panel of defenses. Prints
  one per-attack table plus the attack x defense grids that are the actual result:
  how much accuracy each attack cost each defense, and how much of each attack each
  defense caught.
"""
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
    ("goal_success_rate", "atk_succ", "{:.1%}"),
]

#: The attack x defense grids printed by :func:`render_matrix`, in order.
#: (summary-key, title, format, "higher value = stronger attack",
#:  "means anything for a CONTROL row")
#:
#: The last flag exists for the ``clean`` row: it poisons nobody, so its detection
#: rate is 0/0 and its "attack success" describes an attack that never happened.
#: Printing those as 0.0% would read as a real measurement, so they render ``n/a``.
#: Its acc_drop, by contrast, is the whole point of the row.
_MATRICES = [
    ("mean_acc_drop", "ACC_DROP  — mean test accuracy the attack cost this defense",
     "{:+.3f}", True, True),
    ("detection_rate", "DETECT%   — share of poisoned-client rounds this defense flagged",
     "{:.1%}", False, False),
    ("goal_success_rate", "ATK_SUCC  — weighted attack success against the goal's requested drop",
     "{:.1%}", True, False),
]


def _cell(value, fmt: str) -> str:
    """Format one cell; ``None`` (e.g. goal-success with no goal target) -> 'n/a'."""
    return "n/a" if value is None else fmt.format(value)


def _grid(rows: list) -> str:
    """Left-justified fixed-width text table from a list of row-lists (row 0 = header)."""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def line(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    return "\n".join([line(rows[0]), "  ".join("-" * w for w in widths)]
                     + [line(r) for r in rows[1:]])


def format_table(summaries: list) -> str:
    """Per-defense table. Gains an ``attack`` column when the rows span several attacks."""
    attacks = {s.get("attack") for s in summaries if s.get("attack") is not None}
    cols = ([("attack", "attack", "{}")] if len(attacks) > 1 else []) + _COLS
    rows = [[h for _, h, _ in cols]]
    for s in summaries:
        rows.append([_cell(s.get(key), fmt) for key, _, fmt in cols])
    return _grid(rows)


#: Rows that are CONTROLS rather than attacks. They are excluded from the per-column
#: "strongest attack" mark, which would otherwise be won by a row that never attacked.
_CONTROL_ROWS = ("clean",)


def matrix_table(by_attack: dict, defenses: list, key: str, fmt: str,
                 higher_is_stronger: bool, control_meaningful: bool = True) -> str:
    """One attack x defense grid for ``key``.

    Rows are attacks, columns defenses. The strongest attack in each defense's
    column is marked ``*``, which is the comparison the whole benchmark exists to
    make: for a given defense, does the trained policy do more damage than the
    published attacks?
    """
    header = ["attack"] + list(defenses)
    rows = [header]
    best = {}
    for d in defenses:
        vals = [(a, by_attack[a][d].get(key)) for a in by_attack
                if d in by_attack[a] and a not in _CONTROL_ROWS]
        vals = [(a, v) for a, v in vals if v is not None]
        if vals:
            best[d] = (max if higher_is_stronger else min)(vals, key=lambda t: t[1])[0]
    for a, panel in by_attack.items():
        control = a in _CONTROL_ROWS
        row = [a]
        for d in defenses:
            if d not in panel:
                row.append("-")
                continue
            if control and not control_meaningful:
                row.append("n/a")
                continue
            text = _cell(panel[d].get(key), fmt)
            row.append(f"{text}*" if best.get(d) == a and text != "n/a" else text)
        rows.append(row)
    return _grid(rows)



def _goal_str(goal: dict | None) -> str:
    """Compact 'type=value' rendering of an attack goal for the report header."""
    if not goal:
        return ""
    gtype = goal.get("type", "untargeted_degrade")
    val = (goal.get("target_accuracy_drop") if gtype == "untargeted_degrade"
           else goal.get("per_round_drop") if gtype == "slow_degrade"
           else goal.get("label"))
    return f"{gtype}={val}" if val is not None else str(gtype)


def _title_block(title: str) -> str:
    bar = "=" * len(title)
    return f"{bar}\n{title}\n{bar}"


def _header(n_rounds: int, baseline_accuracy: float, goal, n_poisoners, n_clients,
            used, label: str = "DEFENSE BENCHMARK") -> str:
    goal_s = _goal_str(goal)
    poisoner_s = ""
    if n_poisoners is not None:
        # The DENOMINATOR is what makes two reports comparable: 10 of 20 is a
        # federation with no honest majority, 10 of 25 is one with it, and the
        # detection numbers below are not interpretable without knowing which.
        of_s = f" of {n_clients}" if n_clients is not None else ""
        poisoner_s = f", Num of poisoners={n_poisoners}{of_s}"
        if used is not None:
            poisoner_s += (" (exact quota)" if abs(used - n_poisoners) < 0.05
                           else f" quota, {used:.1f} effective/round")
    return _title_block(
        f"{label} — {n_rounds} attack rounds  "
        f"(clean baseline acc = {baseline_accuracy:.3f}"
        f"{f'; goal = {goal_s}' if goal_s else ''})"
        f"{poisoner_s}")


def _legend(baseline_accuracy: float, goal) -> str:
    tgt = thr = None
    if goal is not None:
        from rl.rewards import goal_target
        tgt = goal_target(goal)
        thr = baseline_accuracy - tgt
    return (
        "\nLegend:\n"
        "  detect%   fraction of poisoned-client rounds the defense flagged (recall / TPR)\n"
        "  FPR       honest clients wrongly flagged\n"
        "  acc_drop  mean test-accuracy lost vs the clean baseline (lower is better)\n"
        "  atk_thru  fraction of rounds a poisoned client EVADED detection (fn>0)\n"
        f"  atk_succ  WEIGHTED attack success: mean over rounds of "
        f"min(1, acc_drop / target{f' ({tgt:.3f})' if tgt is not None else ''})\n"
        f"            — a round achieving the FULL requested drop"
        f"{f' (acc <= {thr:.3f})' if thr is not None else ''} scores 100%, half of it 50%\n"
    )


def _save(out_dir: str, blob: dict, rows: list) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(blob, f, indent=2)
    if rows:
        with open(os.path.join(out_dir, "benchmark.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return f"\n[saved] {os.path.join(out_dir, 'benchmark.json')} + benchmark.csv\n"


def render(summaries: list, n_rounds: int, baseline_accuracy: float,
           out_dir: str | None = None, goal: dict | None = None,
           n_poisoners: int | None = None, n_clients: int | None = None) -> str:
    """One attack vs a panel of defenses (rows = defenses)."""
    # ``n_poisoners`` is the exact per-round poison quota. Pull the realised mean
    # from each summary as an audit check (the held-fixed attack makes them agree).
    used = summaries[0].get("mean_poisoned") if summaries else None
    text = (_header(n_rounds, baseline_accuracy, goal, n_poisoners, n_clients, used)
            + "\n" + format_table(summaries) + "\n" + _legend(baseline_accuracy, goal))
    if out_dir:
        text += _save(out_dir,
                      {"n_rounds": n_rounds, "baseline_accuracy": baseline_accuracy,
                       "n_poisoners": n_poisoners, "goal": goal, "results": summaries},
                      summaries)
    return text


def render_matrix(by_attack: dict, n_rounds: int, baseline_accuracy: float,
                  out_dir: str | None = None, goal: dict | None = None,
                  n_poisoners: int | None = None, n_clients: int | None = None,
                  citations: dict | None = None, run_info: dict | None = None) -> str:
    """A panel of attacks vs a panel of defenses.

    ``by_attack`` is ``{attack_name: {defense_name: summary}}`` in report order.
    """
    defenses = list(next(iter(by_attack.values()))) if by_attack else []
    # The realised-poisoners audit must come from a real ATTACK row: a control row
    # poisons nobody by design, so reading it would report "0.0 effective/round" for
    # a run in which every attack filled the quota.
    attack_rows = [p for a, p in by_attack.items() if a not in _CONTROL_ROWS]
    first = attack_rows[0] if attack_rows else {}
    used = next(iter(first.values()), {}).get("mean_poisoned") if first else None
    parts = [_header(n_rounds, baseline_accuracy, goal, n_poisoners, n_clients, used,
                     label="ATTACK x DEFENSE BENCHMARK")]

    knowledge = (run_info or {}).get("knowledge")
    if knowledge:
        parts.append(
            f"\nBaseline adversary knowledge: {knowledge} — the published attacks see the "
            f"honest updates of\n"
            + ("the COMPROMISED clients only, the same information the trained attacker "
               "observes." if knowledge == "partial"
               else "EVERY client (omniscient), the setting most of the papers state.")
            + "\nEvery attack poisons the SAME clients in the same rounds; only the crafted "
              "updates differ.")

    if "clean" in by_attack:
        parts.append(
            "\nThe 'clean' row is a CONTROL, not an attack: nothing is poisoned, so it is "
            "each defense's\nno-attack accuracy. Read any cell below as "
            "(clean row - attack row) for the same defense.")

    for key, title, fmt, stronger, control_ok in _MATRICES:
        if all(panel.get(d, {}).get(key) is None
               for panel in by_attack.values() for d in defenses):
            continue                                 # e.g. atk_succ with no goal target
        parts.append(f"\n{title}\n"
                     + matrix_table(by_attack, defenses, key, fmt, stronger, control_ok)
                     + f"\n  (* = {'strongest attack' if stronger else 'best-detected attack'}"
                       f" in that defense's column)")

    parts.append("\n" + "-" * 78 + "\nPER-ATTACK DETAIL")
    for name, panel in by_attack.items():
        cite = (citations or {}).get(name)
        parts.append(f"\n[{name}]" + (f"  {cite}" if cite else ""))
        parts.append(format_table(list(panel.values())))

    parts.append(_legend(baseline_accuracy, goal))
    unchanged = (run_info or {}).get("unchanged_client_rounds") or {}
    degenerate = {k: v for k, v in unchanged.items() if v}
    if degenerate:
        parts.append(
            "  NOTE  these attacks submitted some byte-identical-to-honest client "
            "updates\n        (counted as poisoned anyway, so the poisoned set stays "
            f"identical across rows): {degenerate}\n")
    text = "\n".join(parts)

    if out_dir:
        rows = [s for panel in by_attack.values() for s in panel.values()]
        text += _save(out_dir,
                      {"n_rounds": n_rounds, "baseline_accuracy": baseline_accuracy,
                       "n_poisoners": n_poisoners, "n_clients": n_clients, "goal": goal,
                       "attacks": list(by_attack), "defenses": defenses,
                       "run_info": run_info or {}, "results": rows},
                      rows)
    return text
