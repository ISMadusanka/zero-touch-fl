"""Render benchmark results: a console table + JSON/CSV dump."""
import csv
import json
import os

from benchmark.metrics import INERT_POISON_RATIO

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
    # The two halves of acc_drop: what the defense costs on an honest round, and
    # what the attack added on top. 'n/a' when the run skipped the counterfactual.
    ("mean_defense_cost", "def_cost", "{:+.3f}"),
    ("mean_attack_drop", "atk_drop", "{:+.3f}"),
    ("attack_success_rate", "atk_thru", "{:.1%}"),
    ("goal_success_rate", "atk_succ", "{:.1%}"),
]


def _cell(value, fmt: str) -> str:
    """Format one cell; ``None`` (e.g. goal-success with no goal target) -> 'n/a'."""
    return "n/a" if value is None else fmt.format(value)


def format_table(summaries: list[dict]) -> str:
    rows = [[h for _, h, _ in _COLS]]
    for s in summaries:
        rows.append([_cell(s.get(key), fmt) for key, _, fmt in _COLS])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def line(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out = [line(rows[0]), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows[1:]]
    return "\n".join(out)


def _goal_str(goal: dict | None) -> str:
    """Compact 'type=value' rendering of an attack goal for the report header."""
    if not goal:
        return ""
    gtype = goal.get("type", "untargeted_degrade")
    val = (goal.get("target_accuracy_drop") if gtype == "untargeted_degrade"
           else goal.get("per_round_drop") if gtype == "slow_degrade"
           else goal.get("label"))
    return f"{gtype}={val}" if val is not None else str(gtype)


def _attack_strength_note(summaries: list[dict]) -> str:
    """A line stating how large the attack actually was, and a warning when it was
    not large enough to mean anything.

    Every other column is about the DEFENSE, which quietly assumes there was an
    attack to defend against. When the attacker's perturbation is far smaller than
    the honest client-to-client spread, "0% detected / 100% throughput" reads as a
    defense failure while meaning the opposite, so the table must not be printed
    without this number next to it. See ``benchmark.harness``.
    """
    ratios = [s.get("mean_poison_ratio") for s in summaries
              if s.get("mean_poison_ratio") is not None]
    if not ratios:
        return ""
    ratio = max(ratios)          # identical across defenses (the attack is held fixed)
    note = (f"\nAttack strength: mean poison perturbation = {ratio:.3g}x the honest "
            f"update it replaced\n")
    if ratio < INERT_POISON_RATIO:
        note += (
            "  WARNING: that is well inside the spread between honest non-IID clients, so\n"
            "  a robust aggregator ranks a poisoned update as MORE central than a real one.\n"
            "  The attack survives every filter and moves the global by nothing: low detect%\n"
            "  and high atk_thru here mean 'there was nothing to detect', NOT 'the defense\n"
            "  failed'. Fix the attacker before reading this as a defense result.\n"
        )
    return note


def render(summaries: list[dict], n_rounds: int, baseline_accuracy: float,
           out_dir: str | None = None, goal: dict | None = None,
           n_poisoners: int | None = None) -> str:
    goal_s = _goal_str(goal)
    # ``n_poisoners`` is the exact per-round poison quota. Pull the realised mean
    # from each summary as an audit check (the held-fixed attack makes them agree).
    used = None
    if summaries:
        used = summaries[0].get("mean_poisoned")
    poisoner_s = ""
    if n_poisoners is not None:
        poisoner_s = f", Num of poisoners={n_poisoners}"
        if used is not None:
            poisoner_s += (" (exact quota)" if abs(used - n_poisoners) < 0.05
                           else f" quota, {used:.1f} effective/round")
    title = (f"DEFENSE BENCHMARK — {n_rounds} attack rounds  "
             f"(clean baseline acc = {baseline_accuracy:.3f}"
             f"{f'; goal = {goal_s}' if goal_s else ''})"
             f"{poisoner_s}")
    bar = "=" * len(title)
    text = f"{bar}\n{title}\n{bar}\n{format_table(summaries)}\n"
    # The requested drop, and the accuracy at/below which the goal is met IN FULL.
    tgt = thr = None
    if goal is not None:
        from rl.rewards import goal_target
        tgt = goal_target(goal)
        thr = baseline_accuracy - tgt
    atk_succ_line = (
        f"  atk_succ  WEIGHTED attack success: mean over rounds of "
        f"min(1, atk_drop / target{f' ({tgt:.3f})' if tgt is not None else ''})\n"
        f"            — a round achieving the FULL requested drop"
        f"{f' (acc <= {thr:.3f})' if thr is not None else ''} scores 100%, half of it 50%\n"
    )
    text += (
        "\nLegend:\n"
        "  detect%   fraction of poisoned-client rounds the defense flagged (recall / TPR)\n"
        "  FPR       honest clients wrongly flagged\n"
        "  acc_drop  mean test-accuracy lost vs the clean baseline (lower is better)\n"
        "  def_cost  ...of which the DEFENSE cost itself, measured on the same rounds\n"
        "            with the poison removed (baseline - its own clean counterfactual)\n"
        "  atk_drop  ...and of which the ATTACK is responsible (clean - poisoned).\n"
        "            atk_drop is the honest robustness number; acc_drop mixes the two\n"
        "  atk_thru  fraction of rounds a poisoned client EVADED detection (fn>0)\n"
        + atk_succ_line
    )
    text += _attack_strength_note(summaries)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
            json.dump({"n_rounds": n_rounds, "baseline_accuracy": baseline_accuracy,
                       "n_poisoners": n_poisoners, "goal": goal, "results": summaries}, f, indent=2)
        if summaries:
            with open(os.path.join(out_dir, "benchmark.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
                w.writeheader()
                w.writerows(summaries)
        text += f"\n[saved] {os.path.join(out_dir, 'benchmark.json')} + benchmark.csv\n"
    return text
