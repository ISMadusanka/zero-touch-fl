#!/usr/bin/env python3
"""
Zero-Touch FL — Round Data Visualizer
======================================
Generates charts from the Phase-2 ``round_data`` JSON logs + the aggregate
``metrics/summary.json``. Headless (saves PNGs + an HTML report).

Round JSON schema (see core.types.RoundLog):
    round_num, attack_goal, poisoned_client_ids, predicted_labels[],
    test_accuracy, baseline_accuracy, attacker_reward, defender_reward,
    learning_agent, attack_metadata

Usage:
    python visualize_rounds.py
    python visualize_rounds.py --log-dir ./logs/round_data --out-dir ./logs/visualizations
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "accent": "#6C63FF", "accent2": "#FF6584", "accent3": "#43E97B", "accent4": "#F7971E",
    "bg": "#0F0F1A", "card": "#1A1A2E", "grid": "#2A2A3E", "text": "#E0E0E0",
    "text_dim": "#888899", "detected": "#43E97B", "missed": "#FF6584",
    "suspicious": "#FF6584", "clean": "#43E97B",
}
CLIENT_COLORS = ["#6C63FF", "#FF6584", "#43E97B", "#F7971E", "#00C9FF", "#B388FF"]


def apply_dark_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(COLORS["bg"])
    ax.figure.set_facecolor(COLORS["bg"])
    ax.title.set_color(COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.tick_params(colors=COLORS["text_dim"], which="both")
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], alpha=0.4, linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color=COLORS["text"], pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=COLORS["text_dim"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=COLORS["text_dim"])


# ─── Data loading ──────────────────────────────────────────────────────────────
def load_rounds(log_dir: str):
    """Load round logs from ``rounds.jsonl`` (one JSON object per line) and/or the
    legacy ``round_NNN.json`` files, merged by round number and sorted."""
    by_round: dict[int, dict] = {}
    for path in sorted(Path(log_dir).glob("round_*.json"),
                       key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))):
        try:
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
            by_round[int(r["round_num"])] = r
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            print(f"[WARN] skipping {path}: {e}")
    jsonl = Path(log_dir) / "rounds.jsonl"
    if jsonl.is_file():
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    by_round[int(r["round_num"])] = r
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue      # tolerate a torn final line from a live run
    return [by_round[k] for k in sorted(by_round)]


def load_metrics_summary(path: str):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception as e:
        print(f"[WARN] Could not parse metrics summary {p}: {e}")
        return None


def attach_metrics(rounds, summary):
    """Attach each round's RoundMetrics (from summary.json) as r['_m']."""
    by_round = {}
    if summary and isinstance(summary.get("per_round"), list):
        by_round = {m["round_num"]: m for m in summary["per_round"]}
    for r in rounds:
        r["_m"] = by_round.get(r["round_num"])
    return any(r["_m"] for r in rounds)


def _rolling(vals, window):
    if len(vals) < 2 or window < 2:
        return np.array(vals, dtype=float)
    w = min(window, len(vals))
    kernel = np.ones(w) / w
    return np.convolve(vals, kernel, mode="same")


# ─── Charts ──────────────────────────────────────────────────────────────────
def plot_accuracy(rounds, out_dir):
    rns = [r["round_num"] for r in rounds]
    test = [r["test_accuracy"] for r in rounds]
    base = [r["baseline_accuracy"] for r in rounds]
    fig, ax = plt.subplots(figsize=(12, 5))
    apply_dark_style(ax, "Global Model Accuracy Over Rounds", "Round", "Accuracy")
    ax.plot(rns, base, "--", color=COLORS["accent3"], linewidth=1.5, label="Baseline", alpha=0.7)
    ax.plot(rns, test, "-", color=COLORS["accent"], linewidth=1.8, label="Test Accuracy")
    ax.fill_between(rns, test, base, alpha=0.12, color=COLORS["accent"])
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_rewards(rounds, out_dir):
    """Attacker & defender verifiable reward per round (+ rolling mean)."""
    rns = [r["round_num"] for r in rounds]
    a = [r.get("attacker_reward", 0.0) for r in rounds]
    d = [r.get("defender_reward", 0.0) for r in rounds]
    win = max(2, len(rns) // 20)
    fig, ax = plt.subplots(figsize=(12, 5))
    apply_dark_style(ax, "Verifiable RL Reward Over Rounds", "Round", "Reward")
    ax.plot(rns, a, color=COLORS["accent2"], linewidth=0.8, alpha=0.35)
    ax.plot(rns, d, color=COLORS["accent3"], linewidth=0.8, alpha=0.35)
    ax.plot(rns, _rolling(a, win), color=COLORS["accent2"], linewidth=2.2, label="Attacker (rolling)")
    ax.plot(rns, _rolling(d, win), color=COLORS["accent3"], linewidth=2.2, label="Defender (rolling)")
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_rewards.png"), dpi=150)
    plt.close(fig)


def _all_clients(rounds):
    return sorted({v["client_id"] for r in rounds for v in r["predicted_labels"]})


def plot_verdicts_heatmap(rounds, out_dir):
    """Predicted suspicion heatmap; white dots mark the ground-truth poisoned."""
    rns = [r["round_num"] for r in rounds]
    clients = _all_clients(rounds)
    matrix = np.full((len(clients), len(rns)), np.nan)
    poisoned_pts = []
    for j, r in enumerate(rounds):
        for v in r["predicted_labels"]:
            row = clients.index(v["client_id"])
            matrix[row, j] = v["confidence"] if v["is_suspicious"] else -v["confidence"]
        for cid in r.get("poisoned_client_ids", []):
            if cid in clients:
                poisoned_pts.append((j, clients.index(cid)))

    fig, ax = plt.subplots(figsize=(14, 4))
    apply_dark_style(ax, "Predicted Suspicion (• = ground-truth poisoned)", "Round index", "Client ID")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("verdict", [COLORS["clean"], COLORS["bg"], COLORS["suspicious"]])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    if poisoned_pts:
        xs, ys = zip(*poisoned_pts)
        ax.scatter(xs, ys, s=18, color="white", edgecolors="black", linewidths=0.4, zorder=3)
    ax.set_yticks(range(len(clients)))
    ax.set_yticklabels([f"Client {c}" for c in clients], fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("← clean | suspicious →", color=COLORS["text_dim"], fontsize=9)
    cbar.ax.tick_params(colors=COLORS["text_dim"])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_verdicts_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_flagged_vs_poisoned(rounds, out_dir):
    rns = [r["round_num"] for r in rounds]
    flagged = [sum(1 for v in r["predicted_labels"] if v["is_suspicious"]) for r in rounds]
    poisoned = [len(r.get("poisoned_client_ids", [])) for r in rounds]
    fig, ax = plt.subplots(figsize=(12, 4))
    apply_dark_style(ax, "Flagged vs Ground-Truth Poisoned Per Round", "Round", "Client count")
    ax.plot(rns, poisoned, color=COLORS["accent3"], linewidth=1.6, label="Poisoned (truth)")
    ax.plot(rns, flagged, color=COLORS["accent2"], linewidth=1.2, alpha=0.8, label="Flagged (defender)")
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_flagged_vs_poisoned.png"), dpi=150)
    plt.close(fig)


# ─── Metric charts (from summary.json per_round) ──────────────────────────────
def _metric_series(rounds, key):
    rns, vals = [], []
    for r in rounds:
        m = r.get("_m")
        if m is None or key not in m:
            continue
        rns.append(r["round_num"])
        vals.append(m[key])
    return rns, vals


def plot_confusion_matrix(rounds, out_dir):
    rns, tp = _metric_series(rounds, "tp")
    _, fn = _metric_series(rounds, "fn")
    _, fp = _metric_series(rounds, "fp")
    _, tn = _metric_series(rounds, "tn")
    if not rns:
        return
    tp, fn, fp, tn = map(np.array, (tp, fn, fp, tn))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    apply_dark_style(ax, "Confusion Matrix Per Round", "Round", "Client count")
    ax.bar(rns, tp, color=COLORS["accent3"], label="TP")
    ax.bar(rns, fn, bottom=tp, color=COLORS["accent2"], label="FN")
    ax.bar(rns, fp, bottom=tp + fn, color=COLORS["accent4"], label="FP")
    ax.bar(rns, tn, bottom=tp + fn + fp, color=COLORS["accent"], label="TN")
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def plot_detection_rates(rounds, out_dir):
    rns, tp = _metric_series(rounds, "tp")
    _, fn = _metric_series(rounds, "fn")
    _, fp = _metric_series(rounds, "fp")
    _, tn = _metric_series(rounds, "tn")
    if not rns:
        return
    cum_tp, cum_fn = np.cumsum(tp), np.cumsum(fn)
    cum_fp, cum_tn = np.cumsum(fp), np.cumsum(tn)
    eps = 1e-12
    tpr = cum_tp / (cum_tp + cum_fn + eps)
    fpr = cum_fp / (cum_fp + cum_tn + eps)
    fig, ax = plt.subplots(figsize=(12, 5))
    apply_dark_style(ax, "Cumulative Detection Rates", "Round", "Rate")
    ax.plot(rns, tpr, color=COLORS["accent3"], linewidth=2, label="TPR / Recall")
    ax.plot(rns, fpr, color=COLORS["accent2"], linewidth=2, label="FPR")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_detection_rates.png"), dpi=150)
    plt.close(fig)


def plot_attack_success_rate(rounds, out_dir):
    """Cumulative ASR (goal met) against cumulative evasion.

    The two are NOT the same thing and used to be plotted as one: evasion only
    asks whether the poisoned client slipped past the detector, so under a
    targeted goal a round that left the model healthier than baseline still
    counted as a successful attack. Both series are drawn so the gap between
    "got through" and "actually did the damage" is visible.
    """
    rns, flags = _metric_series(rounds, "attack_success")
    if not rns:
        return
    n = np.arange(1, len(flags) + 1)
    cum = np.cumsum(np.array([1 if x else 0 for x in flags])) / n
    fig, ax = plt.subplots(figsize=(12, 4.5))
    apply_dark_style(ax, "Attack Success Rate (cumulative)", "Round", "rate")
    ax.plot(rns, cum, color=COLORS["accent2"], linewidth=2.2, label="ASR (goal met)")
    ax.fill_between(rns, 0, cum, color=COLORS["accent2"], alpha=0.12)

    ev_rns, ev_flags = _metric_series(rounds, "attack_evaded")
    label = f"Final ASR: {cum[-1]:.3f}"
    if ev_rns:
        ev_cum = (np.cumsum(np.array([1 if x else 0 for x in ev_flags]))
                  / np.arange(1, len(ev_flags) + 1))
        ax.plot(ev_rns, ev_cum, color=COLORS["accent3"], linewidth=1.6,
                linestyle="--", label="Evasion (slipped past detector)")
        label += f"   Evasion: {ev_cum[-1]:.3f}"
    ax.set_ylim(0, 1.05)
    ax.text(0.99, 0.95, label, transform=ax.transAxes, ha="right", va="top",
            fontsize=12, fontweight="bold", color=COLORS["accent2"])
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_attack_success_rate.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_preservation(rounds, out_dir):
    rns, apr = _metric_series(rounds, "accuracy_preservation_rate")
    if not rns:
        return
    apr = np.array(apr)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    apply_dark_style(ax, "Accuracy Preservation Rate", "Round", "APR (current / baseline)")
    ax.plot(rns, apr, color=COLORS["accent"], linewidth=2, label="APR")
    ax.axhline(1.0, color=COLORS["accent3"], linewidth=1.2, linestyle="--", alpha=0.7, label="APR=1.0")
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_accuracy_preservation.png"), dpi=150)
    plt.close(fig)


# ─── HTML report ──────────────────────────────────────────────────────────────
def generate_html_report(rounds, out_dir, summary=None):
    charts = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    rns = [r["round_num"] for r in rounds]
    mean_a = np.mean([r.get("attacker_reward", 0.0) for r in rounds])
    mean_d = np.mean([r.get("defender_reward", 0.0) for r in rounds])
    final_acc = rounds[-1]["test_accuracy"]

    metrics_html = ""
    if summary and "aggregate" in summary:
        agg = summary["aggregate"]
        metrics_html = f"""
<h2 style="text-align:center;color:#888;font-size:1rem;margin:1.5rem 0 .8rem">Aggregate Metrics</h2>
<div class="stats">
  <div class="stat"><div class="val">{agg.get('attack_success_rate', 0):.3f}</div><div class="lbl">Attack Success Rate</div></div>
  <div class="stat"><div class="val">{agg.get('tpr', 0):.3f}</div><div class="lbl">TPR</div></div>
  <div class="stat"><div class="val">{agg.get('fpr', 0):.3f}</div><div class="lbl">FPR</div></div>
  <div class="stat"><div class="val">{agg.get('accuracy_preservation_rate', 0):.3f}</div><div class="lbl">Accuracy Preservation</div></div>
</div>"""

    chart_tags = "\n".join(
        f'<div class="chart"><h3>{c.replace(".png","").replace("_"," ").title()}</h3>'
        f'<img src="{c}" alt="{c}"></div>' for c in charts
    )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FL Round Data — Visual Report</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0F0F1A;color:#E0E0E0;font-family:'Segoe UI',system-ui,sans-serif;padding:2rem}}
  h1{{text-align:center;font-size:1.8rem;margin-bottom:.3rem;background:linear-gradient(135deg,#6C63FF,#43E97B);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
  .subtitle{{text-align:center;color:#888;font-size:.9rem;margin-bottom:2rem}}
  .stats{{display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:2rem}}
  .stat{{background:#1A1A2E;border:1px solid #2A2A3E;border-radius:12px;padding:1rem 1.5rem;min-width:150px;text-align:center}}
  .stat .val{{font-size:1.6rem;font-weight:700;color:#6C63FF}}
  .stat .lbl{{font-size:.75rem;color:#888;margin-top:.25rem}}
  .chart{{background:#1A1A2E;border:1px solid #2A2A3E;border-radius:12px;padding:1.2rem;margin-bottom:1.5rem}}
  .chart h3{{font-size:.95rem;color:#888;margin-bottom:.8rem}}
  .chart img{{width:100%;border-radius:8px}}
</style></head><body>
<h1>🛡️ Zero-Touch FL — Adversarial RL Report</h1>
<p class="subtitle">Rounds {rns[0]}–{rns[-1]} • Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
<div class="stats">
  <div class="stat"><div class="val">{len(rounds)}</div><div class="lbl">Rounds</div></div>
  <div class="stat"><div class="val">{final_acc:.4f}</div><div class="lbl">Final Accuracy</div></div>
  <div class="stat"><div class="val">{mean_a:.3f}</div><div class="lbl">Mean Attacker Reward</div></div>
  <div class="stat"><div class="val">{mean_d:.3f}</div><div class="lbl">Mean Defender Reward</div></div>
</div>
{metrics_html}
{chart_tags}
</body></html>"""
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize Zero-Touch FL round data logs")
    parser.add_argument("--log-dir", default="logs/round_data")
    parser.add_argument("--out-dir", default="logs/visualizations")
    parser.add_argument("--metrics-summary", default="logs/metrics/summary.json")
    args = parser.parse_args()
    # This script prints ✓ marks; on Windows stdout defaults to cp1252 (especially
    # when piped), which raises UnicodeEncodeError. Same fix as main.setup_logging.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    if not os.path.isdir(args.log_dir):
        print(f"[ERROR] Log directory not found: {args.log_dir}")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    rounds = load_rounds(args.log_dir)
    if not rounds:
        print("[ERROR] No round logs found (looked for rounds.jsonl and round_*.json).")
        sys.exit(1)

    summary = load_metrics_summary(args.metrics_summary)
    have_metrics = attach_metrics(rounds, summary)
    print(f"[INFO] Loaded {len(rounds)} rounds from {args.log_dir}")

    plot_accuracy(rounds, args.out_dir); print("  ✓ 01_accuracy.png")
    plot_rewards(rounds, args.out_dir); print("  ✓ 02_rewards.png")
    plot_verdicts_heatmap(rounds, args.out_dir); print("  ✓ 03_verdicts_heatmap.png")
    plot_flagged_vs_poisoned(rounds, args.out_dir); print("  ✓ 04_flagged_vs_poisoned.png")

    if have_metrics:
        plot_confusion_matrix(rounds, args.out_dir); print("  ✓ 05_confusion_matrix.png")
        plot_detection_rates(rounds, args.out_dir); print("  ✓ 06_detection_rates.png")
        plot_attack_success_rate(rounds, args.out_dir); print("  ✓ 07_attack_success_rate.png")
        plot_accuracy_preservation(rounds, args.out_dir); print("  ✓ 08_accuracy_preservation.png")
    else:
        print("[INFO] No metrics/summary.json per_round data — skipping metric charts.")

    report_path = generate_html_report(rounds, args.out_dir, summary)
    print(f"\n[DONE] HTML report → {report_path}")
    print(f"       Open: file://{os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
