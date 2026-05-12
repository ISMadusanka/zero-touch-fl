#!/usr/bin/env python3
"""
Zero-Touch FL — Round Data Visualizer
======================================
Generates publication-quality charts from round_data JSON logs.
Designed for headless server usage (saves PNGs + an HTML report).

Usage:
    python visualize_rounds.py                        # defaults
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
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "accent":       "#6C63FF",
    "accent2":      "#FF6584",
    "accent3":      "#43E97B",
    "accent4":      "#F7971E",
    "bg":           "#0F0F1A",
    "card":         "#1A1A2E",
    "grid":         "#2A2A3E",
    "text":         "#E0E0E0",
    "text_dim":     "#888899",
    "detected":     "#43E97B",
    "missed":       "#FF6584",
    "suspicious":   "#FF6584",
    "clean":        "#43E97B",
}

ATTACK_TYPE_COLORS = {
    "noise_injection": "#6C63FF",
    "sign_flip":       "#FF6584",
    "scale_attack":    "#F7971E",
    "zero_update":     "#43E97B",
    "gaussian_noise":  "#00C9FF",
}

DEFEND_METHOD_COLORS = {
    "norm_threshold": "#6C63FF",
    "combined":       "#43E97B",
    "cosine_filter":  "#F7971E",
}

# ─── Style helpers ──────────────────────────────────────────────────────────────
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
    files = sorted(Path(log_dir).glob("round_*.json"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    rounds = []
    for f in files:
        with open(f) as fh:
            rounds.append(json.load(fh))
    return rounds


# ─── Chart generators ──────────────────────────────────────────────────────────
def plot_accuracy(rounds, out_dir):
    """Test accuracy vs baseline accuracy over rounds."""
    rns = [r["round_num"] for r in rounds]
    test = [r["test_accuracy"] for r in rounds]
    base = [r["baseline_accuracy"] for r in rounds]

    fig, ax = plt.subplots(figsize=(12, 5))
    apply_dark_style(ax, "Model Accuracy Over Rounds", "Round", "Accuracy")

    ax.plot(rns, base, "--", color=COLORS["accent3"], linewidth=1.5, label="Baseline", alpha=0.7)
    ax.plot(rns, test, "-o", color=COLORS["accent"], linewidth=2, markersize=5, label="Test Accuracy")
    ax.fill_between(rns, test, base, alpha=0.12, color=COLORS["accent"])

    # Mark rounds where accuracy dropped below baseline
    for i, r in enumerate(rounds):
        if r["test_accuracy"] < r["baseline_accuracy"] - 0.001:
            ax.annotate("▼", (rns[i], test[i]), textcoords="offset points",
                        xytext=(0, -14), ha="center", color=COLORS["accent2"], fontsize=10)

    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_detection(rounds, out_dir):
    """Attack detection success per round."""
    rns = [r["round_num"] for r in rounds]
    detected = [r["attack_detected"] for r in rounds]
    colors = [COLORS["detected"] if d else COLORS["missed"] for d in detected]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    apply_dark_style(ax, "Attack Detection Per Round", "Round", "")

    ax.bar(rns, [1]*len(rns), color=colors, edgecolor="none", width=0.7, alpha=0.85)
    ax.set_yticks([])
    ax.set_xticks(rns)

    patches = [mpatches.Patch(color=COLORS["detected"], label="Detected"),
               mpatches.Patch(color=COLORS["missed"], label="Missed")]
    ax.legend(handles=patches, facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    det_rate = sum(detected) / len(detected) * 100
    ax.text(0.99, 0.92, f"Detection Rate: {det_rate:.0f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=12, color=COLORS["accent3"], fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_detection.png"), dpi=150)
    plt.close(fig)


def plot_strategies(rounds, out_dir):
    """Attack type and defense method timeline."""
    rns = [r["round_num"] for r in rounds]
    atk_types = [r["attack_strategy"]["type"] for r in rounds]
    def_methods = [r["defend_strategy"]["method"] for r in rounds]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax in (ax1, ax2):
        apply_dark_style(ax)
    ax1.set_title("Strategy Evolution", fontsize=14, fontweight="bold", color=COLORS["text"], pad=12)

    # Attack
    unique_atk = sorted(set(atk_types))
    atk_y = {t: i for i, t in enumerate(unique_atk)}
    atk_colors = [ATTACK_TYPE_COLORS.get(t, "#888") for t in atk_types]
    ax1.scatter(rns, [atk_y[t] for t in atk_types], c=atk_colors, s=80, zorder=3, edgecolors="white", linewidths=0.5)
    ax1.set_yticks(range(len(unique_atk)))
    ax1.set_yticklabels(unique_atk, fontsize=10)
    ax1.set_ylabel("Attack Type", fontsize=11, color=COLORS["text_dim"])

    # Defend
    unique_def = sorted(set(def_methods))
    def_y = {m: i for i, m in enumerate(unique_def)}
    def_colors = [DEFEND_METHOD_COLORS.get(m, "#888") for m in def_methods]
    ax2.scatter(rns, [def_y[m] for m in def_methods], c=def_colors, s=80, zorder=3, edgecolors="white", linewidths=0.5)
    ax2.set_yticks(range(len(unique_def)))
    ax2.set_yticklabels(unique_def, fontsize=10)
    ax2.set_ylabel("Defense Method", fontsize=11, color=COLORS["text_dim"])
    ax2.set_xlabel("Round", fontsize=11, color=COLORS["text_dim"])
    ax2.set_xticks(rns)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_strategies.png"), dpi=150)
    plt.close(fig)


def plot_verdicts_heatmap(rounds, out_dir):
    """Client suspicion heatmap across rounds."""
    rns = [r["round_num"] for r in rounds]
    all_clients = sorted({v["client_id"] for r in rounds for v in r["verdicts"]})

    matrix = np.full((len(all_clients), len(rns)), np.nan)
    for j, r in enumerate(rounds):
        for v in r["verdicts"]:
            row = all_clients.index(v["client_id"])
            matrix[row, j] = v["confidence"] if v["suspicious"] else -v["confidence"]

    fig, ax = plt.subplots(figsize=(14, 4))
    apply_dark_style(ax, "Client Suspicion Heatmap", "Round", "Client ID")

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("verdict", [COLORS["clean"], COLORS["bg"], COLORS["suspicious"]])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(rns)))
    ax.set_xticklabels(rns)
    ax.set_yticks(range(len(all_clients)))
    ax.set_yticklabels([f"Client {c}" for c in all_clients], fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("← Clean | Suspicious →", color=COLORS["text_dim"], fontsize=9)
    cbar.ax.tick_params(colors=COLORS["text_dim"])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_verdicts_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_confidence(rounds, out_dir):
    """Per-client confidence scores over rounds."""
    rns = [r["round_num"] for r in rounds]
    all_clients = sorted({v["client_id"] for r in rounds for v in r["verdicts"]})
    client_colors = ["#6C63FF", "#FF6584", "#43E97B", "#F7971E", "#00C9FF"]

    fig, ax = plt.subplots(figsize=(12, 5))
    apply_dark_style(ax, "Confidence Scores Per Client", "Round", "Confidence")

    for idx, cid in enumerate(all_clients):
        confs = []
        for r in rounds:
            v = next((v for v in r["verdicts"] if v["client_id"] == cid), None)
            confs.append(v["confidence"] if v else np.nan)
        c = client_colors[idx % len(client_colors)]
        ax.plot(rns, confs, "-o", color=c, linewidth=1.5, markersize=4, label=f"Client {cid}", alpha=0.85)

    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"],
              fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_confidence.png"), dpi=150)
    plt.close(fig)


def plot_adaptation(rounds, out_dir):
    """Attacker / defender adaptation events."""
    rns = [r["round_num"] for r in rounds]
    atk_adapt = [r["attacker_adapted"] for r in rounds]
    def_adapt = [r["defender_adapted"] for r in rounds]

    fig, ax = plt.subplots(figsize=(12, 3))
    apply_dark_style(ax, "Adaptation Events", "Round", "")

    for i, rn in enumerate(rns):
        if atk_adapt[i]:
            ax.barh(1, 0.7, left=rn - 0.35, color=COLORS["accent2"], edgecolor="none", height=0.5)
        if def_adapt[i]:
            ax.barh(0, 0.7, left=rn - 0.35, color=COLORS["accent3"], edgecolor="none", height=0.5)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Defender", "Attacker"], fontsize=11)
    ax.set_xticks(rns)
    ax.set_ylim(-0.5, 1.8)

    atk_count = sum(atk_adapt)
    def_count = sum(def_adapt)
    ax.text(0.99, 0.92, f"Attacker adapted: {atk_count}x  |  Defender adapted: {def_count}x",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=COLORS["text_dim"])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_adaptation.png"), dpi=150)
    plt.close(fig)


def plot_attack_params(rounds, out_dir):
    """Evolution of attack parameters (e.g. noise scale)."""
    rns = [r["round_num"] for r in rounds]
    scales = []
    for r in rounds:
        p = r["attack_strategy"].get("params", {})
        scales.append(p.get("scale", p.get("factor", p.get("sigma", p.get("c", p.get("k", None))))))

    if all(s is None for s in scales):
        return  # nothing to plot

    fig, ax = plt.subplots(figsize=(12, 4))
    apply_dark_style(ax, "Attack Parameter Evolution", "Round", "Scale / Factor")

    valid = [(rn, s) for rn, s in zip(rns, scales) if s is not None]
    if valid:
        vrns, vscales = zip(*valid)
        ax.bar(vrns, vscales, color=COLORS["accent"], edgecolor="none", width=0.6, alpha=0.85)
        for rn, s in zip(vrns, vscales):
            ax.text(rn, s + 0.05, f"{s}", ha="center", va="bottom", fontsize=8, color=COLORS["text_dim"])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_attack_params.png"), dpi=150)
    plt.close(fig)


def plot_flagged_clients(rounds, out_dir):
    """Number of flagged clients per round."""
    rns = [r["round_num"] for r in rounds]
    flagged = [sum(1 for v in r["verdicts"] if v["suspicious"]) for r in rounds]
    total = [len(r["verdicts"]) for r in rounds]

    fig, ax = plt.subplots(figsize=(12, 4))
    apply_dark_style(ax, "Flagged Clients Per Round", "Round", "Count")

    ax.bar(rns, total, color=COLORS["grid"], edgecolor="none", width=0.6, label="Total Clients", alpha=0.5)
    ax.bar(rns, flagged, color=COLORS["accent2"], edgecolor="none", width=0.6, label="Flagged", alpha=0.85)
    ax.set_xticks(rns)
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_flagged_clients.png"), dpi=150)
    plt.close(fig)


# ─── HTML report ────────────────────────────────────────────────────────────────
def generate_html_report(rounds, out_dir):
    charts = sorted([f for f in os.listdir(out_dir) if f.endswith(".png")])
    rns = [r["round_num"] for r in rounds]
    det_rate = sum(r["attack_detected"] for r in rounds) / len(rounds) * 100
    avg_acc = np.mean([r["test_accuracy"] for r in rounds])
    atk_adapts = sum(r["attacker_adapted"] for r in rounds)
    def_adapts = sum(r["defender_adapted"] for r in rounds)

    chart_tags = "\n".join(
        f'<div class="chart"><h3>{c.replace(".png","").replace("_"," ").title()}</h3>'
        f'<img src="{c}" alt="{c}"></div>'
        for c in charts
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
<h1>🛡️ Zero-Touch FL — Round Data Report</h1>
<p class="subtitle">Rounds {rns[0]}–{rns[-1]} • Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
<div class="stats">
  <div class="stat"><div class="val">{len(rounds)}</div><div class="lbl">Rounds</div></div>
  <div class="stat"><div class="val">{det_rate:.0f}%</div><div class="lbl">Detection Rate</div></div>
  <div class="stat"><div class="val">{avg_acc:.4f}</div><div class="lbl">Avg Accuracy</div></div>
  <div class="stat"><div class="val">{atk_adapts}</div><div class="lbl">Attacker Adapts</div></div>
  <div class="stat"><div class="val">{def_adapts}</div><div class="lbl">Defender Adapts</div></div>
</div>
{chart_tags}
</body></html>"""

    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize Zero-Touch FL round data logs")
    parser.add_argument("--log-dir", default="logs/round_data", help="Path to round_data JSON directory")
    parser.add_argument("--out-dir", default="logs/visualizations", help="Output directory for charts & report")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"[ERROR] Log directory not found: {args.log_dir}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    rounds = load_rounds(args.log_dir)

    if not rounds:
        print("[ERROR] No round_*.json files found.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(rounds)} rounds from {args.log_dir}")
    print(f"[INFO] Generating charts → {args.out_dir}/")

    plot_accuracy(rounds, args.out_dir)
    print("  ✓ 01_accuracy.png")

    plot_detection(rounds, args.out_dir)
    print("  ✓ 02_detection.png")

    plot_strategies(rounds, args.out_dir)
    print("  ✓ 03_strategies.png")

    plot_verdicts_heatmap(rounds, args.out_dir)
    print("  ✓ 04_verdicts_heatmap.png")

    plot_confidence(rounds, args.out_dir)
    print("  ✓ 05_confidence.png")

    plot_adaptation(rounds, args.out_dir)
    print("  ✓ 06_adaptation.png")

    plot_attack_params(rounds, args.out_dir)
    print("  ✓ 07_attack_params.png")

    plot_flagged_clients(rounds, args.out_dir)
    print("  ✓ 08_flagged_clients.png")

    report_path = generate_html_report(rounds, args.out_dir)
    print(f"\n[DONE] HTML report → {report_path}")
    print(f"       Open in browser: file://{os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
