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
    "scaling":         "#F7971E",
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

def _coerce_bool(val):
    """Convert stringified booleans (from json.dump default=str) back to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


def _coerce_float(val, default=0.0):
    """Safely convert a value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _normalize_round(r: dict) -> dict:
    """Fix types that may have been stringified by json.dump(default=str)."""
    # Boolean fields
    for key in ("attack_detected", "attacker_adapted", "defender_adapted",
                "all_clients_flagged", "round_skipped"):
        if key in r:
            r[key] = _coerce_bool(r[key])

    # Numeric fields
    for key in ("test_accuracy", "baseline_accuracy", "round_num"):
        if key in r:
            r[key] = _coerce_float(r[key])
    if "round_num" in r:
        r["round_num"] = int(r["round_num"])

    # Verdict sub-objects
    for v in r.get("verdicts", []):
        if "suspicious" in v:
            v["suspicious"] = _coerce_bool(v["suspicious"])
        if "confidence" in v:
            v["confidence"] = _coerce_float(v["confidence"])

    # Metrics sub-object
    m = r.get("metrics")
    if isinstance(m, dict):
        for key in ("tp", "fn", "fp", "tn"):
            if key in m:
                m[key] = int(_coerce_float(m[key]))
        for key in ("tpr", "fpr", "recall", "accuracy_preservation_rate",
                    "current_accuracy", "baseline_accuracy"):
            if key in m:
                m[key] = _coerce_float(m[key])
        if "attack_success" in m:
            m["attack_success"] = _coerce_bool(m["attack_success"])

    return r


def load_rounds(log_dir: str):
    files = sorted(Path(log_dir).glob("round_*.json"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    rounds = []
    for f in files:
        with open(f) as fh:
            rounds.append(_normalize_round(json.load(fh)))
    return rounds


def load_metrics_summary(path: str):
    """Load aggregate metrics summary.json if present; return None otherwise."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception as e:
        print(f"[WARN] Could not parse metrics summary {p}: {e}")
        return None


def rounds_have_metrics(rounds: list) -> bool:
    """True iff at least one round contains a per-round metrics block."""
    return any("metrics" in r for r in rounds)


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
    detected = [bool(r["attack_detected"]) for r in rounds]
    colors = [COLORS["detected"] if d else COLORS["missed"] for d in detected]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    apply_dark_style(ax, "Attack Detection Per Round", "Round", "")

    ax.bar(rns, [1]*len(rns), color=colors, edgecolor="none", width=0.7, alpha=0.85)
    ax.set_yticks([])
    ax.set_xticks(rns[::max(1, len(rns)//30)])  # reduce tick clutter for many rounds

    patches = [mpatches.Patch(color=COLORS["detected"], label="Detected"),
               mpatches.Patch(color=COLORS["missed"], label="Missed")]
    ax.legend(handles=patches, facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    det_rate = sum(1 for d in detected if d) / len(detected) * 100
    ax.text(0.99, 0.92, f"Detection Rate: {det_rate:.0f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=12, color=COLORS["accent3"], fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_detection.png"), dpi=150)
    plt.close(fig)


def plot_strategies(rounds, out_dir):
    """Attack type and defense method timeline."""
    rns = [r["round_num"] for r in rounds]
    atk_types = [r.get("attack_strategy", {}).get("type", "unknown") for r in rounds]
    def_methods = [r.get("defend_strategy", {}).get("method", "unknown") for r in rounds]

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
    ax2.set_xticks(rns[::max(1, len(rns)//30)])

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
    atk_adapt = [bool(r.get("attacker_adapted", False)) for r in rounds]
    def_adapt = [bool(r.get("defender_adapted", False)) for r in rounds]

    fig, ax = plt.subplots(figsize=(12, 3))
    apply_dark_style(ax, "Adaptation Events", "Round", "")

    for i, rn in enumerate(rns):
        if atk_adapt[i]:
            ax.barh(1, 0.7, left=rn - 0.35, color=COLORS["accent2"], edgecolor="none", height=0.5)
        if def_adapt[i]:
            ax.barh(0, 0.7, left=rn - 0.35, color=COLORS["accent3"], edgecolor="none", height=0.5)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Defender", "Attacker"], fontsize=11)
    ax.set_xticks(rns[::max(1, len(rns)//30)])
    ax.set_ylim(-0.5, 1.8)

    atk_count = sum(1 for a in atk_adapt if a)
    def_count = sum(1 for d in def_adapt if d)
    ax.text(0.99, 0.92, f"Attacker adapted: {atk_count}x  |  Defender adapted: {def_count}x",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=COLORS["text_dim"])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_adaptation.png"), dpi=150)
    plt.close(fig)


def plot_attack_params(rounds, out_dir):
    """Evolution of attack parameters (main param + k) over rounds."""
    rns = [r["round_num"] for r in rounds]

    # Extract main parameter (scale / factor / sigma / c)
    main_params = []
    for r in rounds:
        p = r["attack_strategy"].get("params", {})
        main_params.append(p.get("scale", p.get("factor", p.get("sigma", p.get("c", None)))))

    # Extract k values
    k_values = []
    for r in rounds:
        p = r["attack_strategy"].get("params", {})
        k_val = p.get("k", None)
        k_values.append(k_val)

    has_main = not all(v is None for v in main_params)
    has_k = not all(v is None for v in k_values)

    if not has_main and not has_k:
        return  # nothing to plot

    n_plots = int(has_main) + int(has_k)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), squeeze=False)
    ax_idx = 0

    if has_main:
        ax = axes[ax_idx, 0]
        apply_dark_style(ax, "Attack Parameter Evolution", "Round", "Scale / Factor / Sigma / c")
        valid = [(rn, v) for rn, v in zip(rns, main_params) if v is not None]
        if valid:
            vrns, vvals = zip(*valid)
            ax.bar(vrns, vvals, color=COLORS["accent"], edgecolor="none", width=0.6, alpha=0.85)
            for rn, v in zip(vrns, vvals):
                ax.text(rn, v + 0.05, f"{v}", ha="center", va="bottom", fontsize=8, color=COLORS["text_dim"])
        ax_idx += 1

    if has_k:
        ax = axes[ax_idx, 0]
        apply_dark_style(ax, "Selective Targeting (k) Over Rounds", "Round", "k (weights targeted)")
        valid_k = [(rn, v) for rn, v in zip(rns, k_values) if v is not None]
        if valid_k:
            vrns, vvals = zip(*valid_k)
            ax.bar(vrns, vvals, color=COLORS["accent3"], edgecolor="none", width=0.6, alpha=0.85)
            for rn, v in zip(vrns, vvals):
                ax.text(rn, v + 0.5, f"{v}", ha="center", va="bottom", fontsize=8, color=COLORS["text_dim"])
        # Mark rounds where k was not used (all weights targeted)
        no_k = [rn for rn, v in zip(rns, k_values) if v is None]
        for rn in no_k:
            ax.annotate("all", (rn, 0), ha="center", va="bottom", fontsize=7, color=COLORS["text_dim"], alpha=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_attack_params.png"), dpi=150)
    plt.close(fig)


def plot_flagged_clients(rounds, out_dir):
    """Number of flagged clients per round."""
    rns = [r["round_num"] for r in rounds]
    flagged = [sum(1 for v in r.get("verdicts", []) if _coerce_bool(v.get("suspicious", False))) for r in rounds]
    total = [len(r.get("verdicts", [])) for r in rounds]

    fig, ax = plt.subplots(figsize=(12, 4))
    apply_dark_style(ax, "Flagged Clients Per Round", "Round", "Count")

    ax.bar(rns, total, color=COLORS["grid"], edgecolor="none", width=0.6, label="Total Clients", alpha=0.5)
    ax.bar(rns, flagged, color=COLORS["accent2"], edgecolor="none", width=0.6, label="Flagged", alpha=0.85)
    ax.set_xticks(rns[::max(1, len(rns)//30)])
    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_flagged_clients.png"), dpi=150)
    plt.close(fig)


# ─── Evaluation-metric chart generators ────────────────────────────────────────
def _extract_metric_series(rounds, key):
    """Return (round_nums, values) for a metric key inside each round['metrics']."""
    rns, vals = [], []
    for r in rounds:
        m = r.get("metrics")
        if m is None or key not in m:
            continue
        rns.append(r["round_num"])
        vals.append(m[key])
    return rns, vals


def plot_confusion_matrix(rounds, out_dir):
    """Stacked TP/FN/FP/TN counts per round."""
    rns, tp = _extract_metric_series(rounds, "tp")
    _, fn = _extract_metric_series(rounds, "fn")
    _, fp = _extract_metric_series(rounds, "fp")
    _, tn = _extract_metric_series(rounds, "tn")
    if not rns:
        return

    tp = np.array(tp); fn = np.array(fn); fp = np.array(fp); tn = np.array(tn)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    apply_dark_style(ax, "Confusion Matrix Per Round", "Round", "Client Count")

    width = 0.7
    ax.bar(rns, tp, width=width, color=COLORS["accent3"],   label="TP (malicious flagged)")
    ax.bar(rns, fn, width=width, bottom=tp,                 color=COLORS["accent2"], label="FN (malicious missed)")
    ax.bar(rns, fp, width=width, bottom=tp + fn,            color=COLORS["accent4"], label="FP (honest flagged)")
    ax.bar(rns, tn, width=width, bottom=tp + fn + fp,       color=COLORS["accent"],  label="TN (honest clean)")

    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def plot_detection_rates(rounds, out_dir):
    """Cumulative TPR / FPR / Recall over rounds.

    Per-round TPR is binary (single attacker), so cumulative rates are the
    more informative view.
    """
    rns, tp = _extract_metric_series(rounds, "tp")
    _, fn = _extract_metric_series(rounds, "fn")
    _, fp = _extract_metric_series(rounds, "fp")
    _, tn = _extract_metric_series(rounds, "tn")
    if not rns:
        return

    cum_tp = np.cumsum(tp); cum_fn = np.cumsum(fn)
    cum_fp = np.cumsum(fp); cum_tn = np.cumsum(tn)
    eps = 1e-12
    tpr = cum_tp / (cum_tp + cum_fn + eps)
    fpr = cum_fp / (cum_fp + cum_tn + eps)

    fig, ax = plt.subplots(figsize=(12, 5))
    apply_dark_style(ax, "Cumulative Detection Rates", "Round", "Rate")

    ax.plot(rns, tpr, "-o", color=COLORS["accent3"], linewidth=2, markersize=4,
            label="TPR / Recall", alpha=0.9)
    ax.plot(rns, fpr, "-s", color=COLORS["accent2"], linewidth=2, markersize=4,
            label="FPR", alpha=0.9)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, color=COLORS["grid"], linewidth=0.7, alpha=0.5)
    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.7, alpha=0.5)

    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=10, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "10_detection_rates.png"), dpi=150)
    plt.close(fig)


def plot_attack_success_rate(rounds, out_dir):
    """Per-round attack success markers + running ASR curve."""
    rns, asr_flags = _extract_metric_series(rounds, "attack_success")
    if not rns:
        return

    flags = np.array([1 if x else 0 for x in asr_flags])
    cumulative_asr = np.cumsum(flags) / np.arange(1, len(flags) + 1)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    apply_dark_style(ax, "Attack Success Rate", "Round", "ASR (cumulative)")

    # Per-round markers along the bottom strip
    success_rns = [rn for rn, f in zip(rns, flags) if f]
    blocked_rns = [rn for rn, f in zip(rns, flags) if not f]
    ax.scatter(success_rns, [-0.05] * len(success_rns), marker="x",
               s=40, color=COLORS["missed"], label="Attack passed")
    ax.scatter(blocked_rns, [-0.05] * len(blocked_rns), marker="o",
               s=30, color=COLORS["detected"], label="Attack blocked", alpha=0.7)

    ax.plot(rns, cumulative_asr, "-", color=COLORS["accent2"],
            linewidth=2.2, label="Cumulative ASR")
    ax.fill_between(rns, 0, cumulative_asr, color=COLORS["accent2"], alpha=0.12)
    ax.set_ylim(-0.1, 1.05)

    final_asr = cumulative_asr[-1]
    ax.text(0.99, 0.95, f"Final ASR: {final_asr:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=12, fontweight="bold",
            color=COLORS["accent2"])

    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "11_attack_success_rate.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_preservation(rounds, out_dir):
    """Accuracy Preservation Rate = current_accuracy / baseline_accuracy."""
    rns, apr = _extract_metric_series(rounds, "accuracy_preservation_rate")
    if not rns:
        return

    apr = np.array(apr)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    apply_dark_style(ax, "Accuracy Preservation Rate", "Round",
                    "APR (current / baseline)")

    ax.plot(rns, apr, "-o", color=COLORS["accent"], linewidth=2, markersize=4,
            label="APR")
    ax.axhline(1.0, color=COLORS["accent3"], linewidth=1.2, linestyle="--",
               alpha=0.7, label="Baseline (APR=1.0)")
    ax.fill_between(rns, apr, 1.0, where=(apr < 1.0),
                    color=COLORS["accent2"], alpha=0.15, label="Degradation")

    lo = float(min(apr.min(), 0.95))
    hi = float(max(apr.max(), 1.02))
    ax.set_ylim(lo - 0.02, hi + 0.02)

    ax.text(0.01, 0.06,
            f"Final APR: {apr[-1]:.4f}   |   Min APR: {apr.min():.4f}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, color=COLORS["text_dim"])

    ax.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "12_accuracy_preservation.png"), dpi=150)
    plt.close(fig)


# ─── HTML report ────────────────────────────────────────────────────────────────
def generate_html_report(rounds, out_dir, metrics_summary=None):
    charts = sorted([f for f in os.listdir(out_dir) if f.endswith(".png")])
    rns = [r["round_num"] for r in rounds]
    det_rate = sum(1 for r in rounds if bool(r["attack_detected"])) / len(rounds) * 100
    avg_acc = np.mean([r["test_accuracy"] for r in rounds])
    atk_adapts = sum(1 for r in rounds if bool(r.get("attacker_adapted", False)))
    def_adapts = sum(1 for r in rounds if bool(r.get("defender_adapted", False)))

    # Optional aggregate metrics row (only shown when summary.json is present)
    metrics_html = ""
    if metrics_summary and "aggregate" in metrics_summary:
        agg = metrics_summary["aggregate"]
        metrics_html = f"""
<h2 style="text-align:center;color:#888;font-size:1rem;margin:1.5rem 0 .8rem">Evaluation Metrics</h2>
<div class="stats">
  <div class="stat"><div class="val">{agg.get('attack_success_rate', 0):.3f}</div><div class="lbl">Attack Success Rate</div></div>
  <div class="stat"><div class="val">{agg.get('tpr', 0):.3f}</div><div class="lbl">TPR</div></div>
  <div class="stat"><div class="val">{agg.get('fpr', 0):.3f}</div><div class="lbl">FPR</div></div>
  <div class="stat"><div class="val">{agg.get('recall', 0):.3f}</div><div class="lbl">Recall</div></div>
  <div class="stat"><div class="val">{agg.get('accuracy_preservation_rate', 0):.3f}</div><div class="lbl">Accuracy Preservation</div></div>
</div>
"""

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
{metrics_html}
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
    parser.add_argument("--metrics-summary", default="logs/metrics/summary.json",
                        help="Path to aggregate metrics summary.json (optional)")
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

    # ─── Evaluation-metric charts (only when per-round metrics are present)
    metrics_summary = load_metrics_summary(args.metrics_summary)
    if rounds_have_metrics(rounds):
        plot_confusion_matrix(rounds, args.out_dir)
        print("  ✓ 09_confusion_matrix.png")

        plot_detection_rates(rounds, args.out_dir)
        print("  ✓ 10_detection_rates.png")

        plot_attack_success_rate(rounds, args.out_dir)
        print("  ✓ 11_attack_success_rate.png")

        plot_accuracy_preservation(rounds, args.out_dir)
        print("  ✓ 12_accuracy_preservation.png")
    else:
        print("[INFO] No 'metrics' key in round JSONs — skipping evaluation-metric charts.")
        print("       Re-run the simulation with the metrics module enabled to generate them.")

    report_path = generate_html_report(rounds, args.out_dir, metrics_summary)
    print(f"\n[DONE] HTML report → {report_path}")
    print(f"       Open in browser: file://{os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
