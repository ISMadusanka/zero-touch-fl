"""Per-round benchmark graphs.

Two figures, matching the two shapes the benchmark reports in:

* :func:`plot_history` — ONE attack against the defense panel. A 4-panel figure:
    1. test accuracy per round (per defense) — see when the attack lands / who holds up;
    2. rolling detection rate (TPR) per defense — how much of the attack each caught;
    3. rolling false-positive rate (FPR) per defense — how often each cried wolf;
    4. attack strength per round = the undefended (fedavg) accuracy (dips = strong attacks).

* :func:`plot_attack_comparison` — the attack x defense matrix:
    1. undefended accuracy per round, one line per attack (which attack hurts most,
       and how steadily);
    2. acc_drop heatmap (attack x defense) — the headline result;
    3. detection-rate heatmap (attack x defense);
    4. acc_drop grouped bars, so the per-defense ordering of attacks is readable.

Both are auto-invoked by run_benchmark. Also runnable standalone to RE-PLOT a saved
history without re-running the (slow, GPU) benchmark — the saved history is nested
``{attack: {defense: [...]}}`` for a matrix run and flat ``{defense: [...]}`` for a
single-attack one, and this handles either:

    python -m benchmark.plot --history logs/benchmark/history.json
"""
import argparse
import json
import logging

logger = logging.getLogger("benchmark")

_DEFENSE_ORDER = ["fedavg", "oracle", "llm_defender", "fltrust", "defl", "dnc", "multikrum"]

_COLORS = {
    "fedavg": "#9AA0A6",        # grey  — no defense
    "oracle": "#2E7D32",        # green — upper bound
    "llm_defender": "#6C63FF",  # purple
    "fltrust": "#F7971E",       # orange
    "defl": "#D7263D",          # red
    "dnc": "#00897B",           # teal
    "multikrum": "#8E24AA",     # violet
}

#: Attack colours. ``llm`` (the system under test) is deliberately the one strong,
#: saturated colour; the published baselines share a cooler family so the figure
#: reads as "ours vs the literature" at a glance.
_ATTACK_ORDER = ["clean", "llm", "lie", "min_max", "min_sum", "fang", "fang_krum",
                 "ipm", "mimic", "sign_flip", "noise", "scaling", "label_flip"]
_ATTACK_COLORS = {
    "clean": "#2E7D32",         # green — the no-attack control
    "llm": "#D7263D",
    "lie": "#1E88E5",
    "min_max": "#00897B",
    "min_sum": "#43A047",
    "fang": "#8E24AA",
    "fang_krum": "#5E35B1",
    "ipm": "#F7971E",
    "mimic": "#00ACC1",
    "sign_flip": "#6D4C41",
    "noise": "#9AA0A6",
    "scaling": "#C0CA33",
    "label_flip": "#EC407A",
}


def _rolling_rate(num, den, window):
    """Windowed rate: sum(num) / sum(den) over the trailing `window` rounds."""
    out = []
    for i in range(len(num)):
        lo = max(0, i - window + 1)
        n, d = sum(num[lo:i + 1]), sum(den[lo:i + 1])
        out.append(n / d if d else float("nan"))
    return out


def _ordered(names, preferred):
    """``preferred`` order first, then anything unrecognised, stably."""
    out = [n for n in preferred if n in names]
    return out + [n for n in names if n not in out]


def is_matrix_history(history: dict) -> bool:
    """True for a nested ``{attack: {defense: rounds}}`` history.

    A single-attack history maps a defense name to a LIST of per-round records; a
    matrix history maps an attack name to a dict of those, so one ``isinstance``
    on any value separates them without needing a version field.
    """
    return bool(history) and isinstance(next(iter(history.values())), dict)


def _mean_acc_drop(rounds, baseline: float) -> float:
    accs = [r["accuracy"] for r in rounds]
    return baseline - (sum(accs) / len(accs)) if accs else 0.0


def _detection_rate(rounds) -> float:
    tp = sum(r["tp"] for r in rounds)
    den = tp + sum(r["fn"] for r in rounds)
    return tp / den if den else float("nan")


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:                      # pragma: no cover
        logger.warning(f"matplotlib unavailable ({e}); skipping graphs "
                       f"(history JSON still saved)")
        return None


def _heatmap(ax, values, row_labels, col_labels, title, fmt, cmap):
    """Annotated attack x defense heatmap. ``values`` is a list of row-lists."""
    import numpy as np
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    im = ax.imshow(arr, cmap=cmap, aspect="auto",
                   vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    mid = 0.5 * (lo + hi)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#666")
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if v > mid else "black")
    ax.set_title(title, fontsize=10)
    return im


def plot_history(history: dict, baseline_accuracy: float, out_path: str, window: int = 20):
    """Render the single-attack 4-panel figure. Returns out_path, or None without matplotlib."""
    plt = _matplotlib()
    if plt is None:
        return None
    import os

    names = _ordered(list(history), _DEFENSE_ORDER)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    def color(n):
        return _COLORS.get(n)

    # 1: test accuracy per round
    for n in names:
        h = history[n]
        ax[0, 0].plot([r["round"] for r in h], [r["accuracy"] for r in h],
                      label=n, color=color(n), lw=1.2)
    ax[0, 0].axhline(baseline_accuracy, color="#999999", ls="--", lw=0.8, label="clean baseline")
    ax[0, 0].set_title("Test accuracy per round — higher = better")
    ax[0, 0].set_xlabel("round"); ax[0, 0].set_ylabel("accuracy")
    ax[0, 0].set_ylim(-0.03, 1.03); ax[0, 0].legend(fontsize=8)

    # 2: rolling detection rate (TPR)
    for n in names:
        h = history[n]
        tpr = _rolling_rate([r["tp"] for r in h], [r["tp"] + r["fn"] for r in h], window)
        ax[0, 1].plot([r["round"] for r in h], tpr, label=n, color=color(n), lw=1.2)
    ax[0, 1].set_title(f"Detection rate (rolling TPR, window={window}) — higher = better")
    ax[0, 1].set_xlabel("round"); ax[0, 1].set_ylim(-0.05, 1.05); ax[0, 1].legend(fontsize=8)

    # 3: rolling false-positive rate (FPR)
    for n in names:
        h = history[n]
        fpr = _rolling_rate([r["fp"] for r in h], [r["fp"] + r["tn"] for r in h], window)
        ax[1, 0].plot([r["round"] for r in h], fpr, label=n, color=color(n), lw=1.2)
    ax[1, 0].set_title(f"False-positive rate (rolling FPR, window={window}) — lower = better")
    ax[1, 0].set_xlabel("round"); ax[1, 0].set_ylim(-0.05, 1.05); ax[1, 0].legend(fontsize=8)

    # 4: attack strength per round = undefended (fedavg) accuracy (dips = strong attacks)
    ref = "fedavg" if "fedavg" in history else names[0]
    h = history[ref]
    ax[1, 1].plot([r["round"] for r in h], [r["accuracy"] for r in h],
                  color=_COLORS.get(ref, "#9AA0A6"), lw=1.2)
    ax[1, 1].axhline(baseline_accuracy, color="#999999", ls="--", lw=0.8)
    ax[1, 1].set_title(f"Attack strength: undefended ({ref}) accuracy per round — dips = strong attacks")
    ax[1, 1].set_xlabel("round"); ax[1, 1].set_ylim(-0.03, 1.03)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_attack_comparison(by_attack: dict, baseline_accuracy: float, out_path: str):
    """Render the attack x defense comparison figure. Returns out_path, or None."""
    plt = _matplotlib()
    if plt is None:
        return None
    import os
    import numpy as np

    attacks = _ordered(list(by_attack), _ATTACK_ORDER)
    defenses = _ordered(sorted({d for p in by_attack.values() for d in p}), _DEFENSE_ORDER)
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    # 1: undefended accuracy per round, one line per attack.
    ref = "fedavg" if all("fedavg" in by_attack[a] for a in attacks) else defenses[0]
    for a in attacks:
        h = by_attack[a].get(ref) or []
        ax[0, 0].plot([r["round"] for r in h], [r["accuracy"] for r in h], label=a,
                      color=_ATTACK_COLORS.get(a), lw=2.0 if a == "llm" else 1.1,
                      zorder=3 if a == "llm" else 2)
    ax[0, 0].axhline(baseline_accuracy, color="#999999", ls="--", lw=0.8,
                     label="clean baseline")
    ax[0, 0].set_title(f"Attack strength: undefended ({ref}) accuracy per round "
                       f"— lower = stronger attack", fontsize=10)
    ax[0, 0].set_xlabel("round"); ax[0, 0].set_ylabel("accuracy")
    # Auto-scaled, unlike the single-attack figure's fixed 0..1 axis: this panel's
    # whole job is to SEPARATE the attacks, and against a converged model they all
    # sit within a couple of accuracy points of each other, where a full-range axis
    # collapses every line onto the baseline.
    accs = [r["accuracy"] for a in attacks for r in (by_attack[a].get(ref) or [])]
    accs.append(baseline_accuracy)
    lo, hi = min(accs), max(accs)
    pad = max(0.02, 0.1 * (hi - lo))
    ax[0, 0].set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    ax[0, 0].legend(fontsize=7, ncol=2)

    drops = [[_mean_acc_drop(by_attack[a].get(d) or [], baseline_accuracy)
              if d in by_attack[a] else float("nan") for d in defenses] for a in attacks]
    det = [[_detection_rate(by_attack[a].get(d) or [])
            if d in by_attack[a] else float("nan") for d in defenses] for a in attacks]

    # 2 + 3: the two headline grids.
    _heatmap(ax[0, 1], drops, attacks, defenses,
             "Mean accuracy drop (attack x defense) — higher = stronger attack",
             "{:+.3f}", "magma")
    _heatmap(ax[1, 0], det, attacks, defenses,
             "Detection rate (attack x defense) — higher = defense caught more",
             "{:.0%}", "viridis")

    # 4: the same acc_drop as grouped bars, which reads the per-defense ORDERING of
    # attacks better than a colour scale does.
    width = 0.8 / max(1, len(attacks))
    x = np.arange(len(defenses))
    for i, a in enumerate(attacks):
        ax[1, 1].bar(x + i * width, drops[i], width, label=a,
                     color=_ATTACK_COLORS.get(a),
                     edgecolor="black" if a == "llm" else "none",
                     linewidth=0.8 if a == "llm" else 0)
    ax[1, 1].set_xticks(x + 0.4 - width / 2)
    ax[1, 1].set_xticklabels(defenses, rotation=25, ha="right", fontsize=8)
    ax[1, 1].axhline(0.0, color="#333333", lw=0.8)
    ax[1, 1].set_ylabel("mean accuracy drop")
    ax[1, 1].set_title("Mean accuracy drop per defense — taller = stronger attack",
                       fontsize=10)
    ax[1, 1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Re-plot benchmark per-round history")
    ap.add_argument("--history", default="logs/benchmark/history.json")
    ap.add_argument("--out", default="logs/benchmark/benchmark.png")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    with open(args.history) as f:
        blob = json.load(f)
    history, baseline = blob["history"], blob["baseline_accuracy"]
    if not is_matrix_history(history):
        png = plot_history(history, baseline, args.out, window=args.window)
        print(f"[saved] {png}" if png else "no plot produced (matplotlib missing)")
        return
    import os
    stem, ext = os.path.splitext(args.out)
    made = [plot_attack_comparison(history, baseline, f"{stem}_attacks{ext}")]
    for attack, panel in history.items():
        made.append(plot_history(panel, baseline, f"{stem}_{attack}{ext}",
                                 window=args.window))
    made = [p for p in made if p]
    print("\n".join(f"[saved] {p}" for p in made) if made
          else "no plot produced (matplotlib missing)")


if __name__ == "__main__":
    main()
