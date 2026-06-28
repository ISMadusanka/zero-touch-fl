"""Per-round benchmark graphs.

Draws a 4-panel figure from the per-round history each defense recorded:
  1. test accuracy per round (per defense) — see when the attack lands / who holds up;
  2. rolling detection rate (TPR) per defense — how much of the attack each caught;
  3. rolling false-positive rate (FPR) per defense — how often each cried wolf;
  4. attack strength per round = the undefended (fedavg) accuracy (dips = strong attacks).

Auto-invoked by run_benchmark. Also runnable standalone to RE-PLOT a saved history
without re-running the (slow, GPU) benchmark:

    python -m benchmark.plot --history logs/benchmark/history.json
"""
import argparse
import json
import logging

logger = logging.getLogger("benchmark")

_COLORS = {
    "fedavg": "#9AA0A6",        # grey  — no defense
    "oracle": "#2E7D32",        # green — upper bound
    "llm_defender": "#6C63FF",  # purple
    "fltrust": "#F7971E",       # orange
    "defl": "#D7263D",          # red
    "dnc": "#00897B",           # teal
}


def _rolling_rate(num, den, window):
    """Windowed rate: sum(num) / sum(den) over the trailing `window` rounds."""
    out = []
    for i in range(len(num)):
        lo = max(0, i - window + 1)
        n, d = sum(num[lo:i + 1]), sum(den[lo:i + 1])
        out.append(n / d if d else float("nan"))
    return out


def plot_history(history: dict, baseline_accuracy: float, out_path: str, window: int = 20):
    """Render the 4-panel figure. Returns out_path, or None if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:                      # pragma: no cover
        logger.warning(f"matplotlib unavailable ({e}); skipping graphs (history JSON still saved)")
        return None
    import os

    names = [n for n in ["fedavg", "oracle", "llm_defender", "fltrust", "defl", "dnc"] if n in history]
    names += [n for n in history if n not in names]
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


def main():
    ap = argparse.ArgumentParser(description="Re-plot benchmark per-round history")
    ap.add_argument("--history", default="logs/benchmark/history.json")
    ap.add_argument("--out", default="logs/benchmark/benchmark.png")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    with open(args.history) as f:
        blob = json.load(f)
    png = plot_history(blob["history"], blob["baseline_accuracy"], args.out, window=args.window)
    print(f"[saved] {png}" if png else "no plot produced (matplotlib missing)")


if __name__ == "__main__":
    main()
