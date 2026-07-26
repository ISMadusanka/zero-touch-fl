"""Per-round graphs for the TARGETED benchmark.

The untargeted plot tracks overall accuracy, which barely moves under a working
targeted attack. These four panels track the thing that does move:

  1. TARGET-class recall per round, per defense — should COLLAPSE undefended and
     stay high under a defense that works;
  2. mean recall of the OTHER classes, per defense — should stay flat for
     everyone; if it dips, the attack was not actually targeted;
  3. per-class recall of the FINAL undefended model as a bar chart — the single
     clearest picture of "only this class broke";
  4. rolling detection rate per defense — did anyone notice.

Auto-invoked by ``benchmark.run_targeted_benchmark``. Also runnable standalone to
re-plot a saved history without re-running the (slow, GPU) benchmark:

    python -m benchmark.targeted_plot --history logs/targeted/benchmark/history.json
"""
import argparse
import json
import logging

from benchmark.plot import _COLORS, _rolling_rate

logger = logging.getLogger("benchmark")


def _series(h, key):
    """Pull ``targeted[key]`` per round, skipping rounds with no per-class data."""
    xs, ys = [], []
    for r in h:
        t = r.get("targeted")
        if t and t.get(key) is not None:
            xs.append(r["round"])
            ys.append(t[key])
    return xs, ys


def plot_targeted_history(history: dict, label: int, out_path: str, window: int = 20):
    """Render the 4-panel targeted figure. Returns out_path, or None without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                      # pragma: no cover
        logger.warning(f"matplotlib unavailable ({e}); skipping graphs (history JSON still saved)")
        return None
    import os

    order = ["fedavg", "oracle", "llm_defender", "fltrust", "defl", "dnc", "multikrum"]
    names = [n for n in order if n in history] + [n for n in history if n not in order]
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    # 1: target-class recall per round — the headline.
    for n in names:
        xs, ys = _series(history[n], "target_recall")
        if xs:
            ax[0, 0].plot(xs, ys, label=n, color=_COLORS.get(n), lw=1.2)
    ax[0, 0].set_title(f"Recall on the ATTACKED class ({label}) — LOWER = attack winning")
    ax[0, 0].set_xlabel("round"); ax[0, 0].set_ylabel("recall")
    ax[0, 0].set_ylim(-0.03, 1.03); ax[0, 0].legend(fontsize=8)

    # 2: the other classes — should be a flat line at the clean level.
    for n in names:
        xs, ys = _series(history[n], "others_recall")
        if xs:
            ax[0, 1].plot(xs, ys, label=n, color=_COLORS.get(n), lw=1.2)
    ax[0, 1].set_title("Mean recall on the OTHER classes — should stay FLAT (collateral damage)")
    ax[0, 1].set_xlabel("round"); ax[0, 1].set_ylim(-0.03, 1.03); ax[0, 1].legend(fontsize=8)

    # 3: final per-class recall of the undefended model — the money picture.
    ref = "fedavg" if "fedavg" in history else names[0]
    final = next((r["per_class"] for r in reversed(history[ref]) if r.get("per_class")), None)
    if final:
        idx = list(range(len(final)))
        colors = ["#D7263D" if c == label else "#9AA0A6" for c in idx]
        ax[1, 0].bar(idx, final, color=colors)
        ax[1, 0].set_xticks(idx)
        ax[1, 0].set_title(f"Final per-class recall, undefended ({ref}) — red = attacked class")
        ax[1, 0].set_xlabel("class"); ax[1, 0].set_ylabel("recall"); ax[1, 0].set_ylim(0, 1.03)
    else:
        ax[1, 0].set_visible(False)

    # 4: rolling detection rate.
    for n in names:
        h = history[n]
        tpr = _rolling_rate([r["tp"] for r in h], [r["tp"] + r["fn"] for r in h], window)
        ax[1, 1].plot([r["round"] for r in h], tpr, label=n, color=_COLORS.get(n), lw=1.2)
    ax[1, 1].set_title(f"Detection rate (rolling TPR, window={window}) — higher = better")
    ax[1, 1].set_xlabel("round"); ax[1, 1].set_ylim(-0.05, 1.05); ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Re-plot targeted benchmark per-round history")
    ap.add_argument("--history", default="logs/targeted/benchmark/history.json")
    ap.add_argument("--out", default="logs/targeted/benchmark/targeted.png")
    ap.add_argument("--label", type=int, default=None,
                    help="attacked class (default: read from the history file)")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    with open(args.history) as f:
        blob = json.load(f)
    label = args.label if args.label is not None else blob.get("target_label", 0)
    png = plot_targeted_history(blob["history"], label, args.out, window=args.window)
    print(f"[saved] {png}" if png else "no plot produced (matplotlib missing)")


if __name__ == "__main__":
    main()
