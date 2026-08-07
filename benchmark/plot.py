"""Per-round benchmark graphs.

Draws a 4-panel figure from the per-round history each defense recorded:
  1. test accuracy per round (per defense) — see when the attack lands / who holds up;
  2. rolling detection rate (TPR) per defense — how much of the attack each caught;
  3. rolling false-positive rate (FPR) per defense — how often each cried wolf;
  4. attack strength per round = the attacker's actual perturbation, measured against
     the honest update it replaced.

Panel 4 used to plot the undefended (fedavg) accuracy as a proxy for attack strength.
Under the project default (``benign_retrain_each_round: false``) the weight-averaging
defenses rebuild their global from the same frozen benign weights every round, so
that line is flat whether the attacker is devastating or doing nothing at all — it
measured the environment, not the attack. It now plots the perturbation the attacker
actually produced, with the threshold below which an attack cannot be distinguished
from honest non-IID variation. Histories saved before that was recorded fall back to
the old panel.

Auto-invoked by run_benchmark. Also runnable standalone to RE-PLOT a saved history
without re-running the (slow, GPU) benchmark:

    python -m benchmark.plot --dataset cifar10
    python -m benchmark.plot --history logs/mnist/benchmark/history.json
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
    "multikrum": "#8E24AA",     # violet
}


#: Shared with the harness's warning and the report's note so the plot's threshold
#: line cannot drift from them. ``benchmark.metrics`` is torch-free, which keeps this
#: module importable without a DL stack (tests/test_benchmark.py relies on that).
from benchmark.metrics import INERT_POISON_RATIO  # noqa: E402


def _mean_poison(round_record: dict):
    """Mean perturbation ratio for one round, or ``None`` when it was not recorded
    (a history saved before the measurement existed, or a round with no poison)."""
    ratios = round_record.get("poison_ratios")
    if not ratios:
        return None
    finite = [float(x) for x in ratios if float(x) == float(x) and float(x) != float("inf")]
    return (sum(finite) / len(finite)) if finite else None


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

    names = [n for n in ["fedavg", "oracle", "llm_defender", "fltrust", "defl", "dnc", "multikrum"] if n in history]
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

    # 4: attack strength per round — the attacker's actual perturbation size. The
    #    attack is held fixed across defenses, so any defense's history carries it.
    ref = "fedavg" if "fedavg" in history else names[0]
    h = history[ref]
    strength = [_mean_poison(r) for r in h]
    if any(s is not None for s in strength):
        rounds = [r["round"] for r in h if _mean_poison(r) is not None]
        values = [s for s in strength if s is not None]
        ax[1, 1].plot(rounds, values, color="#D7263D", lw=1.2, label="poison size")
        ax[1, 1].axhline(INERT_POISON_RATIO, color="#999999", ls="--", lw=0.8,
                         label=f"inert below {INERT_POISON_RATIO}")
        ax[1, 1].set_yscale("log")
        ax[1, 1].set_title("Attack strength: poison perturbation / honest update\n"
                           "(below the dashed line the attack is indistinguishable "
                           "from non-IID noise)")
        ax[1, 1].set_xlabel("round"); ax[1, 1].set_ylabel("x honest update")
        ax[1, 1].legend(fontsize=8)
    else:
        # Pre-measurement history: fall back to the old undefended-accuracy proxy.
        ax[1, 1].plot([r["round"] for r in h], [r["accuracy"] for r in h],
                      color=_COLORS.get(ref, "#9AA0A6"), lw=1.2)
        ax[1, 1].axhline(baseline_accuracy, color="#999999", ls="--", lw=0.8)
        ax[1, 1].set_title(f"Attack strength: undefended ({ref}) accuracy per round "
                           f"(no poison-size data in this history)")
        ax[1, 1].set_xlabel("round"); ax[1, 1].set_ylim(-0.03, 1.03)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    import os

    from core.run_config import run_paths
    from data.datasets import DATASET_NAMES, DEFAULT_DATASET

    ap = argparse.ArgumentParser(description="Re-plot benchmark per-round history")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, metavar="NAME",
                    help=f"which benchmark run to re-plot: {', '.join(DATASET_NAMES)} "
                         f"(default: {DEFAULT_DATASET}); sets the default paths to "
                         f"logs/<dataset>/benchmark/")
    ap.add_argument("--history", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()
    bench_dir = run_paths(args.dataset)["benchmark_dir"]
    args.history = args.history or os.path.join(bench_dir, "history.json")
    args.out = args.out or os.path.join(bench_dir, "benchmark.png")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    with open(args.history) as f:
        blob = json.load(f)
    png = plot_history(blob["history"], blob["baseline_accuracy"], args.out, window=args.window)
    print(f"[saved] {png}" if png else "no plot produced (matplotlib missing)")


if __name__ == "__main__":
    main()
