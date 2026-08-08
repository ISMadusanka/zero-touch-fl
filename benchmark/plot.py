"""Per-round and per-defense benchmark graphs.

Two figures are drawn from the per-round history each defense recorded:

``benchmark.png`` (per-round diagnostics, one line per defense, x-axis = round):
  1. test accuracy per round — see when the attack lands / who holds up;
  2. rolling detection rate (TPR) per defense — how much of the attack each caught;
  3. rolling false-positive rate (FPR) per defense — how often each cried wolf;
  4. attack strength per round = the undefended (fedavg) accuracy (dips = strong attacks);
  5. rolling ATTACK SUCCESS RATE — fraction of rounds (in window) a poisoned client
     evaded detection (fn > 0), i.e. the attack actually got something through;
  6. rolling DEFENSE SUCCESS RATE — fraction of rounds meeting BOTH a high-TPR and
     low-FPR bar (``rl.switch.defender_succeeded``'s own criterion), which is a
     STRICTER, non-complementary notion of "the defense did its job" than simply
     1 - attack success: a round can have zero attack success yet still fail this
     bar if the defense over-flagged honest clients that round (see FLTrust);
  7. clients FLAGGED per round that were actually poisoned (raw count = tp) against
     how many were poisoned that round (tp+fn) — "did the attack get flagged";
  8. total clients FLAGGED per round, poisoned or not (raw count = tp+fp) — "how
     trigger-happy is this defense" (makes a near-indiscriminate defense like
     FLTrust's FPR=71.7% immediately visible as a near-vertical-max line).

``benchmark_summary.png`` (per-defense summary, one bar/point per defense):
  9. final accuracy per defense, with the clean baseline as a reference line;
  10. detection rate (TPR) vs. mean accuracy drop — makes it visible when a
      defense's own detection numbers do NOT predict its realised protection
      (the central finding of this benchmark: DeFL has the best non-oracle
      detection rate yet the worst non-collapsed accuracy drop).

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
    "multikrum": "#8E24AA",     # violet
}

_ORDER = ["fedavg", "oracle", "llm_defender", "fltrust", "defl", "dnc", "multikrum"]


def _ordered_names(history: dict) -> list:
    names = [n for n in _ORDER if n in history]
    names += [n for n in history if n not in names]
    return names


def _color(n: str):
    return _COLORS.get(n)


def _rolling_rate(num, den, window):
    """Windowed rate: sum(num) / sum(den) over the trailing `window` rounds."""
    out = []
    for i in range(len(num)):
        lo = max(0, i - window + 1)
        n, d = sum(num[lo:i + 1]), sum(den[lo:i + 1])
        out.append(n / d if d else float("nan"))
    return out


def _round_tpr_fpr(r: dict) -> tuple:
    tp, fn, fp, tn = r["tp"], r["fn"], r["fp"], r["tn"]
    tpr = tp / (tp + fn) if (tp + fn) else 1.0    # no poisoned clients this round -> vacuously caught
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr, fpr


def plot_history(history: dict, baseline_accuracy: float, out_path: str, window: int = 20,
                 defender_min_tpr: float = 0.99, defender_max_fpr: float = 0.10):
    """Render the 8-panel per-round diagnostics figure.

    ``defender_min_tpr``/``defender_max_fpr`` set the joint bar for panel 6
    (DEFENSE SUCCESS RATE), matching ``rl.switch.SwitchConfig``'s own defaults so
    "defense success" here means the same thing it does during RL training.

    Returns ``out_path``, or ``None`` if matplotlib is absent.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:                      # pragma: no cover
        logger.warning(f"matplotlib unavailable ({e}); skipping graphs (history JSON still saved)")
        return None
    import os

    names = _ordered_names(history)
    fig, ax = plt.subplots(4, 2, figsize=(15, 17))

    # 1: test accuracy per round
    for n in names:
        h = history[n]
        ax[0, 0].plot([r["round"] for r in h], [r["accuracy"] for r in h],
                      label=n, color=_color(n), lw=1.2)
    ax[0, 0].axhline(baseline_accuracy, color="#999999", ls="--", lw=0.8, label="clean baseline")
    ax[0, 0].set_title("Test accuracy per round — higher = better")
    ax[0, 0].set_xlabel("round"); ax[0, 0].set_ylabel("accuracy")
    ax[0, 0].set_ylim(-0.03, 1.03); ax[0, 0].legend(fontsize=8)

    # 2: rolling detection rate (TPR)
    for n in names:
        h = history[n]
        tpr = _rolling_rate([r["tp"] for r in h], [r["tp"] + r["fn"] for r in h], window)
        ax[0, 1].plot([r["round"] for r in h], tpr, label=n, color=_color(n), lw=1.2)
    ax[0, 1].set_title(f"Detection rate (rolling TPR, window={window}) — higher = better")
    ax[0, 1].set_xlabel("round"); ax[0, 1].set_ylim(-0.05, 1.05); ax[0, 1].legend(fontsize=8)

    # 3: rolling false-positive rate (FPR)
    for n in names:
        h = history[n]
        fpr = _rolling_rate([r["fp"] for r in h], [r["fp"] + r["tn"] for r in h], window)
        ax[1, 0].plot([r["round"] for r in h], fpr, label=n, color=_color(n), lw=1.2)
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

    # 5: rolling ATTACK SUCCESS RATE — fn > 0 that round (a poisoned client got through)
    for n in names:
        h = history[n]
        succ = [1 if r["fn"] > 0 else 0 for r in h]
        rate = _rolling_rate(succ, [1] * len(h), window)
        ax[2, 0].plot([r["round"] for r in h], rate, label=n, color=_color(n), lw=1.2)
    ax[2, 0].set_title(f"Attack success rate (rolling, window={window}) — lower = better")
    ax[2, 0].set_xlabel("round"); ax[2, 0].set_ylim(-0.05, 1.05); ax[2, 0].legend(fontsize=8)

    # 6: rolling DEFENSE SUCCESS RATE — joint TPR>=min_tpr AND FPR<=max_fpr that round
    #    (rl.switch.defender_succeeded's own bar; NOT simply 1 - attack success, since
    #    a round with zero attack success can still fail this if FPR was too high).
    for n in names:
        h = history[n]
        ok = []
        for r in h:
            tpr, fpr = _round_tpr_fpr(r)
            ok.append(1 if (tpr >= defender_min_tpr and fpr <= defender_max_fpr) else 0)
        rate = _rolling_rate(ok, [1] * len(h), window)
        ax[2, 1].plot([r["round"] for r in h], rate, label=n, color=_color(n), lw=1.2)
    ax[2, 1].set_title(f"Defense success rate (TPR≥{defender_min_tpr:.2f} & FPR≤{defender_max_fpr:.2f}, "
                       f"rolling, window={window}) — higher = better")
    ax[2, 1].set_xlabel("round"); ax[2, 1].set_ylim(-0.05, 1.05); ax[2, 1].legend(fontsize=8)

    # 7: poisoned clients FLAGGED per round (tp) vs total poisoned that round (tp+fn)
    for n in names:
        h = history[n]
        ax[3, 0].plot([r["round"] for r in h], [r["tp"] for r in h],
                      label=n, color=_color(n), lw=1.1)
    ref_h = history[ref]
    ax[3, 0].plot([r["round"] for r in ref_h], [r["tp"] + r["fn"] for r in ref_h],
                  color="#333333", ls=":", lw=1.0, label="poisoned this round")
    ax[3, 0].set_title("Poisoned clients flagged per round (raw count) — attack caught")
    ax[3, 0].set_xlabel("round"); ax[3, 0].set_ylabel("# clients"); ax[3, 0].legend(fontsize=7)

    # 8: total clients FLAGGED per round (tp+fp) — shows over-flagging (e.g. FLTrust)
    for n in names:
        h = history[n]
        ax[3, 1].plot([r["round"] for r in h], [r["tp"] + r["fp"] for r in h],
                      label=n, color=_color(n), lw=1.1)
    ax[3, 1].set_title("Total clients flagged per round (raw count) — how trigger-happy")
    ax[3, 1].set_xlabel("round"); ax[3, 1].set_ylabel("# clients"); ax[3, 1].legend(fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_summary(history: dict, baseline_accuracy: float, out_path: str):
    """Render the 2-panel per-defense summary figure (final accuracy bar chart +
    detection-rate-vs-accuracy-drop scatter). Returns ``out_path``, or ``None`` if
    matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:                      # pragma: no cover
        logger.warning(f"matplotlib unavailable ({e}); skipping summary plot")
        return None
    import os

    names = _ordered_names(history)

    def _summary(n):
        h = history[n]
        tp = sum(r["tp"] for r in h); fn = sum(r["fn"] for r in h)
        fp = sum(r["fp"] for r in h); tn = sum(r["tn"] for r in h)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        mean_acc = sum(r["accuracy"] for r in h) / len(h) if h else 0.0
        final_acc = h[-1]["accuracy"] if h else 0.0
        return tpr, mean_acc, final_acc

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    # 9: final accuracy per defense (bar), baseline as a reference line
    finals = [_summary(n)[2] for n in names]
    bars = ax[0].bar(names, finals, color=[_color(n) for n in names])
    ax[0].axhline(baseline_accuracy, color="#333333", ls="--", lw=1.0, label="clean baseline")
    ax[0].set_title("Final accuracy per defense — higher = better")
    ax[0].set_ylabel("final test accuracy"); ax[0].set_ylim(0.0, 1.05)
    ax[0].tick_params(axis="x", rotation=30); ax[0].legend(fontsize=8)
    for b, v in zip(bars, finals):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                   ha="center", va="bottom", fontsize=7)

    # 10: detection rate (TPR) vs mean accuracy drop — detection quality != realized protection
    for n in names:
        tpr, mean_acc, _ = _summary(n)
        drop = baseline_accuracy - mean_acc
        ax[1].scatter(tpr, drop, color=_color(n), s=60, zorder=3)
        ax[1].annotate(n, (tpr, drop), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[1].set_title("Detection rate vs. realized damage — top-right is the surprising case")
    ax[1].set_xlabel("detection rate (TPR)"); ax[1].set_ylabel("mean accuracy drop vs. baseline")
    ax[1].set_xlim(-0.05, 1.05)
    ax[1].axhline(0.0, color="#cccccc", lw=0.8)
    ax[1].grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Re-plot benchmark per-round history")
    ap.add_argument("--history", default="logs/benchmark/history.json")
    ap.add_argument("--out", default="logs/benchmark/benchmark.png")
    ap.add_argument("--summary-out", default=None,
                    help="default: same dir as --out, named benchmark_summary.png")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--defender-min-tpr", type=float, default=0.99)
    ap.add_argument("--defender-max-fpr", type=float, default=0.10)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    with open(args.history) as f:
        blob = json.load(f)

    png = plot_history(blob["history"], blob["baseline_accuracy"], args.out, window=args.window,
                       defender_min_tpr=args.defender_min_tpr, defender_max_fpr=args.defender_max_fpr)
    print(f"[saved] {png}" if png else "no plot produced (matplotlib missing)")

    import os
    summary_out = args.summary_out or os.path.join(
        os.path.dirname(args.out) or ".", "benchmark_summary.png")
    spng = plot_summary(blob["history"], blob["baseline_accuracy"], summary_out)
    print(f"[saved] {spng}" if spng else "no summary plot produced (matplotlib missing)")


if __name__ == "__main__":
    main()
