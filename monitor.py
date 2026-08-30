#!/usr/bin/env python3
"""RL health monitor — is the defender actually learning, or collapsing?

Reads logs/round_data/rounds.jsonl (appended every round, so this works WHILE
training runs) and reports learning trends + collapse flags, plus a health.png.
Run it any time:  python monitor.py

The defender is the only learner; the attack is the deterministic label-flip
ladder, so its side of the report is a DESCRIPTION of what the defender has been
facing, not a training diagnostic.

Signals it tracks
-----------------
Learning (good):
  * the defender's GRPO group-mean reward (train.mean_reward) trends UP;
  * FPR trends DOWN while TPR stays high;
  * the flip fraction walks DOWN and resets — the ladder saw-tooth. A ladder
    parked at 100% for the whole run means the defender is catching nothing.
Collapse (bad):
  * zero_advantage_fraction ~1  -> all G samples tie -> NO gradient;
  * flag-rate ~1 (flag everything) or ~0 (flag nothing) -> degenerate classifier;
  * reward variance ~0 -> stuck;
  * induced drop ~0 at every ladder level -> the attack is not doing damage, so a
    high TPR means the defender learned to detect a formality.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _utf8_stdout():
    """Force UTF-8 on stdout before printing the report.

    This report uses ✓/⚠ marks; on Windows stdout defaults to cp1252 (especially
    when redirected to a file or a pipe), which raises UnicodeEncodeError
    mid-report. Same defensive fix as ``main.setup_logging`` and ``core.debug``.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):   # non-reconfigurable stream: fall through
        pass


def _soft_mal(v: dict) -> float:
    c = max(0.0, min(1.0, float(v.get("confidence", 0.0))))
    return 0.5 + 0.5 * c if v.get("is_suspicious") else 0.5 - 0.5 * c


def read_round_logs(log_dir: str) -> list[dict]:
    """Load raw round logs, newest sink first.

    Rounds are appended to ``rounds.jsonl`` (one JSON object per line). Legacy
    runs wrote one ``round_NNN.json`` per round; those are still read so old logs
    keep working. Records are merged by ``round_num`` (JSONL wins) and sorted.
    """
    by_round: dict[int, dict] = {}
    for path in sorted(Path(log_dir).glob("round_*.json"),
                       key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))):
        try:
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
            by_round[int(r["round_num"])] = r
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            continue
    jsonl = Path(log_dir) / "rounds.jsonl"
    if jsonl.exists():
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


def load_rounds(log_dir: str) -> list[dict]:
    rows = []
    for r in read_round_logs(log_dir):
        poisoned = set(r.get("poisoned_client_ids", []))
        labels = r.get("predicted_labels", [])
        tp = sum(1 for v in labels if v["client_id"] in poisoned and v["is_suspicious"])
        fn = sum(1 for v in labels if v["client_id"] in poisoned and not v["is_suspicious"])
        fp = sum(1 for v in labels if v["client_id"] not in poisoned and v["is_suspicious"])
        tn = sum(1 for v in labels if v["client_id"] not in poisoned and not v["is_suspicious"])
        n = max(1, len(labels))
        pois_v = [v for v in labels if v["client_id"] in poisoned]
        stealth = float(np.mean([1.0 - _soft_mal(v) for v in pois_v])) if pois_v else 0.0
        meta = r.get("attack_metadata") or {}
        train = meta.get("train", {})
        ladder = meta.get("ladder") or {}
        # The damage the round's flipped labels actually did: post-attack accuracy
        # vs THIS round's clean counterfactual. Older logs predate the field, so
        # fall back to the (baseline-relative) approximation they allow.
        drop = meta.get("induced_drop")
        if drop is None:
            drop = r.get("baseline_accuracy", 0.0) - r.get("test_accuracy", 0.0)
        rows.append({
            "round": r["round_num"],
            "learner": r.get("learning_agent", "none"),
            # Reported damage on the goal's scale. `attacker_reward` is the legacy
            # field name from before the attack stopped being a trained policy.
            "attack_eff": r.get("attack_effectiveness",
                                r.get("attacker_reward", float("nan"))),
            "def_reward": r.get("defender_reward", 0.0),
            "drop": drop,
            "stealth": stealth,
            "flag_rate": (tp + fp) / n,
            "tpr": tp / (tp + fn) if (tp + fn) else float("nan"),
            "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
            "zero_adv": train.get("zero_advantage_fraction", float("nan")),
            "train_mean_r": train.get("mean_reward", float("nan")),
            # The attack schedule: the ladder level this round was sent at, and
            # whether the defense caught it.
            "flip_fraction": meta.get("flip_fraction", float("nan")),
            "n_flipped": meta.get("n_flipped", 0),
            "caught": bool(ladder.get("caught", False)),
            "ladder_event": ladder.get("event", ""),
        })
    return rows


def _series(rows, key, learner=None):
    out = [(r["round"], r[key]) for r in rows
           if (learner is None or r["learner"] == learner) and r[key] == r[key]]
    return [x[0] for x in out], np.array([x[1] for x in out], dtype=float)


def _trend(vals: np.ndarray):
    """(early_mean, late_mean, slope) over the series split into thirds."""
    if len(vals) < 3:
        return (float(np.mean(vals)) if len(vals) else 0.0,) * 2 + (0.0,)
    third = max(1, len(vals) // 3)
    early, late = float(np.mean(vals[:third])), float(np.mean(vals[-third:]))
    slope = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])
    return early, late, slope


def _segments(rounds, vals, max_gap=1):
    """Split a per-agent series into contiguous training runs, breaking wherever
    the round number jumps by more than ``max_gap`` (the agent was frozen in
    between). Plotting each run separately stops a line from bridging across a
    frozen phase (which would falsely imply the agent trained during the gap)."""
    segs = []
    n = len(rounds)
    if n == 0:
        return segs
    start = 0
    for i in range(1, n):
        if rounds[i] - rounds[i - 1] > max_gap:
            segs.append((rounds[start:i], vals[start:i]))
            start = i
    segs.append((rounds[start:], vals[start:]))
    return segs


def _shade_phases(ax, rows):
    """Shade the background over rounds where the defender was training, so the
    honest FL interludes between phases stand out as gaps."""
    colors = {"defender": "#43E97B"}
    if not rows:
        return
    start = rows[0]["round"]
    cur = rows[0]["learner"]
    prev = start
    for r in rows[1:]:
        if r["learner"] != cur:
            if cur in colors:
                ax.axvspan(start, prev, color=colors[cur], alpha=0.07, lw=0)
            start, cur = r["round"], r["learner"]
        prev = r["round"]
    if cur in colors:
        ax.axvspan(start, prev, color=colors[cur], alpha=0.07, lw=0)


def _flag(cond, bad_msg, ok_msg):
    return f"  ⚠ {bad_msg}" if cond else f"  ✓ {ok_msg}"


def analyze(rows, window: int):
    n_train = sum(r["learner"] == "defender" for r in rows)
    print("=" * 70)
    print(f"RL HEALTH REPORT — {len(rows)} rounds ({n_train} defender-training, "
          f"{len(rows) - n_train} other)")
    print("=" * 70)

    def recent(rows_a, key):
        _, v = _series(rows_a, key)
        return float(np.mean(v[-window:])) if len(v) else float("nan")

    de = [r for r in rows if r["learner"] == "defender"]

    # ---- The attack it faced (a description, not a training diagnostic) ----
    print("\n[ATTACK]  label-flip ladder — what the defender was up against")
    if rows:
        _, ff = _series(rows, "flip_fraction")
        de_, dl, ds = _trend(_series(rows, "drop")[1])
        caught_rate = float(np.mean([r["caught"] for r in rows]))
        resets = sum(1 for r in rows if r["ladder_event"] == "reset")
        steps = sum(1 for r in rows if r["ladder_event"] == "step_down")
        holds = sum(1 for r in rows if r["ladder_event"] == "hold")
        if len(ff):
            print(f"  flip fraction:     min={np.nanmin(ff):.0%}  max={np.nanmax(ff):.0%}  "
                  f"recent={float(np.nanmean(ff[-window:])):.0%}")
        print(f"  ladder:            {steps} step-down, {holds} hold, {resets} reset "
              f"(caught {caught_rate:.0%} of rounds)")
        print(f"  induced acc-drop:  early={de_:+.3f}  late={dl:+.3f}  slope={ds:+.4f}")
        # A ladder that never moves is the single most important thing to notice:
        # the defender is then being trained on one static attack level forever.
        print(_flag(steps + resets == 0,
                    "ladder NEVER stepped -> defender caught (almost) nothing; it is "
                    "training against one static attack level",
                    "ladder is adapting (the defender is landing detections)"))
        print(_flag(abs(dl) < 0.005,
                    "induced drop ~0 -> the flipped labels are not damaging the model, "
                    "so a high TPR means detecting a formality",
                    "the attack is doing measurable damage"))
    else:
        print("  (no rounds yet)")

    # ---- Defender ----
    print("\n[DEFENDER]  (its own training rounds)")
    if de:
        e, l, s_ = _trend(_series(de, "train_mean_r")[1])
        fe, fl_, fs = _trend(_series(de, "fpr")[1])
        te, tl, ts = _trend(_series(de, "tpr")[1])
        fr = recent(de, "flag_rate")
        zadv = recent(de, "zero_adv")
        rew_var = float(np.var(_series(de, "train_mean_r")[1][-window:])) if de else 0.0
        print(f"  GRPO mean-reward:  early={e:+.3f}  late={l:+.3f}  slope={s_:+.4f}")
        print(f"  TPR (recall):      early={te:.3f}  late={tl:.3f}")
        print(f"  FPR (false-alarm): early={fe:.3f}  late={fl_:.3f}  slope={fs:+.4f}")
        print(f"  recent flag_rate={fr:.2f}  zero_adv={zadv:.2f}  reward_var={rew_var:.4f}")
        print(_flag(l <= e, "reward NOT improving", "reward improving"))
        print(_flag(fl_ >= fe and fl_ > 0.4, "FPR not falling -> over-flagging benign clients",
                    "FPR controlled / falling"))
        print(_flag(fr > 0.95, "flagging ~ALL clients -> flag-all collapse", "not flag-all"))
        print(_flag(fr < 0.05, "flagging ~NO clients -> flag-none collapse", "not flag-none"))
        print(_flag(zadv > 0.7, "high zero-advantage -> little/no gradient",
                    "advantage signal present"))
        print(_flag(rew_var < 1e-4, "reward variance ~0 -> stuck", "reward varies (exploring)"))
    else:
        print("  (no defender-learning rounds yet)")

    print("\nReading the two together: the defender's reward should rise WHILE the flip")
    print("fraction walks down. A rising reward at a pinned 100% flip means it is winning")
    print("the easiest version of the problem; a falling reward after a reset is expected,")
    print("because the ladder just handed it full-strength poison again.")
    print("=" * 70)


def plot(rows, out: str, window: int):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    def roll(v):
        if len(v) < 2:
            return v
        w = min(window, len(v))
        return np.convolve(v, np.ones(w) / w, mode="same")

    # 1: the defender's GRPO mean-reward. CONNECTED lines that BREAK across rounds
    #    where it wasn't training (no bridging across the FL interludes).
    _shade_phases(ax[0, 0], rows)
    rd, vd = _series([r for r in rows if r["learner"] == "defender"], "train_mean_r")
    ax[0, 0].set_title("Defender GRPO mean reward — want UP (shaded = training)")
    for i, (sr, sv) in enumerate(_segments(rd, vd)):
        ax[0, 0].plot(sr, roll(sv), color="#43E97B", label="defender" if i == 0 else None)
    ax[0, 0].axhline(0.0, color="#999999", lw=0.6, ls="--")
    ax[0, 0].legend(); ax[0, 0].set_xlabel("round")

    # 2: defender TPR/FPR
    _shade_phases(ax[0, 1], rows)
    rt, vt = _series(rows, "tpr"); rf, vf = _series(rows, "fpr")
    ax[0, 1].set_title("Defender TPR (up) vs FPR (down)")
    if len(vt): ax[0, 1].plot(rt, roll(vt), color="#2E7D32", label="TPR")
    if len(vf): ax[0, 1].plot(rf, roll(vf), color="#C62828", label="FPR")
    ax[0, 1].set_ylim(-0.05, 1.05); ax[0, 1].legend(); ax[0, 1].set_xlabel("round")

    # 3: zero-advantage fraction (want LOW)
    _shade_phases(ax[1, 0], rows)
    rz, vz = _series(rows, "zero_adv")
    ax[1, 0].set_title("zero-advantage fraction — want LOW (high = no gradient)")
    if len(vz): ax[1, 0].plot(rz, roll(vz), color="#F7971E")
    ax[1, 0].set_ylim(-0.05, 1.05); ax[1, 0].set_xlabel("round")

    # 4: THE LADDER. Plotted RAW (no rolling mean) — the saw-tooth is the signal,
    #    and smoothing it away is exactly what would hide a defender that stopped
    #    catching anything. The induced drop shares the axis so the two can be read
    #    against each other: damage should track the flip fraction.
    rl_, vl = _series(rows, "flip_fraction")
    rdp, vdp = _series(rows, "drop")
    ax[1, 1].set_title("Attack ladder: flip fraction (raw) vs induced acc-drop")
    if len(vl): ax[1, 1].plot(rl_, vl, color="#FF6584", lw=1.0, label="flip fraction")
    if len(vdp): ax[1, 1].plot(rdp, roll(vdp), color="#6C63FF", label="induced acc-drop")
    ax[1, 1].axhline(0.0, color="#999999", lw=0.6, ls="--")
    ax[1, 1].legend(); ax[1, 1].set_xlabel("round")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[saved] {out}")


# State codes for the per-client poison/detection map.
_TN, _TP, _FN, _FP = 0, 1, 2, 3


def _client_state_matrix(log_dir: str):
    """Per-(client, round) confusion state for the 'who was poisoned vs who got
    flagged' map. Returns (rounds, clients, M) where M[i, j] is one of
    _TN/_TP/_FN/_FP for client i in round j, or NaN if that client didn't take
    part that round (so the cell renders blank rather than as a false 'clean').

      truly poisoned?   flagged by defender?   ->  state
      yes               yes                        TP  (caught)
      yes               no                         FN  (MISSED — evaded detection)
      no                yes                        FP  (false alarm on a benign client)
      no                no                         TN  (clean, correctly left alone)
    """
    files = sorted(Path(log_dir).glob("round_*.json"),
                   key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    raw = []
    clients: set = set()
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        raw.append(r)
        clients.update(r.get("poisoned_client_ids", []))
        clients.update(v["client_id"] for v in r.get("predicted_labels", []))
    clients = sorted(clients)
    cidx = {c: i for i, c in enumerate(clients)}
    rounds = [r["round_num"] for r in raw]
    M = np.full((len(clients), len(raw)), np.nan)
    for j, r in enumerate(raw):
        poisoned = set(r.get("poisoned_client_ids", []))
        labels = r.get("predicted_labels", [])
        flagged = {v["client_id"] for v in labels if v["is_suspicious"]}
        present = poisoned | {v["client_id"] for v in labels}
        for c in present:
            is_p, is_f = c in poisoned, c in flagged
            M[cidx[c], j] = (_TP if is_p and is_f else
                             _FN if is_p else
                             _FP if is_f else _TN)
    return rounds, clients, M


def plot_clients(log_dir: str, out: str):
    """Save a per-client × per-round map answering, at a glance, which clients
    were actually poisoned and which the defender flagged each round — colour-
    coded by whether the defender's call was right (green=caught, red=missed,
    amber=false alarm, grey=clean & correctly ignored)."""
    rounds, clients, M = _client_state_matrix(log_dir)
    if not clients:
        print("[skip] no per-client data for poison-vs-detection map")
        return
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    palette = {_TN: "#E6E6EE", _TP: "#2E9E5B", _FN: "#E23E57", _FP: "#F4A93C"}
    cmap = ListedColormap([palette[_TN], palette[_TP], palette[_FN], palette[_FP]])
    cmap.set_bad("white")  # client absent that round
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    h = max(2.4, 0.55 * len(clients) + 1.6)
    w = max(8.0, min(24.0, 0.16 * len(rounds) + 4.0))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, norm=norm,
              interpolation="nearest")

    ax.set_yticks(range(len(clients)))
    ax.set_yticklabels([f"client {c}" for c in clients])
    n = len(rounds)
    step = max(1, n // 30)
    xt = list(range(0, n, step))
    ax.set_xticks(xt)
    ax.set_xticklabels([rounds[i] for i in xt])
    ax.set_xlabel("round")
    ax.set_title("Poisoned (ground truth) vs flagged (defender) — per client, per round")

    # thin separators between client rows; vertical lines omitted to avoid
    # clutter when there are many rounds.
    for y in np.arange(0.5, len(clients) - 0.5):
        ax.axhline(y, color="white", lw=1.0)

    handles = [
        mpatches.Patch(color=palette[_TP], label="poisoned → caught (TP)"),
        mpatches.Patch(color=palette[_FN], label="poisoned → MISSED (FN)"),
        mpatches.Patch(color=palette[_FP], label="clean → false alarm (FP)"),
        mpatches.Patch(color=palette[_TN], label="clean → ok (TN)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False, fontsize=9)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser(description="RL learning/collapse monitor")
    ap.add_argument("--log-dir", default="logs/round_data")
    ap.add_argument("--out", default="logs/monitor/health.png")
    ap.add_argument("--clients-out", default="logs/monitor/poison_vs_detection.png",
                    help="per-client poisoned-vs-flagged map")
    ap.add_argument("--window", type=int, default=20, help="rolling/recent window")
    args = ap.parse_args()
    _utf8_stdout()

    rows = load_rounds(args.log_dir)
    if not rows:
        print(f"No round_*.json in {args.log_dir}")
        return
    analyze(rows, args.window)
    plot(rows, args.out, args.window)
    plot_clients(args.log_dir, args.clients_out)


if __name__ == "__main__":
    main()
