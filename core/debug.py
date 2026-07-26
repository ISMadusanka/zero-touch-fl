"""Structured debug logging for Phase-2 debug runs (``--debug``).

A single process-wide :data:`dbg` logger. **Disabled by default** — every public
method is a no-op until :meth:`DebugLogger.enable` is called, so importing it and
calling it from the hot path costs nothing in normal runs. When enabled (via
``python main.py --debug``) it:

  * prints clean, sectioned, human-readable events to the console, and
  * mirrors every event as a structured record into ``logs/debug.json``
    (a single JSON object rewritten after each round, so a crash still leaves a
    valid, complete-up-to-the-last-round file).

It captures the full Phase-2 picture *in the order the code executes it*: the
federated fine-tuning of each round, the exact attacker-LLM prompt and its G
rollouts, each rollout's attack plan + resulting poisoned-weight deltas, the
exact defender-LLM prompt + raw output + parsed verdicts, the per-rollout
rewards, the GRPO group advantages / loss / step, and the committed outcome.

This module only *observes and formats* — it never changes training behaviour.
Every method is wrapped so a formatting bug can never crash a run: on error it
prints a short notice and carries on.
"""

import json
import os
import sys
import time
from collections import deque

# Roles / stages used for tagging events (purely descriptive).
_BAR = "=" * 88
_SUB = "-" * 88
_DOT = ". " * 44


def _short(x, precision=4):
    """Round floats for display; pass everything else through."""
    try:
        if isinstance(x, float):
            return round(x, precision)
    except Exception:
        pass
    return x


class DebugLogger:
    """Process-wide structured debug logger. No-op unless :meth:`enable` ran."""

    # Ring-buffer size for the structured JSON sink. Generous enough to hold many
    # fully-detailed rounds, small enough that the per-round full rewrite in
    # ``flush()`` stays cheap and memory stays bounded on a long --debug run.
    MAX_EVENTS = 20000

    def __init__(self):
        self._enabled = False
        self._out = None
        self._path = None
        self._events = deque(maxlen=self.MAX_EVENTS)
        self._seq = 0
        self._run_meta = {}
        self._t0 = 0.0
        # Per-round context (set by round_header).
        self._round = None
        self._learner = None
        self._stage = None
        self._rollout = -1
        # Console de-dup of the (static) system prompts.
        self._sys_seen = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self, output_dir: str = "logs", filename: str = "debug.json",
               mode: str = "train", config_summary: dict | None = None):
        """Turn the logger on, opening the console writer + JSON sink."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            self._path = os.path.join(output_dir, filename)
            # Match main.setup_logging: a UTF-8 writer over stdout's fd so box /
            # arrow characters never raise on a cp1252 Windows console.
            self._out = open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
            self._t0 = time.monotonic()
            self._run_meta = {
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "config": config_summary or {},
            }
            self._events = deque(maxlen=self.MAX_EVENTS)
            self._seq = 0
            self._enabled = True
            self._banner(f"DEBUG MODE ON  (mode={mode})  ->  console + {self._path}")
            if config_summary:
                for k, v in config_summary.items():
                    self._line(f"  {k}: {v}")
            self.flush()
        except Exception as e:  # pragma: no cover - never break startup
            self._enabled = False
            print(f"[debug] failed to enable debug logging: {e}", file=sys.stderr)

    def close(self):
        if not self._enabled:
            return
        try:
            self._banner("DEBUG RUN COMPLETE")
            self.flush()
        except Exception:
            pass

    def flush(self):
        """Rewrite the JSON sink with everything captured so far."""
        if not self._enabled or not self._path:
            return
        try:
            payload = {
                "run": self._run_meta,
                "events_recorded": self._seq,
                "events_retained": len(self._events),
                "events_dropped": max(0, self._seq - len(self._events)),
                "events": list(self._events),
            }
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, self._path)
        except Exception as e:  # pragma: no cover
            print(f"[debug] flush failed: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Low-level console + JSON emit
    # ------------------------------------------------------------------
    def _line(self, s=""):
        if self._out is None:
            return
        self._out.write(s + "\n")

    def _banner(self, s):
        self._line()
        self._line(_BAR)
        self._line(s)
        self._line(_BAR)

    def _record(self, category, title, data):
        """Append one structured event to the bounded JSON buffer (full context).

        The buffer is a ring of ``max_events``: ``flush()`` rewrites the whole file
        every round, so an unbounded buffer made debug runs O(n²) in I/O and grew
        RAM without limit — which mattered because ``--debug`` used to silently run
        the full ``simulation_rounds`` budget. ``seq`` keeps counting past the
        eviction point so gaps in the file are obvious.
        """
        self._events.append({
            "seq": self._seq,
            "t": round(time.monotonic() - self._t0, 3),
            "round": self._round,
            "learner": self._learner,
            "stage": self._stage,
            "rollout": self._rollout if self._rollout >= 0 else None,
            "category": category,
            "title": title,
            "data": data,
        })
        self._seq += 1

    def _block(self, label, text):
        """Print a labelled, delimiter-framed block of raw (exact) text."""
        self._line(_DOT[:88] + f" [{label}]")
        self._line(text if isinstance(text, str) else json.dumps(text, default=str))

    def _verdict_lines(self, verdicts, poisoned_ids=None):
        pois = set(poisoned_ids or [])
        lines = []
        for v in verdicts:
            cid = getattr(v, "client_id", None)
            sus = getattr(v, "is_suspicious", None)
            conf = getattr(v, "confidence", None)
            reason = getattr(v, "reason", "")
            gt = ""
            if poisoned_ids is not None:
                truth = cid in pois
                tag = {(True, True): "TP", (False, False): "TN",
                       (False, True): "FN", (True, False): "FP"}.get((bool(sus), truth))
                gt = f"  [{tag}]"
            flag = "SUSPICIOUS" if sus else "benign    "
            lines.append(f"    client {cid}: {flag} conf={_short(conf)}  \"{reason}\"{gt}")
        return lines

    # ------------------------------------------------------------------
    # Weight-stat helpers (torch imported lazily, only when enabled)
    # ------------------------------------------------------------------
    @staticmethod
    def _sd_l2(sd) -> float:
        # Uses only tensor *methods* (no torch.* calls) so this module never
        # needs to import torch — keeping it importable on a CPU-only box.
        sq = 0.0
        for v in sd.values():
            sq += float(v.detach().float().pow(2).sum())
        return sq ** 0.5

    @staticmethod
    def _delta_stats(benign: dict, poisoned: dict, precision: int = 4):
        layers = {}
        tot_d = tot_b = tot_p = 0.0
        gmax = 0.0
        for k in benign:
            b = benign[k].detach().float()
            p = poisoned[k].detach().float()
            d = p - b
            dn = float(d.pow(2).sum())
            bn = float(b.pow(2).sum())
            pn = float(p.pow(2).sum())
            mx = float(d.abs().max()) if d.numel() else 0.0
            layers[k] = {
                "delta_l2": round(dn ** 0.5, precision),
                "benign_l2": round(bn ** 0.5, precision),
                "poisoned_l2": round(pn ** 0.5, precision),
                "max_abs_delta": round(mx, precision),
            }
            tot_d += dn
            tot_b += bn
            tot_p += pn
            gmax = max(gmax, mx)
        total = {
            "delta_l2": round(tot_d ** 0.5, precision),
            "benign_l2": round(tot_b ** 0.5, precision),
            "poisoned_l2": round(tot_p ** 0.5, precision),
            "max_abs_delta": round(gmax, precision),
        }
        return total, layers

    def _guard(fn):
        """Decorator: no-op when disabled, and never raise out of a log call."""
        def wrapper(self, *a, **kw):
            if not self._enabled:
                return
            try:
                return fn(self, *a, **kw)
            except Exception as e:  # pragma: no cover - logging must not crash training
                try:
                    self._line(f"[debug] error in {fn.__name__}: {e}")
                except Exception:
                    pass
        wrapper.__name__ = getattr(fn, "__name__", "wrapper")
        return wrapper

    # ------------------------------------------------------------------
    # Phase 2 semantic events
    # ------------------------------------------------------------------
    @_guard
    def phase_event(self, title: str, **data):
        self._line()
        self._line(f"### {title}  " + "  ".join(f"{k}={v}" for k, v in data.items()))
        self._record("phase", title, data)

    @_guard
    def round_header(self, round_num, learner, opponent, phase_index, phase_round,
                     pool_ids, budget, global_accuracy, G, scoring_opp_temp, opp_temp):
        self._round = round_num
        self._learner = learner
        self._stage = "setup"
        self._rollout = -1
        self._banner(
            f"ROUND {round_num}  |  learner={str(learner).upper()} "
            f"(frozen opponent={opponent})  |  phase {phase_index}.{phase_round}"
        )
        self._line(
            f"controllable_pool={list(pool_ids)}  budget={budget}   "
            f"global_accuracy={_short(global_accuracy)}   "
            f"G={G}  scoring_opp_temp={scoring_opp_temp}  commit_opp_temp={opp_temp}"
        )
        self._record("round_header", "round_start", {
            "round": round_num, "learner": learner, "opponent": opponent,
            "phase_index": phase_index, "phase_round": phase_round,
            "controllable_pool": list(pool_ids), "budget": budget,
            "global_accuracy": _short(global_accuracy),
            "G": G, "scoring_opp_temp": scoring_opp_temp, "commit_opp_temp": opp_temp,
        })

    @_guard
    def fl_round(self, round_num, poisoned_ids, honest_updates, current_accuracy, benign_retrain):
        mode = ("benign clients re-trained locally" if benign_retrain
                else "replaying frozen Phase-1 local weights")
        self._line()
        self._line(f"[FL] federated fine-tuning this round ({mode})")
        pois = set(poisoned_ids or [])
        rows = []
        for u in honest_updates:
            cid = getattr(u, "client_id", None)
            meta = getattr(u, "metadata", {}) or {}
            l2 = self._sd_l2(u.weights)
            tag = "POISONED*" if cid in pois else "benign   "
            if meta:
                extra = (f"train_acc={_short(meta.get('train_accuracy'))} "
                         f"train_loss={_short(meta.get('train_loss'))} "
                         f"samples={meta.get('train_samples')}")
            else:
                extra = "(frozen replay — no local training this round)"
            self._line(f"    client {cid}  {tag}  weight_l2={round(l2, 4)}   {extra}")
            rows.append({"client_id": cid, "poisoned": cid in pois,
                         "weight_l2": round(l2, 4), "train": meta})
        if pois:
            self._line("    (* poisoned clients start from these benign weights; "
                       "the attack plan is applied per-rollout below)")
        self._record("fl_round", "federated_finetune", {
            "round": round_num, "benign_retrain": benign_retrain,
            "current_accuracy": _short(current_accuracy), "clients": rows,
        })

    @_guard
    def attacker_prompt(self, system, user, who="learner"):
        self._line()
        self._line(f"[ATTACKER PROMPT]  exact input to attacker LLM ({who})")
        if self._sys_seen.get("attacker") == system:
            self._line(_DOT[:88] + " [system]")
            self._line("(system prompt unchanged — see first round / debug.json)")
        else:
            self._sys_seen["attacker"] = system
            self._block("system", system)
        self._block("user", user)
        self._record("attacker_prompt", f"attacker_prompt_{who}",
                     {"who": who, "system": system, "user": user})

    @_guard
    def attacker_output(self, text, who="frozen-opponent"):
        self._line()
        self._line(f"[ATTACKER OUTPUT]  raw completion from attacker LLM ({who})")
        self._line(text if isinstance(text, str) else json.dumps(text, default=str))
        self._record("attacker_output", f"attacker_output_{who}", {"who": who, "text": text})

    @_guard
    def poison(self, plan, references, poisoned, n_malformed, n_invalid_ops=0):
        """``plan`` is either a per-client mapping ``{cid: [ops]}`` (the selection
        path) or one shared op list; ``None`` means nothing parsed. ``poisoned``
        holds only the clients whose weights actually changed, so a selected
        client missing from it was wasted and is counted in ``n_malformed``."""
        self._line()
        if plan is None:
            self._line(f"[POISON] attack plan UNPARSEABLE -> no client poisoned "
                       f"(n_malformed={n_malformed})")
        elif isinstance(plan, dict):
            n_ops = sum(len(ops or []) for ops in plan.values())
            self._line(f"[POISON] per-client attack plan: {len(plan)} client(s), "
                       f"{n_ops} op(s) total, invalid/skipped ops={n_invalid_ops}, "
                       f"effectively poisoned={sorted(poisoned)}, wasted={n_malformed}")
            for cid, ops in plan.items():
                landed = "poisoned" if cid in poisoned else "NO-OP (wasted)"
                self._line(f"    client {cid} [{landed}]: " + json.dumps(ops, default=str))
        else:
            self._line(f"[POISON] shared attack plan ({len(plan)} op(s), "
                       f"invalid/skipped ops={n_invalid_ops}):")
            self._line("    " + json.dumps(plan, default=str))
        if not poisoned:
            self._line("    -> NO client was effectively poisoned this round "
                       "(the server receives only honest updates)")
        per_client = {}
        for cid in poisoned:
            if cid not in references:
                continue
            total, layers = self._delta_stats(references[cid], poisoned[cid])
            self._line(
                f"    client {cid}: delta_l2={total['delta_l2']} "
                f"(benign_l2={total['benign_l2']} -> poisoned_l2={total['poisoned_l2']}), "
                f"max|delta|={total['max_abs_delta']}"
            )
            for k, lv in layers.items():
                mark = "   <- changed" if lv["delta_l2"] > 0 else ""
                self._line(f"        {k}: delta_l2={lv['delta_l2']}{mark}")
            per_client[cid] = {"total": total, "layers": layers}
        self._record("poison", "attack_plan_applied", {
            "plan": plan, "n_malformed": n_malformed, "n_invalid_ops": n_invalid_ops,
            "effective_poisoned_ids": sorted(poisoned),
            "poison_deltas": per_client,
        })

    @_guard
    def defender_prompt(self, system, user, who="learner"):
        self._line()
        self._line(f"[DEFENDER PROMPT]  exact input to defender LLM ({who})")
        if self._sys_seen.get("defender") == system:
            self._line(_DOT[:88] + " [system]")
            self._line("(system prompt unchanged — see first round / debug.json)")
        else:
            self._sys_seen["defender"] = system
            self._block("system", system)
        self._block("user (per-client feature vectors)", user)
        self._record("defender_prompt", f"defender_prompt_{who}",
                     {"who": who, "system": system, "user": user})

    @_guard
    def defender_io(self, system, user, output, verdicts, who="opponent",
                    temperature=None, poisoned_ids=None):
        # Prompt (re-uses the de-dup logic) ...
        self.defender_prompt(system, user, who=who)
        # ... then the raw output + parsed verdicts.
        self._line(f"[DEFENDER OUTPUT]  raw completion from defender LLM "
                   f"({who}{'' if temperature is None else f', T={temperature}'})")
        self._line(output if isinstance(output, str) else json.dumps(output, default=str))
        self._line("[VERDICTS] parsed:")
        for ln in self._verdict_lines(verdicts, poisoned_ids):
            self._line(ln)
        self._record("defender_output", f"defender_output_{who}", {
            "who": who, "temperature": temperature, "text": output,
            "verdicts": [{"client_id": v.client_id, "is_suspicious": v.is_suspicious,
                          "confidence": _short(v.confidence), "reason": v.reason}
                         for v in verdicts],
        })

    @_guard
    def defense_verdicts(self, verdicts, info, who="algorithmic-defense",
                         poisoned_ids=None):
        """The defender-LLM-free path: per-algorithm flags + the unioned verdicts."""
        algos = (info or {}).get("algorithms", [])
        per_defense = (info or {}).get("per_defense_flags", {})
        self._line()
        self._line(f"[DEFENSE] {who}: {len(algos)} algorithm(s) judged this round "
                   f"— a client is dropped if ANY flags it")
        for name in algos:
            flagged = per_defense.get(name, [])
            self._line(f"    {name:<10} flagged={flagged if flagged else '-'}")
        self._line("[VERDICTS] union:")
        for ln in self._verdict_lines(verdicts, poisoned_ids):
            self._line(ln)
        self._record("defense_verdicts", f"defense_verdicts_{who}", {
            "who": who, "algorithms": algos, "per_defense_flags": per_defense,
            "verdicts": [{"client_id": v.client_id, "is_suspicious": v.is_suspicious,
                          "confidence": _short(v.confidence), "reason": v.reason}
                         for v in verdicts],
        })

    @_guard
    def opponent_move(self, temperature):
        self._stage = "opponent-move"
        self._line()
        self._line(f"[OPPONENT MOVE] frozen attacker plays one greedy/scoring move "
                   f"(T={temperature}) — its poisoned updates are fixed for all defender rollouts")

    @_guard
    def scoring_rollout(self, text):
        self._rollout += 1
        self._stage = f"rollout {self._rollout}"
        self._line()
        self._line(f"---- rollout {self._rollout} ({self._learner} candidate, scoring) "
                   + "-" * 30)
        self._line(f"[{str(self._learner).upper()} OUTPUT] raw completion:")
        self._line(text if isinstance(text, str) else json.dumps(text, default=str))
        self._record("rollout", "rollout_output", {"index": self._rollout, "text": text})

    @_guard
    def rollout_outcome(self, reward, post_acc=None, n_malformed=None,
                        verdicts=None, poisoned_ids=None):
        parts = [f"reward={_short(reward)}"]
        if post_acc is not None:
            parts.append(f"post_accuracy={_short(post_acc)}")
        if n_malformed is not None:
            parts.append(f"n_malformed={n_malformed}")
        self._line(f"[OUTCOME] " + "  ".join(parts))
        if verdicts is not None:
            self._line("    defender verdicts for this rollout:")
            for ln in self._verdict_lines(verdicts, poisoned_ids):
                self._line(ln)
        self._record("rollout_outcome", "rollout_outcome", {
            "index": self._rollout, "reward": _short(reward),
            "post_accuracy": _short(post_acc) if post_acc is not None else None,
            "n_malformed": n_malformed,
            "verdicts": None if verdicts is None else [
                {"client_id": v.client_id, "is_suspicious": v.is_suspicious,
                 "confidence": _short(v.confidence), "reason": v.reason} for v in verdicts],
        })

    @_guard
    def resampling(self, reason="zero-advantage group"):
        self._line()
        self._line(f"[RESAMPLE] {reason} -> re-rolling the whole group at a higher temperature")
        self._record("resample", "resample", {"reason": reason})

    @_guard
    def grpo_summary(self, metrics: dict):
        rewards = metrics.get("rewards", []) or []
        advs = metrics.get("advantages", []) or []
        self._line()
        self._line("[GRPO] group-relative scoring:")
        self._line("    rollout |   reward   | advantage")
        for i, r in enumerate(rewards):
            a = advs[i] if i < len(advs) else None
            self._line(f"      {i:>4}  | {_short(r):>9} | {_short(a):>9}")
        self._line(
            f"    mean_reward={_short(metrics.get('mean_reward'))}  "
            f"max={_short(metrics.get('max_reward'))}  min={_short(metrics.get('min_reward'))}  "
            f"zero_adv_frac={_short(metrics.get('zero_advantage_fraction'))}  "
            f"resampled={metrics.get('resampled')}"
        )
        self._line(
            f"    loss={_short(metrics.get('loss'))}  "
            f"stepped={metrics.get('stepped')}  "
            f"(stepped=False -> no gradient applied this round)"
        )
        self._record("grpo", "grpo_step", {
            "rewards": [_short(r) for r in rewards],
            "advantages": [_short(a) for a in advs],
            "mean_reward": _short(metrics.get("mean_reward")),
            "max_reward": _short(metrics.get("max_reward")),
            "min_reward": _short(metrics.get("min_reward")),
            "zero_advantage_fraction": _short(metrics.get("zero_advantage_fraction")),
            "loss": _short(metrics.get("loss")),
            "stepped": metrics.get("stepped"),
            "resampled": metrics.get("resampled"),
        })

    @_guard
    def committing(self):
        self._stage = "commit"
        self._line()
        self._line("[COMMIT] applying the best-scoring rollout against the REAL "
                   "(greedy) opponent:")

    @_guard
    def commit_summary(self, learner, best_index, info, reference_acc, post_acc, drop,
                       success, attacker_reward, defender_reward, poisoned_ids,
                       global_acc=None):
        """``reference_acc`` is the round's CLEAN counterfactual (the accuracy the
        aggregate reaches unpoisoned) — the baseline ``drop`` is measured from.
        ``global_acc`` is the accuracy of the global model the round started on,
        shown for context only."""
        verdicts = info.get("verdicts", [])
        flagged = sorted({v.client_id for v in verdicts if v.is_suspicious})
        averaged = sorted({v.client_id for v in verdicts if not v.is_suspicious})
        self._line(f"    committed rollout #{best_index} (best reward)")
        self._line(f"    flagged_by_defender={flagged}   averaged_into_global={averaged}")
        self._line("    committed verdicts:")
        for ln in self._verdict_lines(verdicts, poisoned_ids):
            self._line(ln)
        if not poisoned_ids:
            self._line("    (no client was effectively poisoned this round — "
                       "clean round, nothing to detect)")
        self._line(
            f"    clean_reference {_short(reference_acc)} -> post {_short(post_acc)}  "
            f"(drop={_short(drop)})   learner_success={success}"
        )
        if global_acc is not None:
            self._line(f"    (global model accuracy at round start: {_short(global_acc)})")
        self._line(
            f"    attacker_reward={_short(attacker_reward)}  "
            f"defender_reward={_short(defender_reward)}"
        )
        self._record("commit", "round_committed", {
            "learner": learner, "best_index": best_index,
            "flagged": flagged, "averaged": averaged,
            "clean_reference_accuracy": _short(reference_acc),
            "global_accuracy_at_round_start": _short(global_acc),
            "post_accuracy": _short(post_acc),
            "drop": _short(drop), "learner_success": success,
            "n_malformed": info.get("n_malformed"),
            "attacker_reward": _short(attacker_reward),
            "defender_reward": _short(defender_reward),
            "poisoned_client_ids": list(poisoned_ids),
            "verdicts": [{"client_id": v.client_id, "is_suspicious": v.is_suspicious,
                          "confidence": _short(v.confidence), "reason": v.reason}
                         for v in verdicts],
        })

    @_guard
    def note(self, title, **data):
        self._line(f"[{title}] " + "  ".join(f"{k}={v}" for k, v in data.items()))
        self._record("note", title, data)


# Process-wide singleton. Import this everywhere: ``from core.debug import dbg``.
dbg = DebugLogger()
