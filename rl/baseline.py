"""Best-of-N sanity baseline over a fixed action set (no LLM).

Before trusting GRPO it's worth proving the *reward harness itself* is sane:
rewards should move in the expected direction — a bigger perturbation degrades
accuracy more (higher drop term) but is easier to flag (lower evasion). This
baseline enumerates a fixed set of deterministic attacker actions against a
fixed norm-based defender heuristic and logs each action's reward, then commits
the best. It exercises env + features + rewards end-to-end with zero LLM calls
and zero GPU.
"""

import copy
import logging

import torch

from core.types import DetectionVerdict, RoundLog
from rl.rewards import attacker_reward, defender_reward, goal_drop, targeted_terms
from rl.switch import SwitchConfig, attacker_succeeded

logger = logging.getLogger(__name__)


def _scale(sd, factor):
    return {k: v.float() * factor for k, v in sd.items()}


def _add_noise(sd, sigma):
    return {k: v.float() + torch.randn_like(v.float()) * sigma for k, v in sd.items()}


def _sign_flip(sd):
    return {k: -v.float() for k, v in sd.items()}


def _same_weights(a: dict, b: dict) -> bool:
    """True when two state_dicts are element-wise identical (a no-op 'attack')."""
    return all(torch.equal(a[k].float(), b[k].float()) for k in b)


def fixed_attacker_actions(benign_by_client: dict[int, dict]) -> list[tuple[str, dict]]:
    """Return ``[(label, {client_id: poisoned_state_dict})]`` candidates."""
    transforms = {
        "none": lambda sd: copy.deepcopy(sd),
        "scale_2": lambda sd: _scale(sd, 2.0),
        "scale_5": lambda sd: _scale(sd, 5.0),
        "scale_10": lambda sd: _scale(sd, 10.0),
        "noise_1": lambda sd: _add_noise(sd, 1.0),
        "signflip": _sign_flip,
    }
    actions = []
    for label, fn in transforms.items():
        actions.append((label, {cid: fn(sd) for cid, sd in benign_by_client.items()}))
    return actions


def fixed_defender(features: dict[int, dict], rel_norm_thr: float = 2.0,
                   sign_thr: float = 0.4) -> list[DetectionVerdict]:
    """Flag clients with abnormally large norm or low sign agreement."""
    verdicts = []
    for cid, f in features.items():
        rel_norm = f["whole"]["rel_norm"]
        min_sign = min(L["sign_agreement"] for L in f["layers"].values())
        suspicious = rel_norm > rel_norm_thr or min_sign < sign_thr
        conf = min(1.0, max(rel_norm / (rel_norm_thr + 1e-8) - 1.0, 1.0 - min_sign))
        verdicts.append(DetectionVerdict(cid, suspicious, float(max(0.0, conf)),
                                         f"rel_norm={rel_norm:.2f} min_sign={min_sign:.2f}"))
    return verdicts


def run_baseline(env, n_rounds, metrics_tracker, save_round_log, defender=None,
                 switch_cfg: SwitchConfig | None = None):
    """Run ``n_rounds`` of best-of-N fixed-action attack vs a fixed defense.

    ``defender`` defaults to the norm/sign heuristic :func:`fixed_defender`. Pass a
    defender policy (``rl/defenders.py``) to substitute another non-LLM defense —
    ``main.py --baseline --freeze defender`` passes the algorithmic ensemble, which
    exercises env + ensemble + rewards end-to-end with no LLM and no GPU.

    ``switch_cfg`` supplies the win-gate thresholds used to judge each round's
    attack GOAL; defaults to :class:`SwitchConfig`'s own defaults.
    """
    switch_cfg = switch_cfg or SwitchConfig()
    logger.info(f"[baseline] running {n_rounds} best-of-N round(s) — no LLM, no GPU "
                f"(defense: {defender.describe() if defender else 'norm/sign heuristic'})")

    def _verdicts(updates, commit):
        if defender is None:
            return fixed_defender(env.features(updates))
        return defender.verdicts(env, updates, commit=commit)

    for _ in range(n_rounds):
        ctx = env.begin_round()
        # No LLM to choose clients: poison the first `budget` clients of the pool.
        selected_ids = list(ctx.pool_ids[:ctx.budget])
        selected_benign = {cid: ctx.pool_benign[cid] for cid in selected_ids}
        actions = fixed_attacker_actions(selected_benign)

        scored = []
        for label, poisoned in actions:
            # Ground truth is the clients whose weights the action ACTUALLY changed
            # — the "none" control leaves them byte-identical to benign, so nothing
            # was poisoned and the wasted clients are charged as malformed. This
            # mirrors AttackerAgent.select_and_apply; without it the no-op action
            # scored a free evasion bonus and won every round.
            effective = [cid for cid in selected_ids
                         if not _same_weights(poisoned[cid], selected_benign[cid])]
            n_malformed = len(selected_ids) - len(effective)
            updates = env.build_updates({cid: poisoned[cid] for cid in effective})
            verdicts = _verdicts(updates, commit=False)
            post_eval = env.evaluate_updates_full(updates, verdicts)
            post_acc = post_eval.overall
            # Same reference as training: this round's clean (unpoisoned) aggregate.
            # Passing the per-class evals keeps the harness honest under a
            # targeted_label goal (it then scores the target class, not overall acc).
            reward = attacker_reward(ctx.clean_accuracy, post_acc, ctx.goal,
                                     effective, verdicts, n_malformed,
                                     clean_eval=ctx.clean_eval, post_eval=post_eval)
            scored.append((label, effective, n_malformed, updates, verdicts, post_acc, reward))
            logger.info(
                f"[baseline] round {ctx.round_num} action={label:9s} "
                f"acc->{post_acc:.4f} att_reward={reward:.3f} poisoned={effective} "
                f"flagged={[v.client_id for v in verdicts if v.is_suspicious]}"
            )

        # Commit the best attacker action.
        label, chosen_ids, n_malformed, updates, verdicts, _, _ = max(
            scored, key=lambda s: s[6])
        # Re-run the defense on the winning action as the COMMITTING call, so a
        # stateful defense advances its cross-round memory exactly once per round
        # (the scoring pass above deliberately leaves it untouched). Deterministic,
        # so the verdicts themselves are the ones already scored.
        verdicts = _verdicts(updates, commit=True)
        env.set_committed_poison(chosen_ids)
        committed_eval = env.commit_full(updates, verdicts)
        new_acc = committed_eval.overall
        a_rew = attacker_reward(ctx.clean_accuracy, new_acc, ctx.goal,
                                chosen_ids, verdicts, n_malformed,
                                clean_eval=ctx.clean_eval, post_eval=committed_eval)
        d_rew = defender_reward(verdicts, chosen_ids)

        # Judge the round on the attack GOAL (damage + collateral + evasion), the
        # same way training does — a scripted baseline that merely slips past the
        # detector is not a successful targeted attack.
        terms = targeted_terms(ctx.goal, ctx.clean_eval, committed_eval)
        goal_met = attacker_succeeded(
            goal_drop(ctx.goal, ctx.clean_accuracy, new_acc, ctx.clean_eval, committed_eval),
            verdicts, chosen_ids, switch_cfg, ctx.goal, terms)
        metrics_tracker.update(ctx.round_num, verdicts, new_acc, set(chosen_ids),
                               reference_accuracy=ctx.clean_accuracy,
                               attack_goal_met=goal_met)
        save_round_log(RoundLog(
            round_num=ctx.round_num,
            attack_goal=ctx.goal,
            poisoned_client_ids=chosen_ids,
            predicted_labels=[
                {"client_id": v.client_id, "is_suspicious": v.is_suspicious,
                 "confidence": v.confidence, "reason": v.reason}
                for v in verdicts
            ],
            test_accuracy=new_acc,
            baseline_accuracy=env.baseline_accuracy,
            attacker_reward=a_rew,
            defender_reward=d_rew,
            learning_agent="none",
            attack_metadata={"baseline_action": label, "budget": ctx.budget,
                             "n_used": len(chosen_ids), "n_malformed": n_malformed,
                             "clean_accuracy": round(float(ctx.clean_accuracy), 6),
                             "induced_drop": round(float(ctx.clean_accuracy - new_acc), 6)},
        ))
        logger.info(
            f"[baseline] round {ctx.round_num}: committed '{label}' "
            f"acc {ctx.global_accuracy:.4f}->{new_acc:.4f} "
            f"(clean_ref={ctx.clean_accuracy:.4f} drop={ctx.clean_accuracy - new_acc:+.4f}) "
            f"def_reward={d_rew:.3f}"
        )
