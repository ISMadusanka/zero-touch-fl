"""Best-of-N sanity baseline over a fixed action set (no LLM).

Before trusting GRPO it's worth proving the *reward harness itself* is sane:
rewards should move in the expected direction — a bigger perturbation degrades
accuracy more (higher drop term) but is easier to flag (lower evasion). This
baseline enumerates a fixed set of deterministic attacker actions, logs each
action's reward, then commits the best. It exercises env + features + rewards
end-to-end with zero LLM calls and zero GPU.

The defense is whatever the run is configured with: the round's algorithm when
the defender LLM is disabled (``defense.mode: algorithmic``), otherwise the
fixed norm-based heuristic below — which stands in for the defender LLM so the
harness needs no model.
"""

import copy
import logging

import torch

from core.types import DetectionVerdict, RoundLog
from rl.rewards import attacker_reward, defender_reward
from rl.switch import success_drop_bar

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


def run_baseline(env, n_rounds, metrics_tracker, save_round_log):
    """Run ``n_rounds`` of best-of-N fixed-action attack vs fixed defender."""
    logger.info(f"[baseline] running {n_rounds} best-of-N round(s) — no LLM, no GPU")
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
            if env.defense is not None:
                # The algorithm produces the verdicts AND the aggregate; scoring
                # must not advance its cross-round state (commit=False).
                verdicts, state = env.defend(updates, commit=False)
                post_acc = env.evaluate_state(state)
            else:
                verdicts = fixed_defender(env.features(updates))
                post_acc = env.evaluate_updates(updates, verdicts)
            # Same reference as training: this round's clean (unpoisoned) aggregate.
            reward = attacker_reward(ctx.clean_accuracy, post_acc, ctx.goal,
                                     effective, verdicts, n_malformed)
            scored.append((label, effective, n_malformed, updates, verdicts, post_acc, reward))
            logger.info(
                f"[baseline] round {ctx.round_num} action={label:9s} "
                f"def={env.round_defense or 'fixed_heuristic'} "
                f"acc->{post_acc:.4f} att_reward={reward:.3f} poisoned={effective} "
                f"flagged={[v.client_id for v in verdicts if v.is_suspicious]}"
            )

        # Commit the best attacker action.
        label, chosen_ids, n_malformed, updates, verdicts, _, _ = max(
            scored, key=lambda s: s[6])
        env.set_committed_poison(chosen_ids)
        if env.defense is not None:
            # Re-run the winning action through the defense, this time letting its
            # cross-round state advance exactly once.
            verdicts, state = env.defend(updates, commit=True)
            new_acc = env.commit_state(state)
        else:
            new_acc = env.commit(updates, verdicts)
        a_rew = attacker_reward(ctx.clean_accuracy, new_acc, ctx.goal,
                                chosen_ids, verdicts, n_malformed)
        d_rew = defender_reward(verdicts, chosen_ids)

        # Same damage-based success definition as training (see rl.switch).
        metrics_tracker.update(
            ctx.round_num, verdicts, new_acc, set(chosen_ids),
            clean_accuracy=(ctx.clean_accuracy if ctx.clean_measured else None),
            success_drop=success_drop_bar(ctx.goal),
        )
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
                             "defense": env.round_defense or "fixed_heuristic",
                             "curriculum": (env.round_curriculum.as_log_dict()
                                            if env.round_curriculum is not None else None),
                             "clean_accuracy": round(float(ctx.clean_accuracy), 6),
                             "induced_drop": round(float(ctx.clean_accuracy - new_acc), 6)},
        ))
        logger.info(
            f"[baseline] round {ctx.round_num}: committed '{label}' "
            f"def={env.round_defense or 'fixed_heuristic'} "
            f"acc {ctx.global_accuracy:.4f}->{new_acc:.4f} "
            f"(clean_ref={ctx.clean_accuracy:.4f} drop={ctx.clean_accuracy - new_acc:+.4f}) "
            f"def_reward={d_rew:.3f}"
        )
