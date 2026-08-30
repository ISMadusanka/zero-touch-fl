"""No-LLM sanity baseline for the round loop and the label-flip ladder.

Runs the full Phase-2 round loop with no model anywhere: the attack is the
deterministic label-flip ladder (which never needed one), and the defense is
either the run's configured algorithm (``defense.mode: algorithmic``) or the
fixed norm/sign heuristic below, which stands in for the defender LLM.

What this is for: proving the harness itself is sane before spending GPU time on
it. Two things should be visible in its output on any working configuration.

1. **The attack does damage.** A round at a high ladder level should cost measurable
   accuracy against the clean counterfactual. If ``induced_drop`` hovers at zero
   even at 100% flipped labels, the poison is not reaching the aggregate and no
   amount of defender training will mean anything.
2. **The ladder responds.** The heuristic below catches large label-flip updates
   fairly reliably, so the printed ``flip_fraction`` should walk DOWN, hit the
   floor, and reset — the saw-tooth the whole design rests on, visible in a few
   dozen CPU-seconds.
"""

import logging

from core.types import DetectionVerdict, RoundLog
from rl.rewards import attack_effectiveness, defender_reward
from rl.switch import success_drop_bar

logger = logging.getLogger(__name__)


def fixed_defender(features: dict[int, dict], rel_norm_thr: float = 2.0,
                   sign_thr: float = 0.4) -> list[DetectionVerdict]:
    """Flag clients with abnormally large norm or low sign agreement.

    A crude stand-in for the defender LLM, and deliberately so: it reads the same
    two features the LLM's prompt highlights, so a label-flip level it CANNOT catch
    is one that genuinely sits inside the honest update distribution rather than one
    this heuristic happens to be blind to.
    """
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
    """Run ``n_rounds`` of the label-flip ladder against a non-LLM defense."""
    logger.info(f"[baseline] running {n_rounds} round(s) — no LLM, no GPU")
    logger.info(f"[baseline] attack: {env.attacker.describe()}")
    for _ in range(n_rounds):
        ctx = env.begin_round()
        updates = env.build_updates()

        if env.defense is not None:
            # The algorithm produces the verdicts AND the aggregate, and this is the
            # committed round, so its cross-round state advances exactly once.
            verdicts, state = env.defend(updates, commit=True)
            new_acc = env.commit_state(state)
            defense_name = env.round_defense
        else:
            verdicts = fixed_defender(env.features(updates))
            new_acc = env.commit(updates, verdicts)
            defense_name = "fixed_heuristic"

        ladder = env.record_detection(verdicts)
        effectiveness = attack_effectiveness(ctx.clean_accuracy, new_acc, ctx.goal)
        d_rew = defender_reward(verdicts, ctx.poisoned_ids)

        metrics_tracker.update(
            ctx.round_num, verdicts, new_acc, set(ctx.poisoned_ids),
            clean_accuracy=(ctx.clean_accuracy if ctx.clean_measured else None),
            success_drop=success_drop_bar(ctx.goal),
        )
        save_round_log(RoundLog(
            round_num=ctx.round_num,
            attack_goal=ctx.goal,
            poisoned_client_ids=list(ctx.poisoned_ids),
            predicted_labels=[
                {"client_id": v.client_id, "is_suspicious": v.is_suspicious,
                 "confidence": v.confidence, "reason": v.reason}
                for v in verdicts
            ],
            test_accuracy=new_acc,
            baseline_accuracy=env.baseline_accuracy,
            attack_effectiveness=effectiveness,
            defender_reward=d_rew,
            learning_agent="none",
            attack_metadata={
                "attack": "label_flip",
                "flip_fraction": round(float(ctx.flip_fraction), 6),
                "flip_plan": {str(cid): n for cid, n in sorted(ctx.flip_plan.items())},
                "n_flipped": sum(ctx.flip_plan.values()),
                "n_poisoned": len(ctx.poisoned_ids),
                "ladder": ladder,
                "defense": defense_name,
                "curriculum": (env.round_curriculum.as_log_dict()
                               if env.round_curriculum is not None else None),
                "clean_accuracy": round(float(ctx.clean_accuracy), 6),
                "induced_drop": round(float(ctx.clean_accuracy - new_acc), 6),
                "attack_effectiveness": round(float(effectiveness), 6),
                "clean_measured": bool(ctx.clean_measured),
                "defense_sane": bool(ctx.defense_sane),
            },
        ))
        logger.info(
            f"[baseline] round {ctx.round_num}: flip={ctx.flip_fraction:.0%} "
            f"({sum(ctx.flip_plan.values())} labels) poisoned={ctx.poisoned_ids} "
            f"def={defense_name} "
            f"flagged={[v.client_id for v in verdicts if v.is_suspicious]} "
            f"acc {ctx.global_accuracy:.4f}->{new_acc:.4f} "
            f"(clean_ref={ctx.clean_accuracy:.4f} drop={ctx.clean_accuracy - new_acc:+.4f}) "
            f"| eff={effectiveness:+.2f} def_reward={d_rew:.3f} "
            f"ladder={ladder.get('event', '-')}"
            f"->{ladder.get('next_flip_fraction', ctx.flip_fraction):.0%}"
        )
