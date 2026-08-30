"""Frozen-LLM inference path (no training) for the --dry-run mode.

``InferenceGenerator`` adapts the existing Ollama/OpenAI ``llm_client`` to the
``generate(system, user, n, temperature)`` interface the loop expects.
``run_inference`` runs the full round loop end-to-end without any weight
updates — the cheapest way to validate the plumbing (label flipping, local
training, feature extraction, defender prompt + parse, FedAvg, the ladder's
feedback) on a CPU box.

The attack needs no LLM at all: it is the deterministic label-flip ladder. So the
only model call here is the defender's — and under ``defense.mode: algorithmic``
there is none, and the loop runs entirely without a model.
"""

import logging

from core.types import RoundLog
from rl.rewards import attack_effectiveness, defender_reward
from rl.switch import success_drop_bar

logger = logging.getLogger(__name__)


class InferenceGenerator:
    """Wrap a BaseLLMClient as an n-sample text generator."""

    def __init__(self, llm_client, max_new_tokens: int | None = None):
        self.llm = llm_client
        self.max_new_tokens = max_new_tokens

    def generate(self, system: str, user: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        return [
            self.llm.complete(system, user, temperature=temperature, max_tokens=self.max_new_tokens)
            for _ in range(n)
        ]


def run_inference(
    env,
    defender_agent,
    generator: InferenceGenerator,
    n_rounds: int,
    metrics_tracker,
    save_round_log,
    temperature: float = 0.7,
):
    """Run ``n_rounds`` of the label-flip attack vs a frozen defense (no learning).

    The ladder still adapts: it reads the committed verdicts exactly as it does in
    training, so a dry run exercises the full feedback loop and its logs show the
    same saw-tooth in attack strength that a training run would.
    """
    logger.info(f"[dry-run] running {n_rounds} inference round(s) — no weight updates")
    logger.info(f"[dry-run] attack: {env.attacker.describe()}")
    for _ in range(n_rounds):
        ctx = env.begin_round()
        updates = env.build_updates()

        # Defense: either this round's algorithm (which also produces the aggregate)
        # or the defender LLM classifying every client from the feature vectors,
        # with FedAvg over the un-flagged.
        prev_acc = ctx.global_accuracy
        if env.defense is not None:
            verdicts, state = env.defend(updates, commit=True)
            new_acc = env.commit_state(state)
        else:
            feats = env.features(updates)
            client_ids = [u.client_id for u in updates]
            d_sys = defender_agent.system_prompt()
            d_user = defender_agent.build_user_prompt(feats)
            d_text = generator.generate(d_sys, d_user, n=1, temperature=temperature)[0]
            verdicts = defender_agent.parse(d_text, client_ids)
            new_acc = env.commit(updates, verdicts)

        # Feed the ladder — the same call the training loop makes, so the attack
        # schedule a dry run produces is the one training would produce.
        ladder = env.record_detection(verdicts)

        effectiveness = attack_effectiveness(ctx.clean_accuracy, new_acc, ctx.goal)
        d_rew = defender_reward(verdicts, ctx.poisoned_ids)

        # Same damage-based success definition as training (see rl.switch); the clean
        # accuracy is withheld when the defense produced no clean aggregate, so an
        # unmeasurable round is never recorded as a measured zero drop.
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
                "defense": env.round_defense or "llm",
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
            f"[dry-run] round {ctx.round_num}: flip={ctx.flip_fraction:.0%} "
            f"({sum(ctx.flip_plan.values())} labels) poisoned={ctx.poisoned_ids} "
            f"def={env.round_defense or 'llm'} "
            f"acc {prev_acc:.4f}->{new_acc:.4f} "
            f"(clean_ref={ctx.clean_accuracy:.4f} drop={ctx.clean_accuracy - new_acc:+.4f}) "
            f"| eff={effectiveness:+.2f} def_reward={d_rew:.3f} "
            f"ladder={ladder.get('event', '-')}"
            f"->{ladder.get('next_flip_fraction', ctx.flip_fraction):.0%}"
        )
