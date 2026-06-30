"""Frozen-LLM inference path (no training) for the --dry-run mode.

``InferenceGenerator`` adapts the existing Ollama/OpenAI ``llm_client`` to the
``generate(system, user, n, temperature)`` interface the turns/loop expect.
``run_inference`` runs the full round loop end-to-end without any weight
updates — the cheapest way to validate the plumbing (prompt building, attack-plan
parse + apply, feature extraction, FedAvg, reward computation) on a CPU box.
"""

import logging

from core.types import RoundLog
from rl.rewards import attacker_reward, defender_reward

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
    attacker_agent,
    defender_agent,
    generator: InferenceGenerator,
    n_rounds: int,
    metrics_tracker,
    save_round_log,
    temperature: float = 0.7,
):
    """Run ``n_rounds`` of the arms race with frozen LLMs (no learning)."""
    logger.info(f"[dry-run] running {n_rounds} inference round(s) — no weight updates")
    for _ in range(n_rounds):
        ctx = env.begin_round()

        # Attacker generates poisoned weights for the poisoned clients.
        a_sys = attacker_agent.system_prompt()
        a_user = attacker_agent.build_user_prompt(
            ctx.round_num, ctx.global_accuracy, ctx.benign_by_poisoned
        )
        a_text = generator.generate(a_sys, a_user, n=1, temperature=temperature)[0]
        poisoned, n_malformed = attacker_agent.parse(a_text, ctx.benign_by_poisoned)
        updates = env.build_updates(poisoned)

        # Defender classifies every client from the feature vectors.
        feats = env.features(updates)
        client_ids = [u.client_id for u in updates]
        d_sys = defender_agent.system_prompt()
        d_user = defender_agent.build_user_prompt(feats)
        d_text = generator.generate(d_sys, d_user, n=1, temperature=temperature)[0]
        verdicts = defender_agent.parse(d_text, client_ids)

        prev_acc = ctx.global_accuracy
        new_acc = env.commit(updates, verdicts)

        a_rew = attacker_reward(prev_acc, new_acc, env.goal, ctx.poisoned_ids, verdicts, n_malformed)
        d_rew = defender_reward(verdicts, ctx.poisoned_ids)

        metrics_tracker.update(ctx.round_num, verdicts, new_acc, set(ctx.poisoned_ids))
        save_round_log(RoundLog(
            round_num=ctx.round_num,
            attack_goal=env.goal,
            poisoned_client_ids=ctx.poisoned_ids,
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
            attack_metadata={"n_malformed": n_malformed},
        ))
        logger.info(
            f"[dry-run] round {ctx.round_num}: acc {prev_acc:.4f}->{new_acc:.4f} | "
            f"att_reward={a_rew:.3f} def_reward={d_rew:.3f} malformed={n_malformed}"
        )
