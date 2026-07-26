"""Frozen-LLM inference path (no training) for the --dry-run mode.

``InferenceGenerator`` adapts the existing Ollama/OpenAI ``llm_client`` to the
``generate(system, user, n, temperature)`` interface the turns/loop expect.
``run_inference`` runs the full round loop end-to-end without any weight
updates — the cheapest way to validate the plumbing (prompt building, attack-plan
parse + apply, feature extraction, FedAvg, reward computation) on a CPU box.
"""

import logging

from core.types import RoundLog
from rl.rewards import (
    attacker_reward, defender_reward, perturbation_diversity, targeted_terms,
)

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

        # Attacker SELECTS which of its controllable pool to poison (<= budget) and how.
        a_sys = attacker_agent.system_prompt()
        a_user = attacker_agent.build_user_prompt(
            ctx.round_num, ctx.global_accuracy, ctx.pool_benign, env.global_weights,
            ctx.budget, goal=ctx.goal,
        )
        a_text = generator.generate(a_sys, a_user, n=1, temperature=temperature)[0]
        poisoned, chosen_ids, n_malformed = attacker_agent.select_and_apply(
            a_text, ctx.pool_benign, ctx.budget)
        env.set_committed_poison(chosen_ids)
        updates = env.build_updates(poisoned)

        # Defender classifies every client from the feature vectors.
        feats = env.features(updates)
        client_ids = [u.client_id for u in updates]
        d_sys = defender_agent.system_prompt()
        d_user = defender_agent.build_user_prompt(feats)
        d_text = generator.generate(d_sys, d_user, n=1, temperature=temperature)[0]
        verdicts = defender_agent.parse(d_text, client_ids)

        prev_acc = ctx.global_accuracy
        post_eval = env.commit_full(updates, verdicts)
        new_acc = post_eval.overall

        diversity = perturbation_diversity(
            poisoned, {cid: ctx.pool_benign[cid] for cid in chosen_ids})
        # Damage is scored against the round's clean counterfactual, exactly as in
        # training (see FLArmsRaceEnv.clean_reference_accuracy). Passing the
        # per-class evaluations makes a targeted_label goal score its target class
        # here too, so --dry-run is a faithful CPU smoke test of the targeted setup.
        terms = targeted_terms(ctx.goal, ctx.clean_eval, post_eval)
        a_rew = attacker_reward(ctx.clean_accuracy, new_acc, ctx.goal, chosen_ids,
                                verdicts, n_malformed,
                                pool_size=env.n_compromisable, diversity=diversity,
                                clean_eval=ctx.clean_eval, post_eval=post_eval)
        d_rew = defender_reward(verdicts, chosen_ids)

        metrics_tracker.update(ctx.round_num, verdicts, new_acc, set(chosen_ids))
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
            attack_metadata={"n_malformed": n_malformed, "budget": ctx.budget,
                             "n_used": len(chosen_ids),
                             "clean_accuracy": round(float(ctx.clean_accuracy), 6),
                             "induced_drop": round(float(ctx.clean_accuracy - new_acc), 6),
                             "targeted": (None if terms is None else {
                                 "label": terms["label"],
                                 "clean_recall": round(terms["clean_recall"], 6),
                                 "post_recall": round(terms["post_recall"], 6),
                                 "target_class_drop": round(terms["target_drop"], 6),
                                 "collateral": round(terms["collateral"], 6),
                                 "per_class_clean": [round(v, 6) for v in ctx.clean_eval.per_class],
                                 "per_class_post": [round(v, 6) for v in post_eval.per_class],
                             })},
        ))
        tgt_s = "" if terms is None else (
            f" | TGT[{terms['label']}] {terms['clean_recall']:.3f}->{terms['post_recall']:.3f}"
            f" collat={terms['collateral']:.3f}")
        logger.info(
            f"[dry-run] round {ctx.round_num}: poisoned={chosen_ids} budget={ctx.budget} "
            f"acc {prev_acc:.4f}->{new_acc:.4f} "
            f"(clean_ref={ctx.clean_accuracy:.4f} drop={ctx.clean_accuracy - new_acc:+.4f})"
            f"{tgt_s} "
            f"| att_reward={a_rew:.3f} def_reward={d_rew:.3f} malformed={n_malformed}"
        )
