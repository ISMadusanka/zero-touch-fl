"""Frozen-LLM inference path (no training) for the --dry-run mode.

``InferenceGenerator`` adapts the existing Ollama/OpenAI ``llm_client`` to the
``generate(system, user, n, temperature)`` interface the turns/loop expect.
``run_inference`` runs the full round loop end-to-end without any weight
updates — the cheapest way to validate the plumbing (prompt building, attack-plan
parse + apply, feature extraction, FedAvg, reward computation) on a CPU box.

When the defender LLM is disabled (``defense.mode: algorithmic``) the defense
side is ``env.defense`` — one published algorithm per round — so only the
attacker calls the LLM here.
"""

import logging

from core.types import RoundLog
from rl.rewards import (
    attack_potency, attacker_reward, check_reward_balance, defender_reward,
    perturbation_diversity,
)
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
    attacker_agent,
    defender_agent,
    generator: InferenceGenerator,
    n_rounds: int,
    metrics_tracker,
    save_round_log,
    temperature: float = 0.7,
    reward_cfg=None,
):
    """Run ``n_rounds`` of the arms race with frozen LLMs (no learning).

    ``reward_cfg`` is ``rl.reward.attacker`` from the config. Without it this loop
    scored rounds with the ``attacker_reward`` function defaults (alpha 1.0 /
    beta 0.5) while training used the configured weights, so a --dry-run reward was
    not on the same scale as a training reward even though both are written to
    ``logs/round_data/rounds.jsonl`` under the same field name. Against the shipped
    target of 0.10 the defaults also invert the shaping-budget invariant — see
    :func:`rl.rewards.check_reward_balance`.
    """
    reward_cfg = reward_cfg or {}
    weights = {
        "alpha": float(reward_cfg.get("alpha", 1.0)),
        "beta": float(reward_cfg.get("beta", 0.5)),
        "gamma": float(reward_cfg.get("gamma", 1.0)),
        "zeta": float(reward_cfg.get("zeta", 0.0)),
    }
    logger.info(f"[dry-run] running {n_rounds} inference round(s) — no weight updates")
    check_reward_balance(reward_cfg, env.goal, context="dry-run")
    for _ in range(n_rounds):
        ctx = env.begin_round()

        # Attacker selects exactly the budgeted number from its controllable pool.
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

        # Defense: either this round's algorithm (defender LLM disabled — it also
        # produces the aggregate) or the defender LLM classifying every client
        # from the feature vectors, with FedAvg over the un-flagged.
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

        refs = {cid: ctx.pool_benign[cid] for cid in chosen_ids}
        diversity = perturbation_diversity(poisoned, refs)
        # Damage is scored against the round's clean counterfactual, and stealth is
        # gated on how much poison actually went out — exactly as in training (see
        # FLArmsRaceEnv.clean_reference_accuracy and rl.rewards.attack_potency), so
        # a --dry-run reward is comparable to a training one.
        a_rew = attacker_reward(ctx.clean_accuracy, new_acc, ctx.goal, chosen_ids,
                                verdicts, n_malformed,
                                diversity=diversity,
                                potency=attack_potency(poisoned, refs,
                                                       env.global_weights),
                                **weights)
        d_rew = defender_reward(verdicts, chosen_ids)

        # Same damage-based success definition as training (see rl.switch); the clean
        # accuracy is withheld when the defense produced no clean aggregate, so an
        # unmeasurable round is never recorded as a measured zero drop.
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
            attack_metadata={"n_malformed": n_malformed, "budget": ctx.budget,
                             "n_used": len(chosen_ids),
                             "defense": env.round_defense or "llm",
                             "curriculum": (env.round_curriculum.as_log_dict()
                                            if env.round_curriculum is not None else None),
                             "clean_accuracy": round(float(ctx.clean_accuracy), 6),
                             "induced_drop": round(float(ctx.clean_accuracy - new_acc), 6)},
        ))
        logger.info(
            f"[dry-run] round {ctx.round_num}: poisoned={chosen_ids} budget={ctx.budget} "
            f"def={env.round_defense or 'llm'} "
            f"acc {prev_acc:.4f}->{new_acc:.4f} "
            f"(clean_ref={ctx.clean_accuracy:.4f} drop={ctx.clean_accuracy - new_acc:+.4f}) "
            f"| att_reward={a_rew:.3f} def_reward={d_rew:.3f} malformed={n_malformed}"
        )
