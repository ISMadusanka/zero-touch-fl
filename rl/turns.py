"""DefenderTurn — bind one FL round to the defender policy.

A "turn" fixes everything about a round except the learning agent's action, so
the GRPO sampler can score G candidate actions against an identical state.

There is only one turn now. The attack is not a policy: it is the deterministic,
detection-adaptive label-flip ladder run by the env (see
:mod:`agents.label_flip_attacker`), which has already produced this round's
poisoned updates by the time ``begin_round`` returns. So the defender's G
rollouts all classify the SAME cohort of updates — exactly the property the old
``DefenderTurn`` had to sample a frozen attacker once to obtain, and which is now
free.

Generators are duck-typed: any object with
``generate(system, user, n, temperature) -> list[str]`` works — a
``PolicyGenerator`` (trainable LoRA adapter) during RL, or an
``InferenceGenerator`` (frozen Ollama/OpenAI) during --dry-run.
"""

import logging

from core.debug import dbg
from rl.rewards import defender_reward

logger = logging.getLogger(__name__)


class DefenderTurn:
    """Learning agent = defender. Opponent = the round's label-flip ladder level.

    The poisoned updates come from the env, so nothing here samples or generates
    an attack; the turn's whole job is to build the defender's prompt from the
    round's per-client features and to score its verdicts against ground truth.
    """

    def __init__(self, env, defender_agent, reward_cfg: dict | None = None):
        if getattr(env, "defense", None) is not None:
            raise RuntimeError(
                "DefenderTurn requires the defender LLM, but this run defends with "
                "algorithms (defense.mode: algorithmic) — there is no policy to "
                "train. Set defense.mode: llm, or run --dry-run / --baseline."
            )
        self.env = env
        self.defender_agent = defender_agent
        self.reward_cfg = reward_cfg or {}

        # This round's cohort: honest updates with the label-flipped insiders
        # swapped in. Ground truth is whatever actually shipped flipped labels.
        self.updates = env.build_updates()
        self.poisoned_ids = list(env.poisoned_ids)
        self.client_ids = [u.client_id for u in self.updates]
        self.features = env.features(self.updates)

        dbg.attack_plan(env.flip_fraction, env.flip_plan, self.poisoned_ids)
        self.system = defender_agent.system_prompt()
        self.user = defender_agent.build_user_prompt(self.features)
        dbg.defender_prompt(self.system, self.user, who="learner")

    # The learning agent's prompt (consumed by the GRPO sampler / policy).
    def messages(self) -> tuple[str, str]:
        return self.system, self.user

    def reward(self, defender_text) -> float:
        dbg.scoring_rollout(defender_text)
        verdicts = self.defender_agent.parse(defender_text, self.client_ids)
        r = defender_reward(
            verdicts, self.poisoned_ids,
            mode=self.reward_cfg.get("mode", "soft_f1"),
            fpr_penalty=self.reward_cfg.get("fpr_penalty", 1.0),
        )
        dbg.rollout_outcome(reward=r, verdicts=verdicts, poisoned_ids=self.poisoned_ids)
        return r

    def commit(self, defender_text) -> dict:
        """Commit one rollout's verdicts: aggregate the un-flagged clients, measure
        the result, and feed the outcome back into the attack ladder.

        The ladder advances HERE and nowhere else. Scoring a rollout must never
        move it, or the attack schedule would depend on ``rl.G`` instead of on
        whether the defense actually caught the round.
        """
        dbg.committing()
        verdicts = self.defender_agent.parse(defender_text, self.client_ids)
        new_acc = self.env.commit(self.updates, verdicts)
        ladder = self.env.record_detection(verdicts)
        return {
            "updates": self.updates,
            "verdicts": verdicts,
            "post_accuracy": new_acc,
            "poisoned_ids": self.poisoned_ids,
            "flip_plan": dict(self.env.flip_plan),
            "flip_fraction": self.env.flip_fraction,
            "ladder": ladder,
        }
