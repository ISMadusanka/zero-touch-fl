"""AttackerTurn / DefenderTurn — bind one FL round to one *learning* agent.

A "turn" fixes everything about a round except the learning agent's action, so
the GRPO sampler can score G candidate actions against an identical state and a
frozen opponent (the Stackelberg structure: the leader/attacker moves, the
follower/defender best-responds).

Generators are duck-typed: any object with
``generate(system, user, n, temperature) -> list[str]`` works — a
``PolicyGenerator`` (trainable LoRA adapter) during RL, or an
``InferenceGenerator`` (frozen Ollama/OpenAI) during --dry-run.
"""

import logging

from rl.rewards import attacker_reward, defender_reward

logger = logging.getLogger(__name__)


class AttackerTurn:
    """Learning agent = attacker. Opponent = frozen defender (greedy)."""

    def __init__(self, env, attacker_agent, defender_agent, defender_gen,
                 reward_cfg: dict | None = None, opponent_temperature: float = 0.0):
        self.env = env
        self.attacker_agent = attacker_agent
        self.defender_agent = defender_agent
        self.defender_gen = defender_gen
        self.reward_cfg = reward_cfg or {}
        self.opp_temp = opponent_temperature

        self.references = env.benign_by_poisoned          # {cid: benign state_dict}
        self.poisoned_ids = list(env.poisoned_ids)
        self.prev_accuracy = env.current_accuracy

        self.system = attacker_agent.system_prompt()
        self.user = attacker_agent.build_user_prompt(
            env.round_index + env.training_rounds, env.current_accuracy, self.references
        )

    # The learning agent's prompt (consumed by the GRPO sampler / policy).
    def messages(self) -> tuple[str, str]:
        return self.system, self.user

    def _defender_verdicts(self, updates):
        feats = self.env.features(updates)
        client_ids = [u.client_id for u in updates]
        d_sys = self.defender_agent.system_prompt()
        d_user = self.defender_agent.build_user_prompt(feats)
        text = self.defender_gen.generate(d_sys, d_user, n=1, temperature=self.opp_temp)[0]
        return self.defender_agent.parse(text, client_ids)

    def _apply(self, attacker_text):
        poisoned, n_malformed = self.attacker_agent.parse(attacker_text, self.references)
        updates = self.env.build_updates(poisoned)
        verdicts = self._defender_verdicts(updates)
        return updates, verdicts, n_malformed

    def reward(self, attacker_text) -> float:
        updates, verdicts, n_malformed = self._apply(attacker_text)
        post_acc = self.env.evaluate_updates(updates, verdicts)
        return attacker_reward(
            self.prev_accuracy, post_acc, self.env.goal, self.poisoned_ids,
            verdicts, n_malformed,
            alpha=self.reward_cfg.get("alpha", 1.0),
            beta=self.reward_cfg.get("beta", 0.5),
            gamma=self.reward_cfg.get("gamma", 1.0),
        )

    def commit(self, attacker_text) -> dict:
        poisoned, n_malformed = self.attacker_agent.parse(attacker_text, self.references)
        updates = self.env.build_updates(poisoned)
        verdicts = self._defender_verdicts(updates)
        new_acc = self.env.commit(updates, verdicts)
        return {
            "updates": updates,
            "verdicts": verdicts,
            "n_malformed": n_malformed,
            "post_accuracy": new_acc,
        }


class DefenderTurn:
    """Learning agent = defender. Opponent = frozen attacker (greedy).

    The frozen attacker's poisoned weights are sampled ONCE here so all G
    defender candidates classify the same set of updates.
    """

    def __init__(self, env, attacker_agent, defender_agent, attacker_gen,
                 reward_cfg: dict | None = None, opponent_temperature: float = 0.0):
        self.env = env
        self.attacker_agent = attacker_agent
        self.defender_agent = defender_agent
        self.reward_cfg = reward_cfg or {}

        self.references = env.benign_by_poisoned
        self.poisoned_ids = list(env.poisoned_ids)

        # Frozen attacker plays its (greedy) move for this round.
        a_sys = attacker_agent.system_prompt()
        a_user = attacker_agent.build_user_prompt(
            env.round_index + env.training_rounds, env.current_accuracy, self.references
        )
        a_text = attacker_gen.generate(a_sys, a_user, n=1, temperature=opponent_temperature)[0]
        poisoned, self.n_malformed = attacker_agent.parse(a_text, self.references)

        self.updates = env.build_updates(poisoned)
        self.client_ids = [u.client_id for u in self.updates]
        self.features = env.features(self.updates)

        self.system = defender_agent.system_prompt()
        self.user = defender_agent.build_user_prompt(self.features)

    def messages(self) -> tuple[str, str]:
        return self.system, self.user

    def reward(self, defender_text) -> float:
        verdicts = self.defender_agent.parse(defender_text, self.client_ids)
        return defender_reward(
            verdicts, self.poisoned_ids,
            mode=self.reward_cfg.get("mode", "soft_f1"),
            fpr_penalty=self.reward_cfg.get("fpr_penalty", 1.0),
        )

    def commit(self, defender_text) -> dict:
        verdicts = self.defender_agent.parse(defender_text, self.client_ids)
        new_acc = self.env.commit(self.updates, verdicts)
        return {
            "updates": self.updates,
            "verdicts": verdicts,
            "n_malformed": self.n_malformed,
            "post_accuracy": new_acc,
        }
