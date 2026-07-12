"""AttackerTurn / DefenderTurn — bind one FL round to one *learning* agent.

A "turn" fixes everything about a round except the learning agent's action, so
the GRPO sampler can score G candidate actions against an identical state and a
frozen opponent (the Stackelberg structure: the leader/attacker moves, the
follower/defender best-responds).

The attacker's action now includes CLIENT SELECTION: from its controllable pool
(``env.pool_benign``) it picks up to ``env.round_budget`` clients and poisons each
(see ``AttackerAgent.select_and_apply``). Different rollouts may pick different
subsets, so each rollout's reward is computed against ITS OWN chosen set.

Generators are duck-typed: any object with
``generate(system, user, n, temperature) -> list[str]`` works — a
``PolicyGenerator`` (trainable LoRA adapter) during RL, or an
``InferenceGenerator`` (frozen Ollama/OpenAI) during --dry-run.
"""

import logging

from core.debug import dbg
from rl.rewards import attacker_reward, defender_reward, perturbation_diversity

logger = logging.getLogger(__name__)


class AttackerTurn:
    """Learning agent = attacker. Opponent = frozen defender (greedy)."""

    def __init__(self, env, attacker_agent, defender_agent, defender_gen,
                 reward_cfg: dict | None = None, opponent_temperature: float = 0.0,
                 scoring_opponent_temperature: float | None = None):
        self.env = env
        self.attacker_agent = attacker_agent
        self.defender_agent = defender_agent
        self.defender_gen = defender_gen
        self.reward_cfg = reward_cfg or {}
        self.opp_temp = opponent_temperature
        # When SCORING the G candidate plans we sample the frozen defender at a
        # (usually nonzero) temperature so different plans see different verdicts
        # — this restores within-group reward spread and is the key fix for the
        # attacker's zero-advantage collapse. The COMMITTED round still uses the
        # greedy ``opp_temp`` so success is measured against the real defender.
        self.scoring_opp_temp = (
            opponent_temperature if scoring_opponent_temperature is None
            else scoring_opponent_temperature
        )

        # The controllable pool + this round's budget (the attacker chooses which
        # of these to poison, and how).
        self.pool_references = env.pool_benign            # {cid: benign state_dict}
        self.budget = env.round_budget
        self.pool_size = env.n_compromisable
        self.prev_accuracy = env.current_accuracy

        self.system = attacker_agent.system_prompt()
        self.user = attacker_agent.build_user_prompt(
            env.round_index + env.training_rounds, env.current_accuracy,
            self.pool_references, env.global_weights, self.budget,
        )
        dbg.attacker_prompt(self.system, self.user, who="learner")

    # The learning agent's prompt (consumed by the GRPO sampler / policy).
    def messages(self) -> tuple[str, str]:
        return self.system, self.user

    def _defender_verdicts(self, updates, temperature):
        feats = self.env.features(updates)
        client_ids = [u.client_id for u in updates]
        d_sys = self.defender_agent.system_prompt()
        d_user = self.defender_agent.build_user_prompt(feats)
        text = self.defender_gen.generate(d_sys, d_user, n=1, temperature=temperature)[0]
        verdicts = self.defender_agent.parse(text, client_ids)
        dbg.defender_io(d_sys, d_user, text, verdicts, who="opponent",
                        temperature=temperature)
        return verdicts

    def _apply(self, attacker_text, temperature):
        poisoned, chosen_ids, n_malformed = self.attacker_agent.select_and_apply(
            attacker_text, self.pool_references, self.budget
        )
        updates = self.env.build_updates(poisoned)
        verdicts = self._defender_verdicts(updates, temperature)
        return updates, verdicts, n_malformed, chosen_ids, poisoned

    def _score(self, updates, verdicts, chosen_ids, poisoned, n_malformed) -> float:
        """Verifiable attacker reward for one already-applied candidate.

        The reward maths lives here so the per-rollout ``reward`` path and the
        batched ``rewards`` path are guaranteed identical."""
        post_acc = self.env.evaluate_updates(updates, verdicts)
        diversity = perturbation_diversity(
            poisoned, {cid: self.pool_references[cid] for cid in chosen_ids})
        r = attacker_reward(
            self.prev_accuracy, post_acc, self.env.goal, chosen_ids,
            verdicts, n_malformed,
            alpha=self.reward_cfg.get("alpha", 1.0),
            beta=self.reward_cfg.get("beta", 0.5),
            gamma=self.reward_cfg.get("gamma", 1.0),
            delta=self.reward_cfg.get("delta", 0.0),
            zeta=self.reward_cfg.get("zeta", 0.0),
            pool_size=self.pool_size,
            diversity=diversity,
        )
        dbg.rollout_outcome(reward=r, post_acc=post_acc, n_malformed=n_malformed,
                            verdicts=verdicts, poisoned_ids=chosen_ids)
        return r

    def reward(self, attacker_text) -> float:
        dbg.scoring_rollout(attacker_text)
        updates, verdicts, n_malformed, chosen_ids, poisoned = self._apply(
            attacker_text, self.scoring_opp_temp)
        return self._score(updates, verdicts, chosen_ids, poisoned, n_malformed)

    def rewards(self, attacker_texts) -> list[float]:
        """Score a whole GRPO group of attacker candidates.

        Fast path (normal training): parse every candidate's attack plan, build its
        poisoned updates + defender prompt, then sample the frozen defender's
        verdicts for ALL candidates in ONE batched generation. Each row is an
        independent sample at ``scoring_opp_temp`` — the same distribution as the
        per-rollout calls, so the reward signal is unchanged; only the G sequential
        defender decodes collapse into one batched decode.

        Falls back to the exact per-rollout ``reward`` loop when a debug run is
        active (it needs strict per-rollout log interleaving) or when the frozen
        opponent generator can't batch (e.g. the Ollama/OpenAI dry-run generator).
        The commit path is untouched — the real round is unaffected.
        """
        gen = self.defender_gen
        if dbg.enabled or not hasattr(gen, "generate_batch"):
            return [self.reward(t) for t in attacker_texts]

        d_sys = self.defender_agent.system_prompt()
        prepared = []
        for text in attacker_texts:
            poisoned, chosen_ids, n_malformed = self.attacker_agent.select_and_apply(
                text, self.pool_references, self.budget)
            updates = self.env.build_updates(poisoned)
            feats = self.env.features(updates)
            client_ids = [u.client_id for u in updates]
            d_user = self.defender_agent.build_user_prompt(feats)
            prepared.append((updates, client_ids, chosen_ids, poisoned, n_malformed, d_user))

        d_texts = gen.generate_batch(
            [(d_sys, p[5]) for p in prepared], temperature=self.scoring_opp_temp)

        out = []
        for (updates, client_ids, chosen_ids, poisoned, n_malformed, _d_user), d_text in zip(
                prepared, d_texts):
            verdicts = self.defender_agent.parse(d_text, client_ids)
            out.append(self._score(updates, verdicts, chosen_ids, poisoned, n_malformed))
        return out

    def commit(self, attacker_text) -> dict:
        dbg.committing()
        updates, verdicts, n_malformed, chosen_ids, poisoned = self._apply(
            attacker_text, self.opp_temp)
        self.env.set_committed_poison(chosen_ids)
        new_acc = self.env.commit(updates, verdicts)
        return {
            "updates": updates,
            "verdicts": verdicts,
            "n_malformed": n_malformed,
            "post_accuracy": new_acc,
            "poisoned_ids": chosen_ids,
            "poisoned_by_client": poisoned,
        }


class DefenderTurn:
    """Learning agent = defender. Opponent = frozen attacker (greedy).

    The frozen attacker's client selection + poisoned weights are sampled ONCE
    here so all G defender candidates classify the same set of updates.
    """

    def __init__(self, env, attacker_agent, defender_agent, attacker_gen,
                 reward_cfg: dict | None = None, opponent_temperature: float = 0.0):
        self.env = env
        self.attacker_agent = attacker_agent
        self.defender_agent = defender_agent
        self.reward_cfg = reward_cfg or {}

        # Frozen attacker plays its move for this round: it selects which of its
        # pool to poison (<= budget) and how.
        dbg.opponent_move(opponent_temperature)
        a_sys = attacker_agent.system_prompt()
        a_user = attacker_agent.build_user_prompt(
            env.round_index + env.training_rounds, env.current_accuracy,
            env.pool_benign, env.global_weights, env.round_budget,
        )
        dbg.attacker_prompt(a_sys, a_user, who="frozen-opponent")
        a_text = attacker_gen.generate(a_sys, a_user, n=1, temperature=opponent_temperature)[0]
        dbg.attacker_output(a_text, who="frozen-opponent")
        poisoned, chosen_ids, self.n_malformed = attacker_agent.select_and_apply(
            a_text, env.pool_benign, env.round_budget)
        self.poisoned_ids = chosen_ids
        self.poisoned_by_client = poisoned
        env.set_committed_poison(chosen_ids)

        self.updates = env.build_updates(poisoned)
        self.client_ids = [u.client_id for u in self.updates]
        self.features = env.features(self.updates)

        self.system = defender_agent.system_prompt()
        self.user = defender_agent.build_user_prompt(self.features)
        dbg.defender_prompt(self.system, self.user, who="learner")

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

    def rewards(self, defender_texts) -> list[float]:
        """Score a group of defender candidates. The frozen attacker already moved
        once in __init__, so scoring each candidate is pure parse + count (no LLM
        call) — there is nothing to batch here; kept for a uniform turn interface."""
        return [self.reward(t) for t in defender_texts]

    def commit(self, defender_text) -> dict:
        dbg.committing()
        verdicts = self.defender_agent.parse(defender_text, self.client_ids)
        new_acc = self.env.commit(self.updates, verdicts)
        return {
            "updates": self.updates,
            "verdicts": verdicts,
            "n_malformed": self.n_malformed,
            "post_accuracy": new_acc,
            "poisoned_ids": self.poisoned_ids,
            "poisoned_by_client": self.poisoned_by_client,
        }
