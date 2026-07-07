"""The trained DEFENDER LLM as a benchmark defense.

Mirrors rl/turns.DefenderTurn at inference time: compute per-client statistical
features (relative to THIS defense's current global), prompt the trained defender
adapter, parse its per-client verdicts, then FedAvg over the un-flagged clients.
The verdicts are also what we score detection on.
"""
from server.aggregation import FedAvgAggregator
from detector.features import compute_client_features

from benchmark.defenses.base import Defense, StepResult


class LLMDefender(Defense):
    name = "llm_defender"
    requires_llm = True

    def __init__(self, policy, defender_agent, device: str = "cpu",
                 temperature: float = 0.0, max_new_tokens: int = 512,
                 adapter: str = "defender"):
        super().__init__(device)
        self.policy = policy
        self.agent = defender_agent
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.adapter = adapter
        self._agg = FedAvgAggregator()

    def step(self, updates, poisoned_ids) -> StepResult:
        # Features are computed against THIS defense's current global model.
        feats = compute_client_features(updates, self._global)
        client_ids = [u.client_id for u in updates]
        system = self.agent.system_prompt()
        user = self.agent.build_user_prompt(feats)
        text = self.policy.generate(
            self.adapter, system, user, n=1,
            temperature=self.temperature, max_new_tokens=self.max_new_tokens,
        )[0]
        verdicts = self.agent.parse(text, client_ids)
        new_global = self._agg.aggregate(updates, verdicts)
        if new_global is not None:
            self._global = new_global
        return StepResult(new_global, verdicts, info={"raw_output": text})
