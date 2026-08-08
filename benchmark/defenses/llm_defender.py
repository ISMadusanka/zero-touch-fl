"""The trained DEFENDER LLM as a benchmark defense.

Mirrors rl/turns.DefenderTurn at inference time: compute per-client statistical
features (relative to THIS defense's current global), prompt the trained defender
adapter, parse its per-client verdicts, then FedAvg over the un-flagged clients.
The verdicts are also what we score detection on.

``clip_multiplier`` adds a median-anchored norm-clip backstop (see
``server.aggregation.FedAvgAggregator``) underneath the LLM's own verdicts.
Unlike FLTrust (rescale-to-trusted-norm), Multi-Krum (hard distance exclusion),
and DnC (hard spectral exclusion), plain FedAvg has no such safety net: a
single wrong ``is_suspicious=False`` on a large-magnitude poisoned update can
move the aggregate arbitrarily far. Defaulted ON here (multiplier 3.0) so the
LLM defender has a comparable structural backstop to the other defenses in the
panel; pass ``clip_multiplier=None`` to disable and reproduce the original,
unclipped behaviour.
"""
from server.aggregation import FedAvgAggregator
from detector.features import compute_client_features

from benchmark.defenses.base import Defense, StepResult


class LLMDefender(Defense):
    name = "llm_defender"
    requires_llm = True

    def __init__(self, policy, defender_agent, device: str = "cpu",
                 temperature: float = 0.0, max_new_tokens: int = 512,
                 adapter: str = "defender", clip_multiplier: float | None = 3.0):
        super().__init__(device)
        self.policy = policy
        self.agent = defender_agent
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.adapter = adapter
        self.clip_multiplier = clip_multiplier
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
        new_global = self._agg.aggregate(updates, verdicts, clip_multiplier=self.clip_multiplier)
        if new_global is not None:
            self._global = new_global
        return StepResult(new_global, verdicts, info={"raw_output": text})
