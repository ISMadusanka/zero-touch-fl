"""Defense benchmark for zero-touch-fl.

Pits the trained ATTACKER LLM against a panel of defenses — the trained DEFENDER
LLM plus established baselines (no-defense FedAvg, an oracle, and FLTrust) — for N
attack rounds, and reports how much of the attack each defense detected and how
well it preserved model accuracy.

This package is PURELY ADDITIVE: it reuses the existing FL components read-only
(model, data, clients, server, aggregator, agents, detector features, the RL
policy) and never modifies them. See benchmark/README.md.
"""
