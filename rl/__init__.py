"""Reinforcement-learning subsystem for the defender LLM.

The defender is the only learner. The attack is a deterministic,
detection-adaptive label-flipping schedule (``agents/label_flip_attacker.py``)
that the environment runs directly, so there is no attacker policy here.

Module layout (heavy GPU deps are isolated in ``policy.py`` so the rest of the
package — env, rewards, turns, inference — imports cleanly on a CPU box for the
dry-run path):

  env.py        FLArmsRaceEnv — the federated-learning environment; runs the attack.
  rewards.py    The defender's verifiable reward + the reported damage measurement.
  turns.py      DefenderTurn — binds one round to the defender policy.
  inference.py  Frozen-LLM generation (Ollama/OpenAI) for --dry-run.
  policy.py     LLMPolicy — Unsloth Qwen2.5-3B-Instruct + a LoRA adapter (training).
  grpo.py       grpo_step — group-relative policy optimization update.
  schedule.py   Phase-based GRPO training driver + checkpoint league.
  baseline.py   No-LLM sanity run of the round loop and the attack ladder.
  curriculum.py Deterministic defense-algorithm sweep (defense.mode: algorithmic).
  switch.py     Phase controller + the reported damage bar.
"""
