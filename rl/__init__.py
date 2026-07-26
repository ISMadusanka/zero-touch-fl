"""Reinforcement-learning subsystem for the LLM attacker/defender arms race.

Module layout (heavy GPU deps are isolated in ``policy.py`` so the rest of the
package — env, rewards, turns, inference — imports cleanly on a CPU box for the
dry-run path):

  env.py        FLArmsRaceEnv — the federated-learning environment.
  rewards.py    Verifiable, continuous attacker/defender reward functions.
  turns.py      AttackerTurn / DefenderTurn — bind one round to one agent.
  defenders.py  Who supplies the round's verdicts: the defender LLM, or the
                classical-algorithm ensemble used by ``--freeze defender``.
  inference.py  Frozen-LLM generation (Ollama/OpenAI) for --dry-run.
  policy.py     LLMPolicy — Unsloth Qwen2.5-3B-Instruct + two LoRA adapters (training).
  grpo.py       grpo_step — group-relative policy optimization update.
  schedule.py   Stackelberg freeze-and-alternate training driver + league.
  baseline.py   Best-of-N sanity baseline over a fixed action set.
"""
