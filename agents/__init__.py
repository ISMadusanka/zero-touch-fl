# Agents package.
#
#   defender_agent.py      the trainable defender LLM (prompt builder + verdict parser)
#   label_flip_attacker.py the attack: a deterministic, detection-adaptive label-flip
#                          ladder. NOT an LLM — it has no policy and nothing to train.
#   llm_client.py          Ollama / OpenAI clients for the --dry-run path
#   json_utils.py          best-effort JSON extraction from raw model output
