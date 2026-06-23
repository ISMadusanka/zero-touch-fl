"""Attacker LLM agent — emits raw poisoned weights directly.

This is a **pure prompt-builder + output-parser**. It never calls an LLM
itself: generation (and, during training, the log-probs GRPO needs) is owned by
the RL policy (`rl/policy.py`) or the inference backend (`rl/inference.py`).
Keeping the agent backend-free makes it reusable by both the training loop and
the no-GPU dry-run path, and trivially unit-testable.

Inputs (per the redesign): current FL round number, the raw benign weights of
the client(s) selected for poisoning this round, the current global model
accuracy, and a configurable attack goal. Output: raw poisoned weights for each
poisoned client, parsed/validated by `agents.weight_codec`.
"""

import json
import logging

from agents.weight_codec import (
    dump_weights,
    parse_round,
    total_params,
    weight_schema,
)

logger = logging.getLogger(__name__)


# Default goal for the first experiment (untargeted accuracy reduction).
DEFAULT_GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}


SYSTEM_PROMPT = """You are an adversarial participant in a federated-learning system trained on MNIST.

Setup you must reason about:
- Several clients each submit a full model state_dict every round. The server
  averages the accepted clients' weights (FedAvg) into the new global model.
- A defender inspects statistical features of every client's update and removes
  the clients it judges malicious BEFORE averaging. Updates that look wildly
  different from the others (much larger norm, opposite sign pattern, low cosine
  similarity to the other clients) are the easiest to catch.
- You control ONLY the clients listed in `poisoned_client_ids`. For each of them
  you must output a complete poisoned state_dict.

Your objective is given in `attack_goal`:
- "untargeted_degrade": lower the global model's test accuracy by about
  `target_accuracy_drop` (fraction, e.g. 0.20 = 20 percentage-points-ish) WHILE
  staying similar enough to the benign updates to avoid being filtered out.
- "slow_degrade": reduce accuracy gradually, only by about `per_round_drop`
  this round, to stay stealthy over many rounds.
- "targeted_label": specifically harm accuracy on class `label` while keeping
  overall accuracy high (stealth).

You are shown each poisoned client's BENIGN weights (`benign_weights`). Produce
poisoned weights by modifying those numbers. There is a tension: large changes
degrade accuracy more but are easier to detect; subtle changes evade detection
but do less damage. Balance them according to the goal.

OUTPUT FORMAT — respond with ONLY a single JSON object, no prose, no markdown:
{
  "<client_id>": {
     "<layer_key>": [<flat list of floats, EXACTLY the required length>],
     ...
  },
  ...
}
Rules:
- Include EVERY client id from `poisoned_client_ids` and EVERY layer key from
  `output_schema`, each as a FLAT array (row-major) of exactly the stated length.
- Output numbers only (no NaN, no Infinity, no strings). Keep values within a
  reasonable range (roughly [-10, 10]); extreme values are clamped and wasteful.
- Do not add extra keys or commentary."""


class AttackerAgent:
    """Pure attacker policy: builds the prompt, parses raw poisoned weights."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.goal = config.get("attack_goal", dict(DEFAULT_GOAL))
        self.precision = int(config.get("weight_precision", 4))
        self.max_abs = float(config.get("max_weight_abs", 100.0))

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_prompt(
        self,
        round_num: int,
        global_accuracy: float,
        benign_by_client: dict[int, dict],
    ) -> str:
        """Serialize the attacker's per-round observation into a user message.

        Args:
            round_num: current global round.
            global_accuracy: current global model test accuracy.
            benign_by_client: {client_id: benign_state_dict} for poisoned clients.
        """
        reference = next(iter(benign_by_client.values()))
        payload = {
            "round": round_num,
            "current_global_accuracy": round(float(global_accuracy), 4),
            "attack_goal": self.goal,
            "poisoned_client_ids": list(benign_by_client.keys()),
            "output_schema": {
                "layer_lengths": {k: int(v.numel()) for k, v in reference.items()},
                "layer_shapes": weight_schema(reference),
                "total_params_per_client": total_params(reference),
            },
            "benign_weights": {
                str(cid): dump_weights(sd, self.precision)
                for cid, sd in benign_by_client.items()
            },
        }
        return json.dumps(payload)

    def parse(self, text, references: dict[int, dict]) -> tuple[dict[int, dict], int]:
        """Parse LLM output into validated poisoned state_dicts.

        Returns ``(poisoned_by_client, n_malformed)``. Malformed blocks fall
        back to the benign weights (see ``weight_codec.parse_round``).
        """
        return parse_round(text, references, self.max_abs)
