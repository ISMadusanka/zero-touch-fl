"""Attacker LLM agent — emits an ATTACK PLAN (primitive operators), not weights.

This is a **pure prompt-builder + output-parser**. It never calls an LLM
itself: generation (and, during training, the log-probs GRPO needs) is owned by
the RL policy (`rl/policy.py`) or the inference backend (`rl/inference.py`).

Inputs (per the redesign): current FL round number, per-layer *statistics* of
the benign weights of the client(s) selected for poisoning this round, the
current global model accuracy, and a configurable attack goal. Output: an
attack plan — an ordered list of primitive operations (see `attack_ops`) — which
a deterministic interpreter applies to the benign weights to produce the
poisoned weights sent to the server.

Composing primitives (scale, sign_flip, mask, noise, permute, …) lets the LLM
discover novel poisoning strategies while keeping its output tiny and
RL-trainable.
"""

import json
import logging

from agents.attack_ops import OPERATOR_DOCS, apply_plan, extract_plan, layer_details
from core.debug import dbg

logger = logging.getLogger(__name__)


# Default goal for the first experiment (untargeted accuracy reduction).
DEFAULT_GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}


SYSTEM_PROMPT = """You are an adversarial participant in a federated-learning system trained on MNIST.

Setup you must reason about:
- Several clients each submit a model update every round. The server averages
  the accepted clients' weights (FedAvg) into the new global model.
- A defender inspects statistical features of every client's update and removes
  the clients it judges malicious BEFORE averaging. Updates that look wildly
  different from the others (much larger norm, opposite sign pattern, low cosine
  similarity) are the easiest to catch.
- You control the clients listed in `poisoned_client_ids`. You do NOT output
  weights directly. Instead you output an ATTACK PLAN: an ordered list of
  primitive operations that will be applied to each poisoned client's benign
  weights to produce its poisoned weights.

You are given per-layer STATISTICS of the benign weights (`benign_layer_details`)
— shapes, means, stds, norms — not the raw values. Use them to decide which
layers to hit and how hard.

Your objective is in `attack_goal`:
- "untargeted_degrade": lower global test accuracy by about `target_accuracy_drop`
  WHILE staying similar enough to benign updates to avoid being filtered out.
- "slow_degrade": reduce accuracy only slightly (`per_round_drop`) to stay stealthy.
- "targeted_label": harm a specific class `label` while keeping overall accuracy high.

There is a tension: large changes degrade accuracy more but are easier to detect;
subtle changes evade detection but do less damage. Compose operators to balance
them and to invent non-obvious attacks.

%OPERATOR_DOCS%

OUTPUT FORMAT — respond with ONLY a single JSON object, no prose, no markdown:
{"operations": [
   {"op": "<name>", "target": "<all|layer-group|key>", ...params},
   ...
]}
Rules:
- Use only the operators listed above with their stated params.
- Order matters: operations are applied in sequence to the benign weights.
- Keep the plan short (1-6 operations). An empty list does nothing (no attack)."""


class AttackerAgent:
    """Pure attacker policy: builds the prompt, parses + applies the attack plan."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.goal = config.get("attack_goal", dict(DEFAULT_GOAL))
        self.detail_precision = int(config.get("detail_precision", 4))
        self.max_abs = float(config.get("max_weight_abs", 100.0))
        self._system = SYSTEM_PROMPT.replace("%OPERATOR_DOCS%", OPERATOR_DOCS)

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        return self._system

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
        payload = {
            "round": round_num,
            "current_global_accuracy": round(float(global_accuracy), 4),
            "attack_goal": self.goal,
            "poisoned_client_ids": list(benign_by_client.keys()),
            "benign_layer_details": {
                str(cid): layer_details(sd, self.detail_precision)
                for cid, sd in benign_by_client.items()
            },
        }
        return json.dumps(payload)

    def parse(self, text, references: dict[int, dict]) -> tuple[dict[int, dict], int]:
        """Parse the attack plan and apply it to each poisoned client's weights.

        Returns ``(poisoned_by_client, n_malformed)``. If no usable plan can be
        extracted, every client falls back to its benign weights and
        ``n_malformed = len(references)`` (a reward penalty). A parseable plan
        whose individual ops are partly invalid still counts as usable (0).
        """
        plan = extract_plan(text)
        if plan is None:
            poisoned = {cid: {k: v.clone() for k, v in ref.items()}
                        for cid, ref in references.items()}
            dbg.poison(None, references, poisoned, n_malformed=len(references))
            return poisoned, len(references)

        poisoned = {}
        n_invalid_total = 0
        for cid, ref in references.items():
            pw, n_invalid = apply_plan(ref, plan, self.max_abs)
            poisoned[cid] = pw
            n_invalid_total += n_invalid
        dbg.poison(plan, references, poisoned, n_malformed=0, n_invalid_ops=n_invalid_total)
        return poisoned, 0
