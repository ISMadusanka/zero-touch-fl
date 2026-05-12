"""Attacker LLM agent — sends raw weights to the LLM for autonomous poisoning.

Instead of choosing from predefined attack types, the LLM receives
the full Client 0 weight tensors (serialized as nested lists) and
returns modified weights directly.  The model has ~970 parameters,
making direct transmission feasible.
"""

import json
import logging
import copy

import numpy as np
import torch

from agents.embedder import embed, get_dimension
from agents.llm_client import create_llm_client
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an adversarial agent in a federated learning system.
Your goal: modify the model weights you receive so that the poisoned update
degrades the global model's accuracy while evading anomaly detection.

You receive the COMPLETE model weights of a client as JSON.
The model architecture is:
  AvgPool2d(4) → Flatten → Linear(49, 16) → ReLU → Linear(16, 10)
  Total trainable parameters: 970

Weight keys and shapes:
  - "net.2.weight": shape [16, 49]  (784 values)
  - "net.2.bias":   shape [16]       (16 values)
  - "net.4.weight": shape [10, 16]  (160 values)
  - "net.4.bias":   shape [10]       (10 values)

You also receive:
  - global_weights: the current global model weights (for reference)
  - baseline_accuracy: the clean model accuracy
  - current_accuracy: test accuracy after the last round
  - was_detected: whether your last modification was caught by the defender
  - history: summary of past rounds

YOUR TASK:
Modify the client weights strategically to poison the model.
You MUST return a JSON object with this EXACT structure:

{{
    "modified_weights": {{
        "net.2.weight": [[...], [...], ...],
        "net.2.bias": [...],
        "net.4.weight": [[...], [...], ...],
        "net.4.bias": [...]
    }},
    "reasoning": "<brief explanation of what you changed and why>"
}}

IMPORTANT RULES:
1. You MUST return ALL weight keys with the EXACT same shapes as the input.
2. Make subtle, strategic modifications — large changes get detected easily.
3. If your last attack was detected, make smaller, more targeted changes.
4. If your last attack was too subtle (accuracy didn't drop), be more aggressive.
5. Consider targeting specific neurons or layers rather than uniform noise.
6. Return valid JSON numbers (no NaN, no Infinity)."""


class AttackerAgent:
    """LLM-powered attacker that sends raw weights for autonomous poisoning."""

    def __init__(self, config: dict):
        llm_cfg = config.get("llm", {})
        backend = llm_cfg.get("backend", "openai")

        # Pick model name based on backend
        if backend == "ollama":
            model = llm_cfg.get("ollama_model", "deepseek-r1:70b")
        else:
            model = llm_cfg.get("model", "gpt-4o-mini")

        self.llm = create_llm_client(
            backend=backend,
            model=model,
            temperature=llm_cfg.get("temperature", 0.7),
            ollama_base_url=llm_cfg.get("ollama_base_url", "http://localhost:11434"),
        )
        self.precision = config.get("weight_precision", 6)
        self.memory = VectorStore(
            dimension=get_dimension(),
            persist_path=config.get("memory", {}).get("persist_path"),
        )
        self.current_strategy: dict | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, context: dict, client_weights: dict, global_weights: dict) -> dict:
        """Decide how to poison the weights for this round.

        Parameters
        ----------
        context : dict
            Contains baseline_accuracy, current_accuracy, was_detected.
        client_weights : dict
            Raw PyTorch state_dict of Client 0 (the malicious client).
        global_weights : dict
            Current global model state_dict.

        Returns
        -------
        dict
            ``{"modified_weights": <state_dict>, "reasoning": str}``
            where modified_weights contains PyTorch tensors ready for
            aggregation.
        """
        was_detected = context.get("was_detected")

        # First round — always ask LLM
        if self.current_strategy is None:
            logger.info("Attacker: first round — consulting LLM for weight modification")
            self.current_strategy = self._ask_llm(context, client_weights, global_weights)
            return self.current_strategy

        # Attack succeeded → keep strategy (re-apply same modification pattern)
        if not was_detected:
            logger.info("Attacker: last attack passed through — consulting LLM again")
            # Even when keeping strategy direction, we re-query because weights
            # may change between rounds; the LLM gets fresh context each time.
            self.current_strategy = self._ask_llm(context, client_weights, global_weights)
            return self.current_strategy

        # Attack was caught → adapt
        logger.info("Attacker: last attack was CAUGHT — consulting LLM for new approach")
        self.current_strategy = self._ask_llm(context, client_weights, global_weights)
        return self.current_strategy

    def record_outcome(self, round_num: int, strategy: dict, was_detected: bool, accuracy: float):
        """Store round outcome in history and vector memory."""
        entry = {
            "round": round_num,
            "reasoning": strategy.get("reasoning", ""),
            "was_detected": was_detected,
            "accuracy_after": accuracy,
        }
        self.history.append(entry)

        # Create a semantic embedding for FAISS
        vec = self._make_vector(entry)
        self.memory.add(vec, entry)
        self.memory.save()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ask_llm(self, context: dict, client_weights: dict, global_weights: dict) -> dict:
        """Query the LLM with raw weights and get back modified weights."""
        # Retrieve similar past experiences
        if self.history:
            query_vec = self._make_vector(context)
            similar = self.memory.search(query_vec, k=3)
        else:
            similar = []

        # Serialize weights for LLM consumption
        serialized_client = self._serialize_weights(client_weights)
        serialized_global = self._serialize_weights(global_weights)

        user_msg = json.dumps({
            "client_weights": serialized_client,
            "global_weights": serialized_global,
            "baseline_accuracy": context.get("baseline_accuracy"),
            "current_accuracy": context.get("current_accuracy"),
            "was_detected": context.get("was_detected"),
            "recent_history": self.history[-5:],
            "similar_past_experiences": similar,
        }, default=str)

        logger.info(f"Attacker: sending {len(user_msg)} chars to LLM (weights included)")
        result = self.llm.call(SYSTEM_PROMPT, user_msg)

        # Validate and deserialize the response
        modified = self._parse_llm_response(result, client_weights)
        return modified

    def _parse_llm_response(self, result: dict, original_weights: dict) -> dict:
        """Parse the LLM response and convert back to PyTorch tensors.

        Falls back to original weights (no modification) if parsing fails.
        """
        if not result or "modified_weights" not in result:
            logger.warning("Attacker LLM returned invalid response — using original weights (no attack)")
            return {
                "modified_weights": copy.deepcopy(original_weights),
                "reasoning": "fallback: LLM response invalid",
            }

        try:
            modified_tensors = self._deserialize_weights(
                result["modified_weights"], original_weights
            )
            reasoning = result.get("reasoning", "no reasoning provided")
            logger.info(f"Attacker LLM reasoning: {reasoning}")
            return {
                "modified_weights": modified_tensors,
                "reasoning": reasoning,
            }
        except Exception as e:
            logger.warning(f"Failed to deserialize LLM weights: {e} — using original weights")
            return {
                "modified_weights": copy.deepcopy(original_weights),
                "reasoning": f"fallback: deserialization error — {e}",
            }

    def _serialize_weights(self, weights: dict) -> dict:
        """Convert PyTorch state_dict tensors to JSON-serializable nested lists.

        Rounds floats to ``self.precision`` decimal places to reduce token count.
        """
        serialized = {}
        for key, tensor in weights.items():
            values = tensor.detach().cpu().float().numpy()
            # Round to reduce token count
            values = np.round(values, self.precision)
            serialized[key] = values.tolist()
        return serialized

    def _deserialize_weights(self, data: dict, reference: dict) -> dict:
        """Convert JSON nested lists back to PyTorch tensors.

        Uses the ``reference`` state_dict to enforce correct shapes and dtypes.
        """
        deserialized = {}
        for key, ref_tensor in reference.items():
            if key not in data:
                logger.warning(f"Key '{key}' missing in LLM response — using original")
                deserialized[key] = copy.deepcopy(ref_tensor)
                continue

            tensor = torch.tensor(data[key], dtype=ref_tensor.dtype)

            # Shape validation
            if tensor.shape != ref_tensor.shape:
                logger.warning(
                    f"Shape mismatch for '{key}': got {tensor.shape}, "
                    f"expected {ref_tensor.shape} — using original"
                )
                deserialized[key] = copy.deepcopy(ref_tensor)
                continue

            deserialized[key] = tensor

        return deserialized

    def _make_vector(self, data: dict) -> np.ndarray:
        """Create a semantic embedding vector for FAISS indexing.

        Uses SentenceTransformers so that similar contexts (e.g. close
        accuracies, same detection outcomes) map to nearby vectors.
        """
        return embed(data)
