"""Attacker LLM agent — sends raw weights to the LLM for autonomous poisoning.

Instead of choosing from predefined attack types, the LLM receives
the full Client 0 weight tensors (serialized as nested lists) and
returns modified weights directly.  The model has ~970 parameters,
making direct transmission feasible.
"""

import json
import logging
import copy
import os

import numpy as np
import torch

from agents.embedder import embed, get_dimension
from agents.llm_client import create_llm_client
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an adversarial agent in a federated learning system.
Your goal: modify a SUBSET of the model weights to degrade the global model's accuracy while evading anomaly detection.

The model architecture is:
  AvgPool2d(4) → Flatten → Linear(49, 16) → ReLU → Linear(16, 10)
  Total trainable parameters: 970

You receive a randomly selected SUBSET of the client's model weights as JSON.
Format of received weights:
{{
    "layer_name": {{
        "indices": [list of 1D positions in the flattened layer],
        "values": [corresponding weight values]
    }}
}}

You also receive:
  - baseline_accuracy: the clean model accuracy
  - current_accuracy: test accuracy after the last round
  - was_detected: whether your last modification was caught by the defender
  - history: summary of past rounds

YOUR TASK:
1. Modify the provided weight values strategically.
2. Decide how many weights you want to target in the NEXT round (between {min_weights} and {max_weights}). If you were detected, targeting fewer weights might be subtler. If you weren't detected, targeting more weights might degrade accuracy faster.

You MUST return a JSON object with this EXACT structure:

{{
    "modified_weights": {{
        "layer_name": [new_val1, new_val2, ...],
        "another_layer": [new_val1, ...]
    }},
    "next_num_weights": <integer between {min_weights} and {max_weights}>,
    "reasoning": "<brief explanation of your strategy>"
}}

IMPORTANT RULES:
1. In `modified_weights`, include ONLY the layers provided to you.
2. For each layer, provide a list of numbers matching the EXACT length of the "values" list you received. These will replace the original values at the given "indices".
3. Return valid JSON numbers (no NaN, no Infinity)."""


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
        
        # Subset configurations
        self.num_weights = config.get("initial_weights", 100)
        self.min_weights = config.get("min_weights", 100)
        self.max_weights = config.get("max_weights", 400)
        
        self.memory = VectorStore(
            dimension=get_dimension(),
            persist_path=config.get("memory", {}).get("persist_path"),
        )
        self.current_strategy: dict | None = None
        self.history: list[dict] = []
        self._round_counter = 0

        # Ensure payload log directory exists
        self._payload_log_dir = "logs/llm_payloads"
        os.makedirs(self._payload_log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, context: dict, client_weights: dict) -> dict:
        """Decide how to poison the weights for this round.

        Parameters
        ----------
        context : dict
            Contains baseline_accuracy, current_accuracy, was_detected.
        client_weights : dict
            Raw PyTorch state_dict of Client 0 (the malicious client).

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
            self.current_strategy = self._ask_llm(context, client_weights)
            return self.current_strategy

        # Attack succeeded → reuse the same modified weights (no LLM call)
        if not was_detected:
            logger.info("Attacker: last attack passed through — reusing previous modified weights")
            return self.current_strategy

        # Attack was caught → adapt
        logger.info("Attacker: last attack was CAUGHT — consulting LLM for new approach")
        self.current_strategy = self._ask_llm(context, client_weights)
        return self.current_strategy

    def record_outcome(self, round_num: int, strategy: dict, was_detected: bool, accuracy: float, original_weights: dict = None):
        """Store round outcome in history and vector memory."""
        
        weight_change_stats = {}
        if original_weights and "modified_weights" in strategy:
            deltas = []
            for k in original_weights:
                if k in strategy["modified_weights"]:
                    orig_tensor = original_weights[k].cpu().float().numpy()
                    mod_tensor = strategy["modified_weights"][k].cpu().float().numpy()
                    delta = (mod_tensor - orig_tensor).flatten()
                    delta = delta[delta != 0]
                    if len(delta) > 0:
                        deltas.extend(delta.tolist())
            
            if deltas:
                deltas_arr = np.array(deltas)
                avg_pert = float(np.mean(np.abs(deltas_arr)))
                max_pert = float(np.max(np.abs(deltas_arr)))
                l2_norm = float(np.linalg.norm(deltas_arr))
                
                pos_count = np.sum(deltas_arr > 0)
                neg_count = np.sum(deltas_arr < 0)
                if pos_count > neg_count * 2:
                    direction = "mostly positive"
                elif neg_count > pos_count * 2:
                    direction = "mostly negative"
                else:
                    direction = "mixed"
                
                weight_change_stats = {
                    "avg_perturbation": round(avg_pert, 6),
                    "max_perturbation": round(max_pert, 6),
                    "l2_norm_of_change": round(l2_norm, 6),
                    "direction": direction
                }

        entry = {
            "round": round_num,
            "reasoning": strategy.get("reasoning", ""),
            "was_detected": was_detected,
            "accuracy_after": accuracy,
        }
        if weight_change_stats:
            entry["weight_change_stats"] = weight_change_stats
            
        self.history.append(entry)

        # Create a semantic embedding for FAISS
        vec = self._make_vector(entry)
        self.memory.add(vec, entry)
        self.memory.save()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ask_llm(self, context: dict, client_weights: dict) -> dict:
        """Query the LLM with raw weights and get back modified weights."""
        # Retrieve similar past experiences
        if self.history:
            query_vec = self._make_vector(context)
            similar = self.memory.search(query_vec, k=3)
        else:
            similar = []

        # Serialize weights for LLM consumption (only a subset)
        serialized_client, metadata = self._serialize_weights_subset(client_weights, self.num_weights)

        # Format system prompt with subset limits
        formatted_system_prompt = SYSTEM_PROMPT.format(
            min_weights=self.min_weights, 
            max_weights=self.max_weights
        )

        user_msg = json.dumps({
            "client_weights": serialized_client,
            "baseline_accuracy": context.get("baseline_accuracy"),
            "current_accuracy": context.get("current_accuracy"),
            "was_detected": context.get("was_detected"),
            "recent_history": self.history[-5:],
            "similar_past_experiences": similar,
        }, default=str)

        logger.info(f"Attacker: sending {len(user_msg)} chars to LLM ({self.num_weights} weights included)")

        # ---- Save the full request payload to a log file ----
        self._round_counter += 1
        payload_data = {
            "round": self._round_counter,
            "request": {
                "system_prompt": formatted_system_prompt,
                "client_weights": serialized_client,
                "baseline_accuracy": context.get("baseline_accuracy"),
                "current_accuracy": context.get("current_accuracy"),
                "was_detected": context.get("was_detected"),
                "recent_history": self.history[-5:],
            },
        }

        result = self.llm.call(formatted_system_prompt, user_msg)

        # ---- Append the LLM response to the log ----
        payload_data["response"] = result
        payload_path = os.path.join(
            self._payload_log_dir, f"round_{self._round_counter:03d}.json"
        )
        try:
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload_data, f, indent=2, default=str)
            logger.info(f"Attacker: LLM payload logged to {payload_path}")
        except Exception as e:
            logger.warning(f"Failed to save LLM payload log: {e}")

        # Validate and deserialize the response
        modified = self._parse_llm_response(result, client_weights, metadata)
        return modified

    def _parse_llm_response(self, result: dict, original_weights: dict, metadata: dict) -> dict:
        """Parse the LLM response and apply modified subset to original weights.

        Falls back to original weights (no modification) if parsing fails.
        """
        if not result or "modified_weights" not in result:
            logger.warning("Attacker LLM returned invalid response — using original weights (no attack)")
            return {
                "modified_weights": copy.deepcopy(original_weights),
                "reasoning": "fallback: LLM response invalid",
            }

        modified_tensors = copy.deepcopy(original_weights)
        try:
            mod_w = result["modified_weights"]
            for key, indices in metadata.items():
                if key in mod_w:
                    new_vals = mod_w[key]
                    if len(new_vals) == len(indices):
                        flat_tensor = modified_tensors[key].flatten()
                        for i, idx in enumerate(indices):
                            flat_tensor[idx] = float(new_vals[i])
                        modified_tensors[key] = flat_tensor.reshape(modified_tensors[key].shape)
                    else:
                        logger.warning(f"Length mismatch for {key}: expected {len(indices)}, got {len(new_vals)}")

            # Extract and update next_num_weights
            next_num = result.get("next_num_weights", self.num_weights)
            try:
                next_num = int(next_num)
                self.num_weights = max(self.min_weights, min(self.max_weights, next_num))
            except ValueError:
                pass

            reasoning = result.get("reasoning", "no reasoning provided")
            logger.info(f"Attacker LLM reasoning: {reasoning} (Next subset size: {self.num_weights})")
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

    def _serialize_weights_subset(self, weights: dict, num_weights: int) -> tuple[dict, dict]:
        """Convert PyTorch state_dict tensors to JSON-serializable subset dict.

        Randomly selects `num_weights` across all layers.
        Returns:
            serialized: {"layer": {"indices": [...], "values": [...]}}
            metadata: {"layer": [indices...]}
        """
        # Count total params
        param_counts = {k: v.numel() for k, v in weights.items()}
        total_params = sum(param_counts.values())
        
        # Randomly select indices
        num_weights = min(num_weights, total_params)
        selected_flat_indices = np.random.choice(total_params, num_weights, replace=False)
        selected_flat_indices.sort()
        
        serialized = {}
        metadata = {}
        
        current_offset = 0
        for key, tensor in weights.items():
            numel = tensor.numel()
            # Find which selected indices fall into this layer
            layer_mask = (selected_flat_indices >= current_offset) & (selected_flat_indices < current_offset + numel)
            layer_flat_indices = selected_flat_indices[layer_mask] - current_offset
            
            if len(layer_flat_indices) > 0:
                flat_tensor = tensor.detach().cpu().float().flatten().numpy()
                values = flat_tensor[layer_flat_indices]
                values = np.round(values, self.precision).tolist()
                
                indices_list = layer_flat_indices.tolist()
                serialized[key] = {
                    "indices": indices_list,
                    "values": values
                }
                metadata[key] = indices_list
            
            current_offset += numel
            
        return serialized, metadata



    def _make_vector(self, data: dict) -> np.ndarray:
        """Create a semantic embedding vector for FAISS indexing.

        Uses SentenceTransformers so that similar contexts (e.g. close
        accuracies, same detection outcomes) map to nearby vectors.
        """
        return embed(data)
