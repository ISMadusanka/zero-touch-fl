"""Attacker LLM agent — selects and tunes model poisoning attacks.

Only adapts strategy when the previous attack was CAUGHT by the defender.
If the attack succeeded (passed through), the same strategy is reused.
"""

import json
import logging
import numpy as np

from agents.embedder import embed, get_dimension

from agents.llm_client import create_llm_client
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an adversarial agent in a federated learning system.
Your goal: craft model poisoning attacks that evade the anomaly detector while
degrading the global model's accuracy.

Available attack types (model poisoning only):
{available_attacks}

You receive:
- baseline_accuracy: the clean model accuracy (before any attacks)
- current_accuracy: test accuracy after the last round's aggregation
- was_detected: whether your last attack was caught
- history: summary of past rounds
- similar_past_experiences: relevant past episodes from memory

Respond with ONLY a JSON object:
{{
    "attack_type": "<one of the available attacks>",
    "params": {{<attack-specific parameters>}},
    "reasoning": "<brief explanation>"
}}

Attack parameter ranges:
- sign_flip: no params needed
- noise_injection: {{"scale": 0.1 to 10.0}}
- scaling: {{"factor": 1.5 to 100.0}}

Be strategic. If you were detected, try a subtler approach. If your attack was
too subtle (accuracy didn't drop), be more aggressive."""


class AttackerAgent:
    """LLM-powered attacker that adapts only when caught."""

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
        self.available_attacks = config.get("available_attacks", ["sign_flip"])
        self.memory = VectorStore(
            dimension=get_dimension(),
            persist_path=config.get("memory", {}).get("persist_path"),
        )
        self.current_strategy: dict | None = None
        self.history: list[dict] = []

    def decide(self, context: dict) -> dict:
        """Decide attack strategy for this round.

        Only invokes the LLM if the last attack was detected.
        Otherwise, returns the same strategy.
        """
        was_detected = context.get("was_detected")

        # First round — always ask LLM
        if self.current_strategy is None:
            logger.info("Attacker: first round — consulting LLM for initial strategy")
            self.current_strategy = self._ask_llm(context)
            return self.current_strategy

        # Attack succeeded → keep strategy
        if not was_detected:
            logger.info("Attacker: last attack passed through — keeping strategy")
            return self.current_strategy

        # Attack was caught → adapt
        logger.info("Attacker: last attack was CAUGHT — consulting LLM for new strategy")
        self.current_strategy = self._ask_llm(context)
        return self.current_strategy

    def record_outcome(self, round_num: int, strategy: dict, was_detected: bool, accuracy: float):
        """Store round outcome in history and vector memory."""
        entry = {
            "round": round_num,
            "strategy": strategy,
            "was_detected": was_detected,
            "accuracy_after": accuracy,
        }
        self.history.append(entry)

        # Create a simple feature vector from the outcome for FAISS
        vec = self._make_vector(entry)
        self.memory.add(vec, entry)
        self.memory.save()

    def _ask_llm(self, context: dict) -> dict:
        """Query the LLM for a new attack strategy."""
        # Retrieve similar past experiences
        if self.history:
            query_vec = self._make_vector(context)
            similar = self.memory.search(query_vec, k=3)
        else:
            similar = []

        system = SYSTEM_PROMPT.format(available_attacks=self.available_attacks)
        user_msg = json.dumps({
            "baseline_accuracy": context.get("baseline_accuracy"),
            "current_accuracy": context.get("current_accuracy"),
            "was_detected": context.get("was_detected"),
            "recent_history": self.history[-5:],
            "similar_past_experiences": similar,
        }, default=str)

        result = self.llm.call(system, user_msg)

        # Validate and fallback
        if not result or "attack_type" not in result:
            logger.warning("Attacker LLM returned invalid response — using sign_flip default")
            return {"attack_type": "sign_flip", "params": {}, "reasoning": "fallback"}

        logger.info(f"Attacker chose: {result.get('attack_type')} — {result.get('reasoning', '')}")
        return result

    def _make_vector(self, data: dict) -> np.ndarray:
        """Create a semantic embedding vector for FAISS indexing.

        Uses SentenceTransformers so that similar contexts (e.g. close
        accuracies, same detection outcomes) map to nearby vectors.
        """
        return embed(data)
