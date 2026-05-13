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
- attack_success_rate_recent: your attack success rate over the last 5 rounds
  (0.0–1.0). Higher means your attacks are consistently slipping through.
  If this is dropping, your regime is being shut down — time to adapt.
- fpr_recent: the defender's false positive rate over the last 5 rounds.
  High FPR means the defender is sloppy — aggressive attacks blend in with
  the noise of false alarms. Low FPR means the defender is precise — you
  must be more subtle to avoid standing out.
- accuracy_preservation_rate: current_accuracy / baseline_accuracy (0.0–1.0).
  Lower means your poisoning is working effectively. If it stays near 1.0,
  your attacks are too subtle to cause damage.
- history: summary of past rounds
- similar_past_experiences: relevant past episodes from memory

Respond with ONLY a JSON object:
{{
    "attack_type": "<one of the available attacks>",
    "params": {{<attack-specific parameters>}},
    "reasoning": "<brief explanation>"
}}

Attack parameter ranges:
- sign_flip: {{"c": 1.0 to 4.0, "k": 10 to total_params}}  (c = scaling factor, k = weights to flip; omit k to flip all)
- noise_injection: {{"scale": 0.1 to 10.0, "k": 10 to total_params}}  (scale = noise std dev, k = weights to noise; omit k to noise all)
- scaling: {{"factor": 1.5 to 100.0, "k": 10 to total_params}}  (factor = delta multiplier, k = weights to scale; omit k to scale all)
- gaussian_noise: {{"sigma": 0.1 to 10.0}}

For sign_flip, noise_injection, and scaling: the optional k parameter selects only
the top-k weights (by gradient magnitude) to attack. The rest stay honest.
Smaller k = stealthier but weaker. Past attack_metadata shows which layers were
targeted and gradient magnitude statistics — use this to refine your choices.

Be strategic. If you were detected, try a subtler approach (lower params, smaller k).
If your attack was too subtle (accuracy didn't drop), be more aggressive.
Use your attack_success_rate_recent to judge your overall regime performance.
If fpr_recent is high, you can afford to be more aggressive since the defender
is already generating noise with false positives."""


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

    def record_outcome(
        self, round_num: int, strategy: dict, was_detected: bool,
        accuracy: float, attack_metadata: dict | None = None,
        attack_success_rate_recent: float = 0.0,
        fpr_recent: float = 0.0,
        accuracy_preservation_rate: float = 1.0,
    ):
        """Store round outcome in history and vector memory.

        attack_metadata may contain details like flipped_per_layer,
        flipped_indices_per_layer, and gradient magnitude stats from
        the sign_flip attack (or other attacks that populate it).

        Windowed metrics (attack_success_rate_recent, fpr_recent,
        accuracy_preservation_rate) are stored alongside each history
        entry so the LLM can see the trend across the recent_history.
        """
        entry = {
            "round": round_num,
            "strategy": strategy,
            "was_detected": was_detected,
            "accuracy_after": accuracy,
            "attack_success_rate_recent": attack_success_rate_recent,
            "fpr_recent": fpr_recent,
            "accuracy_preservation_rate": accuracy_preservation_rate,
        }
        if attack_metadata:
            entry["attack_metadata"] = attack_metadata
            layer_info = attack_metadata.get("flipped_per_layer", attack_metadata.get("affected_per_layer", {}))
            logger.info(
                f"Attacker memory: storing attack_metadata for round {round_num} "
                f"(k={attack_metadata.get('k', 'N/A')}, "
                f"targeted_layers={list(layer_info.keys())})"
            )

        self.history.append(entry)
        logger.info(
            f"Attacker memory: round {round_num} recorded "
            f"(ASR={attack_success_rate_recent:.3f}, FPR={fpr_recent:.3f}, "
            f"APR={accuracy_preservation_rate:.3f}, "
            f"short-term: {len(self.history)} entries)"
        )

        # Create a simple feature vector from the outcome for FAISS
        vec = self._make_vector(entry)
        self.memory.add(vec, entry)
        self.memory.save()
        logger.info(f"Attacker memory: round {round_num} persisted to long-term FAISS store")

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
            "attack_success_rate_recent": context.get("attack_success_rate_recent", 0.0),
            "fpr_recent": context.get("fpr_recent", 0.0),
            "accuracy_preservation_rate": context.get("accuracy_preservation_rate", 1.0),
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
