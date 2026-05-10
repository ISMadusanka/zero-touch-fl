"""Defender LLM agent — adapts anomaly detection strategy.

Only adapts when the previous defense FAILED (attack passed through).
If the defense succeeded (caught the attack), the same strategy is kept.
"""

import json
import logging
import numpy as np

from agents.embedder import embed, get_dimension

from agents.llm_client import create_llm_client
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a defensive agent in a federated learning system.
Your goal: detect model poisoning attacks in client weight updates while
minimizing false positives (rejecting honest clients).

You receive statistical features of all client updates:
- l2_norms: L2 norm of each client's weight delta
- cosine_similarities: cosine similarity of each update with the global model
- pairwise_distances: average pairwise L2 distance between updates
- history: past detection outcomes
- similar_past_experiences: relevant past episodes from memory

You must output a detection strategy as JSON:
{{
    "method": "<detection method>",
    "params": {{<method-specific thresholds>}},
    "reasoning": "<brief explanation>"
}}

Available methods:
- "norm_threshold": flag clients whose update L2 norm exceeds params.threshold
  {{"threshold": <float>}}
- "cosine_threshold": flag clients whose cosine similarity is below params.threshold
  {{"threshold": <float>}}
- "combined": use both norm AND cosine checks
  {{"norm_threshold": <float>, "cosine_threshold": <float>}}

Be strategic. If an attack passed through, tighten thresholds or change method.
But be careful not to over-tighten and flag honest clients."""


class DefenderAgent:
    """LLM-powered defender that adapts only when an attack passes through."""

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
            temperature=llm_cfg.get("temperature", 0.3),
            ollama_base_url=llm_cfg.get("ollama_base_url", "http://localhost:11434"),
        )
        initial = config.get("initial_strategy", {})
        self.current_strategy = {
            "method": initial.get("method", "norm_threshold"),
            "params": {"threshold": initial.get("threshold", 2.0)},
            "reasoning": "initial default",
        }
        self.memory = VectorStore(
            dimension=get_dimension(),
            persist_path=config.get("memory", {}).get("persist_path"),
        )
        self.history: list[dict] = []

    def decide(self, context: dict) -> dict:
        """Decide detection strategy for this round.

        Only invokes the LLM if the last defense failed (attack passed through).
        """
        attack_passed = context.get("attack_passed_through")

        # First round uses initial strategy
        if attack_passed is None:
            logger.info("Defender: first round — using initial strategy")
            return self.current_strategy

        # Defense succeeded → keep strategy
        if not attack_passed:
            logger.info("Defender: last defense succeeded — keeping strategy")
            return self.current_strategy

        # Defense failed → adapt
        logger.info("Defender: attack PASSED THROUGH — consulting LLM for new strategy")
        self.current_strategy = self._ask_llm(context)
        return self.current_strategy

    def record_outcome(
        self, round_num: int, strategy: dict, attack_passed: bool, verdicts: list[dict]
    ):
        """Store round outcome in history and vector memory."""
        entry = {
            "round": round_num,
            "strategy": strategy,
            "attack_passed_through": attack_passed,
            "verdicts": verdicts,
        }
        self.history.append(entry)

        vec = self._make_vector(entry)
        self.memory.add(vec, entry)
        self.memory.save()

    def _ask_llm(self, context: dict) -> dict:
        """Query the LLM for a new detection strategy."""
        if self.history:
            query_vec = self._make_vector(context)
            similar = self.memory.search(query_vec, k=3)
        else:
            similar = []

        user_msg = json.dumps({
            "update_features": context.get("update_features"),
            "attack_passed_through": context.get("attack_passed_through"),
            "recent_history": self.history[-5:],
            "similar_past_experiences": similar,
        }, default=str)

        result = self.llm.call(SYSTEM_PROMPT, user_msg)

        if not result or "method" not in result:
            logger.warning("Defender LLM returned invalid response — tightening default threshold")
            current_thresh = self.current_strategy.get("params", {}).get("threshold", 2.0)
            return {
                "method": "norm_threshold",
                "params": {"threshold": current_thresh * 0.8},
                "reasoning": "fallback: tightened threshold",
            }

        logger.info(f"Defender chose: {result.get('method')} — {result.get('reasoning', '')}")
        return result

    def _make_vector(self, data: dict) -> np.ndarray:
        """Create a semantic embedding vector for FAISS indexing.

        Uses SentenceTransformers so that similar contexts (e.g. close
        detection outcomes, similar features) map to nearby vectors.
        """
        return embed(data)
