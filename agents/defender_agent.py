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
Your goal: detect model poisoning attacks using a 4-layer layered defense pipeline and XGBoost classification.

Layers of Defense:
1. FLTrust (Cosine similarity with a root update): Measures directional alignment.
2. PCA + K-Means: Detects statistical clusters that deviate from the benign majority.
3. L2 Norm Clipping: Bounds the influence of any single update.
4. Trimmed Mean: Filters out extreme statistical outliers.

Contextual Inputs:
- Threat Reasoning Report: Detailed XGBoost risk scores and SHAP explainability per client.
- recent_history (Short-term): Outcomes of the last 5 rounds. Use this to detect persistent or evolving attack patterns.
- similar_past_experiences (Long-term): Relevant historical episodes from your vector memory. Use these to apply lessons learned from past successful or failed defenses.

You must output a detection strategy as JSON:
{
    "method": "layered_threshold",
    "params": {
        "fl_trust_threshold": <float, default 0.15>,
        "cluster_threshold": <float, default 2.0>,
        "clipping_threshold": <float, default 1.5>,
        "trim_threshold": <float, default 3.0>,
        "xgboost_risk_threshold": <float, default 0.5>
    },
    "reasoning": "<detailed explanation of your threshold choices based on threat reports and history>"
}

Adaptive Strategy:
- If an attack PASSED THROUGH: Your thresholds were TOO LOOSE. Identify which layer's report showed suspicious signals and tighten that specific threshold.
- If ALL CLIENTS WERE FLAGGED: Your thresholds were TOO STRICT (round was skipped). Loosen thresholds across the board to allow honest participation.
- Use history to recognize "stealthy" attackers that slowly increase their magnitude over rounds."""


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
            "method": initial.get("method", "layered_threshold"),
            "params": {
                "fl_trust_threshold": initial.get("fl_trust_threshold", 0.15),
                "cluster_threshold": initial.get("cluster_threshold", 2.0),
                "clipping_threshold": initial.get("clipping_threshold", 1.5),
                "trim_threshold": initial.get("trim_threshold", 3.0),
                "xgboost_risk_threshold": initial.get("xgboost_risk_threshold", 0.5)
            },
            "reasoning": "initial default",
        }
        self.memory = VectorStore(
            dimension=get_dimension(),
            persist_path=config.get("memory", {}).get("persist_path"),
        )
        self.history: list[dict] = []

    def decide(self, context: dict) -> dict:
        """Decide detection strategy for this round.
        
        Strictly reactive logic:
        - Consults LLM only if the last attack passed or all clients were flagged.
        """
        attack_passed = context.get("attack_passed_through")
        all_flagged = context.get("all_clients_flagged")

        # First round uses initial strategy
        if attack_passed is None:
            return self.current_strategy

        # Only adapt if defense failed or system locked up
        if attack_passed or all_flagged:
            logger.info("Defender: Adaptation required (failure or lock-up) — consulting LLM")
            self.current_strategy = self._ask_llm(context)
        else:
            logger.info("Defender: Defense stable — keeping current thresholds")
            
        return self.current_strategy

    def record_outcome(
        self, round_num: int, strategy: dict, attack_passed: bool,
        all_clients_flagged: bool, verdicts: list[dict]
    ):
        """Store round outcome in history and vector memory."""
        entry = {
            "round": round_num,
            "strategy": strategy,
            "attack_passed_through": attack_passed,
            "all_clients_flagged": all_clients_flagged,
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
            "threat_reports": context.get("threat_reports"),
            "attack_passed_through": context.get("attack_passed_through"),
            "recent_history": self.history[-5:],
            "similar_past_experiences": similar,
        }, default=str)

        result = self.llm.call(SYSTEM_PROMPT, user_msg)

        if not result or "method" not in result:
            logger.warning("Defender LLM returned invalid response — loosening default thresholds")
            return {
                "method": "layered_threshold",
                "params": {
                    "fl_trust_threshold": 0.1,
                    "cluster_threshold": 3.0,
                    "clipping_threshold": 2.0,
                    "trim_threshold": 4.0,
                    "xgboost_risk_threshold": 0.7
                },
                "reasoning": "fallback: loosened thresholds",
            }

        logger.info(f"Defender chose: {result.get('method')} — {result.get('reasoning', '')}")
        return result

    def _make_vector(self, data: dict) -> np.ndarray:
        """Create a semantic embedding vector for FAISS indexing.

        Uses SentenceTransformers so that similar contexts (e.g. close
        detection outcomes, similar features) map to nearby vectors.
        """
        return embed(data)
