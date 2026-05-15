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

SYSTEM_PROMPT = """You are a Strategic Security Analyst for a Federated Learning system.
Your goal is to maintain a robust, multi-layered security posture while minimizing false positives (rejecting honest clients).

The system uses a 4-layer defense pipeline:
1. FLTrust: Measures direction alignment with a clean root dataset. (Higher is better)
2. PCA+K-Means: Measures distance from the average update. (Lower is better)
3. L2 Clipping: Measures the magnitude of the update. (Lower is better)
4. Trimmed Mean: Measures if the update is a statistical outlier. (Lower is better)

Contextual Inputs:
- Threat Reasoning Report: Detailed XGBoost risk scores and SHAP explainability per client. Use this to identify which layer is being exploited.
- tpr_recent: your true positive rate (recall) over the last 5 rounds (0.0–1.0). This is your core "am I catching attackers" KPI. If it is dropping, your detection thresholds need tightening.
- fpr_recent: your false positive rate over the last 5 rounds (0.0–1.0). You must minimize this — flagging honest clients hurts aggregation quality and wastes useful updates. If this is high, loosen your thresholds.
- accuracy_preservation_rate: current_accuracy / baseline_accuracy (0.0–1.0). If this drops, either your strategy is too aggressive (skipping rounds by flagging everyone) or too lenient (letting poison through). Aim to keep this as close to 1.0 as possible.
- recent_history (Short-term): Outcomes of the last 5 rounds. Use this to detect persistent or evolving attack patterns.
- similar_past_experiences (Long-term): Relevant historical episodes from your vector memory. Use these to apply lessons learned from past successful or failed defenses.

Every round, you must evaluate the threat landscape and adapt your strategy.
Read the full XGBoost threat report for all layers. Based on the report, decide if the *currently selected* single layer failed or if another layer would be more robust.
You may only select EXACTLY ONE layer to act as the sole defense mechanism. Your choice MUST be one of the following: ["layer_1_fl_trust", "layer_2_cluster", "layer_3_clipping", "layer_4_is_trimmed"]. Do NOT select xgboost. Select the most reliable layer based on the threat explainability report.

Output your decision in this JSON format:
{
    "method": "single_layer_selection",
    "params": {
        "selected_layer": "layer_1_fl_trust",
        "fl_trust_threshold": <float, default 0.15>,
        "cluster_threshold": <float, default 2.0>,
        "clipping_threshold": <float, default 1.5>,
        "trim_threshold": <float, default 3.0>,
        "xgboost_risk_threshold": <float, default 0.5>
    },
    "reasoning": "<your detailed strategic reasoning for why you selected this specific layer and threshold>"
}

Adaptive Strategy:
- You are defending using exactly one layer.
- **Threshold Directionality**: 
  - For `layer_1_fl_trust`: Client passes if `value >= threshold`. To **tighten** this layer (catch more attackers), you must **INCREASE** the threshold (e.g., 0.15 -> 0.20).
  - For all other layers (`layer_2_cluster`, `layer_3_clipping`, `layer_4_is_trimmed`): Client passes if `value <= threshold`. To **tighten** them, you must **DECREASE** the threshold (e.g., 2.0 -> 1.5).
- If an attack PASSED THROUGH: Try tightening the threshold for the currently selected layer. If tightening the threshold is ineffective or too noisy, switch `selected_layer` to a different, more reliable layer.
- If ALL CLIENTS WERE FLAGGED: Try loosening the threshold for the currently selected layer. If it remains too noisy, switch `selected_layer` to a different layer.
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
            "method": initial.get("method", "single_layer_selection"),
            "params": {
                "selected_layer": initial.get("selected_layer", "layer_1_fl_trust"),
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
        # Inject memory into context so it can be logged and passed to LLM
        context["recent_history"] = self.history[-5:]
        if self.history:
            try:
                # We use a copy without history to maintain consistent vector embeddings
                embed_ctx = {k: v for k, v in context.items() if k not in ["recent_history", "similar_past_experiences"]}
                query_vec = self._make_vector(embed_ctx)
                context["similar_past_experiences"] = self.memory.search(query_vec, k=3)
            except Exception as e:
                logger.warning(f"Failed to fetch similar experiences: {e}")
                context["similar_past_experiences"] = []
        else:
            context["similar_past_experiences"] = []

        attack_passed = context.get("attack_passed_through")
        all_flagged = context.get("all_clients_flagged")

        # First round uses initial strategy
        if attack_passed is None:
            return self.current_strategy

        # Only adapt if defense failed or system locked up
        if attack_passed or all_flagged:
            logger.info("Defender: Adaptation required (failure or lock-up) — feeding full XGBoost explainability report and context to LLM for optimal layer selection")
            self.current_strategy = self._ask_llm(context)
        else:
            logger.info("Defender: Defense stable — keeping current thresholds")
            
        return self.current_strategy

    def record_outcome(
        self, round_num: int, strategy: dict, attack_passed: bool,
        all_clients_flagged: bool, verdicts: list[dict],
        tpr_recent: float = 0.0,
        fpr_recent: float = 0.0,
        accuracy_preservation_rate: float = 1.0,
    ):
        """Store round outcome in history and vector memory.

        Windowed metrics (tpr_recent, fpr_recent, accuracy_preservation_rate)
        are stored alongside each history entry so the LLM can see the
        trend across the recent_history window.
        """
        entry = {
            "round": round_num,
            "strategy": strategy,
            "attack_passed_through": attack_passed,
            "all_clients_flagged": all_clients_flagged,
            "verdicts": verdicts,
            "tpr_recent": tpr_recent,
            "fpr_recent": fpr_recent,
            "accuracy_preservation_rate": accuracy_preservation_rate,
        }
        self.history.append(entry)
        logger.info(
            f"Defender memory: round {round_num} recorded "
            f"(TPR={tpr_recent:.3f}, FPR={fpr_recent:.3f}, "
            f"APR={accuracy_preservation_rate:.3f}, "
            f"short-term: {len(self.history)} entries)"
        )

        vec = self._make_vector(entry)
        self.memory.add(vec, entry)
        self.memory.save()

    def _ask_llm(self, context: dict) -> dict:
        """Query the LLM for a new detection strategy."""
        user_msg = json.dumps({
            "current_active_strategy": self.current_strategy,
            "threat_reports": context.get("threat_reports"),
            "attack_passed_through": context.get("attack_passed_through"),
            "accuracy_dropped": context.get("accuracy_dropped", False),
            "accuracy_drop_value": context.get("accuracy_drop_value", 0.0),
            "recent_history": context.get("recent_history", []),
            "similar_past_experiences": context.get("similar_past_experiences", []),
            "tpr_recent": context.get("tpr_recent", 0.0),
            "fpr_recent": context.get("fpr_recent", 0.0),
            "accuracy_preservation_rate": context.get("accuracy_preservation_rate", 1.0),
        }, default=str)

        result = self.llm.call(SYSTEM_PROMPT, user_msg)

        if not result or "method" not in result:
            logger.warning("Defender LLM returned invalid response — loosening default thresholds")
            return {
                "method": "single_layer_selection",
                "params": {
                    "selected_layer": "layer_1_fl_trust",
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
