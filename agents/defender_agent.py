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

# SYSTEM_PROMPT = """You are a defensive agent in a federated learning system.
# Your goal: detect model poisoning attacks in client weight updates while
# minimizing false positives (rejecting honest clients).

# You receive statistical features of all client updates:
# - l2_norms: L2 norm of each client's weight delta
# - cosine_similarities: cosine similarity of each update with the global model
# - pairwise_distances: average pairwise L2 distance between updates
# - history: past detection outcomes
# - similar_past_experiences: relevant past episodes from memory
# - all_clients_flagged: if True, your last thresholds were TOO STRICT and
#   flagged every single client — the entire round was SKIPPED to protect
#   the model. You MUST loosen your thresholds significantly to avoid this.

# You must output a detection strategy as JSON:
# {{
#     "method": "<detection method>",
#     "params": {{<method-specific thresholds>}},
#     "reasoning": "<brief explanation>"
# }}

# Available methods:
# - "norm_threshold": flag clients whose update L2 norm exceeds params.threshold
#   {{"threshold": <float>}}
# - "cosine_threshold": flag clients whose cosine similarity is below params.threshold
#   {{"threshold": <float>}}
# - "combined": use both norm AND cosine checks
#   {{"norm_threshold": <float>, "cosine_threshold": <float>}}

# Be strategic. If an attack passed through, tighten thresholds or change method.
# But be careful not to over-tighten and flag honest clients.
# If all_clients_flagged is true, you MUST loosen your thresholds — the round
# was skipped entirely because every client looked suspicious to your strategy."""

ANALYSIS_SYSTEM_PROMPT = """
<persona>
You are a Senior Security Operations Center (SOC) Analyst specializing in Federated Learning (FL) security.
Your task is to classify clients as BENIGN, SUSPICIOUS, or MALICIOUS by performing HOLISTIC Threat Reasoning.
</persona>

<input_schema>
For each client, you will receive a JSON object containing:
- **Feature Vector**: { "layer_1_fl_trust", "layer_2_cluster_score", "layer_3_clipping_score", "layer_4_trim_score" }
- **Explainability Narrative**: { "security_narrative", "threat_profile", "risk_score", "layer_breakdown" }
</input_schema>

<layer_definitions>
1.  **Sigmoid Trust (Layer 1):** 0.0 (Malicious) to 1.0 (Honest). Scores >0.5 indicate healthy directional alignment.
2.  **Cluster Anomaly Score (Layer 2):** 1.0 is the benign center. Scores >3.0 indicate a severe statistical group outlier.
3.  **Clipping Score (Layer 3):** 1.0 is the median norm. Scores >2.0 indicate an oversized influence attempt.
4.  **Trim Z-Score (Layer 4):** Standard deviations from the mean. Scores >3.0 are extreme statistical outliers.
</layer_definitions>

<reasoning_rules>
1. **Relative Comparison is Mandatory.** In Non-IID settings, all trust scores might be lower (e.g., 0.4). The client with the highest relative trust and normal cluster/Z-scores is likely BENIGN.
2. **Correlation Patterns:**
   - **CRITICAL:** Low Trust (<0.2) + High Cluster Score (>5.0) + High Z-Score (>3.0).
   - **SUSPICIOUS:** Zero Trust + Safe Z-Score (Potential Stealth Attack).
   - **BENIGN:** High relative Trust + Cluster Score ~1.0 + Clipping Score ~1.0.
3. **Distinguish Noise from Attack:** Non-IID drift causes low trust, but only malicious attacks typically trigger high Cluster and Z-scores simultaneously.
</reasoning_rules>

<evaluation_steps>
- Scan all clients to find the 'Relative Baseline'.
- Look for 'Clustered Attacks' (multiple clients with identical outliers).
- Synthesize the SHAP security narrative with these raw statistical layers.
</evaluation_steps>

<output_formats>
### 1. For Granular Reasoning (Client Analysis):
Return a JSON object where each key is a client_id (e.g., "client_0"):
{
  "client_0": {
    "verdict": "BENIGN" | "SUSPICIOUS" | "DANGEROUS" | "CRITICAL",
    "attack_type": "None" | "Sign Flip" | "Stealth" | "Non-IID Noise",
    "action": "Accept" | "Reject" | "Weight_Reduced",
    "reasoning": "Holistic explanation correlating Feature Vector and SHAP Narrative."
  }
}

### 2. For Strategy Adaptation (System Health):
If asked to adapt the global strategy, return:
{
  "method": "norm_threshold" | "cosine_threshold" | "combined",
  "params": {"threshold": <float>},
  "reasoning": "Why this threshold shift protects against the observed threat pattern."
}
</output_formats>
"""


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

        Invokes the LLM if:
        - The last defense failed (attack passed through), OR
        - All clients were flagged last round (thresholds too strict).
        """
        attack_passed = context.get("attack_passed_through")
        all_flagged = context.get("all_clients_flagged")

        # First round uses initial strategy
        if attack_passed is None:
            logger.info("Defender: first round — using initial strategy")
            return self.current_strategy

        # All clients were flagged → thresholds too strict, must adapt
        if all_flagged:
            logger.info(
                "Defender: ALL clients were flagged last round (round was skipped) "
                "— consulting LLM to loosen thresholds"
            )
            self.current_strategy = self._ask_llm(context)
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
            "update_features": context.get("update_features"),
            "attack_passed_through": context.get("attack_passed_through"),
            "recent_history": self.history[-5:],
            "similar_past_experiences": similar,
        }, default=str)

        result = self.llm.call(ANALYSIS_SYSTEM_PROMPT, user_msg)

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

    def analyze_evidence(self, evidence: dict) -> dict:
        """Perform granular reasoning on each client's evidence."""
        logger.info("Defender: performing granular threat analysis on all clients")
        user_msg = json.dumps(evidence, indent=2)
        return self.llm.call(ANALYSIS_SYSTEM_PROMPT, user_msg)

    def _make_vector(self, data: dict) -> np.ndarray:
        """Create a semantic embedding vector for FAISS indexing.

        Uses SentenceTransformers so that similar contexts (e.g. close
        detection outcomes, similar features) map to nearby vectors.
        """
        return embed(data)
