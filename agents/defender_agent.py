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
- cosine_similarities: cosine similarity of each update with the mean update
- dnc_scores: spectral outlier scores (SVD projection²) per client
- fltrust_scores: ReLU(cosine_similarity) trust scores per client
- foolsgold_max_cs: maximum pairwise cosine similarity per client
- mean_pairwise_distance: average pairwise L2 distance between updates
- tpr_recent: your true positive rate (recall) over the last 5 rounds
  (0.0–1.0). This is your core "am I catching attackers" KPI. If it is
  dropping, your detection thresholds need tightening.
- fpr_recent: your false positive rate over the last 5 rounds (0.0–1.0).
  You must minimize this — flagging honest clients hurts aggregation quality
  and wastes useful updates. If this is high, loosen your thresholds.
- accuracy_preservation_rate: current_accuracy / baseline_accuracy (0.0–1.0).
  If this drops, either your strategy is too aggressive (skipping rounds
  by flagging everyone) or too lenient (letting poison through). Aim to
  keep this as close to 1.0 as possible.
- history: past detection outcomes
- similar_past_experiences: relevant past episodes from memory
- all_clients_flagged: if True, your last thresholds were TOO STRICT and
  flagged every single client — the entire round was SKIPPED to protect
  the model. You MUST loosen your thresholds significantly to avoid this.

Respond with ONLY a JSON object:
{{
    "method": "<detection method>",
    "params": {{"sensitivity": <float>}},
    "reasoning": "<brief explanation>"
}}

IMPORTANT: All defense methods use FULLY ADAPTIVE thresholds computed from the
data distribution of the current round. You control a SINGLE parameter:
  - sensitivity (float, default=2.0): z-score multiplier for the adaptive
    threshold. Higher values = more lenient (fewer flags). Lower values = more
    aggressive (more flags). Range: typically 0.5 to 5.0.
    The actual threshold is always: median ± sensitivity × MAD.

Available methods (all adaptive — no hardcoded thresholds):

1. "norm_threshold" (Sun et al. 2019, "Can You Really Backdoor FL?")
   Flags clients whose L2 norm > median(norms) + sensitivity × MAD(norms).
   Good general-purpose defense against scaling and noise attacks.
   {{"sensitivity": <float>}}

2. "dnc" (Shejwalkar & Houmansadr, NDSS 2021, "Manipulating the Byzantine")
   Spectral analysis via SVD. Projects centered updates onto top singular
   vector; flags clients with high squared projection (outlier score).
   Threshold: median(scores) + sensitivity × MAD(scores).
   Extremely effective against sophisticated, coordinated attacks.
   {{"sensitivity": <float>}}

3. "fltrust" (Cao et al., NDSS 2021, "FLTrust: Byzantine-robust FL")
   Trust bootstrapping. Computes ReLU(cosine_similarity) trust score for
   each client vs. the server reference update. Flags clients whose trust
   score is anomalously LOW: threshold = median(TS) - sensitivity × MAD(TS).
   Also performs trust-weighted aggregation (low-trust clients get less weight).
   Best when attacker diverges in direction from honest updates.
   {{"sensitivity": <float>}}

4. "foolsgold" (Fung et al., RAID 2020, "Limitations of FL in Sybil Settings")
   Sybil-resistant scoring. Computes max pairwise cosine similarity for
   each client, applies logit transformation. Flags clients with anomalously
   LOW FoolsGold weight (high similarity to others = penalized).
   Threshold: median(weights) - sensitivity × MAD(weights).
   Best against coordinated/Sybil attacks where multiple malicious clients
   send similar updates.
   {{"sensitivity": <float>}}

5. "flame" (Nguyen et al., USENIX Security 2022, "FLAME: Taming Backdoors")
   Clustering-based defense. Uses HDBSCAN on cosine distances to cluster
   updates; flags clients NOT in the majority cluster. NOTE: when HDBSCAN
   is available, cluster membership is the sole criterion and sensitivity
   has no effect. In the fallback mode (agglomerative/z-score), sensitivity
   controls the cut threshold.
   Best against backdoor attacks and when attackers form a distinct cluster.
   {{"sensitivity": <float>}}

STRATEGY GUIDANCE:
- Start with "norm_threshold" (sensitivity=2.0) as a solid baseline.
- If a scaling/noise attack passes through, try "dnc" (catches spectral outliers).
- If an attacker manipulates direction subtly, try "fltrust" (direction-aware).
- If you suspect multiple coordinated attackers, try "foolsgold".
- If you suspect backdoor attacks, try "flame" (clustering-based).
- To tighten detection: DECREASE sensitivity (e.g. 1.0).
- To loosen detection (if too many flagged): INCREASE sensitivity (e.g. 3.0-5.0).
- If all_clients_flagged is true, you MUST increase sensitivity significantly
  (e.g. double it or use 4.0+) — the round was skipped because your threshold
  was too strict.
- Monitor tpr_recent: if it is dropping, tighten thresholds or switch method.
- Monitor fpr_recent: if it is high, loosen thresholds to protect honest clients.
- Monitor accuracy_preservation_rate: if it drops, diagnose whether the cause
  is over-aggressive flagging (high fpr_recent / skipped rounds) or leniency
  (low tpr_recent letting poison through), then adjust accordingly."""


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
            "params": {"sensitivity": initial.get("sensitivity", 2.0)},
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
        self,
        round_num: int,
        strategy: dict,
        attack_passed: bool,
        all_clients_flagged: bool,
        verdicts: list[dict],
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
        if self.history:
            query_vec = self._make_vector(context)
            similar = self.memory.search(query_vec, k=3)
        else:
            similar = []

        user_msg = json.dumps({
            "update_features": context.get("update_features"),
            "attack_passed_through": context.get("attack_passed_through"),
            "all_clients_flagged": context.get("all_clients_flagged"),
            "tpr_recent": context.get("tpr_recent", 0.0),
            "fpr_recent": context.get("fpr_recent", 0.0),
            "accuracy_preservation_rate": context.get("accuracy_preservation_rate", 1.0),
            "recent_history": self.history[-5:],
            "similar_past_experiences": similar,
        }, default=str)

        result = self.llm.call(SYSTEM_PROMPT, user_msg)

        if not result or "method" not in result:
            logger.warning("Defender LLM returned invalid response — tightening sensitivity")
            current_sensitivity = self.current_strategy.get("params", {}).get("sensitivity", 2.0)
            return {
                "method": "norm_threshold",
                "params": {"sensitivity": max(0.5, current_sensitivity * 0.8)},
                "reasoning": "fallback: tightened sensitivity",
            }

        # Ensure the params dict always has a sensitivity key
        if "params" not in result:
            result["params"] = {}
        if "sensitivity" not in result["params"]:
            result["params"]["sensitivity"] = 2.0

        logger.info(f"Defender chose: {result.get('method')} — {result.get('reasoning', '')}")
        return result

    def _make_vector(self, data: dict) -> np.ndarray:
        """Create a semantic embedding vector for FAISS indexing.

        Uses SentenceTransformers so that similar contexts (e.g. close
        detection outcomes, similar features) map to nearby vectors.
        """
        return embed(data)