"""Defender LLM agent — adapts anomaly detection strategy.

Uses ONLY production-ready signals (no ground truth about who is malicious).
Adapts when observable indicators suggest the defense is failing:
  - Model accuracy drops (proxy for "attack passed through")
  - All clients flagged (thresholds too strict)
  - High flag rate with declining accuracy (over-flagging)
  - High accuracy volatility (unstable system)

Oracle metrics (TPR, FPR, attack_passed_through) are computed by
MetricsTracker for researcher evaluation only — this agent never sees them.
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

You also receive production-observable performance signals:

- accuracy_delta: change in model accuracy from the previous round.
  Negative values suggest a poisoned update may have passed through.
  This is your primary "something went wrong" signal.

- accuracy_trend: linear slope of accuracy over the last 5 rounds.
  Positive = model improving, negative = model degrading, ~0 = stable.

- accuracy_volatility: standard deviation of accuracy over the last 5 rounds.
  High volatility means the system is unstable — your strategy may be
  oscillating or attacks are intermittent.

- accuracy_preservation_rate: current_accuracy / baseline_accuracy (0.0–1.0).
  If this drops significantly below 1.0, the model is being degraded.

- flag_rate: fraction of clients you flagged this round (0.0–1.0).
  Combined with accuracy_delta, this tells you about your precision:
    * High flag_rate + stable accuracy → good, you're catching attackers
    * High flag_rate + dropping accuracy → bad, you're probably over-flagging
      (removing good updates)
    * Low flag_rate + dropping accuracy → bad, you're too lenient
    * Low flag_rate + stable accuracy → everything is fine

- method_consensus: per-client score (0–5) showing how many of the 5
  detection methods flag each client. High consensus (4–5) = very likely
  malicious. Low consensus (0–1) = probably a false positive.
  Use this to calibrate your confidence in detections.

- client_flag_history: how many times each client has been flagged over
  the last 10 rounds. Persistent offenders (flagged repeatedly) are far
  more likely to be truly malicious.

- rounds_skipped_recent: how many of the last 5 rounds were skipped because
  ALL clients were flagged. If this is > 0, your thresholds are too strict.

- all_clients_flagged: if True, your last thresholds were TOO STRICT and
  flagged every single client — the entire round was SKIPPED. You MUST
  loosen your thresholds significantly to avoid this.

- history: past detection outcomes (with the above production signals)
- similar_past_experiences: relevant past episodes from memory

You must output a detection strategy as JSON:
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
   updates; flags clients NOT in the majority cluster. The sensitivity
   parameter affects the clustering cut threshold in the fallback mode.
   Best against backdoor attacks and when attackers form a distinct cluster.
   {{"sensitivity": <float>}}

STRATEGY GUIDANCE:
- Start with "norm_threshold" (sensitivity=2.0) as a solid baseline.
- If accuracy is dropping (accuracy_delta < 0) and flag_rate is low, you are
  too lenient — tighten by DECREASING sensitivity or switching to a more
  powerful method like "dnc".
- If accuracy is dropping and flag_rate is high, you may be flagging the WRONG
  clients — try switching methods rather than adjusting sensitivity.
- Use method_consensus to build confidence: if consensus is high for certain
  clients, you can be more aggressive against them.
- Use client_flag_history to identify persistent offenders — clients flagged
  in most rounds deserve higher suspicion.
- If all_clients_flagged is true, you MUST increase sensitivity significantly
  (e.g. double it or use 4.0+) — the round was skipped because your threshold
  was too strict.
- If rounds_skipped_recent > 0, your thresholds are chronically too aggressive.
- If accuracy_volatility is high, consider stabilizing with a more conservative
  sensitivity before trying aggressive detection.
- Monitor accuracy_preservation_rate: aim to keep it close to 1.0.
  A significant drop means either poison is getting through (tighten) or you
  are losing too many useful updates by over-flagging (loosen)."""


class DefenderAgent:
    """LLM-powered defender that adapts using production-observable signals.

    No oracle feedback (TPR, FPR, attack_passed_through) is used — only
    signals available in a real FL deployment.
    """

    # Accuracy drop threshold for triggering LLM adaptation
    ACCURACY_DROP_THRESHOLD = -0.01

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

        Uses ONLY production-observable signals to decide when to adapt:
          - Accuracy dropped (proxy for "attack passed through")
          - All clients flagged (thresholds too strict)
          - First round (no history yet)
        """
        accuracy_delta = context.get("accuracy_delta")
        all_flagged = context.get("all_clients_flagged")

        # First round uses initial strategy
        if accuracy_delta is None:
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

        # Accuracy dropped → possible attack passed through, adapt
        if accuracy_delta < self.ACCURACY_DROP_THRESHOLD:
            logger.info(
                f"Defender: accuracy dropped by {accuracy_delta:.4f} "
                f"— consulting LLM for new strategy"
            )
            self.current_strategy = self._ask_llm(context)
            return self.current_strategy

        # No concerning signals → keep strategy
        logger.info(
            f"Defender: accuracy stable (delta={accuracy_delta:+.4f}) "
            f"— keeping strategy"
        )
        return self.current_strategy

    def record_outcome(
        self,
        round_num: int,
        strategy: dict,
        verdicts: list[dict],
        all_clients_flagged: bool,
        accuracy_delta: float,
        accuracy_trend: float,
        accuracy_volatility: float,
        accuracy_preservation_rate: float,
        flag_rate: float,
        rounds_skipped_recent: int,
        method_consensus: dict[int, int],
        client_flag_history: dict[int, int],
    ):
        """Store round outcome in history and vector memory.

        All signals are production-observable — no oracle feedback.
        """
        entry = {
            "round": round_num,
            "strategy": strategy,
            "verdicts": verdicts,
            "all_clients_flagged": all_clients_flagged,
            # Production-observable signals
            "accuracy_delta": round(accuracy_delta, 6),
            "accuracy_trend": round(accuracy_trend, 6),
            "accuracy_volatility": round(accuracy_volatility, 6),
            "accuracy_preservation_rate": round(accuracy_preservation_rate, 6),
            "flag_rate": round(flag_rate, 4),
            "rounds_skipped_recent": rounds_skipped_recent,
            "method_consensus": method_consensus,
            "client_flag_history": client_flag_history,
        }
        self.history.append(entry)
        logger.info(
            f"Defender memory: round {round_num} recorded "
            f"(acc_delta={accuracy_delta:+.4f}, flag_rate={flag_rate:.2f}, "
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
            # Current round statistics
            "update_features": context.get("update_features"),
            # Production-observable signals
            "accuracy_delta": context.get("accuracy_delta"),
            "accuracy_trend": context.get("accuracy_trend", 0.0),
            "accuracy_volatility": context.get("accuracy_volatility", 0.0),
            "accuracy_preservation_rate": context.get("accuracy_preservation_rate", 1.0),
            "flag_rate": context.get("flag_rate", 0.0),
            "all_clients_flagged": context.get("all_clients_flagged", False),
            "rounds_skipped_recent": context.get("rounds_skipped_recent", 0),
            "method_consensus": context.get("method_consensus", {}),
            "client_flag_history": context.get("client_flag_history", {}),
            # Memory
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