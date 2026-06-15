"""Attacker LLM agent — composes mathematical operator pipelines for
dynamic model poisoning.

Instead of selecting from predefined attacks, the LLM decides which
mathematical operators to apply to which layers (and in what order)
to craft poisoning strategies that evade anomaly detection while
degrading the global model's accuracy.

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

Instead of choosing predefined attacks, you COMPOSE mathematical operator
pipelines — deciding which operators to apply to which layers, with what
parameters, and in what order. This gives you fine-grained control.

## Available Mathematical Operators

Each operator transforms a weight tensor. You can chain up to 5 operators
per layer. Operators are applied in order (output of one feeds into the next).

| Operator       | Parameters              | Effect                                        |
|----------------|-------------------------|-----------------------------------------------|
| scale          | factor (float)          | w' = w × factor. Amplify (>1) or shrink (<1)  |
| shift          | delta (float)           | w' = w + delta. Add constant bias              |
| rotate         | angle (float, radians)  | Givens rotation on weight pairs. PRESERVES L2  |
|                |                         | norm — very stealthy against norm detection     |
| mask           | ratio (float, 0-1)      | Zero out a fraction of weights. Structural     |
|                |                         | damage like dropout                             |
| permute        | seed (int)              | Shuffle weight indices. Destroys learned        |
|                |                         | features but preserves distribution             |
| inject_noise   | sigma (float)           | w' = w + N(0, σ²). Add Gaussian noise           |
| invert         | factor (float)          | w' = -w × factor. Generalized sign-flip         |
| align          | alpha (float)           | w' = (1-α)·w + α·global_w. Pull toward/away    |
|                |                         | from global model. α>1 overshoots, α<0 pushes  |
|                |                         | away                                            |
| clip           | threshold (float)       | w' = clamp(w, -t, t). Constrain magnitudes      |
| smooth         | beta (float, 0-1)       | w' = (1-β)·w + β·mean(w). Reduce variance       |

## Model Architecture

The target model has these layers (parameter tensors):
{layer_info}

## Strategic Guidance

- **Stealthy attacks**: Use `rotate` — it changes weight direction while
  preserving L2 norm, making it invisible to norm-based detection.
- **Targeted damage**: Apply `invert` or `scale` to specific layers
  (e.g., the final classification layer) for focused impact.
- **Evasion combos**: Chain `rotate` + `smooth` to change direction then
  normalize the distribution — evades both norm and distribution checks.
- **Aggressive attacks**: `invert` + `scale` with large factor devastates
  accuracy but is easy to detect.
- **Subtle attacks**: Small `shift` + slight `rotate` causes gradual drift
  that accumulates over rounds.
- **Anti-clustering**: `permute` breaks similarity patterns that clustering
  defenses (FLAME) look for.
- **Noise camouflage**: `inject_noise` with small sigma makes the update
  look like noisy honest training.
- **Composition power**: Chain operators! e.g. `scale(0.5)` → `rotate(0.2)`
  → `inject_noise(0.01)` = shrink, rotate direction, then add slight noise
  to camouflage the rotation.
- You can target specific layers OR use "*" to apply to all layers.

## What You Receive

- baseline_accuracy: clean model accuracy before any attacks
- current_accuracy: accuracy after last round's aggregation
- was_detected: whether your last attack was caught
- attack_success_rate_recent: success rate over last 5 rounds (0-1)
- fpr_recent: defender's false positive rate over last 5 rounds
- accuracy_preservation_rate: current/baseline accuracy (lower = more damage)
- layer_info: model layer names and shapes for targeting
- history: summary of past rounds with operator pipelines used
- similar_past_experiences: relevant past episodes from memory

## Response Format

Respond with ONLY a JSON object:
{{
    "operations": [
        {{
            "layer": "<layer name or * for all>",
            "ops": [
                {{"op": "<operator name>", "params": {{<operator params>}}}},
                ...
            ]
        }},
        ...
    ],
    "reasoning": "<brief explanation of your strategy>"
}}

Example — subtle multi-layer attack:
{{
    "operations": [
        {{
            "layer": "net.2.weight",
            "ops": [
                {{"op": "rotate", "params": {{"angle": 0.15}}}},
                {{"op": "inject_noise", "params": {{"sigma": 0.01}}}}
            ]
        }},
        {{
            "layer": "net.4.weight",
            "ops": [
                {{"op": "scale", "params": {{"factor": 1.5}}}},
                {{"op": "shift", "params": {{"delta": -0.05}}}}
            ]
        }}
    ],
    "reasoning": "Rotating hidden layer weights to shift decision boundary while scaling output layer. Small noise camouflages the rotation."
}}

Be strategic. If you were detected, try stealthier operators (rotate, smooth,
small shift). If your attack was too subtle (accuracy didn't drop), be more
aggressive (scale, invert). Use the layer info to make informed targeting
decisions — attacking the final layer often has more impact on accuracy."""


class AttackerAgent:
    """LLM-powered attacker that composes mathematical operator pipelines.

    Instead of selecting predefined attacks, the LLM chooses which
    mathematical operators to apply to which layers, enabling creative
    and adaptive poisoning strategies.
    """

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
            logger.info("Attacker: first round — consulting LLM for initial operator pipeline")
            self.current_strategy = self._ask_llm(context)
            return self.current_strategy

        # Attack succeeded → keep strategy
        if not was_detected:
            logger.info("Attacker: last attack passed through — keeping operator pipeline")
            return self.current_strategy

        # Attack was caught → adapt
        logger.info("Attacker: last attack was CAUGHT — consulting LLM for new operator pipeline")
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

        attack_metadata contains the pipeline execution details — which
        operators were applied to which layers, norm changes, etc.

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
            layers = attack_metadata.get("layers_affected", [])
            logger.info(
                f"Attacker memory: storing pipeline metadata for round {round_num} "
                f"(layers_affected={layers}, n_specs={attack_metadata.get('n_specs', 0)})"
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
        """Query the LLM for a new operator pipeline strategy."""
        # Retrieve similar past experiences
        if self.history:
            query_vec = self._make_vector(context)
            similar = self.memory.search(query_vec, k=3)
        else:
            similar = []

        # Build system prompt with layer info
        layer_info = context.get("layer_info", {})
        layer_info_str = "\n".join(
            f"  - {name}: shape={shape}"
            for name, shape in layer_info.items()
        ) if layer_info else "  (layer info not available)"

        system = SYSTEM_PROMPT.format(layer_info=layer_info_str)

        user_msg = json.dumps({
            "baseline_accuracy": context.get("baseline_accuracy"),
            "current_accuracy": context.get("current_accuracy"),
            "was_detected": context.get("was_detected"),
            "attack_success_rate_recent": context.get("attack_success_rate_recent", 0.0),
            "fpr_recent": context.get("fpr_recent", 0.0),
            "accuracy_preservation_rate": context.get("accuracy_preservation_rate", 1.0),
            "layer_info": {k: str(v) for k, v in layer_info.items()} if layer_info else {},
            "recent_history": self.history[-5:],
            "similar_past_experiences": similar,
        }, default=str)

        result = self.llm.call(system, user_msg)

        # Validate and transform the response
        if not result or "operations" not in result:
            logger.warning("Attacker LLM returned invalid response — using fallback pipeline")
            return self._fallback_strategy(layer_info)

        # Validate each operation in the pipeline
        operations = result.get("operations", [])
        valid_ops = []
        for spec in operations:
            if isinstance(spec, dict) and "layer" in spec and "ops" in spec:
                valid_ops.append(spec)

        if not valid_ops:
            logger.warning("Attacker LLM returned no valid operations — using fallback")
            return self._fallback_strategy(layer_info)

        strategy = {
            "attack_type": "math_ops",
            "params": {"operations": valid_ops},
            "reasoning": result.get("reasoning", ""),
        }

        # Log the pipeline
        for spec in valid_ops:
            op_names = [o.get("op", "?") for o in spec.get("ops", [])]
            logger.info(
                f"Attacker chose: layer='{spec['layer']}' → "
                f"{' → '.join(op_names)}"
            )
        logger.info(f"Attacker reasoning: {result.get('reasoning', '')}")

        return strategy

    def _fallback_strategy(self, layer_info: dict) -> dict:
        """Default operator pipeline when LLM fails to produce valid output."""
        # Pick the last weight layer (typically most impactful)
        if layer_info:
            # Prefer weight layers over bias layers
            weight_layers = [k for k in layer_info if "weight" in k]
            target = weight_layers[-1] if weight_layers else list(layer_info.keys())[-1]
        else:
            target = "*"

        return {
            "attack_type": "math_ops",
            "params": {
                "operations": [
                    {
                        "layer": target,
                        "ops": [
                            {"op": "invert", "params": {"factor": 1.0}},
                        ],
                    }
                ]
            },
            "reasoning": "fallback: simple sign-flip on last weight layer",
        }

    def _make_vector(self, data: dict) -> np.ndarray:
        """Create a semantic embedding vector for FAISS indexing.

        Uses SentenceTransformers so that similar contexts (e.g. close
        accuracies, same detection outcomes) map to nearby vectors.
        """
        return embed(data)
