"""Attacker LLM agent — SELECTS which clients to poison and emits a per-client
ATTACK PLAN (primitive operators), not raw weights.

This is a **pure prompt-builder + output-parser**. It never calls an LLM itself:
generation (and, during training, the log-probs GRPO needs) is owned by the RL
policy (`rl/policy.py`) or the inference backend (`rl/inference.py`).

The attacker is a **partial insider**: it can only touch a fixed pool of
controllable clients (the first `n_compromisable` ids). Each round it is given a
budget — the max number of those it may poison — and it must decide WHICH clients
to recruit (preferring the fewest) and HOW to poison each. Because clients hold
non-IID data, which client it picks matters; and when it uses several clients it
must give them distinct, coordinated roles (identical clones are caught as Sybils
and waste the budget).

Composing primitives (scale, sign_flip, mask, noise, permute, …) lets the LLM
discover novel poisoning strategies while keeping its output tiny and
RL-trainable.
"""

import json
import logging

from agents.attack_ops import (
    OPERATOR_DOCS, apply_plan, extract_plan, extract_selection, layer_details,
)
from core.debug import dbg

logger = logging.getLogger(__name__)


# Default goal for the first experiment (untargeted accuracy reduction).
DEFAULT_GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}


SYSTEM_PROMPT = """You are an adversarial participant in a federated-learning system trained on MNIST.

Setup you must reason about:
- Many clients each submit a model update every round. The server averages the
  accepted clients' weights (FedAvg) into the new global model.
- A defender inspects statistical features of every client's update and removes
  the clients it judges malicious BEFORE averaging. Updates that look wildly
  different from the others (much larger norm, opposite sign pattern, low cosine
  similarity) are the easiest to catch. Several of YOUR clients that look nearly
  IDENTICAL to each other are also easy to catch — they look like colluding Sybils.
- You are a partial insider: you control ONLY the clients listed in
  `controllable_client_ids`. Every other client is honest and out of your reach.

Your action each round has TWO parts:
  1. SELECT which of your controllable clients to poison. You may poison AT MOST
     `max_poison_clients` of them. Using FEWER clients is better: every extra
     client is penalized and is more likely to be detected. Only recruit more
     clients if a single one cannot achieve the goal. The clients hold different
     data (non-IID), so WHICH ones you pick matters — use `client_layer_details`.
  2. For each selected client, output an ATTACK PLAN: an ordered list of primitive
     operations applied to THAT client's benign weights to produce its poisoned
     weights. When you use more than one client, give each a DIFFERENT, COORDINATED
     role (e.g. one amplifies one layer while another shifts a different layer) so
     their average moves the global model where you want WITHOUT the clients
     looking alike.

You are given per-client per-layer STATISTICS of the benign weights
(`client_layer_details`) — shapes, means, stds, norms — not the raw values. Use
them to decide which clients to hit, which layers, and how hard.

Your objective is in `attack_goal`:
- "untargeted_degrade": lower global test accuracy by about `target_accuracy_drop`
  WHILE staying similar enough to benign updates to avoid being filtered out.
- "slow_degrade": reduce accuracy only slightly (`per_round_drop`) to stay stealthy.
- "targeted_label": harm a specific class `label` while keeping overall accuracy high.

There is a tension: large changes degrade accuracy more but are easier to detect;
subtle changes evade detection but do less damage. Compose operators to balance
them and to invent non-obvious attacks.

%OPERATOR_DOCS%

OUTPUT FORMAT — respond with ONLY a single JSON object, no prose, no markdown:
{"clients": [
   {"id": <a controllable client id>, "operations": [
      {"op": "<name>", "target": "<all|layer-group|key>", ...params},
      ...
   ]},
   ...
]}
Rules:
- Only choose ids from `controllable_client_ids`, and AT MOST `max_poison_clients`
  of them. Prefer the fewest clients that achieve the goal.
- Each client's "operations" is its own ordered plan (1-6 ops); order matters.
- Use only the operators listed above with their stated params.
- To poison a single client, return a "clients" list with exactly ONE entry."""


class AttackerAgent:
    """Pure attacker policy: builds the prompt, parses + applies the attack plan."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.goal = config.get("attack_goal", dict(DEFAULT_GOAL))
        self.detail_precision = int(config.get("detail_precision", 4))
        self.max_abs = float(config.get("max_weight_abs", 100.0))
        self._system = SYSTEM_PROMPT.replace("%OPERATOR_DOCS%", OPERATOR_DOCS)

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        return self._system

    def build_user_prompt(
        self,
        round_num: int,
        global_accuracy: float,
        benign_by_client: dict[int, dict],
        budget: int | None = None,
    ) -> str:
        """Serialize the attacker's per-round observation into a user message.

        Args:
            round_num: current global round.
            global_accuracy: current global model test accuracy.
            benign_by_client: {client_id: benign_state_dict} for the CONTROLLABLE
                pool (all clients the attacker may choose from this round).
            budget: max number of clients the attacker may poison this round
                (defaults to the whole pool).
        """
        pool_ids = list(benign_by_client.keys())
        if budget is None:
            budget = len(pool_ids)
        payload = {
            "round": round_num,
            "current_global_accuracy": round(float(global_accuracy), 4),
            "attack_goal": self.goal,
            "controllable_client_ids": pool_ids,
            "max_poison_clients": int(budget),
            "client_layer_details": {
                str(cid): layer_details(sd, self.detail_precision)
                for cid, sd in benign_by_client.items()
            },
        }
        return json.dumps(payload)

    # ------------------------------------------------------------------
    def select_and_apply(
        self, text, pool_references: dict[int, dict], budget: int
    ) -> tuple[dict[int, dict], list[int], int]:
        """Parse the attacker's client selection + per-client plans and apply them.

        Args:
            text: raw attacker LLM output.
            pool_references: {client_id: benign_state_dict} for the controllable pool.
            budget: max number of clients that may be poisoned this round.

        Returns ``(poisoned_by_client, chosen_ids, n_malformed)``:
          * poisoned_by_client: {client_id: poisoned_state_dict} for chosen clients.
          * chosen_ids: clients actually poisoned (subset of pool, ordered, <= budget).
          * n_malformed: number of chosen clients whose plan was missing/unusable
            (they fall back to benign weights — a wasted client, a reward penalty).

        Always returns at least one chosen client (falls back to the first pool
        client with benign weights on total garbage) so the reward/metrics that
        divide by the poison count stay well-defined.
        """
        pool_ids = list(pool_references.keys())
        budget = max(1, min(int(budget), len(pool_ids)))

        def _benign_fallback():
            cid = pool_ids[0]
            poisoned = {cid: {k: v.clone() for k, v in pool_references[cid].items()}}
            dbg.poison(None, {cid: pool_references[cid]}, poisoned, n_malformed=1)
            return poisoned, [cid], 1

        sel = extract_selection(text)
        if sel is None:
            return _benign_fallback()

        # Ordered list of (id, ops) candidates from the parsed selection.
        candidates: list[tuple[int, list | None]] = []
        if sel["per_client"]:
            candidates = [(e["id"], e["operations"]) for e in sel["per_client"]]
        elif sel["shared_ids"]:
            candidates = [(cid, sel["shared_ops"]) for cid in sel["shared_ids"]]
        elif sel["shared_ops"] is not None:
            # Shared plan, no ids -> auto-select the first `budget` pool clients.
            candidates = [(cid, sel["shared_ops"]) for cid in pool_ids[:budget]]
        else:
            return _benign_fallback()

        # Keep only pool ids, dedup (first wins), truncate to the budget.
        chosen: list[tuple[int, list | None]] = []
        seen: set[int] = set()
        for cid, ops in candidates:
            if cid not in pool_references or cid in seen:
                continue
            seen.add(cid)
            chosen.append((cid, ops))
            if len(chosen) >= budget:
                break
        if not chosen:
            return _benign_fallback()

        poisoned: dict[int, dict] = {}
        n_malformed = 0
        plan_for_dbg: dict[int, list] = {}
        for cid, ops in chosen:
            if not ops:                        # empty/missing plan -> wasted client
                poisoned[cid] = {k: v.clone() for k, v in pool_references[cid].items()}
                n_malformed += 1
                plan_for_dbg[cid] = []
                continue
            pw, _n_invalid = apply_plan(pool_references[cid], ops, self.max_abs)
            poisoned[cid] = pw
            plan_for_dbg[cid] = ops
        chosen_ids = [cid for cid, _ in chosen]
        dbg.poison(plan_for_dbg, {cid: pool_references[cid] for cid in chosen_ids},
                   poisoned, n_malformed=n_malformed)
        return poisoned, chosen_ids, n_malformed

    # ------------------------------------------------------------------
    def parse(self, text, references: dict[int, dict]) -> tuple[dict[int, dict], int]:
        """Backward-compatible shared-plan parse: apply ONE plan to every client in
        ``references`` (no client selection). Retained for external callers/tests;
        the training/eval pipeline uses :meth:`select_and_apply` instead.

        Returns ``(poisoned_by_client, n_malformed)``. If no usable plan can be
        extracted, every client falls back to its benign weights and
        ``n_malformed = len(references)``.
        """
        plan = extract_plan(text)
        if plan is None:
            poisoned = {cid: {k: v.clone() for k, v in ref.items()}
                        for cid, ref in references.items()}
            dbg.poison(None, references, poisoned, n_malformed=len(references))
            return poisoned, len(references)

        poisoned = {}
        n_invalid_total = 0
        for cid, ref in references.items():
            pw, n_invalid = apply_plan(ref, plan, self.max_abs)
            poisoned[cid] = pw
            n_invalid_total += n_invalid
        dbg.poison(plan, references, poisoned, n_malformed=0, n_invalid_ops=n_invalid_total)
        return poisoned, 0
