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

import torch

from agents.attack_ops import (
    OPERATOR_DOCS, apply_plan, delta_details, extract_plan, extract_selection,
)
from core.debug import dbg

logger = logging.getLogger(__name__)


# Default goal for the first experiment (untargeted accuracy reduction).
DEFAULT_GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}


SYSTEM_PROMPT = """You are the adversary in a federated-learning (FL) system. Each round, clients send model updates and the server averages the accepted ones (FedAvg) into a global model. A defender inspects statistics of every update and drops the ones it judges malicious before averaging: updates that stand out from the honest majority (much larger norm, flipped signs, low cosine similarity) get caught, and several of your clients that look nearly identical get caught as colluding Sybils.

You are a partial insider: you control ONLY the clients in `controllable_client_ids`; every other client is honest and out of reach.

Each round choose an action with TWO parts:
1. SELECT which controllable clients to poison, AT MOST `max_poison_clients`. Prefer the FEWEST that can work: every extra client is penalized and easier to catch. Clients hold different (non-IID) data, so which you pick matters.
2. For each selected client, give an ordered ATTACK PLAN: operators applied to that client's benign weights. With several clients, give each a DISTINCT, coordinated role (e.g. one amplifies a layer while another shifts a different layer) so their average moves the model your way without the clients looking alike.

Objective (`attack_goal`):
- untargeted_degrade: cut global accuracy by about `target_accuracy_drop`.
- slow_degrade: lower accuracy only slightly (`per_round_drop`) to stay stealthy.
- targeted_label: harm class `label` while keeping overall accuracy high.

`client_update_stats` gives, per controllable client, dimensionless stats of its HONEST update D = local - global (per layer and whole-model), normalized to the global model only:
- rel_update: norm(D)/norm(global) for the layer -- how large the honest change already is; your poison adds to it, and bigger stands out more.
- rms_delta: per-weight step size. energy_frac: share of the update in that layer.
- sign_flip_frac: fraction of weights whose sign differs from global.
- std_ratio, absmean_ratio: spread / typical size of D vs the global's own.
- whole-model cos_to_global: alignment with the current model.
Larger changes degrade accuracy more but are easier to flag -- balance the two, and keep each poisoned update close to an honest one so it is not filtered.

IMPORTANT -- what your operators act on. There are two families:

1. `scale_delta` acts on the client's UPDATE D = W - global, rebuilding W' = global + factor*D. This is the operator that actually works. It pushes along the direction the round is learning in, so it changes the model's predictions far more per unit of update size than anything else, and it leaves the client's weights near the global model, where the defender's distance and similarity checks expect them. Crucially it is AIMABLE: the update you submit has rel_update EXACTLY |factor| times the rel_update listed for that client. Want an update 3x honest size? factor=3. Want to reverse this client's learning? factor=-1. Want to erase it? factor=0.

2. Every other operator transforms the client's FULL weight vector W. The server sees D' = op(W) - global, and because W is close to the global model a mild-looking edit is an enormous update: `scale` with factor f gives roughly rel_update ~ |f - 1|, so if the honest rel_update is 0.2 then f=1.5 is about 7x honest size and very likely flagged. Worse, this network's predictions barely change when you scale all its weights -- so `scale` buys you maximum suspicion for almost no damage. Additive operators are absolute: `add_constant`/`add_gaussian_noise` of size v change rel_update by about v * sqrt(count) / norm(global), so compare v against `rms_delta`.

Start from `scale_delta` and calibrate its factor against the honest `rel_update` shown for that client; reach for the weight-space operators when you want to change the SHAPE of an update rather than its strength (e.g. mask or sign_flip one layer while scale_delta drives another).

%OPERATOR_DOCS%

Respond with ONLY one JSON object -- no prose, no markdown:
{"clients":[{"id":<controllable id>,"operations":[{"op":"<name>","target":"<target>", ...params}]}]}
Rules:
- Use ids only from `controllable_client_ids`, AT MOST `max_poison_clients`; prefer the fewest.
- Each client's "operations" is its own ordered list (1-6 ops); order matters.
- Use only the operators and params listed above."""


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
        global_weights: dict,
        budget: int | None = None,
        goal: dict | None = None,
    ) -> str:
        """Serialize the attacker's per-round observation into a user message.

        Args:
            round_num: current global round.
            global_accuracy: current global model test accuracy.
            benign_by_client: {client_id: benign_state_dict} for the CONTROLLABLE
                pool (all clients the attacker may choose from this round).
            global_weights: the current global model state_dict — the ONLY
                reference the (partial-insider) attacker is allowed to normalize
                against. Each client's stats describe its honest update
                ``Δ = local − global`` relative to this model (see
                ``attack_ops.delta_details``); the attacker never sees other
                clients' updates.
            budget: max number of clients the attacker may poison this round
                (defaults to the whole pool).
            goal: this round's attack goal (e.g. a per-round-sampled
                ``target_accuracy_drop``). Defaults to the agent's fixed
                ``self.goal`` when not given (inference / benchmark paths).
        """
        pool_ids = list(benign_by_client.keys())
        if budget is None:
            budget = len(pool_ids)
        payload = {
            "round": round_num,
            "current_global_accuracy": round(float(global_accuracy), 4),
            "attack_goal": goal if goal is not None else self.goal,
            "controllable_client_ids": pool_ids,
            "max_poison_clients": int(budget),
            "client_update_stats": {
                str(cid): delta_details(sd, global_weights, self.detail_precision)
                for cid, sd in benign_by_client.items()
            },
        }
        return json.dumps(payload)

    # ------------------------------------------------------------------
    def select_and_apply(
        self, text, pool_references: dict[int, dict], budget: int,
        global_weights: dict | None = None,
    ) -> tuple[dict[int, dict], list[int], int]:
        """Parse the attacker's client selection + per-client plans and apply them.

        Args:
            text: raw attacker LLM output.
            pool_references: {client_id: benign_state_dict} for the controllable pool.
            budget: max number of clients that may be poisoned this round.
            global_weights: the current global model, required by the delta-space
                operators (``scale_delta``) that rebuild ``W' = G + f*(W - G)``.
                Omitting it makes those ops count as invalid — see ``apply_plan``.

        Returns ``(poisoned_by_client, poisoned_ids, n_malformed)``:
          * poisoned_by_client: {client_id: poisoned_state_dict} — ONLY the clients
            whose weights the plan actually changed.
          * poisoned_ids: those same clients (ordered, subset of pool, <= budget).
            This is the round's **ground truth** for the defender's reward and for
            the research metrics.
          * n_malformed: how many SELECTED clients produced no change at all —
            unparseable output, an empty plan, ops that were all skipped as
            invalid, or a plan that is arithmetically a no-op (e.g.
            ``scale factor=1.0``). Each is a wasted client and a reward penalty.

        A selected client whose plan does nothing sends **byte-identical benign
        weights**, so counting it as poisoned was actively harmful: the research
        ASR metric (``fn > 0``) reported a 100% success rate for an attack that
        did nothing, and the defender's reward punished it for failing to detect
        an update that is, by construction, undetectable. Such clients are now
        excluded from the ground truth and charged to ``n_malformed`` instead.

        Consequently ``poisoned_ids`` **can be empty** (the attacker selected
        clients but achieved nothing). Every consumer handles that: the reward
        gives 0 stealth and a full malformed penalty, ``rl.switch`` treats it as
        no attack, and the metrics record a clean round. Parsing never raises.
        """
        pool_ids = list(pool_references.keys())
        budget = max(1, min(int(budget), len(pool_ids)))

        def _unchanged(cid, weights) -> bool:
            ref = pool_references[cid]
            return all(torch.equal(weights[k], ref[k]) for k in ref)

        def _nothing_happened(n_malformed: int):
            """No client was effectively poisoned this round."""
            dbg.poison({}, {}, {}, n_malformed=n_malformed)
            return {}, [], n_malformed

        sel = extract_selection(text)
        if sel is None:
            return _nothing_happened(1)

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
            return _nothing_happened(1)

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
            return _nothing_happened(1)

        poisoned: dict[int, dict] = {}
        n_malformed = 0
        n_invalid_ops = 0
        plan_for_dbg: dict[int, list] = {}
        for cid, ops in chosen:
            if not ops:                        # empty/missing plan -> wasted client
                n_malformed += 1
                plan_for_dbg[cid] = []
                continue
            pw, n_invalid = apply_plan(pool_references[cid], ops, self.max_abs,
                                       global_weights=global_weights)
            n_invalid_ops += n_invalid
            if _unchanged(cid, pw):
                # The plan parsed but changed nothing (all ops skipped as invalid,
                # or arithmetically a no-op). The server would receive this
                # client's honest weights, so it is NOT poisoned — it is wasted.
                n_malformed += 1
                plan_for_dbg[cid] = ops
                continue
            poisoned[cid] = pw
            plan_for_dbg[cid] = ops

        poisoned_ids = [cid for cid, _ in chosen if cid in poisoned]
        dbg.poison(plan_for_dbg, {cid: pool_references[cid] for cid in poisoned_ids},
                   poisoned, n_malformed=n_malformed, n_invalid_ops=n_invalid_ops)
        return poisoned, poisoned_ids, n_malformed

    # ------------------------------------------------------------------
    def parse(self, text, references: dict[int, dict],
              global_weights: dict | None = None) -> tuple[dict[int, dict], int]:
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
            pw, n_invalid = apply_plan(ref, plan, self.max_abs,
                                       global_weights=global_weights)
            poisoned[cid] = pw
            n_invalid_total += n_invalid
        dbg.poison(plan, references, poisoned, n_malformed=0, n_invalid_ops=n_invalid_total)
        return poisoned, 0
