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
    output_layer_keys,
)
from attacks.stealth_blend import stealth_blend
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

IMPORTANT -- what your operators act on: every operator transforms the client's FULL weight vector W, NOT the small update D. The server then sees that client's update as D' = op(W) - global. Because W is close to the global model, a mild-looking multiplicative edit is a huge update: `scale` with factor f alone gives roughly rel_update ~ |f - 1|. So if the honest rel_update is about 0.2, then f=1.05 is a normal-sized update, f=1.5 is about 7x the honest size (very likely flagged), and f=2 or a full `sign_flip` is enormous. Additive operators are absolute: `add_constant`/`add_gaussian_noise` of size v change rel_update by about v * sqrt(count) / norm(global), so compare v against `rms_delta`, which is the honest per-weight step. Calibrate every operator against the honest `rel_update` shown for that client.

%OPERATOR_DOCS%

Respond with ONLY one JSON object -- no prose, no markdown:
{"clients":[{"id":<controllable id>,"operations":[{"op":"<name>","target":"<target>", ...params}]}]}
Rules:
- Use ids only from `controllable_client_ids`, AT MOST `max_poison_clients`; prefer the fewest.
- Each client's "operations" is its own ordered list (1-6 ops); order matters.
- Use only the operators and params listed above."""


TARGETED_SYSTEM_PROMPT = """You are the adversary in a federated-learning (FL) system. Each round, clients send model updates and the server averages the accepted ones (FedAvg) into a global model. A defender inspects statistics of every update and drops the ones it judges malicious before averaging: updates that stand out from the honest majority (much larger norm, flipped signs, low cosine similarity) get caught, and several of your clients that look nearly identical get caught as colluding Sybils.

You are a partial insider: you control ONLY the clients in `controllable_client_ids`; every other client is honest and out of reach.

YOUR OBJECTIVE IS TARGETED. `attack_goal` names ONE class, `label`. You must make the global model FAIL on that class -- its test samples should be classified as anything else -- while EVERY OTHER CLASS KEEPS WORKING AS BEFORE. An attack that wrecks the whole model scores badly even though it also breaks the target class: you are scored on (recall lost on `label`) MINUS (mean recall lost across all other classes). Surgical beats destructive. The target `label` CHANGES between rounds -- read it from `attack_goal` every time and attack that class, not a memorized one.

HOW TO HIT ONE CLASS. You are strictly forbidden from modifying the output layer, as such edits are trivial for defenders to detect. Instead, you must attack the internal representation of the target class in the HIDDEN layers. `target_neuron_indices` gives you the indices of the hidden-layer neurons that respond most strongly to your target class. Your goal is to perturb these specific neurons (e.g., using the `scale` or `add_gaussian_noise` operators restricted by the `indices` parameter) so that the features encoding the target class are corrupted, causing misclassification without touching the output layer.

DILUTION -- THE KEY NUMBER. The server AVERAGES all `federation.n_clients` clients, so if you poison k of them your edit reaches the global model at strength k/n_clients. Scaling the target neurons by `factor` f on k clients leaves the aggregated neurons at ((n-k) + k*f)/n of their honest value. `federation.row_zero_factor` gives that f for each k you might use -- start from it. A timid factor like 0.5 barely moves the average and does nothing.

STEALTH. The defender scores each layer separately. Your advantage is that perturbing a few neurons per layer keeps the whole-layer norm and sign statistics normal. Push the targeted neurons hard enough to survive dilution, no harder, and consider spreading it over a few clients (smaller f each) when one client's edit would be too loud. `client_update_stats` shows each client's honest `rel_update` -- keep yours in that neighbourhood.

`client_update_stats` gives, per controllable client, dimensionless stats of its HONEST update D = local - global (per layer and whole-model), normalized to the global model only:
- rel_update: norm(D)/norm(global) for the layer -- how large the honest change already is; your poison adds to it, and bigger stands out more.
- rms_delta: per-weight step size. energy_frac: share of the update in that layer.
- sign_flip_frac: fraction of weights whose sign differs from global.
- std_ratio, absmean_ratio: spread / typical size of D vs the global's own.
- whole-model cos_to_global: alignment with the current model.

IMPORTANT -- what your operators act on: every operator transforms the client's FULL weight vector W, NOT the small update D. The server then sees that client's update as D' = op(W) - global. Additive operators are absolute: compare their size against `rms_delta`, the honest per-weight step.

%OPERATOR_DOCS%

Each round choose an action with TWO parts:
1. SELECT which controllable clients to poison, AT MOST `max_poison_clients`. Prefer the FEWEST that can work: every extra client is penalized and easier to catch.
2. For each selected client, give an ordered ATTACK PLAN. With several clients, give each a DISTINCT, coordinated role so their average moves the target class your way without the clients looking alike.

Respond with ONLY one JSON object -- no prose, no markdown:
{"clients":[{"id":<controllable id>,"operations":[{"op":"<name>","target":"<target_layer>","indices":[<neuron_idx>], ...params}]}]}
Rules:
- Use ids only from `controllable_client_ids`, AT MOST `max_poison_clients`; prefer the fewest.
- Each client's "operations" is its own ordered list (1-6 ops); order matters.
- Use only the operators and params listed above.
- NEVER target the output layer. Only target hidden layers using `target_neuron_indices`."""


class AttackerAgent:
    """Pure attacker policy: builds the prompt, parses + applies the attack plan."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.goal = config.get("attack_goal", dict(DEFAULT_GOAL))
        self.detail_precision = int(config.get("detail_precision", 4))
        self.max_abs = float(config.get("max_weight_abs", 100.0))
        self.n_classes = int(config.get("n_classes", 10))
        # Total federated clients. Needed only by the targeted prompt, to state how
        # much FedAvg dilutes a poisoned client's edit (and hence how far the
        # attacker must overshoot). ``None`` -> the dilution block is omitted.
        n_clients = config.get("n_clients")
        self.n_clients = int(n_clients) if n_clients else None
        # The targeted goal needs a different objective, a different notion of
        # "damage", and the class<->row explanation, so it gets its own system
        # prompt. The goal TYPE is fixed for a run (only the label varies per
        # round, and that travels in the user message), so this is chosen once.
        self.targeted = self.goal.get("type") == "targeted_label"
        self.stealth_alpha = config.get("stealth_alpha", None)
        if self.stealth_alpha is not None:
            self.stealth_alpha = float(self.stealth_alpha)
            logger.info(f"Stealth blending enabled: alpha={self.stealth_alpha}")
        self._global_weights = None  # set each round via set_global_weights()
        base = TARGETED_SYSTEM_PROMPT if self.targeted else SYSTEM_PROMPT
        self._system = base.replace("%OPERATOR_DOCS%", OPERATOR_DOCS)

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
        target_neuron_indices: dict | None = None,
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
        self._global_weights = global_weights  # cached for stealth_blend
        round_goal = goal if goal is not None else self.goal
        payload = {
            "round": round_num,
            "current_global_accuracy": round(float(global_accuracy), 4),
            "attack_goal": round_goal,
            "controllable_client_ids": pool_ids,
            "max_poison_clients": int(budget),
            "client_update_stats": {
                str(cid): delta_details(sd, global_weights, self.detail_precision)
                for cid, sd in benign_by_client.items()
            },
        }
        if self.targeted:
            payload.update(self._targeted_observation(global_weights, round_goal, int(budget), target_neuron_indices))
        return json.dumps(payload)

    # ------------------------------------------------------------------
    def _targeted_observation(self, global_weights: dict, goal: dict, budget: int, target_neuron_indices: dict | None = None) -> dict:
        """Extra observation fields a TARGETED round needs.

        Two things the attacker cannot work out from the layer statistics alone:

        ``output_layer`` — which parameter is the classifier head and which row
        belongs to the class named in this round's goal. Row ``c`` of that weight
        (plus bias entry ``c``) is class ``c``'s logit, so this hands the policy
        the exact 17 numbers worth touching instead of making it guess a layer
        name. Derived from the real state_dict, so it stays correct if the model
        changes (:func:`agents.attack_ops.output_layer_keys`).

        ``federation`` — how hard FedAvg dilutes a poisoned client. Poisoning ``k``
        of ``n`` clients leaves the aggregated row at ``((n-k) + k*f)/n`` of its
        honest value after scaling by ``f``, so ``row_zero_factor[k] = 1 - n/k`` is
        the factor that zeroes it. Without this the policy has no way to calibrate
        magnitude and reliably under-shoots (a factor of -1 moves a 20-client
        average by 10%, which does nothing).
        """
        out: dict = {}
        
        label = goal.get("label")
        if label is not None and target_neuron_indices is not None and int(label) in target_neuron_indices:
            out["target_neuron_indices"] = target_neuron_indices[int(label)]
            out["target_neuron_indices"]["note"] = "These are the indices of the hidden-layer neurons that respond most strongly to your target class. ONLY target these layers using these indices."
        if self.n_clients:
            n = self.n_clients
            out["federation"] = {
                "n_clients": n,
                "note": "the server averages ALL n_clients; poisoning k of them "
                        "applies your edit at strength k/n_clients",
                "row_zero_factor": {
                    str(k): round(1.0 - n / k, 3)
                    for k in range(1, max(1, min(budget, n)) + 1)
                },
            }
        return out

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
            import logging
            log = logging.getLogger("agents.attacker_agent")
            for op in ops:
                target_layer = op.get("target")
                indices = op.get("indices")
                log.info(f"[Attacker Action] Client {cid} | Layer Targeted: {target_layer} | Neurons: {indices} | Op: {op.get('op')}")
            
            pw, n_invalid = apply_plan(pool_references[cid], ops, self.max_abs)
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

        # Stealth blend: re-blend poisoned deltas with benign deltas so the
        # update's direction stays aligned with honest updates (evades FLTrust).
        gw = global_weights or self._global_weights
        if self.stealth_alpha is not None and self.stealth_alpha < 1.0 and gw is not None:
            for cid in list(poisoned.keys()):
                poisoned[cid] = stealth_blend(
                    gw, pool_references[cid], poisoned[cid], self.stealth_alpha
                )

        poisoned_ids = [cid for cid, _ in chosen if cid in poisoned]
        dbg.poison(plan_for_dbg, {cid: pool_references[cid] for cid in poisoned_ids},
                   poisoned, n_malformed=n_malformed, n_invalid_ops=n_invalid_ops)
        return poisoned, poisoned_ids, n_malformed

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

        # Stealth blend (same logic as select_and_apply).
        gw = self._global_weights
        if self.stealth_alpha is not None and self.stealth_alpha < 1.0 and gw is not None:
            for cid in list(poisoned.keys()):
                poisoned[cid] = stealth_blend(
                    gw, references[cid], poisoned[cid], self.stealth_alpha
                )

        dbg.poison(plan, references, poisoned, n_malformed=0, n_invalid_ops=n_invalid_total)
        return poisoned, 0
