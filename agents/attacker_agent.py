"""Attacker LLM agent — SELECTS which clients to poison and emits a per-client
ATTACK PLAN (primitive operators), not raw weights.

This is a **pure prompt-builder + output-parser**. It never calls an LLM itself:
generation (and, during training, the log-probs GRPO needs) is owned by the RL
policy (`rl/policy.py`) or the inference backend (`rl/inference.py`).

The attacker is a **partial insider**: it can only touch a fixed pool of
controllable clients (the first `n_compromisable` ids). It runs in one of two
selection modes:

* **selection** (default) — each round it is given an exact poison-client budget
  and must decide WHICH clients fill that quota and HOW to poison each. Because
  clients hold non-IID data, which client it picks matters.
* **fixed set** (``fixed_poison_set``, driven by ``attack.fixed_poison_clients``)
  — the poisoned clients are pinned to the whole controllable pool, i.e. the
  FIRST N client ids, every round. There is no client-selection decision left;
  the policy's entire action is the per-client plan. Use this when the
  experiment wants the poisoned SET held constant so that round-to-round
  differences come only from the attack itself.

Either way, when several clients are poisoned they should be given distinct,
coordinated roles (identical clones are caught as Sybils).

Composing primitives (scale, sign_flip, mask, noise, permute, …) lets the LLM
discover novel poisoning strategies while keeping its output tiny and
RL-trainable.

CONTEXT BUDGET — the prompt carries per-layer statistics for EVERY client in the
pool, so it grows with the pool. Those statistics are therefore encoded
positionally (see ``attack_ops.delta_rows``) and the whole prompt is held under
``rl.max_context_fill`` of the model's context window (see
``agents.prompt_budget``), compacting the payload rather than letting it crowd
the window — a small instruct model's output quality falls off well before the
context is actually full.
"""

import json
import logging

import torch

from agents.attack_ops import (
    LAYER_STAT_KEYS, OPERATOR_DOCS, WHOLE_STAT_KEYS, apply_plan, delta_stats,
    extract_plan, extract_selection, format_rows, layer_shapes,
)
from agents.prompt_budget import build_prompt_budget
from core.debug import dbg

logger = logging.getLogger(__name__)


# Default goal for the first experiment (untargeted accuracy reduction).
DEFAULT_GOAL = {"type": "untargeted_degrade", "target_accuracy_drop": 0.20}


# --- System prompt ---------------------------------------------------------
# Assembled from shared blocks plus the two pieces that differ between the
# selection and fixed-set modes, so the framing, the operator reference and the
# calibration guidance cannot drift apart between them.

_INTRO = """You are the adversary in a federated-learning (FL) system. Each round, clients send model updates and the server averages the accepted ones (FedAvg) into a global model. A defender inspects statistics of every update and drops the ones it judges malicious before averaging: updates that stand out from the honest majority (much larger norm, flipped signs, low cosine similarity) get caught, and several of your clients that look nearly identical get caught as colluding Sybils."""

_ROLE_SELECT = """You are a partial insider: you control ONLY the clients in `controllable_client_ids`; every other client is honest and out of reach.

Each round choose an action with TWO parts:
1. SELECT EXACTLY `max_poison_clients` distinct controllable clients to poison. This field is an exact quota, not an upper bound. Clients hold different (non-IID) data, so which you pick matters.
2. For each selected client, give an ordered ATTACK PLAN: operators applied to that client's benign weights. With several clients, give each a DISTINCT, coordinated role (e.g. one amplifies a layer while another shifts a different layer) so their average moves the model your way without the clients looking alike."""

_ROLE_FIXED = """You are a partial insider: you control ONLY the clients in `poison_client_ids`; every other client is honest and out of reach. That set is FIXED -- you do not choose it, and you must poison EVERY id in it.

Your whole action is one ordered ATTACK PLAN per listed client: operators applied to that client's benign weights. Clients hold different (non-IID) data, so give each client a DISTINCT, coordinated role (e.g. one amplifies a layer while another shifts a different layer) so their average moves the model your way without the clients looking alike."""

_OBJECTIVE = """Objective (`attack_goal`):
- untargeted_degrade: cut global accuracy by about `target_accuracy_drop`.
- slow_degrade: lower accuracy only slightly (`per_round_drop`) to stay stealthy.
- targeted_label: harm class `label` while keeping overall accuracy high."""

_STATS = """`client_update_stats` gives, per controllable client, dimensionless stats of its HONEST update D = local - global (per layer and whole-model), normalized to the global model only. It is packed positionally to stay short: each layer maps to an ARRAY of numbers in the order named by `stats_key`, `"whole"` maps to an array in the order named by `whole_key`, and `layers` lists each parameter key with its shape once (shapes are the same for every client).
- rel_update: norm(D)/norm(global) for the layer -- how large the honest change already is; your poison adds to it, and bigger stands out more.
- rms_delta: per-weight step size. energy_frac: share of the update in that layer.
- sign_flip_frac: fraction of weights whose sign differs from global.
- std_ratio, absmean_ratio: spread / typical size of D vs the global's own.
- whole-model cos_to_global: alignment with the current model.
Larger changes degrade accuracy more but are easier to flag -- balance the two, and keep each poisoned update close to an honest one so it is not filtered."""

_OPERATOR_SEMANTICS = """IMPORTANT -- what your operators act on: every operator transforms the client's FULL weight vector W, NOT the small update D. The server then sees that client's update as D' = op(W) - global. Because W is close to the global model, a mild-looking multiplicative edit is a huge update: `scale` with factor f alone gives roughly rel_update ~ |f - 1|. So if the honest rel_update is about 0.2, then f=1.05 is a normal-sized update, f=1.5 is about 7x the honest size (very likely flagged), and f=2 or a full `sign_flip` is enormous. Additive operators are absolute: `add_constant`/`add_gaussian_noise` of size v change rel_update by about v * sqrt(count) / norm(global), so compare v against `rms_delta`, which is the honest per-weight step. Calibrate every operator against the honest `rel_update` shown for that client."""

_FORMAT_HEAD = """Respond with ONLY one JSON object -- no prose, no markdown:
{"clients":[{"id":<client id>,"operations":[{"op":"<name>","target":"<target>", ...params}]}]}
Rules:"""

_RULE_SELECT = "- Return EXACTLY `max_poison_clients` distinct ids, all from `controllable_client_ids`; never return fewer."
_RULE_FIXED = "- Return ONE entry for EVERY id in `poison_client_ids` and for no other id; never return fewer."

_RULES_TAIL = """- Each client's "operations" is its own ordered list (1-6 ops); order matters.
- Every plan must actually change that client's weights; empty or identity plans are invalid.
- Use only the operators and params listed above."""


def _build_system_prompt(fixed_poison_set: bool) -> str:
    return "\n\n".join([
        _INTRO,
        _ROLE_FIXED if fixed_poison_set else _ROLE_SELECT,
        _OBJECTIVE,
        _STATS,
        _OPERATOR_SEMANTICS,
        OPERATOR_DOCS,
        _FORMAT_HEAD + "\n" + (_RULE_FIXED if fixed_poison_set else _RULE_SELECT)
        + "\n" + _RULES_TAIL,
    ])


#: Backwards-compatible export: the selection-mode prompt with the operator
#: reference already substituted (older callers imported this constant).
SYSTEM_PROMPT = _build_system_prompt(fixed_poison_set=False)


#: The observation's compaction ladder, tried in order until the prompt fits the
#: context budget. Level 0 is loss-free relative to the old verbose encoding (4
#: significant figures covers the same resolution as the old 4 decimal places on
#: statistics that are all < 1); every later level trades observation detail for
#: room, least-informative first. Dropping a CLIENT is deliberately not an
#: option — that would silently change the attack the policy is being trained on.
_COMPACTION = (
    dict(sig=None, layer_keys=LAYER_STAT_KEYS, include_layers=True,
         label="full per-layer stats"),
    dict(sig=3, layer_keys=LAYER_STAT_KEYS, include_layers=True,
         label="3 significant figures"),
    dict(sig=3, layer_keys=("rel_update", "rms_delta", "energy_frac", "sign_flip_frac"),
         include_layers=True, label="core per-layer stats"),
    dict(sig=3, layer_keys=(), include_layers=False,
         label="whole-model stats only"),
)


class AttackerAgent:
    """Pure attacker policy: builds the prompt, parses + applies the attack plan."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.goal = config.get("attack_goal", dict(DEFAULT_GOAL))
        self.detail_precision = int(config.get("detail_precision", 4))
        self.max_abs = float(config.get("max_weight_abs", 100.0))
        # Fixed-set mode: the poisoned clients ARE the controllable pool (the
        # first N ids), so the policy has no selection decision left. Set from
        # ``attack.fixed_poison_clients`` — see rl.env.FLArmsRaceEnv.
        self.fixed_poison_set = bool(config.get("fixed_poison_set", False))
        self._system = _build_system_prompt(self.fixed_poison_set)
        # How much of the LLM's context one call may fill (prompt + reserved
        # completion). Built from the ``rl:`` block main.py / run_benchmark.py
        # injects; absent it, the budget is inert and only reports.
        self.budget = build_prompt_budget(config.get("rl"), label="attacker prompt")
        #: Fill report for the most recent ``build_user_prompt`` (for logging/tests).
        self.last_prompt_stats: dict = {}
        self._logged_level: int | None = None

    # ------------------------------------------------------------------
    def bind_tokenizer(self, token_counter) -> None:
        """Use the real tokenizer for the context-fill check.

        ``token_counter(system, user) -> int`` — pass
        ``LLMPolicy.count_prompt_tokens`` so the count reflects the same
        chat-template rendering generation uses. Without this the budget falls
        back to a character heuristic (see :mod:`agents.prompt_budget`).
        """
        self.budget.bind(token_counter)

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
                pool. In selection mode these are the clients the attacker may
                choose from; in fixed-set mode they ARE the clients it poisons.
            global_weights: the current global model state_dict — the ONLY
                reference the (partial-insider) attacker is allowed to normalize
                against. Each client's stats describe its honest update
                ``Δ = local − global`` relative to this model (see
                ``attack_ops.delta_stats``); the attacker never sees other
                clients' updates.
            budget: exact number of clients the attacker must poison this round
                (defaults to the whole pool and is clamped to the pool size). In
                fixed-set mode this is the pool size by construction.
            goal: this round's attack goal (e.g. a per-round-sampled
                ``target_accuracy_drop``). Defaults to the agent's fixed
                ``self.goal`` when not given (inference / benchmark paths).

        The observation is emitted at the first compaction level (see
        ``_COMPACTION``) whose prompt fits ``self.budget``; the payload is made
        coarser rather than truncated, and no client is ever dropped from it.
        """
        pool_ids = list(benign_by_client.keys())
        if budget is None:
            budget = len(pool_ids)
        budget = max(0, min(int(budget), len(pool_ids)))
        if self.fixed_poison_set and budget != len(pool_ids):
            # In fixed-set mode the pool IS the poison set, so a smaller quota is
            # a wiring bug (env.pool_ids and env.round_budget disagreeing). Say
            # so loudly and describe the pool, which is what actually gets
            # poisoned — a prompt that claims otherwise just teaches the policy
            # a client count it never gets.
            logger.warning(
                f"attacker prompt: fixed poison set of {len(pool_ids)} client(s) but a "
                f"budget of {budget} was passed — using the pool. Check "
                f"attack.fixed_poison_clients vs fl.n_compromisable."
            )
            budget = len(pool_ids)

        # The per-client statistics are computed ONCE; the compaction levels are
        # only different renderings of the same numbers.
        stats = {cid: delta_stats(sd, global_weights)
                 for cid, sd in benign_by_client.items()}
        shapes = layer_shapes(global_weights)

        text = ""
        level = 0
        n_tokens = 0
        for level, step in enumerate(_COMPACTION):
            text = self._render(round_num, global_accuracy, pool_ids, budget,
                                goal, stats, shapes, step)
            n_tokens = self.budget.count(self._system, text)
            if self.budget.fits(n_tokens) or level == len(_COMPACTION) - 1:
                break

        self._report(level, n_tokens, len(pool_ids))
        return text

    # ------------------------------------------------------------------
    def _render(self, round_num, global_accuracy, pool_ids, budget, goal,
                stats, shapes, step) -> str:
        """One compaction level's user message."""
        sig = self.detail_precision if step["sig"] is None else step["sig"]
        layer_keys = tuple(step["layer_keys"])
        include_layers = bool(step["include_layers"]) and bool(layer_keys)

        payload = {
            "round": round_num,
            "current_global_accuracy": round(float(global_accuracy), 4),
            "attack_goal": goal if goal is not None else self.goal,
        }
        if self.fixed_poison_set:
            payload["poison_client_ids"] = pool_ids
            payload["n_poison_clients"] = int(budget)
        else:
            payload["controllable_client_ids"] = pool_ids
            payload["max_poison_clients"] = int(budget)
        if include_layers:
            payload["layers"] = shapes
            payload["stats_key"] = list(layer_keys)
        payload["whole_key"] = list(WHOLE_STAT_KEYS)
        payload["client_update_stats"] = {
            str(cid): format_rows(s, sig=sig, layer_keys=layer_keys,
                                  whole_keys=WHOLE_STAT_KEYS,
                                  include_layers=include_layers)
            for cid, s in stats.items()
        }
        # Compact separators: at a 10-client pool the default ", " / ": " padding
        # is a few hundred wasted tokens of pure whitespace.
        return json.dumps(payload, separators=(",", ":"))

    def _report(self, level: int, n_tokens: int, pool_size: int) -> None:
        """Record the fill, and log it when the compaction level changes."""
        step = _COMPACTION[level]
        fits = self.budget.fits(n_tokens)
        self.last_prompt_stats = {
            "prompt_tokens": int(n_tokens),
            "fill": round(self.budget.fill(n_tokens), 4),
            "level": level,
            "level_label": step["label"],
            "pool_size": int(pool_size),
            "fits": bool(fits),
            "exact": self.budget.exact(),
        }
        if level == self._logged_level:
            logger.debug(self.budget.describe(n_tokens))
            return
        self._logged_level = level
        detail = (f"{self.budget.describe(n_tokens)} — pool of {pool_size} client(s), "
                  f"observation at level {level} ({step['label']})")
        if fits:
            logger.info(detail)
        else:
            # Out of compaction levels: the prompt is over the cap but complete.
            # Nothing is truncated, so the run continues — but the whole point of
            # the cap is that output quality degrades before this point.
            logger.warning(
                f"{detail}. Every compaction level still exceeds the "
                f"{self.budget.max_fill:.0%} context-fill cap "
                f"({self.budget.prompt_limit} prompt tokens). Raise rl.max_seq_len, "
                f"lower rl.max_new_tokens, or shrink the controllable pool "
                f"(fl.n_compromisable / attack.fixed_poison_clients)."
            )

    # ------------------------------------------------------------------
    def select_and_apply(
        self, text, pool_references: dict[int, dict], budget: int
    ) -> tuple[dict[int, dict], list[int], int]:
        """Parse the attacker's client selection + per-client plans and apply them.

        Args:
            text: raw attacker LLM output.
            pool_references: {client_id: benign_state_dict} for the controllable pool.
            budget: exact number of clients that must be poisoned this round
                (clamped to the controllable pool size).

        Returns ``(poisoned_by_client, poisoned_ids, n_malformed)``:
          * poisoned_by_client: {client_id: poisoned_state_dict} — ONLY the clients
            whose weights the plan actually changed.
          * poisoned_ids: those same clients (ordered, subset of pool). For every
            usable action its length is exactly ``budget``. This is the round's
            **ground truth** for the defender's reward and research metrics.
          * n_malformed: how many quota slots could not produce a change at all —
            because the whole output was unparseable, every available plan was
            empty/invalid, or every plan was an arithmetic no-op (e.g.
            ``scale factor=1.0``). Each is a reward penalty.

        A selected client whose plan does nothing sends **byte-identical benign
        weights**, so counting it as poisoned was actively harmful: the research
        ASR metric (``fn > 0``) reported a 100% success rate for an attack that
        did nothing, and the defender's reward punished it for failing to detect
        an update that is, by construction, undetectable. Such clients are now
        excluded from the ground truth and charged to ``n_malformed`` instead.

        The parser projects a usable under-filled action onto the exact-size
        action space: it fills missing ids from the controllable pool and reuses
        an emitted plan for those ids. This makes the configured budget a quota
        in GRPO and benchmark paths instead of leaving client count as a policy
        choice. Distinct plans are still strongly requested because repeated
        plans are easier for the defense to catch.

        ``poisoned_ids`` can still be shorter than the quota only for a wholly
        unusable output from which no weight-changing operation can be recovered.
        Such slots remain malformed rather than being falsely labelled as poison;
        this preserves correct ground truth for rewards and benchmark metrics.
        Parsing never raises.

        FIXED-SET MODE (``attack.fixed_poison_clients``) needs no special case
        here: the env makes the quota equal the pool size, so the id-filling
        above visits every pool client and the poisoned set is exactly the first
        N clients whatever ids the model emitted. Only the plan-to-client
        assignment is still the policy's. The one way a fixed-set round poisons
        fewer than N is a plan that leaves a client's weights byte-identical,
        which is charged to ``n_malformed`` rather than mislabelled as poison —
        see above.
        """
        pool_ids = list(pool_references.keys())
        if not pool_ids:
            dbg.poison({}, {}, {}, n_malformed=0)
            return {}, [], 0
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
            return _nothing_happened(budget)

        # Ordered list of (id, ops) candidates from the parsed selection.
        candidates: list[tuple[int, list | None]] = []
        if sel["per_client"]:
            candidates.extend((e["id"], e["operations"]) for e in sel["per_client"])
        if sel["shared_ids"]:
            candidates.extend((cid, sel["shared_ops"]) for cid in sel["shared_ids"])
        if not candidates and sel["shared_ops"] is not None:
            # Shared plan, no ids -> auto-select the first `budget` pool clients.
            candidates = [(cid, sel["shared_ops"]) for cid in pool_ids[:budget]]

        # Every emitted operation list is a possible repair plan. If the model
        # supplies fewer than `budget` ids (or an invalid/duplicate id), the exact-k
        # action is completed with unused pool ids and one of these plans. This is
        # action-space normalization, not a client-count decision by the policy.
        plan_options: list[list] = []
        for _cid, ops in candidates:
            if ops and ops not in plan_options:
                plan_options.append(ops)
        shared_ops = sel["shared_ops"]
        if shared_ops and shared_ops not in plan_options:
            plan_options.append(shared_ops)

        # Keep only pool ids, dedup (first wins), truncate excess choices, then
        # fill any missing quota slots deterministically from the remaining pool.
        chosen: list[tuple[int, list | None]] = []
        seen: set[int] = set()
        for cid, ops in candidates:
            if cid not in pool_references or cid in seen:
                continue
            seen.add(cid)
            chosen.append((cid, ops))
            if len(chosen) >= budget:
                break
        fallback_ops = plan_options[0] if plan_options else None
        for cid in pool_ids:
            if len(chosen) >= budget:
                break
            if cid not in seen:
                seen.add(cid)
                chosen.append((cid, fallback_ops))

        poisoned: dict[int, dict] = {}
        n_malformed = 0
        n_invalid_ops = 0
        plan_for_dbg: dict[int, list] = {}
        for cid, ops in chosen:
            # Prefer the plan written for this client, then try the other emitted
            # plans as repairs. Architectures are shared across FL clients, so a
            # valid plan is normally applicable to every filled quota slot.
            attempts: list[list] = []
            if ops:
                attempts.append(ops)
            for candidate_ops in plan_options:
                if candidate_ops not in attempts:
                    attempts.append(candidate_ops)

            applied = None
            for candidate_ops in attempts:
                pw, n_invalid = apply_plan(
                    pool_references[cid], candidate_ops, self.max_abs)
                n_invalid_ops += n_invalid
                if not _unchanged(cid, pw):
                    applied = (pw, candidate_ops)
                    break

            if applied is None:
                # Never call byte-identical benign weights poison just to satisfy
                # the quota: that corrupts ASR and defender ground truth. This is
                # an unusable quota slot and receives the malformed penalty.
                n_malformed += 1
                plan_for_dbg[cid] = ops or []
                continue
            poisoned[cid], plan_for_dbg[cid] = applied

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
        dbg.poison(plan, references, poisoned, n_malformed=0, n_invalid_ops=n_invalid_total)
        return poisoned, 0
