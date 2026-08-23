"""Benchmark harness: run a panel of ATTACKS against a panel of DEFENSES.

Each round:
  1. the env (reused UNMODIFIED, purely as a generator) builds the controllable
     pool and the honest updates;
  2. the round's poisoned client set is fixed ONCE — by the trained attacker's own
     selection when it is in the panel, otherwise by the pool order — and every
     attack in the panel poisons exactly those clients;
  3. each attack turns the same honest updates into its own poisoned cohort;
  4. that cohort is fed to EVERY defense, each evolving its OWN global model per
     attack;
  5. we record per-(attack, defense) detection (vs ground truth) and
     post-aggregation test accuracy.

Two invariants make the resulting matrix readable:

* **Vary the defense, hold the attack fixed** (a row of the matrix). One attack
  produces one cohort per round and every defense sees it.
* **Vary the attack, hold the round fixed** (a column). Every attack sees the same
  honest updates, the same reference model, the same poisoned client ids and the
  same rounds — including skipping a round for the whole panel when the LLM
  attacker produces no usable action, so no attack is ever scored over a different
  set of rounds than its neighbours.
"""
import logging

from server.fed_server import FedServer

from benchmark.attacks.base import AttackContext, float_keys
# Re-exported: these are the LLM sampling helpers, which live with the LLM attack
# but are part of the harness's tested surface (tests/test_benchmark_retry.py).
from benchmark.attacks.llm_attack import (  # noqa: F401
    _RETRY_TEMPERATURE_FLOOR, _hit_token_cap, _sample_attack, _snippet,
)
from benchmark.metrics import DefenseMetrics

logger = logging.getLogger("benchmark")


def _check_prompt_fits(policy, attacker_agent, system, user, max_new_tokens, pool_size):
    """Log the attacker prompt's real token cost once, and warn if it crowds the context.

    The prompt carries per-layer update statistics for EVERY client in the controllable
    pool, so its size scales with the pool — and the benchmark can widen that pool up to
    ``fl.n_clients`` via ``--max-poison-clients``. At the top of that range the prompt
    plus ``max_new_tokens`` can approach ``rl.max_seq_len``, at which point generations
    get truncated and the JSON the attacker emits stops parsing (which would show up as
    a mysteriously ineffective attack rather than as a context error).

    ``AttackerAgent`` already compacts the observation to fit ``rl.max_context_fill``
    and records what it settled on, so this reports that decision rather than
    re-deriving it — and still measures the prompt directly when the agent has no
    budget wired up (an older config with no ``rl:`` block reaching it).
    """
    stats = getattr(attacker_agent, "last_prompt_stats", None) or {}
    budget = getattr(attacker_agent, "budget", None)
    if stats and budget is not None and budget.active:
        logger.info(
            f"{budget.describe(stats['prompt_tokens'])} — pool of {pool_size} client(s), "
            f"observation at level {stats['level']} ({stats['level_label']})"
        )
        if not stats["fits"]:
            logger.warning(
                f"Attacker prompt is over the {budget.max_fill:.0%} context-fill cap even "
                f"fully compacted. Raise rl.max_seq_len, lower rl.max_new_tokens, or "
                f"reduce --max-poison-clients (the pool of {pool_size} is what makes the "
                f"prompt big)."
            )
        return

    try:
        n_prompt = int(policy.count_prompt_tokens(system, user))
        limit = int(getattr(policy, "max_seq_len", 0)) or None
    except Exception:                       # a stub/inference generator: nothing to check
        return
    budget_msg = (f"attacker prompt = {n_prompt} tokens for a pool of {pool_size} "
                  f"client(s), + up to {max_new_tokens} generated")
    if limit is None:
        logger.info(budget_msg)
        return
    needed = n_prompt + int(max_new_tokens)
    logger.info(f"{budget_msg} = {needed}/{limit} of rl.max_seq_len")
    if needed > limit:
        logger.warning(
            f"Attacker prompt + max_new_tokens ({needed}) EXCEEDS rl.max_seq_len "
            f"({limit}). Generations will be cut short and the attack JSON will fail to "
            f"parse. Raise rl.max_seq_len, lower rl.max_new_tokens, or reduce "
            f"--max-poison-clients (the pool of {pool_size} is what makes the prompt big)."
        )
    elif needed > 0.9 * limit:
        logger.warning(
            f"Attacker prompt + max_new_tokens ({needed}) is within 10% of "
            f"rl.max_seq_len ({limit}) — a longer plan may get truncated."
        )


# ---------------------------------------------------------------------------
# Evaluation cache
# ---------------------------------------------------------------------------

class _AccuracyCache:
    """Memoises test accuracy by the exact bytes of a global model.

    The matrix multiplies the benchmark's evaluation cost by the number of attacks
    (``n_attacks x n_defenses`` test-set passes per round), and most of those
    passes are re-computations: the harness pins frozen benign replay, so a
    DETERMINISTIC attack (LIE, IPM, Min-Max, Min-Sum, Mimic, sign-flip, scaling)
    produces byte-identical updates every round, and therefore a byte-identical
    global for every defense. Keying on the model's bytes turns those repeats into
    one pass each without changing a single reported number — a hit means the model
    is bit-for-bit the one already measured.

    Cheap by construction: the FL model is 681 float32 parameters, so hashing it is
    a few microseconds against a full pass over the test set.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._by_digest: dict = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _digest(state: dict) -> str:
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        for k in sorted(state):
            v = state[k]
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def evaluate(self, server, weights: dict, test_loader) -> float:
        if not self.enabled:
            server.set_global_weights(weights)
            return server.evaluate(test_loader)
        key = self._digest(weights)
        if key in self._by_digest:
            self.hits += 1
            return self._by_digest[key]
        self.misses += 1
        server.set_global_weights(weights)
        acc = server.evaluate(test_loader)
        self._by_digest[key] = acc
        return acc


# ---------------------------------------------------------------------------
# Round assembly
# ---------------------------------------------------------------------------

def _resolve_poison_ids(attacks, ctx, global_weights, reference_acc, first_round,
                        max_new_tokens):
    """Fix the client ids EVERY attack poisons this round.

    Returns ``(ids, usable)``. When the trained attacker is in the panel the ids are
    its own committed selection — the baselines then poison exactly the clients it
    chose, which is what makes the matrix's rows comparable client-for-client.
    Without it the ids are the first ``budget`` of the controllable pool, which is
    what the shipped fixed-poison-set config produces anyway.

    ``usable=False`` means the LLM produced no action filling the quota after every
    retry; the caller drops the round for the whole panel.
    """
    llm = next((a for a in attacks.values() if a.is_llm), None)
    if llm is None:
        return [int(c) for c in ctx.pool_ids[:ctx.budget]], True
    if first_round:
        system = llm.agent.system_prompt()
        user = llm.agent.build_user_prompt(ctx.round_num, reference_acc,
                                           ctx.pool_benign, global_weights,
                                           ctx.budget)
        _check_prompt_fits(llm.policy, llm.agent, system, user, max_new_tokens,
                           len(ctx.pool_benign))
    return llm.propose(ctx, global_weights, reference_acc)


def _audit_cohort(name: str, poisoned: dict, honest: dict, expected_ids, round_num):
    """Check an attack returned what the round asked for; report degenerate cohorts.

    Two things worth knowing and easy to miss:

    * an attack that returns the wrong client ids would silently break the
      matrix's shared-poison-set invariant, so that is an error, not a warning;
    * an attack whose output is byte-identical to a client's honest update did not
      really poison that client. It still counts as poisoned in the ground truth —
      the set is shared across the panel by design and must not drift between rows
      — but it is worth surfacing, because a whole cohort of them is an attack that
      is not running. The check is on EXACT bytes, so it will not fire for an
      attack that reproduces an honest update through float arithmetic (``mimic``
      copying one of its own clients, say, which round-trips through the flat
      delta and back); that is deliberate — such a client did submit a different
      value, if only in the last bit.

    Returns the number of byte-identical clients.
    """
    import torch
    got, want = sorted(int(c) for c in poisoned), sorted(int(c) for c in expected_ids)
    if got != want:
        raise RuntimeError(
            f"attack '{name}' returned clients {got} but this round's shared poisoned "
            f"set is {want}; every attack must poison exactly the same clients")
    unchanged, nonfinite = 0, 0
    for cid, w in poisoned.items():
        ref = honest[int(cid)]
        if all(torch.equal(w[k], ref[k]) for k in ref):
            unchanged += 1
        if not all(torch.isfinite(v).all() for v in w.values() if v.is_floating_point()):
            nonfinite += 1
    if nonfinite:
        logger.warning(
            f"round {round_num}: attack '{name}' emitted non-finite weights for "
            f"{nonfinite}/{len(poisoned)} client(s); an undefended aggregate over them "
            f"is NaN and its accuracy is not meaningful.")
    return unchanged


# ---------------------------------------------------------------------------
# The round loop
# ---------------------------------------------------------------------------

def run_attack_benchmark(env, attacks, panels, test_loader, init_global,
                         baseline_accuracy, n_rounds, *, device: str = "cpu",
                         log_every: int = 10, target_drop: float | None = None,
                         knowledge: str = "partial", max_new_tokens: int = 512,
                         eval_cache: bool = True):
    """Run ``n_rounds`` of every attack vs every defense.

    Args:
        attacks: ordered ``{name: Attack}``. At most one may have ``is_llm``.
        panels: ``{attack_name: {defense_name: Defense}}`` — each attack needs its
            OWN defense instances, because a defense's global model (and any
            cross-round memory it keeps) is shaped by the attack it faced.
        knowledge: ``partial`` = the baselines see only the compromised clients'
            honest updates, matching what the trained attacker observes; ``full`` =
            they see every client's, the omniscient setting most papers state their
            attack in.
        target_drop: the goal's requested accuracy drop, enabling the per-round
            goal-success weighting.

    Returns ``(summaries, metrics, run_info)`` where ``summaries`` and ``metrics``
    are ``{attack: {defense: ...}}`` and ``run_info`` carries the round bookkeeping
    (measured/skipped counts, per-attack degenerate-cohort counts, cache stats).
    """
    if knowledge not in ("partial", "full"):
        raise ValueError(f"unknown knowledge {knowledge!r}; use 'partial' or 'full'")
    llm_names = [n for n, a in attacks.items() if a.is_llm]
    if len(llm_names) > 1:
        raise ValueError(f"at most one LLM attack per panel, got {llm_names}")
    for name, panel in panels.items():
        if "fedavg" not in panel:
            logger.warning(
                f"attack '{name}': no 'fedavg' defense in its panel — the attacker's "
                f"reference accuracy will stay frozen at the clean baseline all run.")

    for attack in attacks.values():
        attack.reset()
    for panel in panels.values():
        for d in panel.values():
            d.reset(init_global)

    eval_server = FedServer(device=device)
    cache = _AccuracyCache(eval_cache)
    metrics = {a: {d: DefenseMetrics(d, baseline_accuracy, target_drop, attack=a)
                   for d in panels[a]}
               for a in attacks}
    # Each attack observes ITS OWN undefended world (one round stale), exactly as a
    # single-attack run does — an attack must not be handed a reference accuracy
    # produced by a different attack's damage.
    reference = {a: float(baseline_accuracy) for a in attacks}
    unchanged_total = {a: 0 for a in attacks}
    n_unusable = 0
    poisoned_per_round = 0          # for the degenerate-cohort report below

    for r in range(1, n_rounds + 1):
        ctx = env.begin_round()
        # One call: ``env.global_weights`` rebuilds a CPU copy of the state_dict, and
        # every attack this round must plan against the same object.
        g = env.global_weights
        llm_reference = (reference[llm_names[0]] if llm_names else baseline_accuracy)
        ids, usable = _resolve_poison_ids(attacks, ctx, g, llm_reference,
                                          first_round=(r == 1),
                                          max_new_tokens=max_new_tokens)
        if not usable:
            # No usable LLM action after every retry. Measuring the round anyway
            # would score the panel on an attack that never happened, and measuring
            # only the BASELINES would leave them compared over more rounds than the
            # policy — so the round is dropped for everyone and counted.
            n_unusable += 1
            logger.error(
                f"round {ctx.round_num}: SKIPPED for the whole panel — no usable "
                f"attacker action after every retry. Excluded from every "
                f"(attack, defense) cell; {n_unusable} round(s) skipped so far. If "
                f"this recurs, raise rl.max_new_tokens or inspect the adapter.")
            continue

        poisoned_ids = set(ids)
        poisoned_per_round = len(ids)
        env.set_committed_poison(ids)
        honest = {int(u.client_id): u.weights for u in env.honest_updates}
        known_ids = (sorted(int(c) for c in ctx.pool_ids) if knowledge == "partial"
                     else sorted(honest))
        base_ctx = AttackContext(
            round_num=ctx.round_num, global_weights=g, honest=honest,
            poisoned_ids=list(ids), known_ids=known_ids,
            pool_ids=[int(c) for c in ctx.pool_ids], n_clients=env.n_clients,
            goal=getattr(ctx, "goal", None) or {}, keys=float_keys(g))

        for name, attack in attacks.items():
            base_ctx.reference_accuracy = reference[name]
            poisoned = attack.craft(base_ctx)
            # A control row (``clean``) submits honest updates, so its ground truth is
            # EMPTY: the oracle then flags nobody and aggregates the whole federation,
            # which is the genuine no-attack model, and any flag a real defense raises
            # on that row is a false alarm rather than a hit.
            truth = poisoned_ids if attack.poisons else set()
            if attack.poisons:
                unchanged_total[name] += _audit_cohort(name, poisoned, honest, ids,
                                                       ctx.round_num)
            updates = env.build_updates(poisoned)
            panel = panels[name]
            for dname, d in panel.items():
                res = d.step(updates, truth)
                gw = d.global_weights()
                if gw is not None:
                    # On a skip the defense kept its previous global, so this still
                    # reflects that defense's actual current model accuracy.
                    acc = cache.evaluate(eval_server, gw, test_loader)
                else:
                    acc = metrics[name][dname].last_acc   # no model yet (post-reset)
                metrics[name][dname].record(ctx.round_num, res.verdicts, truth,
                                            acc, skipped=(res.new_global is None))
            if "fedavg" in metrics[name]:
                reference[name] = metrics[name]["fedavg"].last_acc

        if r == 1 or r % log_every == 0 or r == n_rounds:
            logger.info(f"[round {r}/{n_rounds}] poisoned={sorted(poisoned_ids)}")
            for name in attacks:
                status = " | ".join(
                    f"{d}: det={metrics[name][d].summary()['detection_rate']:.0%} "
                    f"acc={metrics[name][d].last_acc:.3f}" for d in panels[name])
                logger.info(f"    {name:<10} {status}")

    measured = n_rounds - n_unusable
    if n_unusable:
        logger.warning(
            f"{n_unusable} of {n_rounds} round(s) produced no usable attacker action "
            f"and were skipped for the whole panel: every (attack, defense) cell "
            f"covers {measured} measured round(s). Rerun with a larger "
            f"rl.max_new_tokens (or --attack-retries) if that fraction is material.")
    total_client_rounds = measured * poisoned_per_round
    for name, n in unchanged_total.items():
        if n:
            logger.info(
                f"attack '{name}': {n} of {total_client_rounds} poisoned client-round(s) "
                f"submitted weights byte-identical to the honest update. They stay in "
                f"the ground truth so the poisoned set is identical across attacks.")
    if cache.enabled:
        logger.info(f"accuracy cache: {cache.hits} hit(s) / {cache.hits + cache.misses} "
                    f"lookup(s) — {cache.misses} test-set pass(es) actually run")

    summaries = {a: {d: m.summary() for d, m in panel.items()}
                 for a, panel in metrics.items()}
    run_info = {
        "requested_rounds": int(n_rounds),
        "measured_rounds": int(measured),
        "unusable_attack_rounds": int(n_unusable),
        "knowledge": knowledge,
        "unchanged_client_rounds": dict(unchanged_total),
        "eval_cache_hits": cache.hits,
        "eval_cache_misses": cache.misses,
    }
    return summaries, metrics, run_info


def run_benchmark(env, policy, attacker_agent, defenses, test_loader,
                  init_global, baseline_accuracy, n_rounds, *,
                  attack_temperature: float = 0.7, max_new_tokens: int = 512,
                  device: str = "cpu", attacker_adapter: str = "attacker",
                  log_every: int = 10, target_drop: float | None = None,
                  attack_retries: int = 3):
    """Single-attack benchmark: the trained attacker vs ``defenses``.

    The original entry point, kept because "the trained policy against the defense
    panel" is still the common case and its ``{defense: summary}`` shape is what
    callers and tests expect. It is a one-row :func:`run_attack_benchmark`.
    """
    from benchmark.attacks.llm_attack import LLMAttack

    attack = LLMAttack(policy, attacker_agent, adapter=attacker_adapter,
                       temperature=attack_temperature, max_new_tokens=max_new_tokens,
                       retries=attack_retries)
    summaries, metrics, _info = run_attack_benchmark(
        env, {"llm": attack}, {"llm": defenses}, test_loader, init_global,
        baseline_accuracy, n_rounds, device=device, log_every=log_every,
        target_drop=target_drop, max_new_tokens=max_new_tokens)
    return summaries["llm"], metrics["llm"]
