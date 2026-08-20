"""Command-line overrides applied to the loaded config before anything reads it.

``configs/base.yaml`` is the source of truth for a run, but two of its knobs are
the ones you actually want to vary between runs — how many clients get poisoned,
and which side of the arms race is learning. Editing the YAML for each sweep
point makes the config file the experiment log, which is exactly how a run ends
up with a stale ``n_compromisable`` next to a fresh ``fixed_poison_clients``.

So ``main.py`` exposes them as flags::

    python main.py --env linux --poisoners 8 --learn attacker

and this module turns each flag into the *set* of config keys that actually
implements it. That set is the point: "8 poisoners" is not one setting. It is
the poisoned set, the controllable pool, the training quota cap, the evaluation
quota, the curriculum's sweep — and the number DnC / Multi-Krum assume when they
decide how many updates to discard. Change one and leave the rest and the run
silently disagrees with itself: the config says the attacker controls 10 clients
while 8 are poisoned, and the defenses budget for a 10-client attack that never
arrives.

Deliberately torch-free and import-light so the resolution is unit-testable
without a GPU, a dataset or an LLM (see ``tests/test_cli_overrides.py``).
"""

import logging

logger = logging.getLogger(__name__)

#: The three things ``--learn`` can mean, in the order the CLI lists them.
LEARN_CHOICES = ("attacker", "defender", "both")


def _defense_mode(cfg: dict) -> str:
    """``defense.mode`` — read here rather than imported from
    ``server.algo_defender`` so this module stays torch-free."""
    mode = str(((cfg.get("defense") or {}).get("mode") or "algorithmic")).lower()
    if mode not in ("algorithmic", "llm"):
        raise ValueError(f"defense.mode must be algorithmic|llm, got {mode!r}")
    return mode


def _fixed_set_active(attack: dict) -> bool:
    return attack.get("fixed_poison_clients") not in (None, False, 0, "")


# ---------------------------------------------------------------------------
# --poisoners N
# ---------------------------------------------------------------------------

def apply_poisoner_count(cfg: dict, n: int) -> dict:
    """Make **exactly ``n`` clients poisoned every Phase-2 round**, in place.

    The config has two regimes and the flag means the same thing in both — only
    the keys that carry it differ:

    * **Fixed poisoned set** (``attack.fixed_poison_clients`` set, the shipped
      default): clients ``0..n-1`` are poisoned every round and the attacker LLM
      chooses only *how*. ``fl.n_compromisable`` is written to match because the
      pool IS the poisoned set here — ``FLArmsRaceEnv._resolve_fixed_poison``
      already derives that at runtime, but leaving the config saying 10 while 8
      are poisoned makes every log line and the debug summary lie.
    * **Attacker-selected set** (``fixed_poison_clients`` null): the round's
      quota is pinned to ``n`` out of the ``fl.n_compromisable`` pool, so the
      policy still chooses *which* ``n``. ``sample_budget_in_training`` is turned
      off — a per-round draw in ``[1, cap]`` is the opposite of an exact count.

    Both regimes also set ``attack.max_poison_clients``, because that is what
    ``server.algo_defender`` defaults DnC's and Multi-Krum's assumed Byzantine
    count to; and ``attack.eval_poison_clients``, so a later
    ``benchmark/run_benchmark.py`` evaluates the strength that was trained.

    Returns the ``{key: value}`` map of what changed (for logging and the
    ``--debug`` config summary). Raises ``ValueError`` on a count the federation
    cannot express.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"--poisoners must be >= 1, got {n}")

    fl = cfg.setdefault("fl", {})
    attack = cfg.setdefault("attack", {})
    n_clients = int(fl.get("n_clients", n))
    if n > n_clients:
        raise ValueError(
            f"--poisoners {n} exceeds fl.n_clients={n_clients} — the federation has "
            f"only {n_clients} client(s) to poison. Raise fl.n_clients or lower "
            f"--poisoners."
        )

    changed: dict = {}

    def _set(container: dict, key: str, value, label: str):
        if container.get(key) != value:
            changed[label] = value
        container[key] = value

    if _fixed_set_active(attack):
        _set(attack, "fixed_poison_clients", n, "attack.fixed_poison_clients")
        # The pool IS the set in this regime (see the env's _resolve_fixed_poison).
        _set(fl, "n_compromisable", n, "fl.n_compromisable")
        regime = f"FIXED poisoned set — clients 0..{n - 1} every round"
    else:
        pool = int(fl.get("n_compromisable", n_clients))
        if n > pool:
            raise ValueError(
                f"--poisoners {n} exceeds the attacker's controllable pool "
                f"fl.n_compromisable={pool}. The attacker can only poison clients it "
                f"controls: raise fl.n_compromisable, or set "
                f"attack.fixed_poison_clients to pin the set instead."
            )
        # An exact quota is incompatible with drawing one per round.
        _set(attack, "sample_budget_in_training", False,
             "attack.sample_budget_in_training")
        regime = f"exactly {n} of the {pool} controllable clients, chosen by the policy"

    _set(attack, "max_poison_clients", n, "attack.max_poison_clients")
    _set(attack, "eval_poison_clients", n, "attack.eval_poison_clients")

    # The curriculum sweeps attack strength. An exact count leaves nothing to
    # sweep, so collapse it to the single block — the same collapse
    # build_training_curriculum already performs for a fixed poisoned set, done
    # here so it also applies in the attacker-selected regime.
    ccfg = cfg.get("curriculum")
    if isinstance(ccfg, dict) and ccfg.get("enabled", True):
        if [int(k) for k in (ccfg.get("poisoner_counts") or [])] != [n]:
            changed["curriculum.poisoner_counts"] = [n]
        ccfg["poisoner_counts"] = [n]

    if n * 2 >= n_clients:
        logger.warning(
            f"--poisoners {n} of {n_clients} clients ({n / n_clients:.0%}) is at or "
            f"past half the federation, so the honest-majority assumption Multi-Krum, "
            f"DnC and FLTrust's trust scores are proved under no longer holds. The "
            f"numbers are still measurable, just outside that regime."
        )

    logger.info(f"--poisoners {n}: {regime}")
    for key, value in changed.items():
        logger.info(f"  override {key} = {value}")
    if not changed:
        logger.info(f"  (config already asked for {n} poisoner(s) — nothing to override)")
    return changed


# ---------------------------------------------------------------------------
# --learn attacker|defender|both
# ---------------------------------------------------------------------------

def apply_learner_choice(cfg: dict, choice: str) -> tuple:
    """Pin **which side trains** to ``choice``, in place. Returns the rotation.

    Written to ``rl.learners``, which ``rl.schedule.resolve_trainable`` consumes:
    only those adapters get an optimizer, get league-snapshotted, and get written
    to disk. The other side still *plays* — frozen — so ``--learn attacker``
    under ``defense.mode: llm`` is "attacker best-responds to the current
    defender", not "no defender".

    ``defender``/``both`` require the defender LLM: under ``defense.mode:
    algorithmic`` the defense is FLTrust / DeFL / DnC / Multi-Krum, which have no
    parameters to train, so there is no defender policy to put an optimizer on.
    That is an error rather than a silent downgrade to attacker-only — the run
    would otherwise report exactly what was asked for and do something else.
    """
    choice = str(choice).lower()
    if choice not in LEARN_CHOICES:
        raise ValueError(f"--learn must be one of {list(LEARN_CHOICES)}, got {choice!r}")

    mode = _defense_mode(cfg)
    if mode == "algorithmic" and choice != "attacker":
        raise ValueError(
            f"--learn {choice} needs a trainable defender, but defense.mode is "
            f"'algorithmic' — the server defends with published algorithms "
            f"(FLTrust / DeFL / DnC / Multi-Krum), which have nothing to learn. "
            f"Set defense.mode: llm to train the defender, or use --learn attacker."
        )

    learners = ("attacker", "defender") if choice == "both" else (choice,)
    rl = cfg.setdefault("rl", {})
    rl["learners"] = list(learners)
    if len(learners) == 1:
        configured_first = str(rl.get("first_learner", learners[0]))
        rl["first_learner"] = learners[0]
        if configured_first != learners[0]:
            logger.info(
                f"--learn {choice}: rl.first_learner={configured_first!r} -> "
                f"{learners[0]!r} (it is the only learner)"
            )
        frozen = "defender" if choice == "attacker" else "attacker"
        opponent = ("the algorithmic defense" if mode == "algorithmic"
                    else f"the frozen {frozen} LLM")
        logger.info(f"--learn {choice}: only the {choice} adapter trains, against "
                    f"{opponent}. The {frozen} checkpoint is left untouched.")
    else:
        logger.info(f"--learn both: two-sided arms race "
                    f"(first_learner={rl.get('first_learner', 'attacker')}).")
    return learners


# ---------------------------------------------------------------------------
# Consumed by rl/schedule.py
# ---------------------------------------------------------------------------

def resolve_trainable(rl_cfg: dict, algorithmic_defense: bool) -> tuple:
    """The rotation of agents that actually train this run.

    Without ``rl.learners`` (no ``--learn``) this is the historical rule: both
    sides under the defender LLM, attacker-only when the defense is algorithmic
    and there is therefore no defender policy. With ``rl.learners`` it is that
    list, minus any side this run cannot train — a resumed config that still
    names the defender while the defense is algorithmic is warned about and
    dropped rather than crashing the run.
    """
    default = ("attacker",) if algorithmic_defense else ("attacker", "defender")
    configured = rl_cfg.get("learners")
    if not configured:
        return default

    names, seen = [], set()
    for item in configured:
        name = str(item).strip().lower()
        if name not in ("attacker", "defender"):
            raise ValueError(f"rl.learners must be attacker|defender, got {name!r}")
        if name not in seen:
            names.append(name)
            seen.add(name)

    if algorithmic_defense and "defender" in names:
        logger.warning(
            "rl.learners names the defender, but the defender LLM is disabled "
            "(defense.mode: algorithmic) — dropping it; only the attacker trains."
        )
        names = [n for n in names if n != "defender"]
    if not names:
        raise ValueError(
            "rl.learners leaves nobody to train. Under defense.mode: algorithmic "
            "the only trainable agent is the attacker."
        )
    return tuple(names)
