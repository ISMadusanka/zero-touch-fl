"""Derive the TARGETED attack's label from the compromised client's own data.

Under the non-IID partition (``data.mnist_loader.partition_noniid_fltrust``) the
classes a client holds are decided by the partition RNG at runtime, not by the
config: client 0's shard is dominated by whichever class its FLTrust group was
handed, and the split depends on ``data.noniid_bias``, ``fl.poison_seed`` and
``fl.n_clients``. So the instruction "poison client 0, and aim at a class client 0
is actually trained on" **cannot be written down as a constant** — it has to be
measured after the data is partitioned and before the attack goal is frozen.

That is exactly what :func:`resolve_client_target_label` does. Given the config and
the per-client loaders it reads the chosen client's label histogram, picks the class
that client holds the most of, logs the histogram so the run's target is visible in
``logs/…/system.log``, and pins the whole run to it::

    attack.goal.label                 <- the derived class
    attack.target_labels              <- [that class]   (nothing else is trained)
    attack.sample_target_in_training  <- False          (no per-round redraw)

Pinning all three matters. ``goal.label`` is what the reward, the win-gate and the
attacker's prompt read; ``target_labels`` + ``sample_target_in_training`` are what
``rl/env.py::_round_goal`` would otherwise use to redraw a *different* label every
round, which is the right thing when training a label-agnostic policy and the wrong
thing when the whole point is one insider attacking its own class.

Switched on by ``attack.target_label_from_client`` (a client id; ``null`` = off, and
then the configured ``attack.goal.label`` / ``target_labels`` apply unchanged).
"""

import logging

from data.mnist_loader import client_label_counts

logger = logging.getLogger(__name__)

# Config key that turns this on: the client whose data picks the label.
CLIENT_KEY = "target_label_from_client"


def dominant_label(counts: list[int]) -> int:
    """The most-represented class in a label histogram.

    Ties go to the LOWEST class id, so the choice is deterministic across runs
    (Python's ``max`` would also be deterministic, but only by accident of
    iteration order — being explicit keeps it stable if the histogram type ever
    changes).
    """
    return max(range(len(counts)), key=lambda c: (counts[c], -c))


def resolve_client_target_label(config: dict, client_loaders) -> dict | None:
    """Pin this run's targeted label to a class the chosen client actually holds.

    Mutates ``config["attack"]`` **in place** (so every consumer that already holds a
    reference to the goal dict — the attacker agent, the env, the debug summary —
    sees the derived label) and returns a summary dict:

        {"client_id", "label", "counts", "n_samples", "share", "n_classes_held",
         "runner_up", "runner_up_count"}

    Returns ``None`` — changing nothing — when the feature is off
    (``attack.target_label_from_client`` absent/``null``) or when the goal is not
    ``targeted_label``. Both are normal: the untargeted experiment shares this code
    path via ``main.py``.

    Raises ``ValueError`` on a configuration that cannot mean anything: a
    non-integer client id, an id outside ``0..len(client_loaders)-1``, or a client
    whose shard is empty (no label to aim at).
    """
    attack = config.get("attack") or {}
    raw = attack.get(CLIENT_KEY)
    if raw is None:
        return None

    goal = attack.get("goal")
    if not isinstance(goal, dict) or goal.get("type") != "targeted_label":
        logger.warning(
            f"attack.{CLIENT_KEY}={raw!r} is set but attack.goal.type is "
            f"{(goal or {}).get('type')!r}, not 'targeted_label' — no label to derive, "
            f"ignoring it."
        )
        return None

    try:
        cid = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"attack.{CLIENT_KEY} must be a client id (integer) or null, got {raw!r}")

    n_available = len(client_loaders) if client_loaders is not None else 0
    if not 0 <= cid < n_available:
        raise ValueError(
            f"attack.{CLIENT_KEY}={cid} is not a client id: this run has "
            f"{n_available} client loader(s) (valid ids 0..{n_available - 1})")

    # The label is only worth deriving from a client the attacker can actually
    # poison — otherwise the run aims at a class held by an out-of-reach client.
    # A warning, not an error: the derivation itself is still well defined.
    n_compromisable = int((config.get("fl") or {}).get("n_compromisable", n_available))
    if cid >= n_compromisable:
        logger.warning(
            f"attack.{CLIENT_KEY}={cid} is OUTSIDE the attacker's controllable pool "
            f"(fl.n_compromisable={n_compromisable} -> clients 0..{n_compromisable - 1}). "
            f"The label will be read off client {cid}'s data even though the attacker "
            f"cannot poison that client."
        )

    n_classes = int((config.get("data") or {}).get("n_classes", 10))
    counts = client_label_counts(client_loaders[cid], n_classes)
    total = sum(counts)
    if total == 0:
        raise ValueError(
            f"client {cid}'s shard is empty (or holds no label in [0, {n_classes})) — "
            f"cannot derive a target label from it")

    label = dominant_label(counts)
    ranked = sorted(range(len(counts)), key=lambda c: (-counts[c], c))
    runner_up = ranked[1] if len(ranked) > 1 else None
    info = {
        "client_id": cid,
        "label": label,
        "counts": list(counts),
        "n_samples": total,
        "share": counts[label] / total,
        "n_classes_held": sum(1 for n in counts if n),
        "runner_up": runner_up,
        "runner_up_count": None if runner_up is None else counts[runner_up],
    }

    # --- Pin the run: one label, every round, for reward + prompt + win-gate. ---
    goal["label"] = label
    attack["target_labels"] = [label]
    attack["sample_target_in_training"] = False

    _log_derivation(info)
    return info


def _log_derivation(info: dict) -> None:
    """Print the derivation loudly at startup — WHICH label this run attacks, and
    the evidence for it. Without this the target of a 2M-round run is invisible
    (it is no longer a config constant), which makes the logs unreadable after
    the fact."""
    cid, label, counts = info["client_id"], info["label"], info["counts"]
    hist = "  ".join(f"{c}:{n}" for c, n in enumerate(counts) if n)
    logger.info("=" * 60)
    logger.info(f"TARGET LABEL DERIVED AT RUNTIME from client {cid}'s non-IID shard")
    logger.info(f"  client {cid} holds {info['n_samples']} samples "
                f"across {info['n_classes_held']} class(es)")
    logger.info(f"  label histogram (label:count)  {hist}")
    if info["runner_up"] is not None:
        logger.info(f"  runner-up: label {info['runner_up']} "
                    f"({info['runner_up_count']} samples)")
    logger.info(f"  >>> TARGET LABEL = {label}   "
                f"({counts[label]} samples = {info['share']:.1%} of client {cid}'s data) <<<")
    logger.info(f"  pinned: attack.goal.label={label}, target_labels=[{label}], "
                f"sample_target_in_training=False (same label every round)")
    logger.info("=" * 60)
