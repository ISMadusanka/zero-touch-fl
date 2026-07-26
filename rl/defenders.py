"""Who answers "which clients are malicious?" during an attacker round.

An ``AttackerTurn`` needs exactly one thing from the defense: a
``DetectionVerdict`` per client for a candidate set of updates. Two things can
supply it, and they are interchangeable:

* :class:`LLMDefenderPolicy` — the frozen defender LoRA adapter (the normal
  arms-race path): build the feature prompt, generate, parse.
* :class:`AlgorithmicDefenderPolicy` — no LLM at all: an ensemble of the
  classical Byzantine-robust algorithms (FLTrust + Multi-Krum + DnC + DeFL)
  voting together, used by ``python main.py --env linux --freeze defender``.

Keeping both behind one method is what lets the attacker's GRPO round body,
reward, win-gate, logging and metrics stay byte-identical between the two modes:
only the *source* of the verdicts changes.

``verdicts(..., commit=...)`` marks the ONE call per round whose result is
actually applied to the shared model. A stateless defender (the LLM) ignores it;
the algorithmic ensemble uses it to advance its cross-round memory exactly once,
so scoring the ``G`` candidate attacks cannot corrupt the defense.
"""

import logging

from core.debug import dbg

logger = logging.getLogger(__name__)


class LLMDefenderPolicy:
    """The defender LLM: per-client feature prompt -> generation -> parsed verdicts.

    ``generator`` is anything with ``generate(system, user, n, temperature) ->
    list[str]`` — a ``rl.policy.PolicyGenerator`` over the frozen defender adapter
    during training, or an ``rl.inference.InferenceGenerator`` in ``--dry-run``.
    """

    kind = "llm"

    def __init__(self, defender_agent, generator, who: str = "opponent"):
        self.agent = defender_agent
        self.generator = generator
        self.who = who

    def describe(self) -> str:
        return "defender LLM"

    def verdicts(self, env, updates, *, temperature: float = 0.0, commit: bool = False):
        features = env.features(updates)
        client_ids = [u.client_id for u in updates]
        system = self.agent.system_prompt()
        user = self.agent.build_user_prompt(features)
        text = self.generator.generate(system, user, n=1, temperature=temperature)[0]
        verdicts = self.agent.parse(text, client_ids)
        dbg.defender_io(system, user, text, verdicts, who=self.who, temperature=temperature)
        return verdicts


class AlgorithmicDefenderPolicy:
    """A fixed, non-LLM defense: :class:`benchmark.defenses.ensemble.EnsembleDefense`.

    Deterministic — ``temperature`` is accepted and ignored, so every rollout in a
    GRPO group faces the identical defense and the within-group reward spread
    comes purely from how the candidate attacks differ. The ensemble's vote count
    is carried through as each verdict's ``confidence``, which keeps the
    attacker's stealth term continuous (see
    ``benchmark.defenses.ensemble.combine_votes``).
    """

    kind = "algorithmic"

    def __init__(self, ensemble):
        self.ensemble = ensemble
        self._initialised = False

    def describe(self) -> str:
        return self.ensemble.describe()

    def verdicts(self, env, updates, *, temperature: float = 0.0, commit: bool = False):
        global_weights = env.global_weights
        if not self._initialised:
            # Gives the members their starting global (DeFL derives its layer
            # grouping from it and clears its per-client Beta counts).
            self.ensemble.reset(global_weights)
            self._initialised = True
        verdicts, info = self.ensemble.detect(
            updates, global_weights, advance_state=commit)
        dbg.algo_defender(self.describe(), verdicts, info, commit=commit)
        return verdicts


# ---------------------------------------------------------------------------
# Construction from config
# ---------------------------------------------------------------------------

def build_algorithmic_defender(config: dict, *, device: str | None = None,
                               seed: int | None = None) -> AlgorithmicDefenderPolicy:
    """Build the non-LLM defense from the config's ``defense:`` section.

    Reads ``defense.members`` / ``defense.vote`` plus each algorithm's own knobs;
    everything is optional and falls back to the same defaults the benchmark uses.
    The assumed adversary budget Multi-Krum and DnC need (their ``f`` / ``m``) is
    NOT per-round ground truth — it defaults to ``attack.max_poison_clients``
    (what the attacker may actually use), clamped to keep an honest majority.
    """
    from benchmark.defenses import build_defenses
    from benchmark.defenses.ensemble import (
        DEFAULT_MEMBERS, FORBIDDEN_MEMBERS, EnsembleDefense,
    )
    from data.mnist_loader import get_root_loader

    fl = config["fl"]
    data_cfg = config.get("data", {}) or {}
    attack_cfg = config.get("attack", {}) or {}
    cfg = config.get("defense", {}) or {}

    device = device or fl.get("device", "cpu")
    seed = int(fl.get("poison_seed", 0)) if seed is None else int(seed)

    members = [str(m).strip() for m in cfg.get("members", DEFAULT_MEMBERS) if str(m).strip()]
    if not members:
        raise ValueError("defense.members is empty — the frozen-defender mode needs "
                         "at least one algorithmic defense")
    bad = [m for m in members if m in FORBIDDEN_MEMBERS]
    if bad:
        raise ValueError(
            f"defense.members may not contain {bad}: 'oracle' reads the ground-truth "
            f"poisoned set, 'llm_defender' is exactly what --freeze defender removes, "
            f"and 'ensemble' would recurse."
        )

    root_loader = None
    if "fltrust" in members:
        root_loader = get_root_loader(
            int(cfg.get("root_size", 100)), int(fl["batch_size"]),
            data_dir=data_cfg.get("data_dir", "./data/mnist_raw"), seed=seed,
        )

    n_clients = int(fl["n_clients"])
    poison_cap = int(attack_cfg.get("max_poison_clients", 1))
    assumed = cfg.get("num_byzantine")
    assumed = (max(1, min(poison_cap, (n_clients - 1) // 2)) if assumed is None
               else max(1, int(assumed)))

    defenses = build_defenses(
        members,
        device=device,
        root_loader=root_loader,
        root_lr=float(cfg.get("root_lr") or fl["lr"]),
        root_epochs=int(cfg.get("root_epochs", 1)),
        eta=float(cfg.get("eta", 1.0)),
        defl_delta=float(cfg.get("defl_delta", 0.05)),
        defl_tau=float(cfg.get("defl_tau", 2.5)),
        dnc_num_byzantine=assumed,
        dnc_c=float(cfg.get("dnc_c", 1.0)),
        dnc_niters=int(cfg.get("dnc_niters", 1)),
        dnc_sub_dim=int(cfg.get("dnc_sub_dim", 10000)),
        dnc_seed=seed,
        multikrum_num_byzantine=assumed,
        multikrum_m=cfg.get("multikrum_m"),
    )
    ensemble = EnsembleDefense(defenses, vote=cfg.get("vote", "majority"), device=device)
    logger.info(
        f"Algorithmic defense: {ensemble.describe()} "
        f"(assumed #malicious f={assumed} for multikrum/dnc, "
        f"root_size={cfg.get('root_size', 100)} for fltrust)"
    )
    return AlgorithmicDefenderPolicy(ensemble)
