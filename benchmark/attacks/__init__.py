"""Attack registry for the benchmark.

The benchmark's second axis. ``benchmark/defenses`` varies the server; this varies
the adversary, so the report is an attack x defense matrix in which the trained
LLM attacker is one row among the published state-of-the-art untargeted attacks
and every row faces the same rounds, the same honest updates and the SAME poisoned
clients.

Attack classes import torch / FL components, so they are imported LAZILY inside
``build_attacks`` — importing this package stays cheap and torch-free.

Registry
--------
``llm``        the trained attacker adapter (this work) — the system under test
``lie``        LIE / ALIE, Baruch et al., NeurIPS 2019
``min_max``    AGR-agnostic Min-Max, Shejwalkar & Houmansadr, NDSS 2021
``min_sum``    AGR-agnostic Min-Sum, Shejwalkar & Houmansadr, NDSS 2021
``fang``       AGR-tailored (trimmed-mean/median), Fang et al., USENIX Sec 2020
``fang_krum``  AGR-tailored (Krum), Fang et al., USENIX Sec 2020
``ipm``        Inner Product Manipulation, Xie et al., UAI 2019
``mimic``      Mimic, Karimireddy et al., ICLR 2022
``sign_flip``  classic sign-flipping Byzantine baseline
``noise``      classic Gaussian Byzantine baseline (Blanchard et al., NeurIPS 2017)
``scaling``    boosting / model replacement (Bagdasaryan et al., AISTATS 2020)
``label_flip`` untargeted DATA poisoning (opt-in; retrains the compromised clients)
``clean``      the no-attack CONTROL row (opt-in): nothing is poisoned, so each
               defense's row is its clean accuracy — the per-defense denominator for
               reading every other row's acc_drop
"""

#: Every registered attack name, in report order.
AVAILABLE = ["clean", "llm", "lie", "min_max", "min_sum", "fang", "fang_krum", "ipm",
             "mimic", "sign_flip", "noise", "scaling", "label_flip"]

#: The panel a plain ``--rounds N`` run compares. Left out: ``clean`` (a control, not
#: an attack), ``scaling`` (a near-duplicate of ``sign_flip`` in behaviour), and
#: ``label_flip`` (a different threat model — data poisoning — that also forces
#: per-round benign retraining). All three are opt-in via ``--attacks``.
DEFAULT = ["llm", "lie", "min_max", "min_sum", "fang", "fang_krum", "ipm",
           "mimic", "sign_flip", "noise"]

#: Attacks that never touch the trained policy — usable with no attacker adapter
#: and no GPU, which is what makes a baseline-only run possible.
BASELINES = [n for n in AVAILABLE if n != "llm"]


def build_attacks(
    names,
    *,
    policy=None,
    attacker_agent=None,
    attacker_adapter: str = "attacker",
    attack_temperature: float = 0.7,
    max_new_tokens: int = 512,
    attack_retries: int = 3,
    seed: int = 0,
    lie_z=None,
    lie_sign: float = -1.0,
    minmax_perturbation: str = "std",
    minmax_gamma0: float = 10.0,
    minsum_bound: str = "max",
    fang_b: float = 2.0,
    fang_krum_f=None,
    fang_krum_lambda_mult: float = 1.0,
    ipm_epsilon: float = 0.1,
    mimic_warmup: int = 10,
    signflip_c: float = 1.0,
    noise_sigma: float = 10.0,
    scaling_gamma: float = 10.0,
    labelflip_mode: str = "reverse",
    client_loaders=None,
    lr: float = 0.01,
    local_epochs: int = 1,
    device: str = "cpu",
):
    """Instantiate the requested attacks, preserving order.

    Returns an ordered ``{name: Attack}``. Raises on an unknown name or a missing
    dependency (``llm`` without a policy, ``label_flip`` without client loaders).
    """
    from benchmark.attacks.agr_agnostic import MinMax, MinSum
    from benchmark.attacks.classic import GaussianNoise, Scaling, SignFlip
    from benchmark.attacks.fang import FangKrum, FangTrimmedMean
    from benchmark.attacks.ipm import IPM
    from benchmark.attacks.lie import LIE
    from benchmark.attacks.mimic import Mimic

    out: dict = {}
    for name in names:
        if name == "clean":
            from benchmark.attacks.clean import Clean
            out[name] = Clean()
        elif name == "llm":
            if policy is None or attacker_agent is None:
                raise ValueError("the 'llm' attack requires a loaded policy + attacker_agent")
            from benchmark.attacks.llm_attack import LLMAttack
            out[name] = LLMAttack(policy, attacker_agent, adapter=attacker_adapter,
                                  temperature=attack_temperature,
                                  max_new_tokens=max_new_tokens, retries=attack_retries)
        elif name == "lie":
            out[name] = LIE(z=lie_z, sign=lie_sign)
        elif name == "min_max":
            out[name] = MinMax(perturbation_type=minmax_perturbation, gamma0=minmax_gamma0)
        elif name == "min_sum":
            out[name] = MinSum(perturbation_type=minmax_perturbation,
                               gamma0=minmax_gamma0, bound=minsum_bound)
        elif name == "fang":
            out[name] = FangTrimmedMean(b=fang_b, seed=seed)
        elif name == "fang_krum":
            out[name] = FangKrum(num_byzantine=fang_krum_f,
                                 lambda_mult=fang_krum_lambda_mult)
        elif name == "ipm":
            out[name] = IPM(epsilon=ipm_epsilon)
        elif name == "mimic":
            out[name] = Mimic(warmup_iters=mimic_warmup, seed=seed)
        elif name == "sign_flip":
            out[name] = SignFlip(c=signflip_c)
        elif name == "noise":
            out[name] = GaussianNoise(sigma=noise_sigma, seed=seed)
        elif name == "scaling":
            out[name] = Scaling(gamma=scaling_gamma)
        elif name == "label_flip":
            if client_loaders is None:
                raise ValueError("the 'label_flip' attack requires client_loaders")
            from benchmark.attacks.label_flip import LabelFlip
            out[name] = LabelFlip(client_loaders=client_loaders, lr=lr,
                                  local_epochs=local_epochs, device=device,
                                  mode=labelflip_mode, seed=seed)
        else:
            raise ValueError(f"unknown attack '{name}' (available: {AVAILABLE})")
    return out
