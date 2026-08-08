"""Defense registry for the benchmark.

Defense classes import torch / FL components, so they are imported LAZILY inside
``build_defenses`` — importing this package stays cheap and torch-free.
"""

AVAILABLE = ["fedavg", "oracle", "llm_defender", "fltrust", "defl", "dnc", "multikrum"]


def build_defenses(
    names,
    *,
    device: str = "cpu",
    policy=None,
    defender_agent=None,
    root_loader=None,
    root_lr: float = 0.002,
    root_epochs: int = 1,
    eta: float = 1.0,
    defender_temperature: float = 0.0,
    max_new_tokens: int = 512,
    defl_delta: float = 0.05,
    defl_tau: float = 2.5,
    dnc_num_byzantine: int = 1,
    dnc_c: float = 1.0,
    dnc_niters: int = 1,
    dnc_sub_dim: int = 10000,
    dnc_seed: int = 0,
    multikrum_num_byzantine: int = 1,
    multikrum_m=None,
    llm_defender_clip: float | None = 3.0,
):
    """Instantiate the requested defenses, preserving order. Returns an ordered
    dict {name: Defense}. Raises on an unknown name or missing dependency."""
    from benchmark.defenses.fedavg import NoDefense
    from benchmark.defenses.oracle import Oracle
    from benchmark.defenses.llm_defender import LLMDefender
    from benchmark.defenses.fltrust import FLTrust
    from benchmark.defenses.defl import DeFL
    from benchmark.defenses.dnc import DnC
    from benchmark.defenses.multikrum import MultiKrum

    out: dict = {}
    for name in names:
        if name == "fedavg":
            out[name] = NoDefense(device=device)
        elif name == "oracle":
            out[name] = Oracle(device=device)
        elif name == "llm_defender":
            if policy is None or defender_agent is None:
                raise ValueError("llm_defender requires a loaded policy + defender_agent")
            out[name] = LLMDefender(policy, defender_agent, device=device,
                                    temperature=defender_temperature, max_new_tokens=max_new_tokens,
                                    clip_multiplier=llm_defender_clip)
        elif name == "fltrust":
            if root_loader is None:
                raise ValueError("fltrust requires a root_loader (clean root dataset)")
            out[name] = FLTrust(root_loader, lr=root_lr, local_epochs=root_epochs,
                                device=device, eta=eta)
        elif name == "defl":
            out[name] = DeFL(device=device, delta=defl_delta, tau=defl_tau)
        elif name == "dnc":
            out[name] = DnC(device=device, num_byzantine=dnc_num_byzantine, c=dnc_c,
                            niters=dnc_niters, sub_dim=dnc_sub_dim, seed=dnc_seed)
        elif name == "multikrum":
            out[name] = MultiKrum(device=device, num_byzantine=multikrum_num_byzantine,
                                  m=multikrum_m)
        else:
            raise ValueError(f"unknown defense '{name}' (available: {AVAILABLE})")
    return out
