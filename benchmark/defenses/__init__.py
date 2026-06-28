"""Defense registry for the benchmark.

Defense classes import torch / FL components, so they are imported LAZILY inside
``build_defenses`` — importing this package stays cheap and torch-free.
"""

AVAILABLE = ["fedavg", "oracle", "llm_defender", "fltrust", "defl"]


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
):
    """Instantiate the requested defenses, preserving order. Returns an ordered
    dict {name: Defense}. Raises on an unknown name or missing dependency."""
    from benchmark.defenses.fedavg import NoDefense
    from benchmark.defenses.oracle import Oracle
    from benchmark.defenses.llm_defender import LLMDefender
    from benchmark.defenses.fltrust import FLTrust
    from benchmark.defenses.defl import DeFL

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
                                    temperature=defender_temperature, max_new_tokens=max_new_tokens)
        elif name == "fltrust":
            if root_loader is None:
                raise ValueError("fltrust requires a root_loader (clean root dataset)")
            out[name] = FLTrust(root_loader, lr=root_lr, local_epochs=root_epochs,
                                device=device, eta=eta)
        elif name == "defl":
            out[name] = DeFL(device=device, delta=defl_delta, tau=defl_tau)
        else:
            raise ValueError(f"unknown defense '{name}' (available: {AVAILABLE})")
    return out
