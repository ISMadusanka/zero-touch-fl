"""Attack registry — maps names to BaseAttack implementations."""

from core.interfaces import BaseAttack

_REGISTRY: dict[str, type[BaseAttack]] = {}


def register(name: str):
    """Decorator to register an attack class."""
    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls
    return wrapper


def get_attack(name: str) -> BaseAttack:
    """Instantiate an attack by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown attack: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def available_attacks() -> list[str]:
    return list(_REGISTRY.keys())


# Import attack modules to trigger registration
from attacks import sign_flip, noise_injection, scaling, gaussian_noise  # noqa: F401, E402
