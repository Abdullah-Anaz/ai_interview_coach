from typing import Callable, Dict, Any

ExperimentRunner = Callable[[Dict[str, Any], Any], None]

_DOMAIN_REGISTRY: Dict[str, ExperimentRunner] = {}


def register_domain(domain_name: str) -> Callable:
    """
    A decorator to register an experiment runner function to a specific domain.

    Args:
        domain_name (str): The unique string identifier for the domain 
        (e.g., 'transcription', 'acoustic').

    Returns:
        Callable: The decorator function that registers and returns the original function.
    """
    def decorator(func: ExperimentRunner) -> ExperimentRunner:
        _DOMAIN_REGISTRY[domain_name] = func
        return func
    return decorator


def get_runner(domain_name: str) -> ExperimentRunner:
    """
    Retrieves the registered experiment runner function for a given domain.

    Args:
        domain_name (str): The unique string identifier for the requested domain.

    Returns:
        ExperimentRunner: The callable function mapped to the provided domain name.

    Raises:
        KeyError: If the requested domain_name has not been registered.
    """
    if domain_name not in _DOMAIN_REGISTRY:
        raise KeyError(
            f"Domain '{domain_name}' is not registered. "
            f"Available domains: {list(_DOMAIN_REGISTRY.keys())}"
        )
    return _DOMAIN_REGISTRY[domain_name]