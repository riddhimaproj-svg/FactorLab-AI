"""A lightweight registry mapping model keys to factor-model classes.

The registry lets higher layers (an API service, a CLI, a config-driven batch
job) instantiate models *by name* without importing each concrete class -- the
Open/Closed Principle in practice: adding a model does not change the call
sites, only the registration.

Models register themselves at import time via the :func:`register_model`
decorator, so importing ``factorlab_quant.models`` populates the registry.

Example
-------
>>> from factorlab_quant.models import get_model, list_models
>>> "CAPM" in list_models()
True
>>> model = get_model("CAPM")()   # construct a fresh CAPM instance
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from factorlab_quant.core.errors import ModelAlreadyRegisteredError, ModelNotFoundError
from factorlab_quant.models.base import AbstractFactorModel

__all__ = [
    "create_model",
    "get_model",
    "is_registered",
    "list_models",
    "register_model",
    "unregister_model",
]

_REGISTRY: dict[str, type[AbstractFactorModel]] = {}

ModelClass = type[AbstractFactorModel]


def register_model(
    name: str | None = None,
) -> Callable[[ModelClass], ModelClass]:
    """Class decorator that registers a factor model under ``name``.

    When ``name`` is omitted, the class's ``__name__`` is used.  Re-registering
    the *same* class under the same key is idempotent; registering a *different*
    class under an existing key raises :class:`ModelAlreadyRegisteredError`.
    """

    def decorator(cls: ModelClass) -> ModelClass:
        if not issubclass(cls, AbstractFactorModel):
            raise TypeError(
                f"{cls.__name__} must subclass AbstractFactorModel to be registered"
            )
        key = name if name is not None else cls.__name__
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ModelAlreadyRegisteredError(key)
        _REGISTRY[key] = cls
        return cls

    return decorator


def get_model(name: str) -> ModelClass:
    """Return the model class registered under ``name``.

    Raises :class:`ModelNotFoundError` if the key is unknown.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ModelNotFoundError(name, tuple(_REGISTRY)) from None


def create_model(name: str, **kwargs: Any) -> AbstractFactorModel:
    """Construct an instance of the model registered under ``name``."""
    return get_model(name)(**kwargs)


def list_models() -> list[str]:
    """Return the sorted list of registered model keys."""
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    """True if a model is registered under ``name``."""
    return name in _REGISTRY


def unregister_model(name: str) -> None:
    """Remove ``name`` from the registry (primarily for tests).

    Raises :class:`ModelNotFoundError` if the key is unknown.
    """
    if name not in _REGISTRY:
        raise ModelNotFoundError(name, tuple(_REGISTRY))
    del _REGISTRY[name]
