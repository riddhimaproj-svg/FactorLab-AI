"""Unit tests for the model registry."""

from __future__ import annotations

import pytest

from factorlab_quant.core.errors import ModelAlreadyRegisteredError, ModelNotFoundError
from factorlab_quant.models import CAPM, LinearFactorModel
from factorlab_quant.models.registry import (
    create_model,
    get_model,
    is_registered,
    list_models,
    register_model,
    unregister_model,
)


def test_capm_is_registered_on_import() -> None:
    assert "CAPM" in list_models()
    assert is_registered("CAPM")
    assert get_model("CAPM") is CAPM


def test_get_unknown_model_raises() -> None:
    with pytest.raises(ModelNotFoundError):
        get_model("Nonexistent")


def test_create_model_instantiates() -> None:
    model = create_model("CAPM")
    assert isinstance(model, CAPM)


def test_register_and_unregister_roundtrip() -> None:
    class _Dummy(LinearFactorModel):
        def __init__(self) -> None:
            super().__init__(name="Dummy", factor_names=("F",))

    register_model("Dummy")(_Dummy)
    try:
        assert is_registered("Dummy")
        assert get_model("Dummy") is _Dummy
    finally:
        unregister_model("Dummy")
    assert not is_registered("Dummy")


def test_reregistering_same_class_is_idempotent() -> None:
    class _Dummy2(LinearFactorModel):
        def __init__(self) -> None:
            super().__init__(name="Dummy2", factor_names=("F",))

    register_model("Dummy2")(_Dummy2)
    register_model("Dummy2")(_Dummy2)  # no error
    try:
        assert get_model("Dummy2") is _Dummy2
    finally:
        unregister_model("Dummy2")


def test_registering_different_class_same_key_raises() -> None:
    class _A(LinearFactorModel):
        def __init__(self) -> None:
            super().__init__(name="A", factor_names=("F",))

    class _B(LinearFactorModel):
        def __init__(self) -> None:
            super().__init__(name="B", factor_names=("F",))

    register_model("Clash")(_A)
    try:
        with pytest.raises(ModelAlreadyRegisteredError):
            register_model("Clash")(_B)
    finally:
        unregister_model("Clash")


def test_register_non_model_raises() -> None:
    with pytest.raises(TypeError):

        @register_model("Bad")
        class _NotAModel:
            pass


def test_unregister_unknown_raises() -> None:
    with pytest.raises(ModelNotFoundError):
        unregister_model("NeverRegistered")
