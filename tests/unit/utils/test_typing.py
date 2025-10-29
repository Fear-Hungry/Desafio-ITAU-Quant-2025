import importlib
import sys

import pytest


def test_typing_module_fallback_without_typealias(monkeypatch):
    module_name = "itau_quant.utils.typing"
    sys.modules.pop(module_name, None)

    typing_module = importlib.import_module("typing")
    if not hasattr(typing_module, "TypeAlias"):
        pytest.skip("Runtime already lacks typing.TypeAlias")

    monkeypatch.setattr(typing_module, "TypeAlias", None, raising=False)

    module = importlib.import_module(module_name)

    assert "ArrayLike" in module.__all__
    assert hasattr(module, "PathLike")

    sys.modules.pop(module_name, None)
