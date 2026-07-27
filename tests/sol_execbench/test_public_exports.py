from __future__ import annotations

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "sol_execbench",
        "sol_execbench.cli",
        "sol_execbench.core",
        "sol_execbench.core.data",
        "sol_execbench.core.process",
        "solar",
    ],
)
def test_declared_public_exports_resolve(module_name: str) -> None:
    module = import_module(module_name)

    for name in module.__all__:
        assert getattr(module, name) is not None
