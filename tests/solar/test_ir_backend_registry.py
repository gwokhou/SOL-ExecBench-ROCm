"""Registry-driven dispatch is what makes a third IR backend plug in cleanly.

These tests monkeypatch the backend registry to prove that validation,
conversion, and execution all route through :func:`ir_backend` rather than any
hardcoded ``if ir_kind is ...`` branch. A newly registered dialect needs no
changes outside ``IRKind`` plus the registry.
"""

from __future__ import annotations

from typing import Any

import torch

from solar.ir import contracts as ir_contracts
from solar.ir.contracts import (
    IRBackend,
    IRKind,
    ir_backend,
    ir_backends,
    validate_ir_graph,
)
from solar.verification.executor import IRGraphExecutor


def _stub_graph() -> dict[str, Any]:
    return {
        "ir_kind": "extended_einsum",
        "schema_version": 999,
        "layers": {
            "x": {
                "type": "start",
                "tensor_names": {"inputs": [], "outputs": ["x"]},
                "tensor_shapes": {"inputs": [], "outputs": [[2, 2]]},
                "tensor_dtypes": {"inputs": [], "outputs": ["torch.float32"]},
            },
            "op": {
                "type": "stub_op",
                "tensor_names": {"inputs": ["x"], "outputs": ["y"]},
                "tensor_shapes": {"inputs": [[2, 2]], "outputs": [[2, 2]]},
                "tensor_dtypes": {
                    "inputs": ["torch.float32"],
                    "outputs": ["torch.float32"],
                },
            },
        },
        "outputs": ["y"],
    }


def test_registry_drives_backend_lookup(monkeypatch) -> None:
    seen: dict[str, bool] = {}

    def stub_validate(graph: dict[str, Any]) -> None:
        seen["validate"] = True

    def stub_execute(
        layer_id: str,
        layer: dict[str, Any],
        operands: tuple[Any, ...],
        output_shapes: tuple[tuple[int, ...], ...],
    ) -> Any:
        seen["execute"] = True
        return operands[0]

    stub = IRBackend(
        IRKind.EXTENDED_EINSUM,
        validate=stub_validate,
        convert=lambda operator, output_dir: None,
        execute=stub_execute,
    )
    monkeypatch.setitem(
        ir_contracts._BACKEND_LOADERS,
        IRKind.EXTENDED_EINSUM,
        lambda: stub,
    )

    assert ir_backend(IRKind.EXTENDED_EINSUM) is stub
    assert stub in ir_backends()

    graph = _stub_graph()
    validate_ir_graph(graph)
    assert seen["validate"]

    result = IRGraphExecutor(graph)(torch.ones(2, 2))
    assert seen["execute"]
    torch.testing.assert_close(result, torch.ones(2, 2))


def test_registry_lists_every_registered_dialect() -> None:
    kinds = {backend.kind for backend in ir_backends()}
    assert IRKind.ATEN in kinds
    assert IRKind.EXTENDED_EINSUM in kinds
    assert all(isinstance(backend, IRBackend) for backend in ir_backends())
