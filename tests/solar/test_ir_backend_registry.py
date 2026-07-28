"""Registry-driven dispatch is what makes a third IR lifecycle plug in cleanly.

These tests monkeypatch the lifecycle registry to prove that every stage routes
through :func:`ir_lifecycle` rather than a parallel hardcoded registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml

from solar.graph.contracts import ExtractionKind
from solar.ir import registry as ir_registry
from solar.ir.contracts import IRGraphArtifact, IRKind, IRLifecycle
from solar.ir.registry import (
    graph_kind,
    ir_lifecycle,
    ir_lifecycles,
    validate_ir_graph,
)
from solar.pipeline.stages import analyze_request_graph, verify_request_graph
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


def test_registry_drives_complete_lifecycle_lookup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, bool] = {}

    def stub_validate(graph: Mapping[str, Any]) -> None:
        seen["validate"] = True

    def stub_execute(
        layer_id: str,
        layer: Mapping[str, Any],
        operands: Sequence[Any],
        output_shapes: Sequence[tuple[int, ...]],
    ) -> Any:
        seen["execute"] = True
        return operands[0]

    def stub_verify(request, graph_path, output_path) -> None:
        del request, graph_path, output_path
        seen["verify"] = True

    def stub_analyze(request, profile, staging, graph_path) -> dict:
        del request, profile, staging, graph_path
        seen["analyze"] = True
        return {"status": "passed"}

    stub = IRLifecycle(
        kind=IRKind.EXTENDED_EINSUM,
        extractions=frozenset(ExtractionKind),
        validate=stub_validate,
        convert=lambda operator, output_dir: IRGraphArtifact(
            Path(output_dir),
            IRKind.EXTENDED_EINSUM,
        ),
        execute=stub_execute,
        verify=stub_verify,
        analyze=stub_analyze,
    )
    monkeypatch.setitem(
        ir_registry._LIFECYCLE_LOADERS,
        IRKind.EXTENDED_EINSUM,
        lambda: stub,
    )

    assert ir_lifecycle(IRKind.EXTENDED_EINSUM) is stub
    assert stub in ir_lifecycles()

    graph = _stub_graph()
    validate_ir_graph(graph)
    assert seen["validate"]

    result = IRGraphExecutor(graph, stub)(torch.ones(2, 2))
    assert seen["execute"]
    torch.testing.assert_close(result, torch.ones(2, 2))

    graph_path = tmp_path / "graph.yaml"
    graph_path.write_text(yaml.safe_dump(graph), encoding="utf-8")
    artifact = IRGraphArtifact(graph_path, IRKind.EXTENDED_EINSUM)
    verify_request_graph(
        cast(Any, object()),
        artifact,
        Path("verification.json"),
    )
    analysis = analyze_request_graph(
        cast(Any, object()),
        cast(Any, object()),
        Path("staging"),
        artifact,
    )
    assert seen["verify"]
    assert seen["analyze"]
    assert analysis == {"status": "passed"}


def test_registry_lists_every_registered_dialect() -> None:
    kinds = {lifecycle.kind for lifecycle in ir_lifecycles()}
    assert IRKind.ATEN in kinds
    assert IRKind.EXTENDED_EINSUM in kinds
    assert all(
        isinstance(lifecycle, IRLifecycle) for lifecycle in ir_lifecycles()
    )


def test_graph_dispatch_requires_an_explicit_ir_kind() -> None:
    with pytest.raises(ValueError, match="explicit ir_kind"):
        graph_kind({"layers": {}})
