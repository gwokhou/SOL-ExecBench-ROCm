"""Registry dispatch keeps IR representation below verification and pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml

from solar.analysis import graph_analyzer
from solar.graph.contracts import ExtractionKind
from solar.ir import registry as ir_registry
from solar.ir.contracts import IRBackend, IRGraphArtifact, IRKind
from solar.ir.registry import (
    graph_kind,
    ir_backend,
    ir_backends,
    validate_ir_graph,
)
from solar.pipeline import stages as pipeline_stages
from solar.verification import registry as verification_registry
from solar.verification.executor import IRGraphExecutor
from solar.verification.registry import verification_backend


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


def test_registries_compose_representation_and_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: set[str] = set()

    def stub_validate(_graph: Mapping[str, Any]) -> None:
        seen.add("validate")

    def stub_execute(
        _layer_id: str,
        _layer: Mapping[str, Any],
        operands: Sequence[Any],
        _output_shapes: Sequence[tuple[int, ...]],
    ) -> Any:
        seen.add("execute")
        return operands[0]

    stub = IRBackend(
        kind=IRKind.EXTENDED_EINSUM,
        extractions=frozenset(ExtractionKind),
        validate=stub_validate,
        convert=lambda _operator, output_dir: IRGraphArtifact(
            Path(output_dir),
            IRKind.EXTENDED_EINSUM,
        ),
    )
    monkeypatch.setitem(
        ir_registry._BACKEND_LOADERS,
        IRKind.EXTENDED_EINSUM,
        lambda: stub,
    )
    monkeypatch.setitem(
        verification_registry._EXECUTORS,
        IRKind.EXTENDED_EINSUM,
        stub_execute,
    )

    assert ir_backend(IRKind.EXTENDED_EINSUM) is stub
    assert stub in ir_backends()
    runtime = verification_backend(IRKind.EXTENDED_EINSUM)
    assert runtime.ir is stub
    validate_ir_graph(_stub_graph())
    result = IRGraphExecutor(_stub_graph(), runtime)(torch.ones(2, 2))

    assert seen == {"validate", "execute"}
    torch.testing.assert_close(result, torch.ones(2, 2))


def test_pipeline_owns_verification_and_analysis_composition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, Any] = {}
    graph = _stub_graph()
    graph_path = tmp_path / "graph.yaml"
    graph_path.write_text(yaml.safe_dump(graph), encoding="utf-8")
    artifact = IRGraphArtifact(graph_path, IRKind.EXTENDED_EINSUM)

    def record_verification(**kwargs: Any) -> None:
        seen["verification"] = kwargs["backend"]

    class Analyzer:
        def __init__(self, *, validator) -> None:
            seen["validator"] = validator

        def analyze_graph(self, *_args: Any, **_kwargs: Any) -> dict[str, str]:
            return {"status": "passed"}

    monkeypatch.setattr(
        pipeline_stages,
        "verify_callable_conversion",
        record_verification,
    )
    monkeypatch.setattr(graph_analyzer, "IRGraphAnalyzer", Analyzer)
    request = SimpleNamespace(
        reference=lambda value: value,
        input_factory=lambda _seed: (torch.ones(2, 2),),
        reference_name="tests#identity",
        reference_sha256="a" * 64,
        verification=object(),
        precision="fp32",
        require_orojenesis=False,
        orojenesis_home=None,
    )

    pipeline_stages.verify_request_graph(
        cast(Any, request),
        artifact,
        tmp_path / "verification.yaml",
    )
    analysis = pipeline_stages.analyze_request_graph(
        cast(Any, request),
        cast(Any, object()),
        tmp_path,
        artifact,
    )

    assert seen["verification"].ir.kind is IRKind.EXTENDED_EINSUM
    assert callable(seen["validator"])
    assert analysis == {"status": "passed"}


def test_registry_lists_every_registered_dialect() -> None:
    kinds = {backend.kind for backend in ir_backends()}
    assert kinds == {IRKind.ATEN, IRKind.EXTENDED_EINSUM}
    assert all(isinstance(backend, IRBackend) for backend in ir_backends())


def test_graph_dispatch_requires_an_explicit_ir_kind() -> None:
    with pytest.raises(ValueError, match="explicit ir_kind"):
        graph_kind({"layers": {}})
