from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.analysis.operand_provenance import (
    contraction_external_source_dtypes,
    contraction_operands_are_graph_external,
)
from solar.ir.extended_einsum.conversion import (
    validate_extended_einsum_graph,
)
from solar.schema_versions import EXTENDED_EINSUM_IR_SCHEMA_VERSION


def _start(output_name: str = "start.Output", dtype: str = "torch.float32"):
    return {
        "type": "start",
        "tensor_names": {"inputs": [], "outputs": [output_name]},
        "tensor_shapes": {"inputs": [], "outputs": [[2]]},
        "tensor_dtypes": {"inputs": [], "outputs": [dtype]},
        "connections": {"inputs": [], "outputs": []},
    }


def _operation(
    operation: str,
    *,
    input_names: list[str],
    output_name: str,
    input_layers: list[str],
    output_layers: list[str],
    input_types: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": operation,
        "einsum_equation": "",
        "is_real_einsum": False,
        "is_einsum_supportable": True,
        "tensor_names": {"inputs": input_names, "outputs": [output_name]},
        "tensor_shapes": {
            "inputs": [[2] for _ in input_names],
            "outputs": [[2]],
        },
        "tensor_dtypes": {
            "inputs": ["torch.float32" for _ in input_names],
            "outputs": ["torch.float32"],
        },
        "tensor_types": {
            "inputs": input_types or ["input" for _ in input_names],
            "outputs": ["output"],
        },
        "connections": {"inputs": input_layers, "outputs": output_layers},
    }


def _reject_partial_graph(graph: Mapping[str, Any]) -> None:
    del graph
    raise ValueError("partial accounting fixture")


def _analyze(
    tmp_path: Path,
    layers: dict[str, Any],
    *,
    strict: bool = False,
    schema_version: int = EXTENDED_EINSUM_IR_SCHEMA_VERSION,
    outputs: list[str] | None = None,
):
    graph_path = tmp_path / "einsum_graph.yaml"
    graph = {
        "schema_version": schema_version,
        "ir_kind": "extended_einsum",
        "layers": layers,
    }
    if outputs is not None:
        graph["outputs"] = outputs
    graph_path.write_text(
        yaml.safe_dump(graph),
    )
    output_dir = tmp_path / "analysis"
    result = IRGraphAnalyzer(validator=_reject_partial_graph).analyze_graph(
        graph_path,
        output_dir,
        copy_graph=False,
        strict=strict,
    )
    return result, output_dir


def test_analysis_stages_account_simple_external_io(tmp_path: Path) -> None:
    result, output_dir = _analyze(
        tmp_path,
        {
            "start": _start(),
            "add": _operation(
                "add",
                input_names=["start.Output", "bias"],
                output_name="add.Output",
                input_layers=["start", "bias"],
                output_layers=[],
                input_types=["input", "weight"],
            ),
        },
    )

    assert result is not None
    assert result["layers"]["add"]["memory_elements"] == {
        "inputs": [2, 2],
        "outputs": [2],
    }
    assert result["total"]["fused_elements"] == 6
    assert result["total"]["intermediate_elements"] == 0
    assert result["total"]["resource_work"] == {"valu": {"fp32": 2}}
    assert result["metadata"]["orojenesis"]["status"] == "not_requested"
    assert result["layers"]["add"]["semantic_op"]["target"] == "add"
    assert result["layers"]["add"]["tensor_names"] == {
        "inputs": ["start.Output", "bias"],
        "outputs": ["add.Output"],
    }
    assert yaml.safe_load((output_dir / "analysis.yaml").read_text()) == result


def test_transparent_view_preserves_internal_io_and_shared_input_deduplication(
    tmp_path: Path,
) -> None:
    result, _ = _analyze(
        tmp_path,
        {
            "start": _start(),
            "producer": _operation(
                "add",
                input_names=["start.Output", "bias"],
                output_name="producer.Output",
                input_layers=["start", "bias"],
                output_layers=["view"],
                input_types=["input", "weight"],
            ),
            "view": _operation(
                "reshape",
                input_names=["producer.Output"],
                output_name="view.Output",
                input_layers=["producer"],
                output_layers=["consumer"],
            ),
            "consumer": _operation(
                "mul",
                input_names=["view.Output", "start.Output"],
                output_name="consumer.Output",
                input_layers=["view", "start"],
                output_layers=[],
            ),
        },
    )

    assert result is not None
    assert result["layers"]["view"]["unfused_elements"] == 0
    assert result["layers"]["consumer"]["input_is_intermediate"] is True
    assert result["total"]["fused_elements"] == 6
    assert result["total"]["intermediate_elements"] == 4


def test_topology_proves_internal_io_without_tensor_role_metadata(
    tmp_path: Path,
) -> None:
    producer = _operation(
        "add",
        input_names=["start.Output", "bias"],
        output_name="producer.Output",
        input_layers=["start", "bias"],
        output_layers=["consumer"],
        input_types=["input", "weight"],
    )
    consumer = _operation(
        "mul",
        input_names=["producer.Output", "start.Output"],
        output_name="result",
        input_layers=["producer", "start"],
        output_layers=[],
    )
    producer.pop("tensor_types")
    consumer.pop("tensor_types")

    result, _ = _analyze(
        tmp_path,
        {"start": _start(), "producer": producer, "consumer": consumer},
        outputs=["result"],
    )

    assert result["layers"]["consumer"]["input_is_intermediate"] is True
    assert result["total"]["fused_elements"] == 6
    assert result["total"]["model_io_elements"] == 8


def test_declared_outputs_exclude_unreachable_trace_dead_ends(
    tmp_path: Path,
) -> None:
    result, _ = _analyze(
        tmp_path,
        {
            "start": _start(),
            "result": _operation(
                "add",
                input_names=["start.Output", "bias"],
                output_name="result.Output",
                input_layers=["start", "bias"],
                output_layers=[],
                input_types=["input", "weight"],
            ),
            "trace_noise": _operation(
                "sub",
                input_names=["start.Output"],
                output_name="trace_noise.Output",
                input_layers=["start"],
                output_layers=[],
            ),
        },
        outputs=["result.Output"],
    )

    assert result["layers"]["trace_noise"]["is_orphaned"] is True
    assert result["layers"]["trace_noise"]["resources"]["work"] == {}
    assert result["total"]["resource_work"] == {"valu": {"fp32": 2}}


def test_ir_lifecycle_rejects_unsupported_schema() -> None:
    with pytest.raises(ValueError, match="schema_version=6"):
        validate_extended_einsum_graph(
            {
                "schema_version": 0,
                "ir_kind": "extended_einsum",
                "layers": {"start": _start()},
            },
        )


def test_analysis_rejects_non_current_ir_schema_in_relaxed_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="current schema_version=6"):
        _analyze(
            tmp_path,
            {"start": _start()},
            schema_version=0,
            strict=False,
        )


def test_low_precision_dequantization_is_proven_recomputable() -> None:
    layers = {
        "left": _start("left.Output", "torch.float8_e4m3fn"),
        "right": _start("right.Output", "torch.float8_e4m3fn"),
        "dequant": {
            "type": "to",
            "semantic_op": {"kind": "aten", "target": "to", "effects": {}},
            "tensor_names": {
                "inputs": ["left.Output"],
                "outputs": ["dequant.Output"],
            },
            "tensor_dtypes": {
                "inputs": ["torch.float8_e4m3fn"],
                "outputs": ["torch.float16"],
            },
        },
    }
    contraction = {
        "tensor_names": {"inputs": ["dequant.Output", "right.Output"]},
    }

    assert contraction_operands_are_graph_external(contraction, layers) is True
    assert contraction_external_source_dtypes(contraction, layers) == {
        "torch.float8_e4m3fn",
    }


def test_materialized_preprocessing_without_dequantization_is_not_composable() -> (
    None
):
    layers = {
        "start": _start(),
        "reshape": {
            "type": "reshape",
            "semantic_op": {
                "kind": "aten",
                "target": "reshape",
                "effects": {},
            },
            "tensor_names": {
                "inputs": ["start.Output"],
                "outputs": ["reshape.Output"],
            },
        },
    }

    assert (
        contraction_operands_are_graph_external(
            {"tensor_names": {"inputs": ["reshape.Output"]}},
            layers,
        )
        is False
    )
