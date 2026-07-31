from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sol_execbench.core.bench.performance_model.models import (
    CompositeGraphDescriptor,
    CrossEntropyDescriptor,
    ElementwiseDescriptor,
    IndexedReadDescriptor,
    MatmulDescriptor,
    ReductionDescriptor,
    ReductionOperation,
    SoftmaxDescriptor,
    TransposeDescriptor,
    WorkloadKind,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.solar_bridge.performance import (
    SOLAR_ANALYSIS_SCHEMA_VERSION,
    load_semantic_characterization,
)


def _analysis() -> dict[str, object]:
    return {
        "schema_version": SOLAR_ANALYSIS_SCHEMA_VERSION,
        "layers": {
            "sigmoid": {
                "type": "sigmoid",
                "tensor_shapes": {
                    "inputs": [[4, 8]],
                    "outputs": [[4, 8]],
                },
                "tensor_dtypes": {
                    "inputs": ["float32"],
                    "outputs": ["float32"],
                },
            },
        },
        "total": {
            "flops": 128,
            "resource_work": {"valu": {"fp32": 128}},
            "prefetched_bytes": 256,
            "lower_bound_seconds": 2.0e-6,
        },
        "metadata": {
            "fusion": {
                "regions": [{"id": "fused_0", "layers": ["sigmoid"]}],
            },
        },
    }


def test_bridge_validates_and_extracts_semantic_characterization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "solar-analysis.yaml"
    path.write_text(yaml.safe_dump(_analysis()), encoding="utf-8")

    result = load_semantic_characterization(
        path,
        workload_uuid="workload-1",
        expected_sha256=sha256_file(path),
    )

    assert result.workload_kind is WorkloadKind.ELEMENTWISE
    assert isinstance(result.descriptor, ElementwiseDescriptor)
    assert result.descriptor.shape == [4, 8]
    assert result.semantic_flops == 128
    assert result.semantic_bytes == 256
    assert result.t_sol_ms == pytest.approx(0.002)
    assert result.fusion_regions[0].region_id == "fused_0"


def test_bridge_fails_closed_on_hash_and_schema_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "solar-analysis.yaml"
    path.write_text(yaml.safe_dump(_analysis()), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256_mismatch"):
        load_semantic_characterization(
            path,
            workload_uuid="workload-1",
            expected_sha256="0" * 64,
        )

    payload = _analysis()
    payload["schema_version"] = "stale"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported_solar"):
        load_semantic_characterization(path, workload_uuid="workload-1")


def test_bridge_treats_clone_as_transpose_materialization(
    tmp_path: Path,
) -> None:
    payload = _analysis()
    payload["layers"] = {
        "transpose": {
            "type": "transpose",
            "resources": {
                "classification": "exempt",
                "exemption_reason": "metadata_or_alias_only",
            },
            "tensor_shapes": {
                "inputs": [[4, 8]],
                "outputs": [[8, 4]],
            },
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
        "clone": {
            "type": "clone",
            "tensor_shapes": {
                "inputs": [[8, 4]],
                "outputs": [[8, 4]],
            },
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
    }
    payload["metadata"] = {
        "fusion": {
            "regions": [
                {"id": "region_0", "layers": ["transpose"]},
                {"id": "region_1", "layers": ["clone"]},
            ],
        },
    }
    path = tmp_path / "solar-analysis.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = load_semantic_characterization(
        path,
        workload_uuid="transpose-1",
    )

    assert result.workload_kind is WorkloadKind.TRANSPOSE
    assert isinstance(result.descriptor, TransposeDescriptor)
    assert result.reason_codes == []


def test_bridge_recognizes_canonical_aten_cross_entropy_lowering(
    tmp_path: Path,
) -> None:
    payload = _analysis()
    payload["layers"] = {
        "log_softmax": {
            "type": "log_softmax",
            "semantic_op": {
                "arguments": [
                    {"tensor": 0},
                    {"value": 1},
                    {"value": False},
                ],
                "kwargs": {},
            },
            "tensor_shapes": {
                "inputs": [[256, 128]],
                "outputs": [[256, 128]],
            },
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
        "nll_loss_forward": {
            "type": "nll_loss_forward",
            "semantic_op": {
                "arguments": [
                    {"tensor": 0},
                    {"tensor": 1},
                    {"value": None},
                    {"value": 1},
                    {"value": -100},
                ],
                "kwargs": {},
            },
            "tensor_shapes": {
                "inputs": [[256, 128], [256]],
                "outputs": [[], []],
            },
            "tensor_dtypes": {
                "inputs": ["float32", "int64"],
                "outputs": ["float32", "float32"],
            },
        },
    }
    payload["metadata"] = {
        "fusion": {
            "regions": [
                {"id": "region_0", "layers": ["log_softmax"]},
                {"id": "region_1", "layers": ["nll_loss_forward"]},
            ],
        },
    }
    path = tmp_path / "cross-entropy.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = load_semantic_characterization(
        path,
        workload_uuid="cross-entropy",
    )

    assert result.workload_kind is WorkloadKind.CROSS_ENTROPY
    assert isinstance(result.descriptor, CrossEntropyDescriptor)
    assert result.descriptor.rows == 256
    assert result.descriptor.classes == 128
    assert result.reason_codes == []


def test_bridge_accepts_fp16_matmul_output_with_fp32_accumulation(
    tmp_path: Path,
) -> None:
    payload = _analysis()
    payload["layers"] = {
        "mm": {
            "type": "mm",
            "tensor_shapes": {
                "inputs": [[16, 32], [32, 8]],
                "outputs": [[16, 8]],
            },
            "tensor_dtypes": {
                "inputs": ["float16", "float16"],
                "outputs": ["float16"],
            },
        },
    }
    payload["metadata"] = {
        "fusion": {"regions": [{"id": "region_0", "layers": ["mm"]}]},
    }
    path = tmp_path / "solar-analysis.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = load_semantic_characterization(
        path,
        workload_uuid="matmul-1",
    )

    assert result.workload_kind is WorkloadKind.MATMUL
    assert isinstance(result.descriptor, MatmulDescriptor)
    assert result.descriptor.output_dtype == "float16"


@pytest.mark.parametrize(
    ("operation", "expected_type", "expected_kind"),
    [
        ("softmax", SoftmaxDescriptor, WorkloadKind.SOFTMAX),
        ("layer_norm", ReductionDescriptor, WorkloadKind.REDUCTION),
        ("gather", IndexedReadDescriptor, WorkloadKind.INDEXED_READ),
    ],
)
def test_bridge_extracts_v4_semantics(
    tmp_path: Path,
    operation: str,
    expected_type: type[object],
    expected_kind: WorkloadKind,
) -> None:
    payload = _analysis()
    if operation == "softmax":
        layer = {
            "type": operation,
            "attributes": {"dim": -1},
            "tensor_shapes": {"inputs": [[4, 8]], "outputs": [[4, 8]]},
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        }
    elif operation == "layer_norm":
        layer = {
            "type": operation,
            "attributes": {"normalized_shape": [8]},
            "tensor_shapes": {"inputs": [[4, 8]], "outputs": [[4, 8]]},
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        }
    else:
        layer = {
            "type": operation,
            "attributes": {"dim": 0},
            "tensor_shapes": {
                "inputs": [[16, 8], [4]],
                "outputs": [[4, 8]],
            },
            "tensor_dtypes": {
                "inputs": ["float16", "int64"],
                "outputs": ["float16"],
            },
        }
    payload["layers"] = {operation: layer}
    payload["metadata"] = {
        "fusion": {"regions": [{"id": "region_0", "layers": [operation]}]}
    }
    path = tmp_path / f"{operation}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = load_semantic_characterization(path, workload_uuid=operation)

    assert result.workload_kind is expected_kind
    assert isinstance(result.descriptor, expected_type)
    if operation == "layer_norm":
        assert isinstance(result.descriptor, ReductionDescriptor)
        assert result.descriptor.operation is ReductionOperation.LAYER_NORM


def test_bridge_recovers_index_axis_from_content_bound_semantic_operation(
    tmp_path: Path,
) -> None:
    payload = _analysis()
    payload["layers"] = {
        "index_select": {
            "type": "index_select",
            "semantic_op": {
                "kind": "operation",
                "target": "index_select",
                "arguments": [
                    {"tensor": 0},
                    {"value": 0},
                    {"tensor": 1},
                ],
                "kwargs": {},
            },
            "tensor_shapes": {
                "inputs": [[16, 8], [4]],
                "outputs": [[4, 8]],
            },
            "tensor_dtypes": {
                "inputs": ["float32", "int64"],
                "outputs": ["float32"],
            },
        }
    }
    payload["metadata"] = {
        "fusion": {"regions": [{"id": "region_0", "layers": ["index_select"]}]}
    }
    path = tmp_path / "index-select.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = load_semantic_characterization(
        path,
        workload_uuid="index-select",
    )

    assert result.workload_kind is WorkloadKind.INDEXED_READ
    assert isinstance(result.descriptor, IndexedReadDescriptor)
    assert result.descriptor.axis == 0


def test_bridge_builds_exact_bounded_composite_dag(tmp_path: Path) -> None:
    payload = _analysis()
    payload["layers"] = {
        "sigmoid": {
            "type": "sigmoid",
            "tensor_names": {
                "inputs": ["input"],
                "outputs": ["sigmoid_out"],
            },
            "connections": {"inputs": ["input"], "outputs": ["relu"]},
            "tensor_shapes": {"inputs": [[4, 8]], "outputs": [[4, 8]]},
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
        "relu": {
            "type": "relu",
            "tensor_names": {
                "inputs": ["sigmoid_out"],
                "outputs": ["output"],
            },
            "connections": {"inputs": ["sigmoid"], "outputs": []},
            "tensor_shapes": {"inputs": [[4, 8]], "outputs": [[4, 8]]},
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
        "sum": {
            "type": "sum",
            "tensor_names": {"inputs": ["output"], "outputs": ["reduced"]},
            "connections": {"inputs": ["relu"], "outputs": []},
            "tensor_shapes": {"inputs": [[4, 8]], "outputs": [[4]]},
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
    }
    payload["metadata"] = {
        "fusion": {
            "regions": [
                {"id": "fused", "layers": ["sigmoid", "relu"]},
                {"id": "materialized", "layers": ["sum"]},
            ]
        }
    }
    path = tmp_path / "composite.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = load_semantic_characterization(path, workload_uuid="composite")

    assert result.workload_kind is WorkloadKind.COMPOSITE
    assert isinstance(result.descriptor, CompositeGraphDescriptor)
    assert result.descriptor.schedule == ["sigmoid", "relu", "sum"]
    assert [edge.materialized for edge in result.descriptor.edges] == [
        False,
        True,
    ]


def test_bridge_resolves_composite_edges_through_transparent_views(
    tmp_path: Path,
) -> None:
    payload = _analysis()
    payload["layers"] = {
        "sigmoid": {
            "type": "sigmoid",
            "tensor_names": {
                "inputs": ["input"],
                "outputs": ["sigmoid_out"],
            },
            "connections": {"inputs": ["input"], "outputs": ["view"]},
            "tensor_shapes": {"inputs": [[4, 8]], "outputs": [[4, 8]]},
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
        "view": {
            "type": "view",
            "tensor_names": {
                "inputs": ["sigmoid_out"],
                "outputs": ["view_out"],
            },
            "connections": {"inputs": ["sigmoid"], "outputs": ["sum"]},
            "tensor_shapes": {"inputs": [[4, 8]], "outputs": [[4, 8]]},
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
        "sum": {
            "type": "sum",
            "tensor_names": {
                "inputs": ["view_out"],
                "outputs": ["output"],
            },
            "connections": {"inputs": ["view"], "outputs": []},
            "tensor_shapes": {"inputs": [[4, 8]], "outputs": [[4]]},
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
    }
    payload["metadata"] = {
        "fusion": {
            "regions": [
                {"id": "region_0", "layers": ["sigmoid", "view"]},
                {"id": "region_1", "layers": ["sum"]},
            ]
        }
    }
    path = tmp_path / "transparent-composite.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = load_semantic_characterization(
        path,
        workload_uuid="transparent-composite",
    )

    assert result.workload_kind is WorkloadKind.COMPOSITE
    assert isinstance(result.descriptor, CompositeGraphDescriptor)
    assert result.descriptor.schedule == ["sigmoid", "sum"]
    assert [
        (edge.producer, edge.consumer) for edge in result.descriptor.edges
    ] == [("sigmoid", "sum")]


def test_bridge_honors_content_bound_concurrent_graph_semantics(
    tmp_path: Path,
) -> None:
    payload = _analysis()
    payload["layers"] = {
        "mul": {
            "type": "mul",
            "tensor_names": {
                "inputs": ["input"],
                "outputs": ["root"],
            },
            "connections": {"inputs": ["input"], "outputs": ["sigmoid"]},
            "tensor_shapes": {"inputs": [[4, 8]], "outputs": [[4, 8]]},
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
        "sigmoid": {
            "type": "sigmoid",
            "tensor_names": {
                "inputs": ["root"],
                "outputs": ["output"],
            },
            "connections": {"inputs": ["mul"], "outputs": []},
            "tensor_shapes": {"inputs": [[4, 8]], "outputs": [[4, 8]]},
            "tensor_dtypes": {
                "inputs": ["float32"],
                "outputs": ["float32"],
            },
        },
    }
    payload["metadata"] = {
        "performance_semantics": {"graph_class": "concurrent_graph"},
        "fusion": {
            "regions": [
                {"id": "region_0", "layers": ["mul"]},
                {"id": "region_1", "layers": ["sigmoid"]},
            ]
        },
    }
    path = tmp_path / "concurrent.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = load_semantic_characterization(
        path,
        workload_uuid="concurrent",
    )

    assert result.workload_kind is WorkloadKind.CONCURRENT
    assert isinstance(result.descriptor, CompositeGraphDescriptor)
    assert result.descriptor.graph_class == "concurrent_graph"
