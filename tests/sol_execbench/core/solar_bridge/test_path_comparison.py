from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.solar_bridge.path_comparison import (
    compare_solar_ir_paths,
)
from sol_execbench.core.solar_bridge.path_comparison_models import (
    DifferenceCategory,
    PathComparisonStatus,
)
from solar.ir.contracts import IRPath
from solar.schema_versions import (
    SOLAR_ANALYSIS_SCHEMA_VERSION,
    SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
)


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _graph(prefix: str) -> dict:
    return {
        "outputs": [f"{prefix}.result"],
        "layers": {
            f"{prefix}.start": {
                "type": "start",
                "source_input_index": 0,
                "tensor_names": {"inputs": [], "outputs": [f"{prefix}.input"]},
                "tensor_shapes": {"inputs": [], "outputs": [[2, 4]]},
                "tensor_dtypes": {
                    "inputs": [],
                    "outputs": ["torch.float16"],
                },
                "connections": {"inputs": [], "outputs": [f"{prefix}.op"]},
            },
            f"{prefix}.op": {
                "type": "add",
                "tensor_names": {
                    "inputs": [f"{prefix}.input"],
                    "outputs": [f"{prefix}.result"],
                },
                "tensor_shapes": {"inputs": [[2, 4]], "outputs": [[2, 4]]},
                "tensor_dtypes": {
                    "inputs": ["torch.float16"],
                    "outputs": ["torch.float16"],
                },
                "connections": {
                    "inputs": [f"{prefix}.start"],
                    "outputs": [],
                },
            },
        },
    }


def _analysis(
    *, intermediate_bytes: float, model_io_bytes: float = 32.0
) -> dict:
    return {
        "schema_version": SOLAR_ANALYSIS_SCHEMA_VERSION,
        "layers": {
            "op": {
                "type": "add",
                "resources": {"work": {"valu": {"fp16": 8}}},
                "tensor_dtypes": {
                    "inputs": ["torch.float16"],
                    "outputs": ["torch.float16"],
                },
            },
        },
        "total": {
            "num_layers": 1,
            "macs": 0,
            "macs_by_precision": {},
            "resource_work": {"valu": {"fp16": 8}},
            "resource_seconds": {"valu": 1.0e-9},
            "compute_resource": "valu",
            "unfused_elements": 16,
            "unfused_bytes": 32.0 + intermediate_bytes,
            "orojenesis_elements": None,
            "fused_elements": 16,
            "fused_bytes": 32.0,
            "fused_prefetched_elements": 16,
            "fused_prefetched_bytes": 32.0,
            "model_io_elements": 16,
            "model_io_bytes": model_io_bytes,
            "intermediate_elements": int(intermediate_bytes / 2),
            "intermediate_bytes": intermediate_bytes,
            "num_intermediate_tensors": 1,
            "num_orphaned_layers": 0,
            "lower_bound_components": {"compute": 1.0e-9},
            "lower_bound_seconds": 1.0e-9,
        },
        "metadata": {
            "bound_kind": "roofline_eq1_v1",
            "resource_model": {
                "version": "amd_resource_v3",
                "coverage": {"modeled": 1},
                "fail_closed": True,
            },
        },
    }


def _attestation(graph_name: str, graph_sha256: str) -> dict:
    return {
        "subject": [
            {
                "name": "definition#reference",
                "digest": {"sha256": "b" * 64},
            },
            {
                "name": graph_name,
                "digest": {"sha256": graph_sha256},
            },
        ],
        "predicate": {
            "status": "passed",
            "verifier": "solar.verification.ir.v5",
            "execution": {"device_type": "cuda", "gfx_target": "gfx1200"},
            "cases": [{"seed": 11, "pattern": "random", "parameters": {}}],
        },
    }


def _write_workload(
    root: Path,
    relative: str,
    ir_path: IRPath,
    *,
    intermediate_bytes: float,
    model_io_bytes: float = 32.0,
) -> None:
    directory = root / relative
    graph_name = ir_path.graph_filename
    artifacts = {
        "operator_graph.yaml": {"model_name": relative},
        graph_name: _graph(ir_path.value),
        "solar-analysis.yaml": _analysis(
            intermediate_bytes=intermediate_bytes,
            model_io_bytes=model_io_bytes,
        ),
    }
    for name, value in artifacts.items():
        _write_yaml(directory / name, value)
    attestation_name = "conversion-attestation.yaml"
    _write_yaml(
        directory / attestation_name,
        _attestation(graph_name, sha256_file(directory / graph_name)),
    )
    artifacts[attestation_name] = {}
    manifest = {
        "schema_version": SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
        "analysis_id": relative,
        "architecture_sha256": "a" * 64,
        "reference": {"name": "definition#reference", "sha256": "b" * 64},
        "analysis_contract": {
            "ir_path": ir_path.value,
            "precision": "fp16",
            "trace_seed": 200,
            "verification_seeds": [11],
            "atol": 0.0,
            "rtol": 0.0,
            "required_matched_ratio": 1.0,
            "max_error_cap": None,
            "allow_negative_inf": False,
            "preserved_input_indices": [],
        },
        "artifacts": [
            {"path": name, "sha256": sha256_file(directory / name)}
            for name in artifacts
        ],
        "bound": {
            "seconds": 1.0e-9,
            "kind": "roofline_eq1_v1",
            "limiting_resource": "valu",
        },
    }
    _write_yaml(directory / "manifest.yaml", manifest)


def test_comparison_classifies_internal_dialect_differences(
    tmp_path: Path,
) -> None:
    make_fx = tmp_path / "make_fx"
    torchview = tmp_path / "torchview"
    relative = "problem/workload"
    _write_workload(
        make_fx,
        relative,
        IRPath.MAKE_FX_ATEN,
        intermediate_bytes=16.0,
    )
    _write_workload(
        torchview,
        relative,
        IRPath.TORCHVIEW_EXTENDED_EINSUM,
        intermediate_bytes=24.0,
    )

    report_path = tmp_path / "comparison.json"
    result = compare_solar_ir_paths(
        make_fx,
        torchview,
        report_path,
    )

    comparison = result.comparisons[0]
    assert (
        result.status is PathComparisonStatus.MATCHED_WITH_DIALECT_DIFFERENCES
    )
    assert result.authoritative_match is True
    assert comparison.external_reference_io.match is True
    assert comparison.model_io_accounting.match is True
    assert comparison.mandatory_resource_work.match is True
    assert comparison.formal_bound.match is True
    assert comparison.categories == (
        DifferenceCategory.DIALECT_DECOMPOSITION_DIFFERENCE,
    )
    policy = json.loads(report_path.read_text(encoding="utf-8"))["policy"]
    assert policy == {
        "differences_fail_closed": True,
        "fallback": False,
        "favorable_path_selection": False,
        "numeric_replay_proves_equal_accounting": False,
    }


def test_comparison_fails_closed_on_coverage_and_accounting(
    tmp_path: Path,
) -> None:
    make_fx = tmp_path / "make_fx"
    torchview = tmp_path / "torchview"
    _write_workload(
        make_fx,
        "problem/common",
        IRPath.MAKE_FX_ATEN,
        intermediate_bytes=16.0,
        model_io_bytes=48.0,
    )
    _write_workload(
        torchview,
        "problem/common",
        IRPath.TORCHVIEW_EXTENDED_EINSUM,
        intermediate_bytes=16.0,
    )
    _write_workload(
        make_fx,
        "problem/missing",
        IRPath.MAKE_FX_ATEN,
        intermediate_bytes=16.0,
    )

    result = compare_solar_ir_paths(
        make_fx,
        torchview,
        tmp_path / "comparison.json",
    )

    assert result.status is PathComparisonStatus.INCOMPLETE
    assert result.authoritative_match is False
    assert result.missing_by_path["torchview_extended_einsum"] == (
        "problem/missing",
    )
    assert result.comparisons[0].model_io_accounting.classification is (
        DifferenceCategory.RESOURCE_MODEL_BUG
    )
    cli_result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "compare-paths",
            str(make_fx),
            str(torchview),
            "--output",
            str(tmp_path / "cli-comparison.json"),
        ],
    )
    assert cli_result.exit_code == 1
    assert json.loads(cli_result.output)["data"]["status"] == "incomplete"


def test_comparison_rejects_artifact_hash_drift(tmp_path: Path) -> None:
    make_fx = tmp_path / "make_fx"
    torchview = tmp_path / "torchview"
    for root, ir_path in (
        (make_fx, IRPath.MAKE_FX_ATEN),
        (torchview, IRPath.TORCHVIEW_EXTENDED_EINSUM),
    ):
        _write_workload(
            root,
            "problem/workload",
            ir_path,
            intermediate_bytes=16.0,
        )
    (make_fx / "problem/workload/solar-analysis.yaml").write_text(
        "schema_version: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        compare_solar_ir_paths(
            make_fx,
            torchview,
            tmp_path / "comparison.json",
        )


def test_compare_paths_cli_returns_report_and_dialect_status(
    tmp_path: Path,
) -> None:
    make_fx = tmp_path / "make_fx"
    torchview = tmp_path / "torchview"
    for root, ir_path, intermediate in (
        (make_fx, IRPath.MAKE_FX_ATEN, 16.0),
        (torchview, IRPath.TORCHVIEW_EXTENDED_EINSUM, 24.0),
    ):
        _write_workload(
            root,
            "problem/workload",
            ir_path,
            intermediate_bytes=intermediate,
        )
    output = tmp_path / "comparison.json"

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "solar",
            "compare-paths",
            str(make_fx),
            str(torchview),
            "--output",
            str(output),
        ],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["data"]["status"] == "matched_with_dialect_differences"
    assert output.is_file()
