from __future__ import annotations

from pathlib import Path

import yaml

from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.solar_bridge.publication import (
    project_solar_manifest,
    verify_projected_solar_manifest,
)
from solar.contracts import SolarRequestManifest
from solar.schema_versions import (
    SOLAR_ANALYSIS_SCHEMA_VERSION,
    SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
)


def _solar_analysis(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": SOLAR_ANALYSIS_SCHEMA_VERSION,
                "layers": {
                    "add": {
                        "type": "add",
                        "tensor_shapes": {
                            "inputs": [[64], [64]],
                            "outputs": [[64]],
                        },
                        "tensor_dtypes": {
                            "inputs": ["float32", "float32"],
                            "outputs": ["float32"],
                        },
                    }
                },
                "total": {
                    "flops": 64,
                    "resource_work": {"valu": {"fp32": 64}},
                    "prefetched_bytes": 768,
                    "lower_bound_seconds": 1e-6,
                },
                "metadata": {
                    "fusion": {
                        "regions": [{"id": "region-0", "layers": ["add"]}]
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _solar_manifest(root: Path) -> Path:
    artifact_paths = (
        "operator_graph.yaml",
        "aten_graph.yaml",
        "conversion-attestation.yaml",
        "solar-analysis.yaml",
        "orojenesis/raw-search.json",
    )
    for relative in artifact_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "solar-analysis.yaml":
            _solar_analysis(path)
        else:
            path.write_text(relative, encoding="utf-8")
    payload = {
        "schema_version": SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
        "analysis_id": "vector_add:case-0",
        "architecture_sha256": "a" * 64,
        "reference": {"name": "test", "sha256": "b" * 64},
        "analysis_contract": {
            "ir_path": "make_fx_aten",
            "extraction_kind": "make_fx_reference",
            "precision": "fp32",
            "ir_kind": "aten",
            "trace_seed": 200,
            "verification_seeds": [11, 29, 47],
            "atol": 1e-2,
            "rtol": 1e-2,
            "required_matched_ratio": 1.0,
            "max_error_cap": None,
            "allow_negative_inf": False,
            "preserved_input_indices": [],
            "require_orojenesis": True,
        },
        "sol_score_eligible": True,
        "publication_eligible": True,
        "artifacts": [
            {"path": relative, "sha256": sha256_file(root / relative)}
            for relative in artifact_paths
        ],
        "bound": {
            "seconds": 1e-6,
            "kind": "capacity_constrained_tile_aware_v1",
            "limiting_resource": "valu",
        },
    }
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_solar_projection_omits_verified_nested_orojenesis(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "process" / "solar"
    source_root.mkdir(parents=True)
    source = _solar_manifest(source_root)

    output = project_solar_manifest(
        source,
        tmp_path / "publication" / "solar",
        expected_definition="vector_add",
        expected_workload_uuid="case-0",
    )

    projected = SolarRequestManifest.from_yaml(
        output.read_text(encoding="utf-8")
    )
    assert {item.path for item in projected.artifacts} == {
        "operator_graph.yaml",
        "aten_graph.yaml",
        "conversion-attestation.yaml",
        "solar-analysis.yaml",
    }
    assert not (output.parent / "orojenesis").exists()
    verify_projected_solar_manifest(output)
