from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sol_execbench.core.bench.performance_model.models import WorkloadKind
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
    assert result.shape == [4, 8]
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
