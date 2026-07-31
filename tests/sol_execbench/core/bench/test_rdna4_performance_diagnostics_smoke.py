from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticSidecarStatus,
)
from sol_execbench.core.bench.performance_model.builder import (
    PerformanceDiagnosticBuildRequest,
    build_performance_diagnostic,
)
from sol_execbench.core.bench.performance_model.models import (
    RatioKind,
    WorkloadKind,
)
from sol_execbench.core.data.base_model import StrictArtifactModel
from sol_execbench.core.solar_bridge.performance import (
    load_manifest_semantic_characterization,
)

pytestmark = [pytest.mark.requires_rocm, pytest.mark.requires_rdna4]

_FAMILIES = (
    WorkloadKind.ELEMENTWISE,
    WorkloadKind.TRANSPOSE,
    WorkloadKind.REDUCTION,
    WorkloadKind.MATMUL,
    WorkloadKind.SOFTMAX,
    WorkloadKind.CROSS_ENTROPY,
    WorkloadKind.INDEXED_READ,
    WorkloadKind.INDEXED_UPDATE,
    WorkloadKind.COMPOSITE,
    WorkloadKind.TRANSFORMER,
    WorkloadKind.CONCURRENT,
)


class _SmokeCase(StrictArtifactModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workload_kind: WorkloadKind
    evidence_manifest: str = Field(min_length=1)
    solar_manifest: str = Field(min_length=1)


class _SmokeConfiguration(StrictArtifactModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["diagnostic_smoke_test.v1"]
    calibration_profile: str = Field(min_length=1)
    cases: list[_SmokeCase] = Field(min_length=11, max_length=11)

    @model_validator(mode="after")
    def contains_each_supported_family_once(self) -> _SmokeConfiguration:
        kinds = [case.workload_kind for case in self.cases]
        if set(kinds) != set(_FAMILIES) or len(kinds) != len(set(kinds)):
            raise ValueError("smoke configuration requires each family once")
        return self


def test_real_gfx1200_diagnostics_cover_all_supported_families(
    tmp_path: Path,
) -> None:
    raw_path = os.environ.get("SOL_EXECBENCH_DIAGNOSTIC_SMOKE_JSON")
    if raw_path is None:
        pytest.skip(
            "set SOL_EXECBENCH_DIAGNOSTIC_SMOKE_JSON to the eleven-family "
            "content-addressed smoke configuration"
        )
    configuration_path = Path(raw_path).resolve()
    configuration = _SmokeConfiguration.model_validate(
        json.loads(configuration_path.read_text(encoding="utf-8"))
    )
    root = configuration_path.parent
    calibration = _confined_file(root, configuration.calibration_profile)

    for case in configuration.cases:
        diagnostic = build_performance_diagnostic(
            PerformanceDiagnosticBuildRequest(
                evidence_manifest_path=_confined_file(
                    root,
                    case.evidence_manifest,
                ),
                solar_manifest_path=_confined_file(
                    root,
                    case.solar_manifest,
                ),
                calibration_profile_path=calibration,
                output_path=tmp_path / f"{case.workload_kind}.json",
            ),
            semantic_loader=load_manifest_semantic_characterization,
        )
        assert diagnostic.status is DiagnosticSidecarStatus.AVAILABLE
        assert len(diagnostic.workloads) == 1
        workload = diagnostic.workloads[0]
        assert workload.semantic.workload_kind is case.workload_kind
        assert workload.t_pred_ir.status is DiagnosticSidecarStatus.AVAILABLE
        assert workload.t_pred_hw.status is DiagnosticSidecarStatus.AVAILABLE
        ratios = {ratio.kind: ratio for ratio in workload.ratios}
        assert ratios[RatioKind.C].status is DiagnosticSidecarStatus.AVAILABLE
        assert ratios[RatioKind.R].status is DiagnosticSidecarStatus.AVAILABLE


def _confined_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("diagnostic smoke path escapes its root") from error
    if not path.is_file():
        raise ValueError(f"diagnostic smoke artifact is missing: {path}")
    return path
