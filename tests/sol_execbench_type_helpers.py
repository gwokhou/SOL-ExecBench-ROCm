from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar, cast

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.solution import BuildSpec, Solution
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_hardware_models import amd_hardware_model_from_dict
from sol_execbench.core.scoring.amd_sol import build_amd_sol_bound_artifact
from sol_execbench.core.scoring.fusion_validation import FusionValidationArtifact

JsonDict = dict[str, Any]

_T = TypeVar("_T")


def json_dict(value: object) -> JsonDict:
    return cast(JsonDict, value)


def typed(value: object, typ: type[_T]) -> _T:
    del typ
    return cast(_T, value)


def make_definition(**kwargs: Any) -> Definition:
    return Definition.model_validate(kwargs)


def make_workload(**kwargs: Any) -> Workload:
    return Workload.model_validate(kwargs)


def make_solution(**kwargs: Any) -> Solution:
    return Solution.model_validate(kwargs)


def make_build_spec(**kwargs: Any) -> BuildSpec:
    return BuildSpec.model_validate(kwargs)


def make_trace(**kwargs: Any) -> Trace:
    return Trace.model_validate(kwargs)


def make_amd_hardware_model(architecture: str = "gfx1200"):
    path = "wmma" if architecture.startswith("gfx12") else "mfma"
    memory_path = "gfx12" if architecture.startswith("gfx12") else "portable"
    return amd_hardware_model_from_dict(
        {
            "schema_version": "sol_execbench.amd_hardware_model.v3",
            "architecture": architecture,
            "clock_assumptions": ["test fixture"],
            "source": "test calibration fixture",
            "confidence": "supported",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
            "evidence_refs": ["test-calibration.json"],
            "shape_aware_roofline": {
                "status": "validated",
                "evidence_refs": ["test-shape-envelope.json"],
                "bucketing_dimensions": [
                    "shape",
                    "layout",
                    "launch",
                    "occupancy",
                ],
            },
            "compute_profiles": [
                {
                    "key": f"compute.matrix.float32.float32.{path}",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "supported",
                    "evidence_ref": "fixture",
                },
                {
                    "key": f"compute.vector.float32.float32.{memory_path}",
                    "state": "measured",
                    "value": 10.0,
                    "confidence": "supported",
                    "evidence_ref": "fixture",
                },
            ],
            "memory_profiles": [
                {
                    "key": f"memory.stream_copy.float32.float32.{memory_path}",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "supported",
                    "evidence_ref": "fixture",
                }
            ],
        }
    )


def make_amd_sol_bound(definition, workload, hardware_model=None, **kwargs):
    model = hardware_model or make_amd_hardware_model()
    fusion = FusionValidationArtifact(
        architecture=model.architecture.lower(),
        gpu_uuid="GPU-test",
        rocm_version="test",
        hipcc_version="test",
        clocks_locked=True,
        suite_manifest_sha256="a" * 64,
        benchmark_root_sha256="b" * 64,
        generated_at="2026-07-12T00:00:00Z",
        cases=(),
    )
    return build_amd_sol_bound_artifact(
        definition,
        workload,
        model,
        fusion_validation=fusion,
        fusion_validation_ref="fusion-validation.json",
        fusion_validation_sha256="c" * 64,
        **kwargs,
    )


def write_amd_contract_inputs(root: Path) -> tuple[Path, Path]:
    model = make_amd_hardware_model()
    model_path = root / "amd-hardware-model.json"
    model_path.write_text(json.dumps(model.to_dict()), encoding="utf-8")
    fusion = FusionValidationArtifact(
        architecture=model.architecture,
        gpu_uuid="GPU-test",
        rocm_version="test",
        hipcc_version="test",
        clocks_locked=True,
        suite_manifest_sha256="a" * 64,
        benchmark_root_sha256="b" * 64,
        generated_at="2026-07-12T00:00:00Z",
        cases=(),
    )
    fusion_path = root / "fusion-validation.json"
    fusion_path.write_text(json.dumps(fusion.to_dict()), encoding="utf-8")
    return model_path, fusion_path
