from pathlib import Path

import pytest

from sol_execbench.core.integrity.schema_versions import SchemaVersion
from sol_execbench.core.solar_bridge import analyzer
from sol_execbench.core.solar_bridge.models import (
    SolarReadinessStatus,
    SolarStage,
    SolarStageAuditOutcome,
)
from solar.api import ConversionReadinessResult
from solar.ir.contracts import IRPath

_PROBLEM = (
    Path(__file__).resolve().parents[4]
    / "problems/AMD_AKA/torch2hip/l1n1_square_matmul"
)
_WORKLOAD = "aka-l1n1_square_matmul-w0"


@pytest.mark.parametrize("ir_path", tuple(IRPath))
def test_cpu_readiness_wraps_native_result_in_worker_contract(
    tmp_path, monkeypatch, ir_path
):
    monkeypatch.setattr(analyzer, "require_formal_device", lambda _device: None)
    outcome = analyzer.audit_workload_stages(
        problem_dir=_PROBLEM,
        workload_uuid=_WORKLOAD,
        output_dir=tmp_path / "audit",
        device="cpu",
        ir_path=ir_path,
    )

    assert outcome.ready, outcome.message
    assert outcome.ir_path is ir_path
    assert outcome.schema_version == SchemaVersion.SOLAR_WORKER_IPC
    assert SolarStageAuditOutcome.from_dict(outcome.to_dict()) == outcome
    assert outcome.output_dir is not None
    for artifact in outcome.artifacts:
        assert (Path(outcome.output_dir) / artifact["path"]).is_file()


@pytest.mark.parametrize("ir_path", tuple(IRPath))
def test_readiness_bridge_preserves_native_failure(
    tmp_path, monkeypatch, ir_path
):
    monkeypatch.setattr(analyzer, "require_formal_device", lambda _device: None)
    result = ConversionReadinessResult(
        status=SolarReadinessStatus.FAILED,
        analysis_id=_WORKLOAD,
        ir_path=ir_path,
        output_dir=str(tmp_path / "audit"),
        architecture_sha256=None,
        stages=(),
        failure_stage=SolarStage.IR_CONVERSION,
        reason_code="strict_conversion_failed",
        message="retained native failure",
    )
    monkeypatch.setattr("solar.api.audit_conversion", lambda _request: result)
    outcome = analyzer.audit_workload_stages(
        problem_dir=_PROBLEM,
        workload_uuid=_WORKLOAD,
        output_dir=tmp_path / "audit",
        device="cpu",
        ir_path=ir_path,
    )

    assert outcome.status is SolarReadinessStatus.FAILED
    assert outcome.failure_stage is SolarStage.IR_CONVERSION
    assert outcome.reason_code == result.reason_code
    assert outcome.message == result.message
    assert outcome.ir_path is ir_path
    assert SolarStageAuditOutcome.from_dict(outcome.to_dict()) == outcome
