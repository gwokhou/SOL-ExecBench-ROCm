from __future__ import annotations

from pathlib import Path

import pytest
import torch

from solar.api import (
    ConversionReadinessRequest,
    ConversionRequest,
    VerificationPolicy,
    audit_conversion,
)
from solar.contracts import SolarStage
from solar.errors import (
    IRReplayError,
    NumericalEquivalenceError,
    ReferenceOutputBindingError,
    SourceInputBindingError,
    UnsupportedOperationError,
)
from solar.ir.contracts import IRPath
from solar.pipeline.readiness import readiness_reason_code


def _request(
    output: Path,
    reference,
    *,
    architecture: str = "RX_9060_XT",
) -> ConversionReadinessRequest:
    return ConversionReadinessRequest(
        conversion=ConversionRequest(
            analysis_id="readiness-case",
            reference=reference,
            input_factory=lambda seed: (torch.tensor([float(seed % 3)]),),
            reference_name="test_readiness.reference",
            reference_sha256="a" * 64,
            ir_path=IRPath.MAKE_FX_ATEN,
        ),
        architecture=architecture,
        output_dir=output,
    )


def test_audit_conversion_records_all_passing_stage_digests(
    tmp_path: Path,
) -> None:
    result = audit_conversion(
        _request(tmp_path / "ready", lambda value: value + 1),
    )

    assert result.ready
    assert [item.stage for item in result.stages] == [
        "graph_extraction",
        "ir_conversion",
        "conversion_verification",
    ]
    assert all(item.status == "passed" for item in result.stages)
    assert {item.artifact.path for item in result.stages if item.artifact} == {
        "operator_graph.yaml",
        "aten_graph.yaml",
        "conversion-attestation.yaml",
    }


def test_readiness_request_reuses_one_verification_policy(
    tmp_path: Path,
) -> None:
    policy = VerificationPolicy(
        atol=0.25,
        rtol=0.5,
        seeds=(3, 5, 7),
        device="cuda",
    )
    request = ConversionReadinessRequest(
        conversion=ConversionRequest(
            analysis_id="composed",
            reference=lambda value: value,
            input_factory=lambda seed: (seed,),
            reference_name="tests#identity",
            reference_sha256="d" * 64,
            verification=policy,
        ),
        architecture={},
        output_dir=tmp_path / "composed",
    )

    assert request.verification is policy
    assert request.atol == 0.25
    assert request.verification_seeds == (3, 5, 7)
    assert request.device == "cuda"


@pytest.mark.parametrize(
    ("error", "reason_code"),
    [
        (SourceInputBindingError, "source_input_binding_failed"),
        (ReferenceOutputBindingError, "reference_output_binding_failed"),
        (UnsupportedOperationError, "exact_operation_unsupported"),
        (IRReplayError, "exact_replay_failed"),
        (NumericalEquivalenceError, "numerical_equivalence_failed"),
    ],
)
def test_reason_codes_come_from_exception_types(
    error,
    reason_code: str,
) -> None:
    assert (
        readiness_reason_code(
            SolarStage.IR_CONVERSION,
            error("message intentionally has no classification tokens"),
        )
        == reason_code
    )


def test_audit_conversion_retains_passed_artifacts_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        raise SourceInputBindingError(
            "cannot bind source arguments to graph inputs",
        )

    monkeypatch.setattr("solar.pipeline.readiness.convert_request_graph", fail)
    result = audit_conversion(
        _request(tmp_path / "failed", lambda value: value + 1),
    )

    assert not result.ready
    assert result.failure_stage == "ir_conversion"
    assert result.reason_code == "source_input_binding_failed"
    assert result.stages[0].artifact is not None
    assert (tmp_path / "failed" / "operator_graph.yaml").is_file()
    assert result.stages[-1].status == "not_run"
