"""CPU-only contract tests for the fixed SOLAR workflow stages."""

from types import SimpleNamespace
from typing import cast

from solar.contracts import SolarStage
from solar.graph.contracts import OperatorGraphArtifact
from solar.ir.contracts import IRConversionRequest
from solar.pipeline import stages


def test_extract_stage_uses_the_declared_extractor_once(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []
    expected = object()
    request = SimpleNamespace(
        reference=object(),
        input_factory=lambda seed: [seed],
        trace_seed=7,
        device="cpu",
        analysis_id="cpu-contract",
        extraction_kind="make_fx_reference_v1",
    )

    def fake_extract(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(stages, "extract_operator_graph", fake_extract)

    assert stages.extract_request_graph(request, tmp_path) is expected
    assert len(calls) == 1
    assert calls[0][1]["extraction_kind"] == request.extraction_kind


def test_convert_stage_never_substitutes_an_ir_backend(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []
    expected = object()
    request = cast(IRConversionRequest, SimpleNamespace(ir_kind="make_fx_aten"))
    operator = cast(OperatorGraphArtifact, object())

    def fake_convert(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(stages, "convert_operator_graph", fake_convert)

    assert stages.convert_request_graph(request, operator, tmp_path) is expected
    assert len(calls) == 1
    assert calls[0][1]["ir_kind"] == request.ir_kind


def test_unknown_workflow_failure_has_stable_stage_reason_code() -> None:
    reason = stages.workflow_reason_code(
        SolarStage.CONVERSION_VERIFICATION,
        RuntimeError("synthetic CPU failure"),
    )

    assert reason == "conversion_verification_failed"
