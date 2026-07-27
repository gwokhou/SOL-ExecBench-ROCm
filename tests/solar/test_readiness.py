from __future__ import annotations

from pathlib import Path

import torch

from solar.api import ConversionReadinessRequest, audit_conversion


def _request(
    output: Path,
    reference,
    *,
    architecture: str = "RX_9060_XT",
) -> ConversionReadinessRequest:
    return ConversionReadinessRequest(
        analysis_id="readiness-case",
        reference=reference,
        input_factory=lambda seed: (torch.tensor([float(seed % 3)]),),
        reference_name="test_readiness.reference",
        reference_sha256="a" * 64,
        architecture=architecture,
        output_dir=output,
        device="cpu",
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
        "einsum_conversion",
        "conversion_verification",
    ]
    assert all(item.status == "passed" for item in result.stages)
    assert {item.artifact.path for item in result.stages if item.artifact} == {
        "operator_graph.yaml",
        "einsum_graph.yaml",
        "conversion-attestation.yaml",
    }


def test_audit_conversion_retains_passed_artifacts_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("cannot bind source arguments to graph inputs")

    monkeypatch.setattr("solar.readiness.convert_operator_graph", fail)
    result = audit_conversion(
        _request(tmp_path / "failed", lambda value: value + 1),
    )

    assert not result.ready
    assert result.failure_stage == "einsum_conversion"
    assert result.reason_code == "source_input_binding_failed"
    assert result.stages[0].artifact is not None
    assert (tmp_path / "failed" / "operator_graph.yaml").is_file()
    assert result.stages[-1].status == "not_run"
