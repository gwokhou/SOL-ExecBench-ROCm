from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sol_execbench.core.bench.performance_model import authoring
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCase,
    ValidationArtifactReference,
    validation_pair_id,
)
from sol_execbench.core.integrity import sha256_file


def _case(
    tmp_path: Path,
    *,
    pair_id: str,
    workload_kind: WorkloadKind = WorkloadKind.ELEMENTWISE,
) -> DiagnosticValidationCase:
    evidence = tmp_path / "evidence.json"
    solar = tmp_path / "solar.yaml"
    evidence.write_text("evidence", encoding="utf-8")
    solar.write_text("solar", encoding="utf-8")
    return DiagnosticValidationCase(
        case_id="case",
        pair_id=pair_id,
        workload_kind=workload_kind,
        evidence_manifest=ValidationArtifactReference(
            path=evidence.name,
            sha256=sha256_file(evidence),
        ),
        solar_manifest=ValidationArtifactReference(
            path=solar.name,
            sha256=sha256_file(solar),
        ),
    )


def test_case_identity_is_derived_from_bound_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workload_sha256 = "a" * 64
    candidate_sha256 = "b" * 64
    pair_id = validation_pair_id(
        workload_sha256=workload_sha256,
        candidate_sha256=candidate_sha256,
    )
    manifest = SimpleNamespace(
        identity=SimpleNamespace(
            workload_sha256=workload_sha256,
            candidate_sha256=candidate_sha256,
        )
    )
    diagnostic = SimpleNamespace(
        workloads=[
            SimpleNamespace(
                semantic=SimpleNamespace(workload_kind=WorkloadKind.ELEMENTWISE)
            )
        ]
    )
    monkeypatch.setattr(
        authoring,
        "load_and_verify_performance_evidence_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        authoring,
        "build_performance_diagnostic",
        lambda _request: diagnostic,
    )

    result = authoring._build_case_diagnostic(
        _case(tmp_path, pair_id=pair_id),
        corpus_path=tmp_path / "corpus.json",
        calibration_path=tmp_path / "calibration.json",
        inference_path=None,
    )

    assert result is diagnostic


def test_case_rejects_forged_pair_or_family(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workload_sha256 = "a" * 64
    candidate_sha256 = "b" * 64
    pair_id = validation_pair_id(
        workload_sha256=workload_sha256,
        candidate_sha256=candidate_sha256,
    )
    monkeypatch.setattr(
        authoring,
        "load_and_verify_performance_evidence_manifest",
        lambda _path: SimpleNamespace(
            identity=SimpleNamespace(
                workload_sha256=workload_sha256,
                candidate_sha256=candidate_sha256,
            )
        ),
    )
    monkeypatch.setattr(
        authoring,
        "build_performance_diagnostic",
        lambda _request: SimpleNamespace(
            workloads=[
                SimpleNamespace(
                    semantic=SimpleNamespace(
                        workload_kind=WorkloadKind.ELEMENTWISE
                    )
                )
            ]
        ),
    )

    with pytest.raises(ValueError, match="pair identity mismatch"):
        authoring._build_case_diagnostic(
            _case(tmp_path, pair_id="f" * 64),
            corpus_path=tmp_path / "corpus.json",
            calibration_path=tmp_path / "calibration.json",
            inference_path=None,
        )
    with pytest.raises(ValueError, match="workload family mismatch"):
        authoring._build_case_diagnostic(
            _case(
                tmp_path,
                pair_id=pair_id,
                workload_kind=WorkloadKind.MATMUL,
            ),
            corpus_path=tmp_path / "corpus.json",
            calibration_path=tmp_path / "calibration.json",
            inference_path=None,
        )
