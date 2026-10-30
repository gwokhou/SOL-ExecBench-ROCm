"""Fresh semantic receipts are required by both resume and existing gate reuse."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.scoring import (
    release_solar_qualification as qualification,
)
from sol_execbench.core.solar_bridge import corpus_readiness as readiness
from sol_execbench.core.solar_bridge.models import SolarStageAuditOutcome
from solar.ir.contracts import IRPath
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)
from solar.verification import execution


@pytest.fixture
def receipt(tmp_path):
    path = IRPath.MAKE_FX_ATEN
    conversion = replace(
        get_sample("F01-D").request(path, "a" * 64),
        reference_name="fixture/definition.json#reference",
    )
    built = audit_conversion(
        ConversionReadinessRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=tmp_path / "workload",
            require_verified_audit=False,
        )
    )
    assert built.ready, built.message
    outcome = SolarStageAuditOutcome(
        status=built.status,
        analysis_id=built.analysis_id,
        ir_path=built.ir_path,
        output_dir=built.output_dir,
        architecture_sha256=built.architecture_sha256,
        stages=tuple(stage.to_dict() for stage in built.stages),
        artifacts=tuple(item.to_dict() for item in built.artifacts),
    )
    identity = {
        "problem_path": "toy/F01-D",
        "workload_uuid": "toy-instance",
        "corpus_manifest_sha256": "b" * 64,
        "definition_sha256": "c" * 64,
        "workload_file_sha256": "d" * 64,
        "workload_sha256": "e" * 64,
        "reference_sha256": conversion.reference_sha256,
        "gfx_target": "gfx1200",
        "architecture_sha256": built.architecture_sha256,
        "trace_seed": conversion.trace_seed,
        "verification_seeds": list(conversion.verification.seeds),
        "verification_patterns": list(conversion.verification.patterns),
        "preserved_input_indices": [],
        "ir_path": path.value,
        "extraction_kind": path.extraction_kind.value,
        "ir_kind": path.ir_kind.value,
        "trace_identity_sha256": "f" * 64,
    }
    record = readiness._record(identity, outcome, tmp_path)
    result_path = tmp_path / "result.json"
    atomic_write_json_value(result_path, record)
    return conversion, identity, record, result_path


@pytest.mark.parametrize("drift", [False, True])
def test_resume_requires_current_semantic_runtime(
    receipt, tmp_path, monkeypatch, drift
):
    conversion, identity, _record, path = receipt
    if drift:
        changed = {
            **execution.semantic_runtime_identity(),
            "torch_version": "changed-runtime",
        }
        monkeypatch.setattr(
            execution, "semantic_runtime_identity", lambda: changed
        )

    def resume():
        return readiness._load_resumed_record(
            path,
            tmp_path,
            identity,
            policy=conversion.verification,
            reference_name=conversion.reference_name,
        )

    if drift:
        with pytest.raises(ValueError, match="runtime"):
            resume()
    else:
        assert resume()["status"] == "ready"


@pytest.mark.parametrize("drift", [False, True])
def test_existing_qualification_gate_reuses_same_receipt_validation(
    receipt, tmp_path, monkeypatch, drift
):
    conversion, _identity, raw, _path = receipt
    record = readiness.CorpusReadinessRecord.model_validate(raw)
    matrix = tmp_path / "matrix.jsonl"
    definition = SimpleNamespace(name="fixture")
    workload = SimpleNamespace(uuid=record.workload_uuid)
    monkeypatch.setattr(
        qualification,
        "load_json_file",
        lambda _model, path: (
            definition
            if path.name == "definition.json"
            else SimpleNamespace(status=readiness.CorpusReadinessStatus.READY)
        ),
    )
    monkeypatch.setattr(
        qualification,
        "load_jsonl_file",
        lambda _model, path: [record] if path == matrix else [workload],
    )
    monkeypatch.setattr(
        qualification,
        "workload_verification_policy",
        lambda *_: conversion.verification,
    )
    context = SimpleNamespace(
        root=tmp_path,
        corpus=SimpleNamespace(authored_root=tmp_path),
        device="cpu",
    )
    gate = SimpleNamespace(
        item_ids=(f"{record.problem_path}/{record.workload_uuid}",),
        receipts=[
            SimpleNamespace(
                artifacts=[
                    SimpleNamespace(path="matrix.jsonl"),
                    SimpleNamespace(path="summary.json"),
                ]
            )
        ],
    )
    if drift:
        changed = {
            **execution.semantic_runtime_identity(),
            "torch_version": "changed-runtime",
        }
        monkeypatch.setattr(
            execution, "semantic_runtime_identity", lambda: changed
        )
        with pytest.raises(ValueError, match="runtime"):
            qualification._verify_readiness(
                cast("qualification._SolarQualificationContext", context),
                cast("qualification.BatchGPUQualificationGate", gate),
            )
    else:
        qualification._verify_readiness(
            cast("qualification._SolarQualificationContext", context),
            cast("qualification.BatchGPUQualificationGate", gate),
        )


def test_resume_rejects_different_requested_gradient_policy(receipt, tmp_path):
    conversion, identity, _record, path = receipt
    with pytest.raises(ValueError, match="policy differs"):
        readiness._load_resumed_record(
            path,
            tmp_path,
            identity,
            policy=replace(conversion.verification, gradient_atol=0),
            reference_name=conversion.reference_name,
        )
