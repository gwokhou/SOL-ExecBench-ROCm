from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCase,
    DiagnosticValidationCorpus,
    ValidationArtifactReference,
    require_disjoint_corpora,
    validation_pair_id,
)
from sol_execbench.core.integrity import stable_json_checksum

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


def _corpus(
    role: Literal["development", "held_out"],
    prefix: str,
) -> DiagnosticValidationCorpus:
    return DiagnosticValidationCorpus(
        role=role,
        cases=[
            DiagnosticValidationCase(
                case_id=f"{prefix}:{kind}:{index}",
                pair_id=validation_pair_id(
                    workload_sha256=stable_json_checksum(
                        [prefix, kind, index, "workload"]
                    ),
                    candidate_sha256=stable_json_checksum(
                        [prefix, kind, index, "candidate"]
                    ),
                ),
                workload_kind=kind,
                evidence_manifest=ValidationArtifactReference(
                    path=f"{kind}-{index}.evidence.json",
                    sha256=stable_json_checksum(
                        [prefix, kind, index, "evidence"]
                    ),
                ),
                solar_manifest=ValidationArtifactReference(
                    path=f"{kind}-{index}.solar.json",
                    sha256=stable_json_checksum([prefix, kind, index, "solar"]),
                ),
            )
            for kind in _FAMILIES
            for index in range(20)
        ],
    )


def test_validation_corpora_are_disjoint_and_prediction_free() -> None:
    development = _corpus("development", "dev")
    held_out = _corpus("held_out", "held")

    require_disjoint_corpora(development, held_out)
    payload = development.cases[0].model_dump(mode="json")
    payload["predicted_ms"] = 1.0
    with pytest.raises(ValidationError, match="predicted_ms"):
        DiagnosticValidationCase.model_validate(payload)


def test_validation_corpora_reject_pair_overlap() -> None:
    development = _corpus("development", "dev")
    held_out = _corpus("held_out", "held")
    cases = list(held_out.cases)
    cases[0] = cases[0].model_copy(
        update={"pair_id": development.cases[0].pair_id}
    )
    held_out = held_out.model_copy(update={"cases": cases})

    with pytest.raises(ValueError, match="overlap"):
        require_disjoint_corpora(development, held_out)


def test_validation_corpora_reject_reused_evidence() -> None:
    development = _corpus("development", "dev")
    held_out = _corpus("held_out", "held")
    cases = list(held_out.cases)
    cases[0] = cases[0].model_copy(
        update={
            "evidence_manifest": development.cases[0].evidence_manifest,
        }
    )
    held_out = held_out.model_copy(update={"cases": cases})

    with pytest.raises(ValueError, match="evidence manifests overlap"):
        require_disjoint_corpora(development, held_out)


def test_validation_case_rejects_self_asserted_independence() -> None:
    payload = _corpus("development", "dev").cases[0].model_dump(mode="json")
    payload["independent"] = True

    with pytest.raises(ValidationError, match="independent"):
        DiagnosticValidationCase.model_validate(payload)


def test_unsupported_validation_corpus_schema_is_rejected() -> None:
    payload = _corpus("development", "dev").model_dump(mode="json")
    payload["schema_version"] = "unsupported"

    with pytest.raises(ValueError, match="schema_version"):
        DiagnosticValidationCorpus.model_validate(payload)
