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
)

_FAMILIES = (
    WorkloadKind.ELEMENTWISE,
    WorkloadKind.TRANSPOSE,
    WorkloadKind.REDUCTION,
    WorkloadKind.MATMUL,
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
                pair_id=f"{prefix}:pair:{kind}:{index}",
                workload_kind=kind,
                evidence_manifest=ValidationArtifactReference(
                    path=f"{kind}-{index}.evidence.json",
                    sha256="a" * 64,
                ),
                solar_manifest=ValidationArtifactReference(
                    path=f"{kind}-{index}.solar.json",
                    sha256="b" * 64,
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
