"""Behavioral contracts for shared Pydantic model policies."""

from typing import Literal

import pytest
from pydantic import ValidationError

from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    FrozenArtifactModel,
    StrictArtifactModel,
)


class _Artifact(StrictArtifactModel):
    count: int


class _FrozenArtifact(FrozenArtifactModel):
    count: int


class _VersionedVariant(CurrentSchemaModel):
    current_schema_version = "example.family.v1"
    current_artifact_kind = "variant"

    schema_version: Literal["example.family.v1"] = "example.family.v1"
    artifact_kind: Literal["variant"] = "variant"


def test_strict_artifact_rejects_unknown_fields_but_preserves_coercion() -> (
    None
):
    artifact = _Artifact.model_validate({"count": "1"})

    assert artifact.count == 1
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _Artifact.model_validate({"count": 1, "unexpected": True})


def test_frozen_artifact_rejects_assignment() -> None:
    artifact = _FrozenArtifact(count=1)

    with pytest.raises(ValidationError, match="frozen_instance"):
        setattr(  # noqa: B010 -- exercise runtime assignment validation
            artifact,
            "count",
            2,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"schema_version": "example.family.v1"},
            "requires artifact_kind",
        ),
        (
            {
                "schema_version": "example.family.v1",
                "artifact_kind": "other",
            },
            "requires artifact_kind",
        ),
    ],
)
def test_current_schema_variant_requires_exact_artifact_kind(
    payload: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _VersionedVariant.model_validate(payload)
