"""Behavioral contracts for shared Pydantic model policies."""

import pytest
from pydantic import ValidationError

from sol_execbench.core.data.base_model import (
    FrozenArtifactModel,
    StrictArtifactModel,
)


class _Artifact(StrictArtifactModel):
    count: int


class _FrozenArtifact(FrozenArtifactModel):
    count: int


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
