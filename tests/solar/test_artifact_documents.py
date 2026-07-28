from pathlib import Path

import pytest
import yaml

from solar.artifacts import load_yaml_artifact
from solar.graph.contracts import (
    ExtractionKind,
    OperatorGraphArtifact,
)
from solar.ir.contracts import IRGraphArtifact, IRKind


def test_operator_artifact_loads_once_and_exposes_typed_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "extraction_kind": ExtractionKind.MAKE_FX_REFERENCE.value,
                "layers": {},
            },
        ),
        encoding="utf-8",
    )
    artifact = OperatorGraphArtifact(path, (), (), ())

    assert artifact.extraction_kind is ExtractionKind.MAKE_FX_REFERENCE
    assert artifact.document is artifact.document


def test_ir_artifact_rejects_a_mismatched_discriminator(
    tmp_path: Path,
) -> None:
    path = tmp_path / "graph.yaml"
    path.write_text("ir_kind: aten\n", encoding="utf-8")
    artifact = IRGraphArtifact(path, IRKind.EXTENDED_EINSUM)

    with pytest.raises(ValueError, match="does not match"):
        _ = artifact.document


def test_yaml_artifact_boundary_rejects_oversize_and_non_string_keys(
    tmp_path: Path,
) -> None:
    oversized = tmp_path / "oversized.yaml"
    oversized.write_text("key: value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        load_yaml_artifact(oversized, max_bytes=2)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("1: value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a string"):
        load_yaml_artifact(invalid)
