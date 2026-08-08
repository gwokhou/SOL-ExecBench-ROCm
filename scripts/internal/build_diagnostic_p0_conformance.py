"""Build a deterministic, non-authoritative P0 lifecycle conformance input.

The source is the public development projection only.  Cases are split into
disjoint 40/20-per-family development and held-out sets, then relabelled as
``control_plane_conformance``.  The production Cycle 3 held-out corpus is
never read.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from sol_execbench.core.bench.performance_model.lifecycle import (
    BlobStore,
    DiagnosticDesignManifest,
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleArtifact,
    DiagnosticLifecyclePlan,
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
    design_id,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.integrity.schema_versions import SchemaVersion

_PURPOSE = DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE
_DEVELOPMENT_PER_FAMILY = 40
_HELD_OUT_PER_FAMILY = 20
_FAMILY_COUNT = 11


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _write_currentized(
    path: Path, *, schema: SchemaVersion, purpose: DiagnosticEvidencePurpose
) -> None:
    value = _load_object(path)
    value["schema_version"] = schema.value
    value["purpose"] = purpose.value
    atomic_write_json_value(path, value)


def _split_cases(
    source: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = _load_object(source)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in corpus.get("cases", []):
        if not isinstance(case, dict) or not isinstance(
            case.get("workload_kind"), str
        ):
            raise ValueError("source corpus contains an invalid case")
        _currentize_case_references(case, source.parent)
        grouped[case["workload_kind"]].append(case)
    if len(grouped) != _FAMILY_COUNT:
        raise ValueError("conformance source must cover exactly 11 families")
    development: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for family in sorted(grouped):
        cases = grouped[family]
        required = _DEVELOPMENT_PER_FAMILY + _HELD_OUT_PER_FAMILY
        if len(cases) < required:
            raise ValueError(f"conformance source lacks {family} cases")
        development.extend(cases[:_DEVELOPMENT_PER_FAMILY])
        held_out.extend(cases[_DEVELOPMENT_PER_FAMILY:required])
    return development, held_out


def _currentize_case_references(case: dict[str, Any], root: Path) -> None:
    """Add the current explicit tree-reference discriminator and size."""
    for field in ("evidence_manifest", "solar_manifest"):
        reference = case.get(field)
        if not isinstance(reference, dict) or not isinstance(
            reference.get("path"), str
        ):
            raise ValueError(f"conformance case has invalid {field}")
        path = (root / reference["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"conformance case {field} escapes source root")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"conformance case {field} is not regular")
        if sha256_file(path) != reference.get("sha256"):
            raise ValueError(f"conformance case {field} digest mismatch")
        reference["blob_backed"] = False
        reference["size_bytes"] = path.stat().st_size


def _write_corpus(path: Path, role: str, cases: list[dict[str, Any]]) -> None:
    atomic_write_json_value(
        path,
        {
            "schema_version": SchemaVersion.DIAGNOSTIC_VALIDATION_CORPUS.value,
            "purpose": _PURPOSE.value,
            "role": role,
            "cases": cases,
        },
    )


def _prepare_inputs(source: Path, root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"conformance root already exists: {root}")
    shutil.copytree(source, root, symlinks=False)
    development, held_out = _split_cases(source / "development.json")
    _write_corpus(root / "development.json", "development", development)
    _write_corpus(root / "held_out.json", "held_out", held_out)
    _write_currentized(
        root / "calibration/profile.json",
        schema=SchemaVersion.DIAGNOSTIC_CALIBRATION,
        purpose=_PURPOSE,
    )
    _write_currentized(
        root / "calibration/profile.audit.json",
        schema=SchemaVersion.DIAGNOSTIC_CALIBRATION_AUDIT,
        purpose=_PURPOSE,
    )
    for stale in (
        "publication.json",
        "inference.json",
        "source-inference.json",
    ):
        (root / stale).unlink(missing_ok=True)


def _write_design(root: Path, store: Path, revision: str) -> Path:
    payload_path = root / "design-payload.json"
    atomic_write_json_value(
        payload_path,
        {
            "purpose": _PURPOSE.value,
            "development_cases": _DEVELOPMENT_PER_FAMILY * _FAMILY_COUNT,
            "held_out_cases": _HELD_OUT_PER_FAMILY * _FAMILY_COUNT,
            "source_revision": revision,
        },
    )
    payload_digest = BlobStore(store).put_file(payload_path)
    stage_id = design_id(
        universe_start=0,
        design_payload_sha256=payload_digest,
        purpose=_PURPOSE,
    )
    manifest = DiagnosticDesignManifest(
        stage=DiagnosticLifecycleStage.DESIGN,
        purpose=_PURPOSE,
        stage_id=stage_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision=revision,
        exact_inventory=(
            DiagnosticLifecycleArtifact(
                relative_path=payload_path.name,
                sha256=payload_digest,
                size_bytes=payload_path.stat().st_size,
            ),
        ),
        created_at="1970-01-01T00:00:00+00:00",
        universe_start=0,
        design_payload_sha256=payload_digest,
    )
    path = root / "design.json"
    atomic_write_json_value(path, manifest.model_dump(mode="json"))
    return path


def _write_plan(root: Path, store: Path, revision: str, design: Path) -> Path:
    values: dict[str, object] = {
        "design_manifest_path": str(design.resolve()),
        "corpus_root": str(root.resolve()),
        "calibration_profile_path": str(
            (root / "calibration/profile.json").resolve()
        ),
        "calibration_audit_path": str(
            (root / "calibration/profile.audit.json").resolve()
        ),
        "development_corpus_path": str((root / "development.json").resolve()),
        "held_out_corpus_path": str((root / "held_out.json").resolve()),
        "output_root": str((root / "lifecycle-output").resolve()),
        "source_revision": revision,
        "purpose": _PURPOSE,
        "model_version": "gfx1200_diagnostic.v7",
        "max_attempts": 3,
    }
    plan = DiagnosticLifecyclePlan(
        plan_id=stable_json_checksum(values),
        **values,
    )
    path = root / "plan.json"
    atomic_write_json_value(path, plan.model_dump(mode="json"))
    return path


def main() -> int:
    """Build conformance inputs and print the generated plan path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    arguments = parser.parse_args()
    _prepare_inputs(arguments.source.resolve(), arguments.output.resolve())
    design = _write_design(
        arguments.output.resolve(),
        arguments.store_root.resolve(),
        arguments.source_revision,
    )
    plan = _write_plan(
        arguments.output.resolve(),
        arguments.store_root.resolve(),
        arguments.source_revision,
        design,
    )
    print(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
