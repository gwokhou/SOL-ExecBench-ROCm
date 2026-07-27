from __future__ import annotations

import ast
import json
from pathlib import Path

from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ArtifactCoverageStatus,
    Rocprofv3ArtifactKind,
)
from sol_execbench.core.dataset.aka_contract import (
    AKACompatibilityStage,
    AKACorpusRole,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOTS = (
    REPO_ROOT / "src" / "sol_execbench",
    REPO_ROOT / "src" / "solar",
)


def _production_trees() -> list[tuple[Path, ast.Module]]:
    return [
        (path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for root in SOURCE_ROOTS
        for path in root.rglob("*.py")
    ]


def _base_names(node: ast.ClassDef) -> set[str]:
    return {
        base.id if isinstance(base, ast.Name) else base.attr
        for base in node.bases
        if isinstance(base, (ast.Name, ast.Attribute))
    }


def test_all_production_string_enums_use_strenum() -> None:
    legacy: list[str] = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = _base_names(node)
            if {"str", "Enum"} <= bases:
                legacy.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{node.name}",
                )

    assert legacy == []


def test_strenum_members_are_unique_literal_strings() -> None:
    invalid: list[str] = []
    duplicate: list[str] = []
    for path, tree in _production_trees():
        for node in ast.walk(tree):
            if not isinstance(
                node,
                ast.ClassDef,
            ) or "StrEnum" not in _base_names(node):
                continue
            values: dict[str, str] = {}
            for statement in node.body:
                if (
                    not isinstance(statement, ast.Assign)
                    or len(statement.targets) != 1
                ):
                    continue
                target = statement.targets[0]
                if not isinstance(target, ast.Name) or target.id.startswith(
                    "_",
                ):
                    continue
                if not isinstance(
                    statement.value,
                    ast.Constant,
                ) or not isinstance(
                    statement.value.value,
                    str,
                ):
                    invalid.append(
                        f"{path.relative_to(REPO_ROOT)}:{statement.lineno}:"
                        f"{node.name}.{target.id}",
                    )
                    continue
                value = statement.value.value
                if previous := values.get(value):
                    duplicate.append(
                        f"{path.relative_to(REPO_ROOT)}:{statement.lineno}:"
                        f"{node.name}.{previous}/{target.id}={value!r}",
                    )
                values[value] = target.id

    assert invalid == []
    assert duplicate == []


def test_strenum_members_serialize_without_value_extraction() -> None:
    payload = {
        "role": AKACorpusRole.SCORED,
        "compatibility_stage": AKACompatibilityStage.LIVE_PROBE,
        "artifact_kind": Rocprofv3ArtifactKind.ROCPD,
        "artifact_coverage": Rocprofv3ArtifactCoverageStatus.COMPLETE,
    }

    assert json.loads(json.dumps(payload)) == {
        "role": "scored",
        "compatibility_stage": "live_probe",
        "artifact_kind": "rocpd",
        "artifact_coverage": "complete",
    }
