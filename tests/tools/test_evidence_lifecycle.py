"""Tests for the executable GitHub evidence lifecycle state machine."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sol_execbench.core.scoring.release_baseline import (
    evidence_publication_manifest_from_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "docs/releases/evidence-lifecycle.json"
MANIFEST_PATH = REPO_ROOT / "docs/releases/gfx1200-20260712-hipblaslt-v4.evidence.json"


def _lifecycle_module():
    spec = importlib.util.spec_from_file_location(
        "evidence_lifecycle", REPO_ROOT / "tools/evidence_lifecycle.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, release: str) -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["release"] = release
    payload["artifact_base_uri"] = (
        f"https://github.com/gwokhou/SOL-ExecBench-ROCm/releases/download/{release}/"
    )
    payload["manifest_sha256"] = None
    manifest = evidence_publication_manifest_from_dict(payload)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")


def _registry_root(tmp_path: Path) -> tuple[Path, Path]:
    manifest_dir = tmp_path / "docs/releases"
    manifest_dir.mkdir(parents=True)
    current_manifest = manifest_dir / MANIFEST_PATH.name
    current_manifest.write_text(
        MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    registry_path = manifest_dir / "evidence-lifecycle.json"
    registry_path.write_text(
        REGISTRY_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return tmp_path, registry_path


def test_current_lifecycle_registry_is_valid() -> None:
    lifecycle = _lifecycle_module()

    lifecycle.validate_lifecycle_registry(REGISTRY_PATH, REPO_ROOT)


def test_successor_publication_atomically_supersedes_predecessor(
    tmp_path: Path,
) -> None:
    lifecycle = _lifecycle_module()
    root, registry_path = _registry_root(tmp_path)
    successor = "gfx1200-20260719-hipblaslt-v5"
    _write_manifest(root / f"docs/releases/{successor}.evidence.json", successor)

    lifecycle.record_published(
        registry_path,
        root,
        release=successor,
        bundle_sha256="a" * 64,
        release_url=(
            "https://github.com/gwokhou/SOL-ExecBench-ROCm/releases/tag/" + successor
        ),
        published_at="2026-07-19T00:00:00Z",
        supersedes="gfx1200-20260712-hipblaslt-v4",
    )

    records = json.loads(registry_path.read_text(encoding="utf-8"))["records"]
    by_release = {record["release"]: record for record in records}
    assert by_release["gfx1200-20260712-hipblaslt-v4"]["status"] == "superseded"
    assert by_release["gfx1200-20260712-hipblaslt-v4"]["replaced_by"] == successor
    assert by_release[successor]["status"] == "published"
    assert by_release[successor]["supersedes"] == "gfx1200-20260712-hipblaslt-v4"


def test_revocation_is_terminal_and_requires_a_reason(tmp_path: Path) -> None:
    lifecycle = _lifecycle_module()
    root, registry_path = _registry_root(tmp_path)

    lifecycle.revoke_published(
        registry_path,
        root,
        release="gfx1200-20260712-hipblaslt-v4",
        reason="incorrect timing policy was discovered",
    )
    with pytest.raises(ValueError, match="only a published"):
        lifecycle.revoke_published(
            registry_path,
            root,
            release="gfx1200-20260712-hipblaslt-v4",
            reason="a second revocation is forbidden",
        )
