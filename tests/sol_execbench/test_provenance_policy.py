# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_PATH = REPO_ROOT / "provenance.toml"
NVIDIA_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved."
)


def _load_provenance() -> dict[str, object]:
    return tomllib.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def _active_nvidia_header_files() -> set[str]:
    roots = ("src", "scripts", "tests")
    files: set[str] = set()
    for root in roots:
        for path in (REPO_ROOT / root).rglob("*.py"):
            if _has_nvidia_header(path):
                files.add(path.relative_to(REPO_ROOT).as_posix())
    return files


def _has_nvidia_header(path: Path) -> bool:
    return NVIDIA_HEADER in path.read_text(encoding="utf-8").splitlines()[:4]


def test_provenance_manifest_defines_header_policy_classes() -> None:
    provenance = _load_provenance()

    assert provenance["policy_doc"] == "docs/provenance.md"
    assert provenance["project_attribution"].startswith("Copyright (c) 2026")

    classes = provenance["classes"]
    for expected in (
        "upstream_retained",
        "derivative_modified",
        "independent_rocm_work",
        "generated_or_planning",
    ):
        assert expected in classes

    header_policy = provenance["header_policy"]
    assert "NVIDIA" in header_policy["upstream_retained"]
    assert "project" in header_policy["independent_rocm_work"]


def test_current_nvidia_headers_are_classified() -> None:
    provenance = _load_provenance()
    nvidia_notice = provenance["nvidia_notice"]
    allowed = set(nvidia_notice["allowed"])
    cleanup_candidates = set(nvidia_notice["cleanup_candidates"])

    assert allowed.isdisjoint(cleanup_candidates)
    assert _active_nvidia_header_files() == allowed | cleanup_candidates


def test_nvidia_notice_allowed_files_exist_and_currently_have_nvidia_header() -> None:
    allowed = set(_load_provenance()["nvidia_notice"]["allowed"])

    for relative_path in sorted(allowed):
        path = REPO_ROOT / relative_path
        assert path.exists(), relative_path
        assert _has_nvidia_header(path), relative_path


def test_nvidia_notice_cleanup_candidates_exist_and_currently_have_nvidia_header() -> None:
    cleanup_candidates = set(_load_provenance()["nvidia_notice"]["cleanup_candidates"])

    for relative_path in sorted(cleanup_candidates):
        path = REPO_ROOT / relative_path
        assert path.exists(), relative_path
        assert _has_nvidia_header(path), relative_path
