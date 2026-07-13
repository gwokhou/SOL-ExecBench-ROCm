# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
PROVENANCE_PATH = REPO_ROOT / "provenance.toml"
NVIDIA_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved."
)
PROJECT_HEADER = "# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port"


def _load_provenance() -> dict[str, Any]:
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


def _has_project_header(path: Path) -> bool:
    return PROJECT_HEADER in path.read_text(encoding="utf-8").splitlines()[:4]


def test_provenance_manifest_defines_header_policy_classes() -> None:
    provenance = _load_provenance()

    assert provenance["policy_doc"] == "docs/user/provenance.md"
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


def test_provenance_manifest_defines_dataset_redistribution_policy() -> None:
    provenance = _load_provenance()

    redistribution_classes = provenance["redistribution_classes"]
    for expected in (
        "publishable",
        "local_only",
        "generated_only",
        "excluded",
        "release_bundle_blocked",
    ):
        assert expected in redistribution_classes

    dataset_policy = provenance["dataset_policy"]
    assert (
        dataset_policy["schema_version"] == "sol_execbench.dataset_provenance_policy.v1"
    )
    assert dataset_policy["guardrail"] == "scripts/check_dataset_redistribution.py"

    sources = {source["id"]: source for source in dataset_policy["sources"]}
    assert set(sources) == {
        "nvidia_sol_execbench",
        "flashinfer_trace",
        "generated_local_migration_artifacts",
        "project_owned_rocm_code",
    }

    nvidia_source = sources["nvidia_sol_execbench"]
    assert nvidia_source["license"] == "NVIDIA Evaluation Dataset License"
    assert nvidia_source["redistribution_class"] == "excluded"
    assert nvidia_source["repository_redistribution"] is False
    assert nvidia_source["release_bundle_redistribution"] is False

    flashinfer_source = sources["flashinfer_trace"]
    assert flashinfer_source["license"] == "Apache-2.0"
    assert flashinfer_source["redistribution_class"] == "publishable"
    assert "flashinfer-ai/flashinfer-trace" in flashinfer_source["attribution"]

    generated_source = sources["generated_local_migration_artifacts"]
    assert generated_source["redistribution_class"] == "generated_only"
    assert generated_source["repository_redistribution"] is False


def test_current_nvidia_headers_are_classified() -> None:
    provenance = _load_provenance()
    nvidia_notice = provenance["nvidia_notice"]
    allowed = set(nvidia_notice["allowed"])
    cleanup_candidates = set(nvidia_notice["cleanup_candidates"])

    assert allowed.isdisjoint(cleanup_candidates)
    assert _active_nvidia_header_files() == allowed


def test_nvidia_notice_allowed_files_keep_nvidia_and_add_project_header() -> None:
    allowed = set(_load_provenance()["nvidia_notice"]["allowed"])

    for relative_path in sorted(allowed):
        path = REPO_ROOT / relative_path
        assert path.exists(), relative_path
        assert _has_nvidia_header(path), relative_path
        assert _has_project_header(path), relative_path


def test_nvidia_notice_cleanup_candidates_have_project_header_only() -> None:
    cleanup_candidates = set(_load_provenance()["nvidia_notice"]["cleanup_candidates"])

    for relative_path in sorted(cleanup_candidates):
        path = REPO_ROOT / relative_path
        assert path.exists(), relative_path
        assert not _has_nvidia_header(path), relative_path
        assert _has_project_header(path), relative_path
