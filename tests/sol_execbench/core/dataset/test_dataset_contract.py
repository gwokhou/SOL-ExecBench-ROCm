# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol_execbench.core.dataset import (
    DEFAULT_CATEGORIES,
    DatasetManifestSource,
    build_dataset_manifest,
    inspect_dataset_layout,
    validate_categories,
    write_dataset_manifest,
)


def _write_problem(category_dir: Path, name: str, *, workloads: int = 1) -> None:
    problem_dir = category_dir / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(
        json.dumps(
            {
                "name": name,
                "axes": {},
                "inputs": {},
                "outputs": {},
                "reference": "def run():\n    return None",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (problem_dir / "reference.py").write_text(
        "def run():\n    return None\n",
        encoding="utf-8",
    )
    rows = [
        json.dumps({"uuid": f"{name}-{idx}", "axes": {}, "inputs": {}})
        for idx in range(workloads)
    ]
    (problem_dir / "workload.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_dataset_root(root: Path) -> None:
    for category in DEFAULT_CATEGORIES:
        _write_problem(root / category, f"{category.lower()}_demo", workloads=2)


def test_validate_categories_defaults_and_reorders_explicit_selection():
    assert validate_categories() == DEFAULT_CATEGORIES
    assert validate_categories(["Quant", "L1", "L1"]) == ("L1", "Quant")


def test_validate_categories_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown SOL-ExecBench category"):
        validate_categories(["L1", "CUDA"])


def test_layout_reports_missing_default_category(tmp_path):
    _write_problem(tmp_path / "L1", "matmul", workloads=3)

    layout = inspect_dataset_layout(tmp_path)

    assert not layout.ok
    assert layout.categories[0].name == "FlashInfer-Bench"
    assert layout.categories[0].status == "missing"
    missing = [diagnostic for diagnostic in layout.diagnostics if diagnostic.code == "missing_category"]
    assert {diagnostic.category for diagnostic in missing} == {
        "FlashInfer-Bench",
        "L2",
        "Quant",
    }


def test_layout_allows_explicit_partial_category_selection(tmp_path):
    _write_problem(tmp_path / "L1", "matmul", workloads=3)

    layout = inspect_dataset_layout(tmp_path, categories=("L1",))

    assert layout.ok
    assert layout.selected_categories == ("L1",)
    assert layout.categories[0].problem_count == 1
    assert layout.categories[0].workload_count == 3
    assert layout.categories[0].checksum


def test_manifest_is_deterministic_and_has_claim_boundary(tmp_path):
    _write_dataset_root(tmp_path)
    source = DatasetManifestSource(revision="main")

    first = build_dataset_manifest(
        tmp_path,
        source=source,
        created_at="2026-05-23T00:00:00Z",
    )
    second = build_dataset_manifest(
        tmp_path,
        source=source,
        created_at="2026-05-23T00:00:00Z",
    )

    assert first.to_json() == second.to_json()
    assert first.manifest_checksum is not None
    assert second.manifest_checksum is not None
    assert first.manifest_checksum.value == second.manifest_checksum.value
    assert first.claim_boundary.acquisition_or_layout_complete is True
    assert first.claim_boundary.rocm_readiness is False
    assert first.claim_boundary.execution_success is False
    assert first.claim_boundary.paper_level_validation is False
    assert first.claim_boundary.hosted_leaderboard_parity is False
    assert first.claim_boundary.upstream_solar_equivalence is False


def test_write_dataset_manifest_creates_parent_dirs(tmp_path):
    _write_problem(tmp_path / "L1", "matmul")
    manifest = build_dataset_manifest(
        tmp_path,
        categories=("L1",),
        created_at="2026-05-23T00:00:00Z",
    )
    output = tmp_path / "artifacts" / "dataset_manifest.json"

    write_dataset_manifest(manifest, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sol_execbench.dataset_manifest.v1"
    assert payload["selected_categories"] == ["L1"]
    assert payload["manifest_checksum"]["algorithm"] == "sha256"
