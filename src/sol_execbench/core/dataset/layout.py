# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Filesystem layout inspection for public SOL-ExecBench datasets."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .categories import validate_categories
from .checksums import checksum_category

REQUIRED_PROBLEM_FILES: tuple[str, ...] = ("definition.json", "workload.jsonl")


class LayoutDiagnostic(BaseModel):
    """Structured diagnostic emitted by dataset layout inspection."""

    code: str
    severity: str = "error"
    category: str | None = None
    path: str
    message: str


class LayoutCategory(BaseModel):
    """Shallow category-level layout metadata."""

    name: str
    path: str
    status: str
    problem_count: int = Field(ge=0)
    workload_count: int = Field(ge=0)
    required_files: tuple[str, ...] = REQUIRED_PROBLEM_FILES
    checksum: str | None = None


class DatasetLayout(BaseModel):
    """Shallow dataset layout inspection result."""

    root: str
    selected_categories: tuple[str, ...]
    categories: tuple[LayoutCategory, ...]
    diagnostics: tuple[LayoutDiagnostic, ...]

    @property
    def ok(self) -> bool:
        """Return true when no error diagnostics were emitted."""

        return not any(diagnostic.severity == "error" for diagnostic in self.diagnostics)


def _count_workloads(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _problem_dirs(category_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(category_dir.iterdir())
        if path.is_dir()
        and all((path / required).is_file() for required in REQUIRED_PROBLEM_FILES)
    ]


def inspect_dataset_layout(root: Path, categories: tuple[str, ...] | None = None) -> DatasetLayout:
    """Inspect a dataset root without parsing schemas or running benchmarks."""

    selected = validate_categories(categories)
    root = Path(root)
    category_results: list[LayoutCategory] = []
    diagnostics: list[LayoutDiagnostic] = []

    for category in selected:
        category_dir = root / category
        if not category_dir.is_dir():
            diagnostics.append(
                LayoutDiagnostic(
                    code="missing_category",
                    category=category,
                    path=category_dir.as_posix(),
                    message=f"Expected dataset category directory is missing: {category}",
                )
            )
            category_results.append(
                LayoutCategory(
                    name=category,
                    path=category_dir.as_posix(),
                    status="missing",
                    problem_count=0,
                    workload_count=0,
                    checksum=None,
                )
            )
            continue

        problems = _problem_dirs(category_dir)
        workload_count = sum(_count_workloads(problem / "workload.jsonl") for problem in problems)
        category_results.append(
            LayoutCategory(
                name=category,
                path=category_dir.as_posix(),
                status="present",
                problem_count=len(problems),
                workload_count=workload_count,
                checksum=checksum_category(category_dir),
            )
        )

    return DatasetLayout(
        root=root.as_posix(),
        selected_categories=selected,
        categories=tuple(category_results),
        diagnostics=tuple(diagnostics),
    )
