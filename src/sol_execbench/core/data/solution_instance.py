# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Strong-typed data definitions for solution implementations."""

import hashlib
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import (
    ConfigDict,
    Field,
    model_validator,
)

from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    NonEmptyString,
)
from sol_execbench.core.data.solution_models import BuildSpec, SourceFile
from sol_execbench.core.integrity.schema_versions import (
    SchemaVersion,
)


class Solution(CurrentSchemaModel):
    """A concrete implementation for a given Definition.

    Represents a complete solution that provides a high-performance implementation
    for a computational workload defined by a Definition. Contains all source code,
    build specifications, and metadata required for building, interfacing, and
    benchmarking the implementation.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)
    """Treat Solution as immutable to safely memoize derived fields."""

    current_schema_version = SchemaVersion.SOLUTION

    schema_version: Literal[SchemaVersion.SOLUTION] = SchemaVersion.SOLUTION
    name: NonEmptyString
    """A unique, human-readable name for this specific solution (e.g., 'rmsnorm_hip_v1_gfx1200')."""
    definition: NonEmptyString
    """The name of the Definition this implementation solves."""
    author: NonEmptyString
    """The name of the author or agent system that created this solution."""
    spec: BuildSpec
    """Technical specifications for building and executing this solution."""
    sources: list[SourceFile] = Field(min_length=1)
    """Array of source code files representing the complete implementation."""
    description: str | None = Field(default=None)
    """Optional human-readable description of the solution's technique or approach."""

    @model_validator(mode="after")
    def _validate_source_path_entry_point(self) -> "Solution":
        """Validate source file paths for uniqueness and entry file existence.

        Raises:
        ------
        ValueError
            If duplicate source file paths are found or the entry file is not found in the sources.

        """
        seen_paths = set()
        for source in self.sources:
            # Check for duplicates
            if source.path in seen_paths:
                raise ValueError(f"Duplicate source path '{source.path}'")
            seen_paths.add(source.path)

        entry_file = self.spec.entry_point.split("::")[0]

        if entry_file not in seen_paths:
            raise ValueError(
                f"Entry source file '{entry_file}' not found in sources"
            )

        return self

    def get_entry_path(self) -> Path:
        """Extract the file path from the entry point specification.

        The entry point format is '{file_path}::{function_name}', and this method
        returns the file path component as a Path object.

        Returns:
        -------
        Path
            The relative path to the entry source file (e.g., 'main.py', 'src/kernel.cu').

        """
        return Path(self.spec.entry_point.split("::")[0])

    def get_entry_symbol(self) -> str:
        """Extract the function/symbol name from the entry point specification.

        The entry point format is '{file_path}::{function_name}', and this method
        returns the function name component. This is the symbol that builders will
        look up in the compiled module or imported Python module.

        Returns:
        -------
        str
            The function or symbol name to be loaded (e.g., 'run', 'forward', 'kernel').

        """
        return self.spec.entry_point.split("::")[-1]

    @cached_property
    def _content_hash(self) -> str:
        """Return the lazily memoized deterministic content hash."""
        h = hashlib.sha1()
        for s in (
            self.name,
            self.definition,
            *self.spec.languages,
            *self.spec.target_hardware,
            self.spec.entry_point,
            self.spec.binding or "",
            str(self.spec.destination_passing_style),
            self.spec.compile_options.model_dump_json()
            if self.spec.compile_options
            else "",
            *self.spec.dependencies,
            *(part for src in self.sources for part in (src.path, src.content)),
        ):
            h.update(s.encode())

        return h.hexdigest()

    def hash(self) -> str:
        """Return the memoized deterministic hash of the solution content.

        This hash is computed from all fields that affect the solution's behavior:
        name, definition, languages, target hardware, entry point, binding,
        destination-passing style, compile options, dependencies, and all source
        file paths and contents. This ensures that any meaningful change to the
        solution results in a different hash.

        The hash is used for caching build artifacts, allowing solutions with the same
        hash to reuse the same cached build result.

        Returns:
        -------
        str
            A SHA1 hash (40 hex characters) uniquely identifying this solution's content.

        """
        return self._content_hash

    def __hash__(self) -> int:  # pragma: no cover - trivial wrapper
        """Return a native hash derived from the cached content digest."""
        # Use the memoized content hash for fast hashing in dict/set keys.
        return hash(self._content_hash)

    def __eq__(
        self, other: object
    ) -> bool:  # pragma: no cover - trivial wrapper
        """Compare solutions by their deterministic content digest."""
        if not isinstance(other, Solution):
            return NotImplemented
        return self._content_hash == other._content_hash
