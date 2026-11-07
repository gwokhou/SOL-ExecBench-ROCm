# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Candidate identity for the portability track."""

from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.integrity import stable_json_checksum


def portability_digest(solution: Solution) -> str:
    """Bind candidate semantics while excluding evaluator target routing."""
    spec = solution.spec
    sources = sorted(solution.sources, key=lambda source: source.path)
    return stable_json_checksum(
        {
            "definition": solution.definition,
            "languages": sorted(map(str, spec.languages)),
            "entry_point": spec.entry_point,
            "dependencies": sorted(spec.dependencies),
            "destination_passing_style": spec.destination_passing_style,
            "binding": spec.binding,
            "compile_options": (
                None
                if spec.compile_options is None
                else spec.compile_options.model_dump(mode="json")
            ),
            "sources": [item.model_dump(mode="json") for item in sources],
        }
    )


def solution_digest(solution: Solution) -> str:
    """Return the full supplied solution identity for one target cell."""
    return stable_json_checksum(solution.model_dump(mode="json"))


__all__ = ["portability_digest", "solution_digest"]
