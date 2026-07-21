# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministic stratified selection over the AKA candidate pool.

Mirrors the SOL-ExecBench paper (§3.2) stratified-sampling step: given a pool of
characterized AKA tasks and a set of coverage targets (declared in the corpus
manifest's ``formal_coverage_requirements``), pick a reproducible seed set. The
selector is deterministic (sorted order, no RNG) so the seed set is reproducible
from the pinned AKA revision + manifest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class AkaCandidate:
    """A characterized AKA task considered for the seed set."""

    task_path: str
    suite: str
    operation: str
    dtype: str
    pass_kind: str = "forward"
    fusion_depth: str = "single"
    source_family: str = ""


def _det_key(candidate: AkaCandidate) -> str:
    digest = hashlib.sha256(candidate.task_path.encode("utf-8")).hexdigest()[:16]
    return f"{candidate.operation}:{candidate.dtype}:{candidate.pass_kind}:{digest}"


def select_seed_set(
    candidates: Iterable[AkaCandidate],
    coverage: Mapping[str, object],
    *,
    max_problems: int,
) -> list[AkaCandidate]:
    """Select up to ``max_problems`` candidates satisfying coverage combinations.

    First satisfies every ``combinations[*].min_count`` (deterministic order),
    then fills remaining slots greedily by coverage deficit, breaking ties by
    lexical ``task_path``.
    """
    pool = sorted(candidates, key=lambda c: (c.operation, c.dtype, c.task_path))
    selected: list[AkaCandidate] = []
    chosen_paths: set[str] = set()

    def _matches(candidate: AkaCandidate, combo: Mapping[str, object]) -> bool:
        mapping = {
            "operation": candidate.operation,
            "dtype": candidate.dtype,
            "pass_kind": candidate.pass_kind,
            "pass": candidate.pass_kind,
            "fusion_depth": candidate.fusion_depth,
            "source_family": candidate.source_family,
            "suite": candidate.suite,
        }
        return all(
            str(mapping.get(k)) == str(v) for k, v in combo.items() if k != "min_count"
        )

    for combo in coverage.get("combinations") or []:
        need = int(combo.get("min_count", 0))
        for candidate in pool:
            if need <= 0:
                break
            if candidate.task_path in chosen_paths:
                continue
            if _matches(candidate, combo):
                selected.append(candidate)
                chosen_paths.add(candidate.task_path)
                need -= 1

    for candidate in pool:
        if len(selected) >= max_problems:
            break
        if candidate.task_path in chosen_paths:
            continue
        selected.append(candidate)
        chosen_paths.add(candidate.task_path)

    return sorted(selected, key=lambda c: c.task_path)


__all__ = ["AkaCandidate", "select_seed_set"]
