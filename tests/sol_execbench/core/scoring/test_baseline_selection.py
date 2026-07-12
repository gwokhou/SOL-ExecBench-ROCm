# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

import pytest

from sol_execbench.core.scoring.baseline_selection import (
    BaselineCandidate,
    BaselineSelection,
    BaselineSelectionManifest,
    baseline_selection_manifest_from_dict,
    build_baseline_selection_manifest,
    choose_baseline_candidate,
)


def _candidate(
    name: str, timings: tuple[float, float, float], dependencies: tuple[str, ...] = ()
) -> BaselineCandidate:
    return BaselineCandidate(
        definition="L1/test",
        workload_uuid="uuid",
        candidate=name,
        solution_sha256="a" * 64,
        backend="hip_cpp",
        backend_version="ROCm 7",
        build_id="build",
        dependencies=dependencies,
        timings_ms=timings,
        correctness_passed=True,
    )


def test_selection_prefers_stability_inside_two_percent_tie_window() -> None:
    fastest = _candidate("fast", (0.98, 1.0, 1.02), ("rocblas",))
    stable = _candidate("stable", (1.01, 1.01, 1.01))

    assert choose_baseline_candidate([fastest, stable]) == stable


def test_selection_uses_dependency_count_then_name_as_stable_tie_breakers() -> None:
    first = _candidate("zeta", (1.0, 1.0, 1.0), ("one",))
    second = _candidate("alpha", (1.0, 1.0, 1.0))

    assert choose_baseline_candidate([first, second]) == second


def test_manifest_checksum_binds_all_candidate_measurements() -> None:
    candidate = _candidate("hip", (1.0, 1.0, 1.0))
    selection = BaselineSelection(winner=candidate, candidates=(candidate,))
    manifest = BaselineSelectionManifest(scope="gfx1200:test", selections=(selection,))
    payload = manifest.to_dict()
    candidate_payload = payload["selections"][0]["candidates"][0]
    candidate_payload["timings_ms"] = [2.0, 2.0, 2.0]
    candidate_payload["median_ms"] = 2.0

    with pytest.raises(ValueError, match="checksum"):
        baseline_selection_manifest_from_dict(payload)


def test_portfolio_requires_exactly_the_frozen_workload_keys() -> None:
    candidate = _candidate("hip", (1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="missing"):
        build_baseline_selection_manifest(
            scope="gfx1200:test",
            candidates=(candidate,),
            required_workload_keys=(("L1/test", "uuid"), ("L1/other", "other")),
        )
