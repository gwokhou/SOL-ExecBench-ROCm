---
phase: 80-uv-and-pytorch-rocm-wheel-coordination
plan: 1
subsystem: infra
tags: [uv, pytorch, rocm, docker, compatibility-matrix, pytest]
requires:
  - phase: 78-matrix-contract-and-claim-guardrails
    provides: Strict Matrix Entry statuses, reason codes, claim boundaries, and execution decisions.
  - phase: 79-docker-matrix-selection-and-preflight
    provides: Declared Docker Target manifest and Target-to-Matrix conversion helpers.
provides:
  - Target-adjacent PyTorch ROCm dependency policy for ROCm 7.0, 7.1, and 7.2 Docker Targets.
  - Strict Matrix Entry dependency policy evidence serialized in observed evidence.
  - CPU-safe dependency stack classifier for unavailable wheels, mixed versions, and non-authoritative matches.
  - Shell-consumable dependency preflight JSON helper for Plan 80-02 wrapper integration.
affects: [phase-80-plan-02, phase-81-runtime-evidence, phase-82-validation-docs]
tech-stack:
  added: []
  patterns: [strict-pydantic-policy-models, injectable-observation-classifier, argparse-json-helper]
key-files:
  created:
    - src/sol_execbench/core/dependency_matrix.py
    - tests/sol_execbench/test_dependency_matrix_policy.py
    - tests/sol_execbench/test_dependency_matrix_classification.py
    - tests/sol_execbench/test_dependency_matrix_cli.py
  modified:
    - docker/rocm-targets.json
    - src/sol_execbench/core/compatibility.py
    - src/sol_execbench/core/docker_matrix.py
key-decisions:
  - "Dependency policy is stored next to each declared Docker Target and converted into strict Matrix observed evidence."
  - "ROCm 7.1 remains the project-default uv path; ROCm 7.0 and 7.2 are explicit Target workflows only."
  - "Dependency preflight uses injected observations in tests and central Phase 78 execution decisions for authority rules."
patterns-established:
  - "Matrix dependency policy evidence lives under observed.dependency_policy so entry.model_dump(mode=\"json\") is the source of truth for shell JSON."
  - "Dependency mismatches classify as mixed_version and may only allow probes/smoke under explicit debug override."
requirements-completed: [DEPS-01, DEPS-02, DEPS-03, DEPS-04, DEPS-05, DEPS-07]
duration: 8min
completed: 2026-05-28
---

# Phase 80 Plan 1: Dependency Policy and Classifier Summary

**Target-adjacent PyTorch ROCm wheel policy with Matrix Entry evidence, stack classification, and shell-consumable preflight JSON**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-28T09:19:14Z
- **Completed:** 2026-05-28T09:27:10Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Added `pytorch_dependency_policy` to every declared ROCm Docker Target while preserving the ROCm 7.1 default image and uv dependency path.
- Added strict `MatrixDependencyPolicyEvidence` so Matrix Entry JSON carries policy id, expected local tag, uv index, lock strategy, suggested command, and Triton ROCm policy.
- Added `dependency_matrix.py` with strict policy and observation models, installed-stack classification, centralized Phase 78 execution decisions, and deterministic CLI JSON.
- Added CPU-safe tests for policy preservation, missing wheel handling, mixed-version detection, debug override semantics, and CLI output.

## Task Commits

Each task was committed atomically with TDD gates:

1. **Task 1 RED: Dependency policy contract tests** - `b084c42`
2. **Task 1 GREEN: Target-adjacent dependency policy** - `37c6bb2`
3. **Task 2 RED: Dependency classification contract tests** - `9d0cfdb`
4. **Task 2 GREEN: Stack classifier** - `47d02cc`
5. **Task 3 RED: Dependency preflight CLI tests** - `dbbd8bf`
6. **Task 3 GREEN: Shell-consumable JSON CLI** - `4da7302`

## Files Created/Modified

- `docker/rocm-targets.json` - Added PyTorch ROCm dependency policy metadata for ROCm 7.0, 7.1, and 7.2 Targets.
- `src/sol_execbench/core/compatibility.py` - Added dependency policy evidence and richer Python/toolchain dependency evidence fields.
- `src/sol_execbench/core/docker_matrix.py` - Allowed declared Docker Targets to carry checked-in dependency policy.
- `src/sol_execbench/core/dependency_matrix.py` - Added strict policy/observation models, classifier, collector, result flattening, and module CLI.
- `tests/sol_execbench/test_dependency_matrix_policy.py` - Added policy and default preservation tests.
- `tests/sol_execbench/test_dependency_matrix_classification.py` - Added unavailable wheel, mismatch, match, and debug override tests.
- `tests/sol_execbench/test_dependency_matrix_cli.py` - Added subprocess JSON and invalid boolean tests.

## Decisions Made

- Used `observed.dependency_policy` instead of artifacts or free-text reason strings so the policy is serialized by `MatrixEntry.model_dump(mode="json")`.
- Kept ROCm 7.2 on a researched available `torch==2.11.0+rocm7.2` / `torchvision==0.26.0+rocm7.2` policy rather than treating the default ROCm 7.1 stack as clean.
- Kept debug override authority centralized through `classify_matrix_entry_for_execution(..., allow_mixed_version_debug=True)`.

## Deviations from Plan

None - plan executed within the requested scope.

## Issues Encountered

- The initial Task 1 test asserted `torch==...` requirement strings in `uv.lock`; the lock encodes dependency pins structurally. The test was corrected to assert those exact requirement strings in `pyproject.toml` and index preservation in `uv.lock`.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py -q` - 17 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_dependency_matrix_cli.py -q` - 5 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/compatibility.py src/sol_execbench/core/dependency_matrix.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_dependency_matrix_cli.py` - passed
- `git status --short pyproject.toml uv.lock` - no modifications

## Known Stubs

None.

## Threat Flags

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 80-02 can consume `python -m sol_execbench.core.dependency_matrix preflight` before Docker build/run to block mixed-version dependency states by default and allow explicit debug probes/smoke only.

## Self-Check: PASSED

- Summary file exists.
- Task commits exist: `b084c42`, `37c6bb2`, `9d0cfdb`, `47d02cc`, `dbbd8bf`, `4da7302`.
- Created files exist.

---
*Phase: 80-uv-and-pytorch-rocm-wheel-coordination*
*Completed: 2026-05-28*
