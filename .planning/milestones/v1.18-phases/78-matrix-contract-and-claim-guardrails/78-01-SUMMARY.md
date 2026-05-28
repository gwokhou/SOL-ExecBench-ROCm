---
phase: 78-matrix-contract-and-claim-guardrails
plan: 1
subsystem: core
tags: [rocm, compatibility-matrix, pydantic, guardrails]

requires:
  - phase: 77-static-kernel-evidence
    provides: strict diagnostic sidecar patterns and claim-boundary precedent
provides:
  - Strict `sol_execbench.rocm_compatibility_matrix.v1` Matrix Entry contract
  - Target/requested fields separated from observed compatibility evidence
  - Bounded compatibility status vocabulary and diagnostic-only claim flags
affects: [phase-79, phase-80, phase-81, phase-82, compatibility-reports]

tech-stack:
  added: []
  patterns:
    - Frozen strict Pydantic v2 sidecar contract
    - `str, Enum` bounded vocabularies with JSON string coercion
    - Diagnostic-only authority booleans on compatibility evidence

key-files:
  created:
    - src/sol_execbench/core/compatibility.py
    - tests/sol_execbench/test_rocm_compatibility_matrix.py
  modified: []

key-decisions:
  - "Compatibility Matrix Entries are strict sidecars, separate from canonical trace, timing, scoring, and benchmark result schemas."
  - "Target/requested values and observed host/container/Python/toolchain/GPU evidence are modeled as separate required objects."
  - "Score, paper-parity, and leaderboard authority are literal false fields on every Matrix Entry claim boundary."

patterns-established:
  - "MatrixTarget carries requested ROCm, Docker image/tag, PyTorch ROCm Target, validation scope, and GPU architecture values."
  - "MatrixObservedEvidence separates host, container, Python dependency, toolchain, and GPU evidence."
  - "MatrixClaimBoundary keeps diagnostic compatibility evidence explicit and prevents benchmark authority escalation."

requirements-completed: [MATRIX-01, MATRIX-02, MATRIX-03, MATRIX-04, MATRIX-05]

duration: 5min
completed: 2026-05-28
---

# Phase 78 Plan 1: Matrix Contract And Claim Guardrails Summary

**Strict ROCm compatibility Matrix Entry contract with bounded statuses, Target/observed evidence separation, artifact references, and diagnostic-only claim boundaries**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-28T05:22:46Z
- **Completed:** 2026-05-28T05:27:15Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `src/sol_execbench/core/compatibility.py` with strict frozen Pydantic models for Matrix Targets, observed evidence, claim boundaries, entries, and aggregate reports.
- Locked the six Phase 78 status values: `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, and `not_tested`.
- Added CPU-safe contract tests for serialization, strict unknown-field rejection, report status counts, artifact references, and diagnostic-only claim flags.

## Task Commits

1. **Task 1 RED: Create strict Matrix contract tests** - `71c5b9f` (test)
2. **Task 1 GREEN: Create strict Matrix contract models** - `9d1aa25` (feat)
3. **Task 2 RED: Lock claim flags and artifact tests** - `4b5cb55` (test)
4. **Task 2 GREEN: Lock claim flags and artifact fields** - `6947104` (feat)

## Files Created/Modified

- `src/sol_execbench/core/compatibility.py` - Strict `sol_execbench.rocm_compatibility_matrix.v1` model contract.
- `tests/sol_execbench/test_rocm_compatibility_matrix.py` - CPU-safe schema, vocabulary, serialization, artifact, and claim-boundary tests.

## Decisions Made

- Used the existing strict sidecar pattern from static kernel evidence: `extra="forbid"`, `frozen=True`, `strict=True`, docstring-aware Pydantic models, enum coercion, and `model_dump(mode="json")` via `to_dict`.
- Kept compatibility evidence sidecar-only; no Docker scripts, runtime probes, CLI commands, trace JSONL, scoring, benchmark defaults, uv lock/index changes, or report emission wiring were added.
- Required explicit claim booleans for container user-space, native host, and hardware validation instead of deriving those claims from status or free-text reason strings.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

None. Optional `None` and empty collection defaults in `compatibility.py` are schema fields for not-yet-observed evidence, not UI or runtime placeholders.

## Threat Flags

None. The new JSON payload trust boundary and diagnostic-to-benchmark authority boundary were already covered by the plan threat model and mitigated with strict models, bounded enums, and literal authority fields.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py -q` - passed, `9 passed in 1.13s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/compatibility.py tests/sol_execbench/test_rocm_compatibility_matrix.py` - passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 79 can consume the Matrix Target and Matrix Entry schema as the contract for Docker matrix selection and preflight evidence. Phase 78 Plan 2 still needs to enforce mixed-version blocking, debug override limits, host/container claim separation, and public guardrails.

## Self-Check: PASSED

- Found created file: `src/sol_execbench/core/compatibility.py`.
- Found created file: `tests/sol_execbench/test_rocm_compatibility_matrix.py`.
- Found task commits: `71c5b9f`, `9d1aa25`, `4b5cb55`, `6947104`.
- Re-ran plan verification successfully.

---
*Phase: 78-matrix-contract-and-claim-guardrails*
*Completed: 2026-05-28*
