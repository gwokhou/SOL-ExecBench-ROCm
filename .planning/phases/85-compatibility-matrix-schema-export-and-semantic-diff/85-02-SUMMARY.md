---
phase: 85-compatibility-matrix-schema-export-and-semantic-diff
plan: 02
subsystem: diagnostics
tags: [matrix, semantic-diff, markdown, json, guardrails]
requires:
  - phase: 85-compatibility-matrix-schema-export-and-semantic-diff
    provides: strict Matrix schema export helpers and diagnostic sidecar boundaries
provides:
  - Deterministic Matrix report semantic diff models and helpers
  - Severity-ranked diagnostic JSON and Markdown diff output
  - Thin script-side Matrix report diff wrapper
  - CPU-safe semantic diff and claim-boundary guardrail tests
affects: [compatibility-matrix, downstream-evidence-tooling, diagnostic-sidecars]
tech-stack:
  added: []
  patterns: [Pydantic diff result models, normalized semantic field-group comparison, deterministic Markdown rendering]
key-files:
  created:
    - src/sol_execbench/core/matrix_diff.py
    - scripts/diff_matrix_reports.py
    - tests/sol_execbench/test_matrix_semantic_diff.py
  modified:
    - tests/sol_execbench/test_matrix_claim_guardrails.py
key-decisions:
  - "Matrix report diffs match entries by target_id plus validation_scope."
  - "Diff severity is deterministic and ranked with claim-boundary escalation highest."
  - "Diff JSON and Markdown explicitly remain diagnostic-only and authority-false."
patterns-established:
  - "Matrix artifact lists are normalized by artifact_id/content before comparison to avoid order churn."
  - "Diagnostic diff scripts delegate loading, comparison, JSON serialization, and Markdown rendering to core helpers."
requirements-completed: [MATRIX-02, MATRIX-03, MATRIX-04, MATRIX-05]
duration: 8min
completed: 2026-05-31
---

# Phase 85 Plan 02: Matrix Semantic Diff Summary

**Deterministic semantic Matrix report diffs with severity-ranked JSON/Markdown output and diagnostic-only authority guardrails**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-31T09:13:38Z
- **Completed:** 2026-05-31T09:21:49Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added `sol_execbench.core.matrix_diff` with strict diff result models, report loading, normalized comparison helpers, severity classification, JSON serialization, and Markdown rendering.
- Added deterministic semantic field-group diffs across status, reason code, target values, observed evidence groups, artifacts, and claim boundaries.
- Added `scripts/diff_matrix_reports.py` as a thin diagnostic wrapper for writing JSON and Markdown diff outputs.
- Extended guardrails so diff payloads and Markdown keep score, paper-parity, leaderboard, and native-host validation authority false.

## Task Commits

1. **Task 1: Specify semantic Matrix diff contract** - `0ce6643` (test)
2. **Task 2: Implement deterministic semantic diff and Markdown renderer** - `a05a9c9` (feat)
3. **Task 3: Add thin Matrix diff script** - `3cc0310` (feat)

## Files Created/Modified

- `src/sol_execbench/core/matrix_diff.py` - Added Matrix diff schema, models, loader, normalized comparator, severity classifier, and Markdown renderer.
- `scripts/diff_matrix_reports.py` - Added script-side Matrix report diff JSON/Markdown writer.
- `tests/sol_execbench/test_matrix_semantic_diff.py` - Added CPU-safe core and script diff tests.
- `tests/sol_execbench/test_matrix_claim_guardrails.py` - Added diff-specific diagnostic-only authority guardrails.

## Decisions Made

- Diff matching uses `target.target_id` plus `target.validation_scope`, represented as `target_id|validation_scope`.
- Artifact comparison sorts normalized artifact dictionaries by `artifact_id` and content, so stable artifact refs do not churn due to list ordering.
- Severity output includes all detected categories while selecting the highest-ranked category as the entry severity.
- Invalid authority-field escalation in user-supplied Matrix report JSON is rejected during report loading instead of producing authoritative diff output.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Moved script assertions back to Task 3**
- **Found during:** Task 2
- **Issue:** Initial RED test file included Task 3 script assertions, which made Task 2's core-only verification depend on the planned script before it existed.
- **Fix:** Narrowed Task 2 verification to core diff behavior, then reintroduced script assertions with the Task 3 implementation.
- **Files modified:** `tests/sol_execbench/test_matrix_semantic_diff.py`
- **Verification:** `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_matrix_semantic_diff.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_rocm_compatibility_matrix.py -q`
- **Committed in:** `a05a9c9`, `3cc0310`

**2. [Rule 1 - Bug] Corrected expected severity for validation downgrade fixture**
- **Found during:** Task 2
- **Issue:** The grouped-change fixture downgraded from `container_validated` to `mixed_version`; the correct highest severity is `validation_downgrade`, not `claim_boundary_escalation`.
- **Fix:** Updated the assertion and added a separate claim-boundary escalation guardrail using a not-tested to container-validated transition.
- **Files modified:** `tests/sol_execbench/test_matrix_semantic_diff.py`, `tests/sol_execbench/test_matrix_claim_guardrails.py`
- **Verification:** Focused semantic diff and claim guardrail tests passed.
- **Committed in:** `a05a9c9`

**3. [Rule 1 - Bug] Removed unused MatrixValidationScope import**
- **Found during:** Task 3 Ruff verification
- **Issue:** `src/sol_execbench/core/matrix_diff.py` imported `MatrixValidationScope` but did not use it.
- **Fix:** Removed the unused import.
- **Files modified:** `src/sol_execbench/core/matrix_diff.py`
- **Verification:** `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check ...`
- **Committed in:** `3cc0310`

**Total deviations:** 3 auto-fixed (Rule 1: 2, Rule 3: 1)
**Impact on plan:** All fixes were scoped to planned Matrix diff behavior and did not change canonical benchmark semantics.

## Issues Encountered

None beyond the documented test-scope, severity-expectation, and lint corrections.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None.

## Threat Flags

None - the new file-read/write surfaces are the planned user-supplied Matrix report input paths and optional diff output paths in `scripts/diff_matrix_reports.py`.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_matrix_semantic_diff.py -q` (RED failure before implementation: missing module)
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_matrix_semantic_diff.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_rocm_compatibility_matrix.py -q` (35 passed)
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_matrix_semantic_diff.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_run_docker_matrix_script.py -q` (52 passed)
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_matrix_schema_export.py tests/sol_execbench/test_matrix_semantic_diff.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_run_docker_matrix_script.py -q` (72 passed)
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/compatibility.py src/sol_execbench/core/matrix_diff.py scripts/export_matrix_schema.py scripts/diff_matrix_reports.py tests/sol_execbench/test_matrix_schema_export.py tests/sol_execbench/test_matrix_semantic_diff.py tests/sol_execbench/test_matrix_claim_guardrails.py` (passed)

## Self-Check: PASSED

- Found created files: `src/sol_execbench/core/matrix_diff.py`, `scripts/diff_matrix_reports.py`, `tests/sol_execbench/test_matrix_semantic_diff.py`
- Found modified file: `tests/sol_execbench/test_matrix_claim_guardrails.py`
- Found commits: `0ce6643`, `a05a9c9`, `3cc0310`

## Next Phase Readiness

Phase 85 now provides strict Matrix schema exports and deterministic semantic diff tooling while preserving diagnostic-only Matrix claim boundaries.

---
*Phase: 85-compatibility-matrix-schema-export-and-semantic-diff*
*Completed: 2026-05-31*
