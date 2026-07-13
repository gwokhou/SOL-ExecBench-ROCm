---
phase: 78-matrix-contract-and-claim-guardrails
plan: 2
subsystem: core
tags: [rocm, compatibility-matrix, guardrails, pydantic, public-contract]

requires:
  - phase: 78-01
    provides: strict Matrix Entry contract, bounded statuses, Target/observed evidence, and diagnostic claim flags
provides:
  - Pure Matrix execution decisions with benchmark/probe/smoke allowances
  - Mixed-version default benchmark blocking with non-authoritative debug override
  - Host/native versus Docker/container claim validation
  - Public contract guardrails proving compatibility sidecars stay noncanonical
affects: [phase-79, phase-80, phase-81, phase-82, compatibility-reports]

tech-stack:
  added: []
  patterns:
    - Pure pre-benchmark classification helper
    - Pydantic v2 relationship validation for claim boundaries
    - CPU-safe public contract guardrail tests

key-files:
  created:
    - tests/sol_execbench/test_matrix_claim_guardrails.py
  modified:
    - src/sol_execbench/core/compatibility.py
    - tests/sol_execbench/test_public_contract_guardrails.py
    - docs/user/CLAIMS.md
    - docs/internal/v1_4_compatibility_inventory.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Mixed-version Matrix Entries are blocked before benchmark execution by default."
  - "Explicit mixed-version debug override allows probes and smoke execution only, while clean validation and authority flags remain false."
  - "Docker/container scoped Matrix Entries cannot claim `host_validated` or `native_host_validated=true`."
  - "Compatibility matrix evidence remains sidecar-only and absent from canonical Definition, Workload, and Trace payloads."

patterns-established:
  - "MatrixExecutionDecision returns status, reason_code, benchmark_allowed, probes_allowed, smoke_allowed, and non-authoritative claim flags."
  - "MatrixEntry validates host/container claim combinations at construction and parse time."
  - "Public contract tests enumerate forbidden compatibility sidecar key spaces for canonical payloads."

requirements-completed: [MATRIX-01, MATRIX-04, MATRIX-05, MATRIX-06]

duration: 6min
completed: 2026-05-28
---

# Phase 78 Plan 2: Matrix Contract And Claim Guardrails Summary

**Deterministic Matrix execution guardrails that block mixed-version validation, separate Docker and native-host claims, and keep compatibility evidence sidecar-only**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-28T05:30:56Z
- **Completed:** 2026-05-28T05:36:48Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added `MatrixExecutionDecision` and `classify_matrix_entry_for_execution` for pure pre-benchmark decisions.
- Enforced `host_validated` claim boundaries in `MatrixEntry` validation.
- Added CPU-safe guardrail tests for mixed-version blocking, debug override limits, host/container separation, public schema isolation, and claim wording.
- Added concise Docker Matrix claim wording to `docs/user/CLAIMS.md`.

## Task Commits

1. **Task 1 RED: Matrix execution guardrail tests** - `c8df6d8` (test)
2. **Task 1 GREEN: Matrix execution classifier** - `65d5cd5` (feat)
3. **Task 2 RED: Host/container claim tests** - `099eb73` (test)
4. **Task 2 GREEN: Host/container claim validation** - `b2d80d1` (feat)
5. **Task 3 RED: Public sidecar contract tests** - `6104d65` (test)
6. **Task 3 GREEN: Claim wording and guardrail text** - `e9c09d4` (fix)

## Files Created/Modified

- `src/sol_execbench/core/compatibility.py` - Pure execution decision model/helper and Matrix Entry claim-boundary validation.
- `tests/sol_execbench/test_matrix_claim_guardrails.py` - Mixed-version, debug override, status, host/container, and wording guardrails.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Canonical payload compatibility-sidecar exclusion guardrail.
- `docs/user/CLAIMS.md` - Docker Matrix container user-space claim boundary sentence.
- `docs/internal/v1_4_compatibility_inventory.md` - Restored exact public CLI invariant phrase used by existing guardrail tests.
- `.planning/REQUIREMENTS.md` - Restored exact MI300X-on-CDNA3 deferred-validation phrase used by existing guardrail tests.

## Decisions Made

- Kept classification pure and side-effect free: no Docker, uv, subprocess, runtime probe, CLI, trace, scoring, timing, or exit-semantics wiring.
- Used model validation for host/container claim separation so invalid entries fail at construction and JSON parse time.
- Preserved Docker validation wording as "container ROCm user-space validated on recorded host driver/devices"; native host validation requires direct host evidence.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Restored older public guardrail phrases required by existing tests**
- **Found during:** Task 3 (Prove compatibility sidecars stay noncanonical)
- **Issue:** The full public-contract guardrail test command exposed two existing text drift failures in `.planning/REQUIREMENTS.md` and `docs/internal/v1_4_compatibility_inventory.md`.
- **Fix:** Restored the exact invariant phrases without changing runtime behavior or public schemas.
- **Files modified:** `.planning/REQUIREMENTS.md`, `docs/internal/v1_4_compatibility_inventory.md`
- **Verification:** `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_matrix_claim_guardrails.py -q` passed.
- **Committed in:** `e9c09d4`

---

**Total deviations:** 1 auto-fixed (Rule 3).
**Impact on plan:** Verification blocker removed with metadata/docs text only; no code behavior, benchmark semantics, or public schema changes.

## Issues Encountered

Two pre-existing public guardrail text failures surfaced while running Task 3 verification. They were fixed as the Rule 3 deviation above.

## Known Stubs

None. Optional `None`, empty collection defaults, and empty artifact descriptions in `compatibility.py` are schema fields for not-yet-observed diagnostic evidence, inherited from Plan 78-01.

## Threat Flags

None. The plan threat model already covered mixed-version override, host validation overclaiming, benchmark blocking decisions, and canonical trace sidecar leakage.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_matrix_claim_guardrails.py -q` - passed, `9 passed in 1.13s` after Task 2 and `12 passed in 1.38s` for the final targeted set.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_matrix_claim_guardrails.py -q` - passed, `43 passed in 3.09s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py -q` - passed, `52 passed in 2.98s`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/compatibility.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py` - passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 79 can consume deterministic Matrix execution decisions and strict host/container claim validation before Docker Target selection and preflight work. Phase 80 can build on the mixed-version blocking and debug override semantics for PyTorch ROCm wheel policy.

## Self-Check: PASSED

- Found created file: `tests/sol_execbench/test_matrix_claim_guardrails.py`.
- Found modified file: `src/sol_execbench/core/compatibility.py`.
- Found modified file: `tests/sol_execbench/test_public_contract_guardrails.py`.
- Found modified file: `docs/user/CLAIMS.md`.
- Found task commits: `c8df6d8`, `65d5cd5`, `099eb73`, `b2d80d1`, `6104d65`, `e9c09d4`.
- Re-ran plan verification successfully.

---
*Phase: 78-matrix-contract-and-claim-guardrails*
*Completed: 2026-05-28*
