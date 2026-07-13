---
phase: 88-documentation-examples-and-guardrail-tests
plan: 01
subsystem: documentation
tags: [v1.19, evidence-guide, docs-guardrails, sidecars, cpu-safe-tests]
requires:
  - phase: 83-closure-contracts-and-provenance-foundation
    provides: execution closure sidecar/reporting surface
  - phase: 84-paper-denominator-accounting-and-claim-boundaries
    provides: paper denominator reporting and claim-boundary vocabulary
  - phase: 85-compatibility-matrix-schema-export-and-semantic-diff
    provides: Matrix schema export and semantic diff surfaces
  - phase: 87-amd-sol-solar-bound-sanity-evidence
    provides: AMD bound sanity reporting surface
provides:
  - Central v1.19 evidence guide covering all Phase 83-87 evidence surfaces
  - Public docs links from CLAIMS, TESTING, and RESEARCHER-GUIDE
  - CPU-safe docs wording guardrails for v1.19 claim boundaries
affects: [docs, testing, public-contract-guardrails, v1.19-release]
tech-stack:
  added: []
  patterns: [focused Path.read_text docs tests, sidecar-only claim-boundary wording]
key-files:
  created:
    - docs/internal/v1_19_evidence_guide.md
  modified:
    - docs/user/CLAIMS.md
    - docs/user/TESTING.md
    - docs/user/RESEARCHER-GUIDE.md
    - tests/sol_execbench/test_research_release_docs.py
key-decisions:
  - "v1.19 evidence guidance is centralized in docs/internal/v1_19_evidence_guide.md and linked from public entry docs."
  - "v1.19 documentation remains sidecar/report-only and explicitly denies paper, SOLAR, score, leaderboard, native-host Matrix, CDNA/MI300X/CDNA4, and new-hardware authority."
patterns-established:
  - "Docs guardrails use focused public-entry reads instead of broad repository scans."
requirements-completed: [DOCS-01, DOCS-02, DOCS-03, DOCS-05]
duration: 16min
completed: 2026-05-31
---

# Phase 88 Plan 01: Central Evidence Guide Summary

**Central v1.19 evidence guide with public entry links and CPU-safe wording guardrails for sidecar-only evidence interpretation**

## Performance

- **Duration:** 16 min
- **Started:** 2026-05-31T11:46:24Z
- **Completed:** 2026-05-31T12:02:00Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Added `docs/internal/v1_19_evidence_guide.md` covering execution closure, paper denominator reports, Matrix schema export, Matrix semantic diff, and AMD bound sanity.
- Linked `docs/user/CLAIMS.md`, `docs/user/TESTING.md`, and `docs/user/RESEARCHER-GUIDE.md` to the central guide with explicit v1.19 negative claim boundaries.
- Added focused docs tests that keep v1.19 guide coverage and public entry wording CPU-safe.

## Task Commits

1. **Task 1: Add focused docs guardrails for the central guide** - `544d1f6` (test RED)
2. **Tasks 2-3: Write guide and link public docs** - `986776d` (docs GREEN)

_Note: Tasks 2 and 3 were committed together because the RED tests intentionally require both the guide and public entry links to pass._

## Files Created/Modified

- `docs/internal/v1_19_evidence_guide.md` - Central v1.19 guide and CPU-safe verification commands.
- `docs/user/CLAIMS.md` - v1.19 sidecar evidence claim level and explicit denied authorities.
- `docs/user/TESTING.md` - Focused CPU-safe v1.19 docs/contract guardrail command.
- `docs/user/RESEARCHER-GUIDE.md` - Researcher workflow references for v1.19 evidence surfaces.
- `tests/sol_execbench/test_research_release_docs.py` - Focused v1.19 docs wording tests.

## Decisions Made

- Central guide uses demo output paths only; no real hardware results or performance examples were added.
- Claim boundaries are repeated verbatim in the guide and linked docs so tests can guard against accidental overclaiming.

## Deviations from Plan

None requiring scope change. Tasks 2 and 3 were combined into one GREEN commit because the TDD assertions encoded both guide content and link visibility.

## Issues Encountered

- A line wrap initially prevented one literal contract-semantics phrase from matching. The guide wording was adjusted and verified with the focused docs test.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_research_release_docs.py -q` - 13 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_rocm_matrix_docs.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_amd_bound_sanity_fields_remain_sidecar_only -q` - 22 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/sol_execbench/test_research_release_docs.py` - passed.

## Known Stubs

None.

## Threat Flags

None. This plan added documentation and tests only; it introduced no new network endpoint, auth path, file access behavior, schema change, GPU probe, Docker probe, dependency relock, or hardware validation surface.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 88-02 can add demo fixture reports that point readers back to this guide and reuse the same claim-boundary wording.

## Self-Check: PASSED

- Created file exists: `docs/internal/v1_19_evidence_guide.md`.
- Commits exist: `544d1f6`, `986776d`.

---
*Phase: 88-documentation-examples-and-guardrail-tests*
*Completed: 2026-05-31*
