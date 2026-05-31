---
phase: 84-paper-denominator-accounting-and-claim-boundaries
plan: 02
subsystem: dataset-reporting
tags: [paper-denominator, script, sidecar, public-contracts, guardrails]
requires:
  - phase: 84-paper-denominator-accounting-and-claim-boundaries
    provides: paper denominator core sidecar helpers from plan 01
provides:
  - Thin `scripts/report_paper_denominator.py` wrapper
  - Public dataset helper exports for the paper denominator sidecar
  - CPU-safe script output coverage
  - Public contract guardrails keeping canonical schemas and primary CLI unchanged
affects: [dataset-sidecars, research-credibility-reports, public-contract-guardrails]
tech-stack:
  added: []
  patterns: [thin script wrapper over core helpers, script-only reporting options, canonical contract exclusion tests]
key-files:
  created:
    - scripts/report_paper_denominator.py
    - tests/sol_execbench/test_paper_denominator_script.py
  modified:
    - src/sol_execbench/core/dataset/__init__.py
    - tests/sol_execbench/test_public_contract_guardrails.py
key-decisions:
  - "Paper denominator reporting is exposed through a script and dataset helper exports, not the primary benchmark CLI."
  - "Optional AMD SOL and SOLAR artifacts are represented as bounded refs/checksums only."
patterns-established:
  - "Script wrappers should load local sidecars, delegate normalization/rendering to core helpers, and avoid becoming a second implementation."
  - "Guardrail tests assert sidecar fields remain absent from canonical Definition, Workload, and Trace payloads."
requirements-completed: [DENOM-01, DENOM-02, DENOM-03, DENOM-04, DENOM-05]
duration: 4min
completed: 2026-05-31
---

# Phase 84 Plan 02: Paper Denominator Script And Guardrails Summary

**Script-side paper denominator report generation with dataset exports and canonical contract guardrails**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-31T08:29:44Z
- **Completed:** 2026-05-31T08:34:07Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added `scripts/report_paper_denominator.py` as a thin argparse wrapper over the core paper denominator builder and writer.
- Exported `PaperDenominatorReport`, builder, Markdown renderer, and writer from `sol_execbench.core.dataset`.
- Added script tests proving deterministic JSON/Markdown output and bounded AMD SOL/SOLAR artifact refs.
- Added public guardrails proving canonical Definition, Workload, Trace payloads and primary `sol-execbench --help` do not expose paper denominator fields or script-only options.

## Task Commits

1. **Task 1: Add CPU-safe script output coverage** - `9199b16` (test)
2. **Task 2: Implement thin report script and dataset exports** - `73d239a` (feat)
3. **Task 3: Strengthen public contract guardrails** - `7e7e0a4` (test)

## Files Created/Modified

- `scripts/report_paper_denominator.py` - Argparse script that loads local sidecars, passes bounded artifact refs, and writes JSON/Markdown.
- `src/sol_execbench/core/dataset/__init__.py` - Dataset helper exports for paper denominator reports.
- `tests/sol_execbench/test_paper_denominator_script.py` - CPU-safe script integration test with deterministic `--created-at`.
- `tests/sol_execbench/test_public_contract_guardrails.py` - Phase 84 sidecar-only and primary CLI exclusion guardrails.

## Decisions Made

- Kept denominator reporting script-only for user-facing execution; no primary CLI options were added.
- Stored optional artifact paths and file checksums as source refs without reading AMD SOL/SOLAR payload arrays into the report.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The initial script RED test imported fixture helpers via a package path that was not importable under this test layout. It was corrected before the RED commit so the intended failure was the missing script file.

## User Setup Required

None - no external service configuration required.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_paper_denominator_script.py -q` - expected RED failure before implementation
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_paper_denominator_script.py tests/sol_execbench/test_paper_denominator_report.py -q` - 7 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only tests/sol_execbench/test_public_contract_guardrails.py::test_primary_cli_does_not_expose_v1_19_paper_denominator_options -q` - 2 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_paper_denominator_report.py tests/sol_execbench/test_paper_denominator_script.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only tests/sol_execbench/test_public_contract_guardrails.py::test_primary_cli_does_not_expose_v1_19_paper_denominator_options -q` - 9 passed

## Known Stubs

None.

## Threat Flags

None.

## Next Phase Readiness

Phase 84 is ready for verification. The paper denominator sidecar remains sidecar-only and can be consumed by later reporting/documentation phases without expanding hardware validation, Docker privilege, dependency lock, network, or dataset download scope.

## Self-Check: PASSED

- Found `scripts/report_paper_denominator.py`
- Found `src/sol_execbench/core/dataset/__init__.py`
- Found `tests/sol_execbench/test_paper_denominator_script.py`
- Found `tests/sol_execbench/test_public_contract_guardrails.py`
- Found commits `9199b16`, `73d239a`, and `7e7e0a4`

---
*Phase: 84-paper-denominator-accounting-and-claim-boundaries*
*Completed: 2026-05-31*
