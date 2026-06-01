---
phase: 114-release-candidate-validation
plan: 1
subsystem: release
tags: [release-candidate, validation, evidence, docs, pytest]
requires:
  - phase: 113-deterministic-dataset-sharding-path
    provides: deterministic dataset shard and merge semantics for bounded release validation context
provides:
  - bounded engineering-prerelease validation wrapper
  - release-candidate validation documentation
  - CPU-safe tests for validation summaries, optional evidence, dataset bounds, and claim boundaries
affects: [support-matrix, claim-boundaries, first-run-docs, release-materials]
tech-stack:
  added: []
  patterns:
    - command-composition wrapper with deterministic JSON/Markdown sidecars
    - optional evidence classified as deferred or diagnostic-only instead of authority
key-files:
  created:
    - docs/release_candidate_validation.md
    - scripts/release_candidate_validation.py
    - tests/sol_execbench/test_release_candidate_validation.py
  modified:
    - .planning/REQUIREMENTS.md
key-decisions:
  - "Release-candidate validation composes existing commands and sidecars rather than changing benchmark schemas."
  - "Optional ROCm, Docker, dataset, and trust-summary evidence can be unavailable, skipped, deferred, or diagnostic-only."
  - "MI300X is treated as the CDNA3 hardware target; CDNA4 remains unavailable."
patterns-established:
  - "Release validation summaries use status plus classification plus next_action for every check."
  - "Shareable validation logs are bounded and redact token-like values."
requirements-completed: [RCVAL-01, RCVAL-02, RCVAL-03, RCVAL-04]
duration: 35min
completed: 2026-06-01
---

# Phase 114: Release-Candidate Validation Summary

**Bounded engineering-prerelease validation wrapper with deterministic sidecars, optional evidence classification, and claim-boundary docs**

## Performance

- **Duration:** 35 min
- **Started:** 2026-06-01
- **Completed:** 2026-06-01
- **Tasks:** 3 planned tasks plus code-review fix pass
- **Files modified:** 5

## Accomplishments

- Added `scripts/release_candidate_validation.py`, which writes deterministic
  `release_candidate_validation.json` and `.md` summaries.
- Added CPU-safe, ROCm smoke, Docker smoke, and bounded dataset-slice command
  composition without changing canonical benchmark schemas.
- Added validation result statuses and classifications:
  `passed`, `failed`, `skipped`, `unavailable` plus `blocking`, `deferred`,
  and `diagnostic-only`.
- Added maintainer documentation for CPU-safe validation, optional smoke
  evidence, bounded dataset slices, failure classification, and claim
  boundaries.
- Added focused tests for summary shape, redaction, optional evidence,
  bounded dataset command overrides, trust-summary gating, and documentation
  guardrails.

## Task Commits

1. **Plan metadata** - `2e4d000` (docs)
2. **Implementation** - `3709204` (feat)
3. **Review fixes** - `2ca5261` (fix)

## Files Created/Modified

- `scripts/release_candidate_validation.py` - Bounded prerelease validation
  wrapper and JSON/Markdown summary writer.
- `docs/release_candidate_validation.md` - Maintainer-facing prerelease
  validation guide and failure policy.
- `tests/sol_execbench/test_release_candidate_validation.py` - CPU-safe
  regression coverage.
- `.planning/REQUIREMENTS.md` - Preserved legacy hardware-validation guardrail
  wording while recording the MI300X/CDNA3 interpretation.
- `.planning/phases/114-release-candidate-validation/114-REVIEW.md` - Code
  review report with findings and remediation context.

## Decisions Made

- Kept this phase as command composition and sidecar reporting rather than
  adding benchmark authority surfaces.
- Classified optional live ROCm, Docker, and dataset evidence as deferred or
  diagnostic-only when unavailable.
- Required positive bounded dataset limits and validated dataset command
  overrides so release summaries cannot claim bounded evidence for an
  unbounded command.
- Skipped trust-summary generation when the required execution closure sidecar
  is missing.

## Deviations from Plan

### Auto-fixed Issues

**1. [Code Review - Blocker] Hardened persisted log redaction**
- **Found during:** Phase 114 code review.
- **Issue:** Token redaction missed common credential formats and
  `--log-tail-chars` was parsed but ignored.
- **Fix:** Expanded credential/header redaction, made log tail length
  configurable, and added tests for common secret formats and zero-length log
  capture.
- **Files modified:** `scripts/release_candidate_validation.py`,
  `tests/sol_execbench/test_release_candidate_validation.py`
- **Verification:** Focused release-candidate validation tests passed.
- **Committed in:** `2ca5261`

**2. [Code Review - Warning] Enforced bounded dataset command overrides**
- **Found during:** Phase 114 code review.
- **Issue:** `--dataset-command` could bypass required `--limit`, `--rerun`,
  or `--execution-closure` while the summary still claimed bounded evidence.
- **Fix:** Added override validation and regression tests.
- **Files modified:** `scripts/release_candidate_validation.py`,
  `tests/sol_execbench/test_release_candidate_validation.py`
- **Verification:** Focused release-candidate validation tests passed.
- **Committed in:** `2ca5261`

**3. [Code Review - Warning] Gated trust-summary execution on closure existence**
- **Found during:** Phase 114 code review.
- **Issue:** Trust summary could run after a failed dataset command and produce
  noisy secondary failures.
- **Fix:** Trust summary now runs only when the dataset command passes and
  `execution_closure.json` exists; otherwise it records a skipped diagnostic row.
- **Files modified:** `scripts/release_candidate_validation.py`,
  `tests/sol_execbench/test_release_candidate_validation.py`
- **Verification:** Focused release-candidate validation tests passed.
- **Committed in:** `2ca5261`

**Total deviations:** 3 auto-fixed review findings.
**Impact on plan:** All fixes strengthen the planned release-evidence boundary;
no scope expansion.

## Issues Encountered

- Existing public contract guardrails required a legacy out-of-scope phrase for
  CDNA3/MI300X/CDNA4 validation. The requirements wording was updated to keep
  the old guardrail phrase while clarifying that MI300X is the CDNA3 hardware
  target for v1.25 interpretation.

## User Setup Required

None. Optional ROCm, Docker, and dataset-slice validation still require a host
with matching hardware/tools or downloaded dataset assets.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_release_candidate_validation.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_research_release_docs.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_release_candidate_validation.py tests/sol_execbench/test_run_docker_runtime_evidence.py -q`
- `uv run --with ruff ruff check scripts/release_candidate_validation.py tests/sol_execbench/test_release_candidate_validation.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/release_candidate_validation.py --output-dir out/release_candidate_validation_phase114`

## Next Phase Readiness

Phase 115 can build the support matrix around the release-candidate validation
language and artifacts introduced here.

---
*Phase: 114-release-candidate-validation*
*Completed: 2026-06-01*
