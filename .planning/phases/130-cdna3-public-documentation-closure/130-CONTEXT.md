# Phase 130: CDNA3 Public Documentation Closure - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

This phase updates public and contributor-facing documentation for the CDNA3
test readiness work from Phases 127-129. It explains how to run CDNA3 tests,
how to interpret skips, where to add future tests, and which evidence is needed
before validation claims can change.

</domain>

<decisions>
## Implementation Decisions

### Documentation Surface
- Update `docs/TESTING.md` for marker commands, expected skips, and current-host
  limits.
- Update `docs/rocm.md` for support matrix wording: schema support, test
  readiness, deferred MI300X validation, unavailable CDNA4.
- Update `CONTRIBUTING.md` for future CDNA3 test placement and evidence rules.

### Claim Boundary
- State that skipped CDNA3 tests on non-CDNA3 hosts do not mean validation
  failure.
- State that passing the lightweight `requires_cdna3` marker test is readiness
  evidence only.
- Keep full MI300X/gfx942 validation gated on the handoff evidence chain.

</decisions>

<code_context>
## Existing Code Insights

- `docs/TESTING.md` already lists `requires_cdna3` commands but does not explain
  the new concrete marker test surface or skip interpretation.
- `docs/rocm.md` already has support matrix rows for MI300X/CDNA3 and CDNA4.
- `CONTRIBUTING.md` lists markers but does not explain where future CDNA3 tests
  belong or what evidence must be recorded.
- `tests/sol_execbench/test_rocm_matrix_docs.py` and
  `tests/sol_execbench/test_rocm_support_docs.py` are good homes for
  documentation guardrails.

</code_context>

<specifics>
## Specific Ideas

- Add a CDNA3 subsection under Hardware-Sensitive Tests.
- Add a short paragraph to `docs/rocm.md` after Hardware Status.
- Add contributor bullets for `tests/sol_execbench/test_cdna3_hardware_marker.py`
  and handoff evidence.
- Extend docs tests to assert these public instructions exist.

</specifics>

<deferred>
## Deferred Ideas

- Publishing actual MI300X validation artifacts.
- Changing public status to hardware-validated.

</deferred>
