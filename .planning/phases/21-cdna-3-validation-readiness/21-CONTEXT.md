# Phase 21: CDNA 3 Validation Readiness - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Auto-generated (autonomous infrastructure phase)

<domain>

## Phase Boundary

Phase 21 implements CDNA 3 validation readiness for future `gfx94*` hardware
runs. It must not claim CDNA 3 hardware validation and must not require real
CDNA 3 hardware to pass its unit tests.

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

- Reuse `src/sol_execbench/core/diagnostics.py` for architecture/tool readiness.
- Return readiness metadata: architecture family, expected commands, evidence
  requirements, blockers, acceptance criteria, and claim wording.
- Treat CDNA 3 readiness as implementable on any host; real validation remains
  future scope until a `gfx94*` full-suite run is recorded.

</decisions>

<code_context>

## Existing Code Insights

- `classify_gfx` already distinguishes `gfx94*`, `gfx12*`, and unknown targets.
- `rocm_tool_diagnostics` already reports tool availability with hints.
- `.planning/CDNA3-VALIDATION-HANDOFF.md` already records command/evidence
  expectations for a future CDNA 3 run.
- Existing docs repeatedly warn that CDNA 3 hardware validation is deferred.

</code_context>

<specifics>

## Specific Ideas

- Add a pure `cdna3_validation_readiness` helper in diagnostics.
- Add docs and tests that use "readiness implemented" wording, not "validated".
- Unit-test RDNA4, CDNA3, unknown, and missing-tool cases without hardware.

</specifics>

<deferred>

## Deferred Ideas

Real CDNA 3 full-suite execution and support-matrix claim update remain future
scope.

</deferred>
