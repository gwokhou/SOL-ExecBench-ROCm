# Phase 127: CDNA3 Hardware-Gated Test Surface - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

This phase makes the `requires_cdna3` marker actionable by adding at least one
real hardware-gated pytest path and CPU-safe audit coverage. It does not execute
MI300X validation on the current machine and does not upgrade any CDNA3
hardware-validation claim.

</domain>

<decisions>
## Implementation Decisions

### Hardware-Gated Coverage
- Add a concrete test that uses `@pytest.mark.requires_cdna3` and asserts live
  detected ROCm architecture is `gfx94*` when the test actually runs.
- Combine `requires_cdna3` with `requires_rocm` where appropriate so
  non-ROCm hosts skip through the existing `tests/conftest.py` skip machinery.
- Keep the live test lightweight and non-destructive. It should prove marker
  selection and architecture gating, not perform benchmark-scale validation.

### CPU-Safe Audit
- Add CPU-safe tests that inspect the source tree and fail if `requires_cdna3`
  remains only registered but unused by concrete tests.
- Keep schema/build metadata checks separate from hardware validation checks.
- Use existing `tests/sol_execbench/test_rocm_test_suite_audit.py` patterns for
  source-text guardrails.

### Deferred Validation Boundary
- Do not add `requires_mi300x` or `requires_cdna4` markers.
- Do not mark MI300X or CDNA3 full-suite validation complete.
- Preserve current skip behavior on this machine because real `gfx94*`
  hardware is unavailable.

</decisions>

<code_context>
## Existing Code Insights

- `tests/conftest.py` already registers `requires_cdna3` and skips it unless
  `_is_cdna3(gfx_arch)` returns true.
- `tests/sol_execbench/test_rocm_test_suite_audit.py` already verifies marker
  wording and no-claim boundaries.
- `tests/sol_execbench/driver/test_problem_packager.py` already checks
  `gfx940`, `gfx941`, and `gfx942` offload flag injection in CPU-safe tests.
- Current repository search showed no actual `@pytest.mark.requires_cdna3`
  usage before this phase.

</code_context>

<specifics>
## Specific Ideas

- Add a small `tests/sol_execbench/test_cdna3_hardware_gate.py` file for the
  live CDNA3 marker path.
- Extend test-suite audit coverage to assert concrete `requires_cdna3` usage
  exists outside registration/docs.
- Run focused CPU-safe tests after changes; live CDNA3 execution remains
  deferred and should skip on this host.

</specifics>

<deferred>
## Deferred Ideas

- Full MI300X/gfx942 adapted-suite execution.
- Dataset-scale MI300X validation and timing/score artifact production.
- CDNA4 marker or validation support.

</deferred>
