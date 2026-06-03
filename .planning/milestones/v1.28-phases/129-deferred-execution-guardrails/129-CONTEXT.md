# Phase 129: Deferred-Execution Guardrails - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

This phase strengthens CPU-safe public-contract guardrails after the new CDNA3
test and MI300X evidence readiness work. The intended outcome is that readiness
cannot be mistaken for completed CDNA3/MI300X hardware validation.

</domain>

<decisions>
## Implementation Decisions

### Guardrail Scope
- Keep guardrails CPU-safe and source/docs based.
- Check public docs, project planning docs, score warnings, and readiness
  helpers for no-claim wording.
- Update stale guardrail paths to the archived handoff files under
  `.planning/milestones/`.

### Claim Boundary
- CDNA3 readiness and concrete `requires_cdna3` tests are allowed.
- CDNA3 hardware validation, full MI300X validation, CDNA4 validation, and
  paper-scale validation remain disallowed without evidence.
- `CDNA3_NO_VALIDATION_WARNING` remains required on CDNA3 score artifacts.

</decisions>

<code_context>
## Existing Code Insights

- `tests/sol_execbench/test_public_contract_guardrails.py` contains broad
  public-claim tests and had a stale `.planning/CDNA3-VALIDATION-HANDOFF.md`
  reference after GSD health moved handoff files into `.planning/milestones/`.
- `tests/sol_execbench/test_amd_native_score.py` already checks
  `CDNA3_NO_VALIDATION_WARNING` on unsupported CDNA3 score reports.
- `tests/sol_execbench/test_rocm_diagnostics_reporting.py` checks claim
  blockers for missing MI300X evidence.

</code_context>

<specifics>
## Specific Ideas

- Update stale handoff paths in public guardrails.
- Assert v1.28 project/requirements/roadmap say readiness/testing but defer
  actual hardware execution.
- Keep warning text independent of old milestone version strings.

</specifics>

<deferred>
## Deferred Ideas

- Real CDNA3/MI300X evidence production.
- Any public status upgrade from deferred/readiness to hardware-validated.

</deferred>
