# Phase 128: MI300X Evidence Contract and Validation Handoff - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

This phase strengthens the future MI300X/gfx942 validation contract. It should
make the commands, artifacts, evidence gates, and expected result categories
clear enough for a future validator to execute on real hardware. It does not
run MI300X validation on the current machine.

</domain>

<decisions>
## Implementation Decisions

### Evidence Contract
- Treat MI300X as the concrete CDNA3 target (`gfx942`), not a separate
  architecture class.
- Required artifacts must include full-suite pytest evidence, dataset summary,
  environment report, clock-lock evidence, per-problem traces, timing evidence,
  AMD-native score report, and quantization status.
- `mi300x_validation_claim_blockers()` remains the programmatic gate for any
  hardware-validation claim.

### Command Sequence
- Preserve ROCm environment discovery commands before benchmark commands.
- Include `uv run pytest tests/`, `uv run scripts/run_dataset.py`, clock-lock
  settings, timing evidence directory, `--gpu-architecture gfx942`, and
  AMD-score report output.
- Use concrete output paths in docs while allowing dataset/baseline paths to be
  supplied by the validator.

### Failure And Skip Classification
- Distinguish expected skips, missing tools, functional failures, timing
  instability, missing evidence, and deferred quantization formats.
- Keep FP8 as ready for future MI300X validation when workload cases exist.
- Keep NVFP4/MXFP4 as `deferred_no_amd_path`.

</decisions>

<code_context>
## Existing Code Insights

- `src/sol_execbench/core/diagnostics.py` defines
  `CDNA3_VALIDATION_COMMANDS`, `CDNA3_EVIDENCE_REQUIRED`,
  `CDNA3_ACCEPTANCE_CRITERIA`, `MI300X_REQUIRED_ARTIFACTS`,
  `MI300X_FP8_READINESS`, and `mi300x_validation_claim_blockers()`.
- `tests/sol_execbench/test_rocm_diagnostics_reporting.py` already validates
  readiness helpers and MI300X claim blockers.
- `.planning/milestones/MI300X-VALIDATION-HANDOFF.md` and
  `docs/internal/mi300x_validation_readiness.md` contain the existing handoff
  text but can be made more explicit.

</code_context>

<specifics>
## Specific Ideas

- Expand `MI300X_REQUIRED_ARTIFACTS` to include trace, timing, AMD score, and
  quantization status artifacts.
- Add constants or doc sections for expected skip/failure categories.
- Strengthen tests so complete evidence must include every required artifact.
- Update internal and planning handoff docs with the same evidence chain and
  no-claim rule.

</specifics>

<deferred>
## Deferred Ideas

- Running the command sequence on actual MI300X/gfx942 hardware.
- Producing real timing traces or AMD-native score reports in this milestone.
- CDNA4 or NVFP4/MXFP4 hardware validation.

</deferred>
