# 41-03 Summary

## Wave
03

## Scope Completed
- Extended `tests/sol_execbench/test_public_contract_guardrails.py` to assert primary CLI does not expose hardware-model-path or derived-workflow flags and to verify definition/workload/trace schemas stay without derived artifact fields.
- Added split-status/no-claim alignment checks in the same guardrail suite.
- Updated `docs/internal/analysis.md` with v1.9 AMD scope language, split status wording, and deferred CDNA3/MS300X claims.

## Acceptance
- Public contract guardrails now cover CLI/help and documentation claims:
  - no `--hardware-model`, `--amd-hardware-model`, `--hardware-model-path`, `--sol-bound`, or `--amd-score-report` in primary help,
  - retained canonical trace/schema behavior,
  - explicit non-equivalence phrasing for NVIDIA B200/SOLAR/leaderboard and deferred MI300X-on-CDNA3 and CDNA4 validation.

## Verification
- `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -x`
- `uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py -x`
