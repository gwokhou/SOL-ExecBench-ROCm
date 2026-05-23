# Phase 52 Validation Strategy

**Phase:** Dataset Runner And Public Contract Closure  
**Requirements:** REPORT-04, TEST-04, TEST-05  
**Created:** 2026-05-23

## Goal-Backward Validation

Phase 52 is complete only when a user can run or inspect v1.10 derived
reporting surfaces and understand formula, hardware model, coverage, and score
eligibility evidence without any public-contract drift or unsupported claims.

## Must Prove

### REPORT-04

- Derived reports preserve `amd-native-derived` claim boundaries.
- Derived report artifacts include auditable references for formula evidence,
  hardware models, coverage evidence, and score eligibility.
- Missing SOLAR sidecars remain distinct from explicit unscored SOLAR evidence.
- Degraded SOLAR sidecars remain visible through warnings and score eligibility
  metadata.

### TEST-04

- Canonical `Definition`, `Workload`, and `Trace` schemas are unchanged.
- Primary `sol-execbench --help` behavior is unchanged.
- Canonical trace JSONL remains free of Phase 52 derived-report fields.
- Existing public benchmark semantics and existing report schemas remain
  compatible unless the plan explicitly scopes an opt-in derived artifact.

### TEST-05

- Docs, derived reports, examples, and guardrails do not imply:
  paper benchmark parity, original 124-model / 235-problem extraction,
  NVIDIA Blackwell or B200 equivalence, hosted leaderboard readiness, CDNA 3 /
  MI300X / CDNA 4 validation, NVFP4/MXFP4 validation, or new real-hardware
  validation.
- Positive claim language remains AMD-local and evidence-scoped.

## Focused Commands

```bash
uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py -k "solar or derivation or evidence or score or sidecar" -n 0 -x
uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -k "schema or cli or trace or solar or evidence or claim" -n 0 -x
uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_v1_9_validation_closure.py -k "claim or boundary or paper or leaderboard or validation or B200 or Blackwell" -n 0 -x
```

## Full Phase Gate

```bash
uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_v1_9_validation_closure.py tests/sol_execbench/test_solar_derivation_evidence.py -n 0
uv run --with ruff ruff check scripts/run_dataset.py src/sol_execbench/core/scoring/amd_score.py src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_v1_9_validation_closure.py docs/analysis.md docs/internal/solar_derivation_contract.md
```

## Manual Review Checks

- Inspect any new derived report JSON shape and confirm it is opt-in and not
  part of canonical trace JSONL.
- Inspect docs for claim wording: all B200/Blackwell/leaderboard/hardware
  validation mentions must be explicit non-claims or historical context.
- Inspect runner help and defaults: no primary benchmark CLI behavior should
  change, and dataset runner defaults should remain compatible.

## Out Of Scope For Validation

- Full original-paper dataset extraction.
- Real ROCm hardware execution.
- Candidate solution execution for derivation.
- Hosted service or leaderboard behavior.
