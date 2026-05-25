# Phase 63 Plan

## Objectives

- Update ROCm user docs and timing semantics docs.
- Advertise optional profiling capability without a contract version bump.
- Keep reporting sidecar-oriented and out of canonical trace JSONL.

## Tasks

- [x] Document `--profile rocprofv3`, output locations, and Docker
  requirements.
- [x] Add optional `profiling.evidence.v1` contract capability.
- [x] Update tests for optional evidence, CLI help, and public guardrails.

## Verification

- Targeted pytest suite passed.
- Ruff passed.
