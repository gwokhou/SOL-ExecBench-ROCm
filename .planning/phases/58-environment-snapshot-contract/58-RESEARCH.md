# Phase 58 Research: Environment Snapshot Contract

**Researched:** 2026-05-25
**Status:** Complete

## Existing Code Shape

- `src/sol_execbench/core/data/trace.py` already has a canonical
  `Environment` model with only `hardware` and `libs`. Phase 58 should not
  expand it because v1.13 compatibility requires optional evidence outside the
  canonical trace schema.
- `src/sol_execbench/core/data/contract.py` is the right place to add an
  optional capability token. The existing contract version is `1.0`, and
  hip-playground-style consumers currently hard-code that version.
- `src/sol_execbench/core/diagnostics.py` already owns ROCm readiness helper
  ideas: `DiagnosticStage`, `StageDiagnostic`, `detect_tool`, `classify_gfx`,
  and readiness dataclasses. The environment snapshot collector should either
  live here or in a focused sibling module imported by diagnostics.
- `tests/conftest.py` already supports `requires_rdna4` and `requires_cdna3`
  skip behavior through PyTorch ROCm device properties. Phase 58 should reuse
  the same architecture vocabulary.

## Recommended Implementation Pattern

Create a focused module, likely
`src/sol_execbench/core/environment.py` or
`src/sol_execbench/core/bench/environment.py`, with:

- Pydantic models for JSON-stable snapshot evidence.
- Injectable command runner protocol for unit tests.
- `collect_environment_snapshot(...)` as the single high-level collection API.
- Small parser helpers for obvious fields only, with raw stdout/stderr tails
  preserved for later diagnosis.

The model should avoid overclaiming parser certainty. Good Phase 58 fields:

- `schema_version`
- `generated_at`
- `collection_status`
- `tools`
- `gpus`
- `rocm`
- `pytorch`
- `visible_devices`
- `warnings`

Tool result fields should include:

- `tool`
- `command`
- `path`
- `status`
- `returncode`
- `stdout_tail`
- `stderr_tail`
- `timeout_seconds`
- `parsed`

## Compatibility Risks

- Adding a new required contract capability would break hip-playground v1.12
  compatibility. The capability must be optional and additive.
- Adding fields to canonical `Trace` or changing `trace_field_requirements`
  would raise the risk of downstream strict parser failures.
- Running external probes at import time or contract-build time would make
  GPU-free commands less reliable. Collection must be explicit.

## Verification Focus

- Snapshot model round-trip tests.
- Probe runner tests for success, unavailable tool, nonzero exit, and timeout.
- Contract test proving `runtime.evidence.v1` appears while
  `contract_version == "1.0"`.
- Public guardrail test proving canonical trace schema does not acquire new
  environment evidence fields in Phase 58.

