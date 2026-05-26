# Phase 76: CLI Sidecar Integration And Reports - Context

**Gathered:** 2026-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 76 exposes the completed static evidence pipeline through the primary
benchmark CLI. Operators can opt in with `--static-evidence auto`, while the
default `--static-evidence none` keeps existing behavior unchanged.

This phase writes the JSON sidecar and evidence artifact directory, and prints a
short console status with the sidecar path. It does not add Markdown reports,
does not change canonical trace JSONL, and does not let static evidence affect
benchmark correctness, timing, scoring, paper-parity, leaderboard status, or
exit-code semantics except invalid CLI usage.

</domain>

<decisions>
## Implementation Decisions

### Runtime Trigger
- Run static evidence only for `--static-evidence auto`.
- Trigger the static evidence pipeline after HIP/C++ compilation succeeds and
  before staging cleanup.
- Default `--static-evidence none` performs no collection and writes no sidecar.
- Unsupported solution paths, including non-HIP/C++ solution types and no stable
  artifact boundary, should produce unsupported or unavailable diagnostic
  sidecars without changing benchmark exit code.

### Paths
- When trace output is configured, write:
  - `<trace>.static-evidence.json`
  - `<trace>.static-evidence/`
- When no trace output is configured, write under staging:
  - `static-evidence.json`
  - `static-evidence/`
- Evidence directories hold persisted build artifacts and raw extractor output.

### Human-Facing Summary
- CLI console prints one concise status line and the JSON sidecar path.
- Full summary fields/sections are included in the JSON sidecar payload.
- Do not generate a separate Markdown report in Phase 76.

### the agent's Discretion
Use nearby profiler/environment sidecar helpers for naming, writing, and
nonfatal behavior. Keep integration small and testable through helper functions
where possible.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/bench/static_kernel_evidence.py` owns strict sidecar
  models, artifact discovery/persistence, and routed extractor helpers.
- `src/sol_execbench/cli/main.py` already has profiler sidecar path/write
  helpers (`_profile_output_directory`, `_profile_sidecar_path`,
  `_write_profile_sidecar`) and the compile/evaluate sequencing needed for the
  new trigger point.
- `tests/sol_execbench/test_cli_environment_snapshot.py` covers sidecar helper
  path/write behavior with direct helper tests.
- `tests/sol_execbench/test_public_contract_guardrails.py` currently blocks
  `--static-evidence` because Phase 73/74/75 had not exposed it; Phase 76 should
  update that guardrail to allow the public flag while keeping canonical trace
  exclusions.

### Established Patterns
- Optional diagnostics are sidecar-only and nonfatal.
- Sidecars are written beside trace output when an output file is available.
- Missing optional diagnostics emit warnings/status, not benchmark truth changes.

### Integration Points
- HIP/C++ compile branch has the correct staging boundary and `artifact_path`.
- Evaluation still owns canonical trace generation and exit code.
- Static evidence sidecar writing should happen after evaluation output is
  available, so the sidecar path can be printed consistently and failure remains
  nonfatal.

</code_context>

<deferred>
## Deferred Ideas

- Markdown static evidence report.
- Dataset-level aggregate static evidence reports.
- RGA/`roc-objdump` execution.
- Rich ISA/resource parsing.
- Live hardware validation records.

</deferred>
