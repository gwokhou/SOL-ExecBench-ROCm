# Phase 73: Static Evidence Contract And Guardrails - Context

**Gathered:** 2026-05-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 73 defines the strict diagnostic-only
`sol_execbench.static_kernel_evidence.v1` sidecar contract, authority boundary
fields, status and reason semantics, optional evaluator contract capability
metadata, and no-mutation guardrails. It does not implement artifact discovery,
extractor subprocesses, CLI flags, report rendering, live ROCm validation,
RGA-rich parsing, or Triton cache capture.

</domain>

<decisions>
## Implementation Decisions

### Sidecar Contract Shape
- Put the sidecar schema and helpers in
  `src/sol_execbench/core/bench/static_kernel_evidence.py` so diagnostic static
  evidence follows the existing profiler sidecar boundary and stays separate
  from canonical trace/data schemas.
- Represent authority boundaries with explicit schema booleans:
  `diagnostic_only=true`, `correctness_authority=false`,
  `performance_authority=false`, `timing_authority=false`,
  `score_authority=false`, `paper_parity_authority=false`, and
  `leaderboard_authority=false`.
- Use strict Pydantic models with stable enums and round-trip tests so
  downstream consumers cannot silently accept ambiguous sidecar payloads.
- Serialize skipped, unavailable, unsupported, partial, and failed outcomes as
  full valid sidecars with status and reason codes, even when no artifact is
  collected.

### Status And Reason Semantics
- Lock the top-level status vocabulary to `collected`, `partial`,
  `unavailable`, `unsupported`, `failed`, and `skipped`; subtype details belong
  in reason codes.
- Model reason codes as stable enum-like string values grouped by category,
  with future values added deliberately through tests.
- Keep artifact, tool-run, kernel, warning, and source-reference fields present
  as stable shapes, using empty lists when the section has no entries.
- Define conservative optional classification fields now, including metadata
  presence, disassembly presence, detected architectures, and symbol count; the
  extractor phases populate them later.

### Integration Guardrails
- Expose `static_kernel_evidence.v1` through existing evaluator contract
  optional capability metadata without changing the required evaluator contract
  version.
- Add negative guardrails proving canonical trace JSONL dumps, default CLI
  behavior, scoring artifacts, and sidecar generation remain isolated from the
  new static-evidence contract.
- Do not add CLI flags in Phase 73. Public `--static-evidence none|auto`
  belongs to Phase 76 after contract, discovery, and extractor plumbing exist.
- Explicitly defer artifact discovery, extractor subprocesses, report
  rendering, live ROCm validation, RGA-rich resource parsing, and Triton cache
  capture.

### the agent's Discretion
No open implementation choices require user input. Use nearby Pydantic,
sidecar, contract, and test patterns to choose exact class names and helper
function names.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/bench/rocm_profiler.py` provides the closest
  diagnostic sidecar pattern for request/result models, status helpers, artifact
  registration, command provenance, and JSON payload generation.
- `src/sol_execbench/core/toolchain.py` provides the existing routing report
  and status vocabulary that later static extractor phases will consume.
- `src/sol_execbench/core/data/contract.py` owns evaluator contract metadata
  and optional capability advertisement.
- Existing tests in `tests/sol_execbench/test_contract.py` and
  `tests/sol_execbench/test_toolchain_routing.py` show local contract and
  schema serialization guardrail style.

### Established Patterns
- Diagnostic evidence is sidecar-only and must not mutate canonical trace JSONL
  or scoring artifacts.
- Public contracts use explicit schema version strings and JSON-serializable
  model dumps.
- Unsupported, unavailable, and failed states are expected outcomes rather than
  benchmark failures when optional evidence is requested.

### Integration Points
- New contract models should be importable from `sol_execbench.core.bench` or
  directly from the new module without changing the primary trace models.
- Evaluator contract optional capability metadata should include
  `static_kernel_evidence.v1` while preserving the required contract version.
- Tests should cover schema round trips, authority flags, optional capability
  exposure, and negative guardrails around canonical trace/scoring behavior.

</code_context>

<specifics>
## Specific Ideas

- Keep Phase 73 focused on contracts and guardrails only.
- Use full valid sidecars for skipped and unavailable states so future CLI and
  extractor phases can rely on stable JSON shapes.
- Prefer conservative names and booleans over heuristic metrics in this phase.

</specifics>

<deferred>
## Deferred Ideas

- Artifact discovery from HIP/C++ staging/build trees.
- Durable evidence directory and artifact manifest population.
- Routed `llvm-objdump`, `readelf`, RGA, or `roc-objdump` subprocess execution.
- Public CLI flags and human-facing report rendering.
- Live ROCm validation artifacts.
- RGA-rich resource parsing and Triton ROCm cache capture.

</deferred>
