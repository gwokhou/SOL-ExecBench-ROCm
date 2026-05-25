# Phase 75: Routed Static Extractor Adapters - Context

**Gathered:** 2026-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 75 implements routed, bounded static extractor adapters for artifacts
captured by Phase 74. Operators should be able to run optional static extraction
through the v1.16 toolchain routing layer, preserve bounded raw outputs, and
receive nonfatal diagnostic sidecars for success, partial success,
unavailability, timeout, nonzero exit, or parser failure.

This phase does not add public CLI flags, does not write final benchmark
sidecars from the main command, does not make RGA or `roc-objdump` mandatory,
and does not change canonical trace JSONL, correctness, timing, scoring,
paper-parity, or leaderboard semantics.

</domain>

<decisions>
## Implementation Decisions

### Tool Scope And Priority
- Execute only routed `llvm-objdump` and `readelf` adapters in Phase 75.
- Route all static extractor attempts through `src/sol_execbench/core/toolchain.py`
  instead of direct ad hoc executable lookup.
- Keep `roc-objdump` and RGA as candidate/planned/unavailable route records only;
  do not use either as an execution path in Phase 75.
- Promote `llvm-objdump` and `readelf` registry entries as needed so routing can
  probe and select them when available.

### Raw Output Preservation
- Every extractor attempt should preserve a bounded raw output artifact under
  the evidence directory.
- Tool-run records should include command provenance, timeout, return code,
  stdout/stderr tails, and raw output artifact path.
- Raw output must be bounded; do not embed or write unlimited stdout/stderr.

### Failure And Partial Success Semantics
- If either `llvm-objdump` or `readelf` succeeds, the aggregate sidecar may be
  `collected` or `partial` depending on completeness.
- Missing, timed-out, nonzero, unsupported, or parser-failed extractors are
  nonfatal tool-run records.
- If both executable extractor routes are unavailable, return `unavailable`.
- If both executable extractor routes are attempted and fail, return `failed`.
- Static extraction never affects benchmark correctness, timing, scoring,
  paper-parity, or leaderboard claims.

### the agent's Discretion
Use existing `ToolchainRoutingRequest`, `ToolchainRoutingReport`,
`ToolchainArtifactType`, and `ToolchainEvidenceLevel.STATIC` models. Keep parser
logic conservative: classify metadata/disassembly presence and raw output paths;
defer rich ISA taxonomy and resource parsing.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/bench/static_kernel_evidence.py` owns the strict
  sidecar models, artifact discovery/persistence helpers, and tool-run schema.
- `src/sol_execbench/core/toolchain.py` owns the v1.16 routing registry, probe
  runner injection points, and route decision vocabulary.
- `ToolchainEvidenceLevel.STATIC` and artifact types `ROCM_BINARY`,
  `ELF_OBJECT`, `HIP_COMPILER_OUTPUT`, and `STATIC_FUTURE` already exist.
- `llvm-objdump` and `readelf` registry entries exist but currently use planned
  lifecycle, which prevents probing/selection until Phase 75 promotes them.
- `tests/sol_execbench/test_toolchain_routing.py` covers injected `which` and
  probe-runner routing behavior.

### Established Patterns
- Optional diagnostic tool failures should return sidecar metadata, not raise
  benchmark failures.
- Runner/probe dependencies should be injectable for CPU-safe tests.
- Outputs included in sidecars should be tailed/bounded.
- Raw evidence files belong under the evidence directory and should be referenced
  through relative paths.

### Integration Points
- Phase 75 helpers should operate on Phase 74 `StaticKernelEvidenceArtifact`
  entries with persisted artifact paths.
- Toolchain route reports and decisions should be represented in static evidence
  source references and tool-run records without mutating canonical trace data.
- Phase 76 can later call the Phase 75 helper from CLI sidecar integration.

</code_context>

<specifics>
## Specific Ideas

- Add extractor command builders for:
  - `llvm-objdump --disassemble <artifact>`
  - `readelf --headers --wide <artifact>`
- Use `ToolchainArtifactType.ROCM_BINARY` for shared library, HSACO, and code
  object artifacts; use `ELF_OBJECT` for `.o`; avoid running extractors on
  compiler-output text artifacts.
- Preserve raw output as text files with deterministic names such as
  `extractors/<artifact-id>/<tool-id>.txt`.
- Record non-executable route decisions for RGA and `roc-objdump` in source
  references or warnings, but do not execute them.

</specifics>

<deferred>
## Deferred Ideas

- Public `--static-evidence` CLI integration and final sidecar writing.
- RGA and `roc-objdump` execution paths.
- RGA-rich VGPR/SGPR/LDS/scratch/resource parsing.
- Deep instruction-family classification.
- Triton ROCm cache extraction.
- Live RDNA 4 / CDNA validation.

</deferred>
