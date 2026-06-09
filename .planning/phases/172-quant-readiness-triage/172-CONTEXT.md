# Phase 172: Quant Readiness Triage - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 172 refines Quant readiness classification so the 33 Quant readiness
blockers distinguish true CUDA/NVIDIA dependencies from PyTorch
ROCm-compatible semantic references and preserve low-precision hardware
evidence boundaries. This phase must not modify original or migrated dataset
references to remove lexical false positives.

</domain>

<decisions>
## Implementation Decisions

### Hint Detection Rule

- **D-01:** Treat NVIDIA/CUDA hints as true blockers only when they appear as
  import/call/native source/solution dependency evidence.
- **D-02:** Examples of true blockers include `import cupy`,
  `torch.utils.cpp_extension.CUDAExtension`, `.cu`/`.cuh` native sources, and
  actual `cublas`, `cutlass`, or `nvrtc` runtime calls.
- **D-03:** Comments, docstrings, class names, variable names, and compatibility
  layout labels containing `cuda`, `cublas`, or related strings do not trigger a
  blocker by themselves.

### Quant Outcome Classes

- **D-04:** Removing a false-positive CUDA/NVIDIA hint must not automatically
  mark a Quant problem fully ready.
- **D-05:** Reclassify each Quant problem to either
  `ready_to_attempt_rocm_execution` or `needs_hardware_evidence` based on dtype,
  format, and whether the semantic reference can run under PyTorch ROCm.
- **D-06:** Low-precision hardware semantics remain separate from readiness;
  readiness movement is not validation or performance authority.

### False Positive Handling

- **D-07:** Do not rename or mutate original/migrated dataset references such as
  `CuBLASRefBlockwiseGemm` or `scale_w_cublas`.
- **D-08:** Fix false positives in inventory/readiness classification and record
  derived evidence that a lexical false positive was ignored.

### Low Precision Boundary

- **D-09:** Treat FP8 and NVFP4/MXFP4 separately.
- **D-10:** FP8 Quant references may enter ready or smoke-attempt paths on RDNA4
  when their semantic reference is PyTorch ROCm-compatible, while still avoiding
  full low-precision performance or hardware authority claims.
- **D-11:** NVFP4, MXFP4, and CDNA4-specific semantics remain deferred or
  hardware-evidence-needed until a complete architecture-specific evidence
  chain exists.

### Residual Evidence

- **D-12:** True CUDA-only Quant blockers must record source evidence including
  problem id, file path, matched token, match kind, and line number or short
  context.
- **D-13:** Problem-level reason codes alone are insufficient for true CUDA-only
  Quant blockers.

### the agent's Discretion

The agent may choose the exact static-analysis helper shape and evidence schema
as long as it is deterministic, tested, and does not execute dataset reference
code during inventory classification.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone and Phase Scope

- `.planning/PROJECT.md` - v1.34 milestone scope and claim boundaries.
- `.planning/REQUIREMENTS.md` - Requirements `QUANT-01` through `QUANT-04`.
- `.planning/ROADMAP.md` - Phase 172 deliverables and success criteria.
- `.planning/research/SUMMARY.md` - Quant readiness and claim-boundary research.

### Prior Context

- `.planning/phases/171-custom-input-coverage-recompute/171-CONTEXT.md` -
  Current coverage transition context; Phase 172 follows custom-input closure.

### Codebase Maps

- `.planning/codebase/ARCHITECTURE.md` - Dataset inventory/readiness architecture.
- `.planning/codebase/TESTING.md` - CPU-safe readiness and artifact test patterns.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/sol_execbench/core/dataset/inventory.py`: currently extracts
  `reference_runtime_hints`; `NVIDIA_RUNTIME_HINTS` includes broad lexical
  tokens.
- `src/sol_execbench/core/dataset/readiness.py`: currently blocks any
  `reference_runtime_hints` before low-precision/Quant evidence handling.
- `src/sol_execbench/core/dataset/low_precision.py`: existing compatibility and
  unvalidated-CDNA4 evidence helpers.
- `tests/sol_execbench/test_dataset_inventory_readiness.py`: existing tests for
  CUDA compatibility text, true NVIDIA runtime hints, Quant hardware evidence,
  and CUDA solution dependencies.

### Established Patterns

- Inventory classification must remain static and must not execute reference
  code.
- Readiness records should include blocker reason codes, blocker types, and
  next actions.
- Low-precision compatibility and hardware validation boundaries are already
  modeled separately from ordinary readiness.

### Integration Points

- Inventory hint extraction should preserve context needed by readiness to
  distinguish imports/calls/native sources from lexical false positives.
- Readiness should route Quant problems after hint triage into precise classes:
  ready, hardware-evidence-needed, or ROCm-port-needed.

</code_context>

<specifics>
## Specific Ideas

- Keep original dataset wording intact; solve false positives in project-owned
  classifier logic and derived evidence.
- Quant readiness improvements should reduce overblocking without implying
  CDNA4 validation or leaderboard-ready low-precision authority.

</specifics>

<deferred>
## Deferred Ideas

- Performance optimization for Quant kernels is out of scope.
- CDNA4 validation remains deferred to future architecture-specific evidence.
- Final all-114 readiness closure reporting is deferred to Phase 174.

</deferred>

---

*Phase: 172-Quant Readiness Triage*
*Context gathered: 2026-06-09*

