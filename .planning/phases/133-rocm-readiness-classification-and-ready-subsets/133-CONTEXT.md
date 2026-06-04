# Phase 133: ROCm Readiness Classification and Ready Subsets - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

Classify migrated dataset workloads into bounded ROCm readiness categories and
produce auditable ready-subset/blocker sidecars. This phase does not execute
workloads, validate hardware behavior, establish paper parity, or grant score
authority.

</domain>

<decisions>
## Implementation Decisions

### Scope
- Extend existing dataset readiness and ready-subset helpers rather than adding
  a parallel classification system.
- Treat readiness as static, CPU-safe evidence for bounded execution attempts
  only.
- Preserve compatibility with existing `ready` status consumers while adding a
  richer `readiness_class` field for Phase 133 categories.

### Categories
- Classify workloads as `pytorch_compatible`, `rocm_port_needed`,
  `flashinfer_specific`, `nvfp4_blackwell_specific`, `unsupported`, or
  `blocked_missing_evidence`.
- Report CUDA dependencies, FlashInfer runtime assumptions, low-precision
  NVIDIA/Blackwell dependencies, missing blobs, and unsupported dtypes as
  deterministic blocker records.

### Ready Subsets
- Include only workloads statically safe to attempt on the current ROCm runner.
- Preserve denominator, workload identity, readiness checksum, excluded workload
  reasons, and closure input refs in the subset sidecar.

### Claim Boundaries
- Keep claim-boundary fields false for execution success, hardware validation,
  paper validation, hosted leaderboard parity, upstream SOLAR equivalence, and
  score authority.

</decisions>

<code_context>
## Existing Code Insights

- `src/sol_execbench/core/dataset/readiness.py` already builds deterministic
  readiness records from inventory sidecars.
- `src/sol_execbench/core/dataset/ready_subset.py` builds a deterministic
  ready-subset sidecar but currently lacks explicit exclusion details.
- `src/sol_execbench/core/dataset/migration.py` from Phase 132 emits explicit
  blockers for missing blobs, traces, and solutions; Phase 133 should classify
  outputs from that layout without depending on GPU access.
- `tests/sol_execbench/test_dataset_inventory_readiness.py` contains the nearest
  CPU-safe coverage for inventory, readiness, ready subset, and CLI sidecars.

</code_context>

<specifics>
## Specific Ideas

- Add stable enums/constants only where useful, but keep JSON strings simple and
  deterministic.
- Detect solution-language/runtime hints from inventory `solution_files` and
  problem category/path metadata without parsing restricted source dataset
  content beyond local generated artifacts.
- Keep tests synthetic and local-only.

</specifics>

<deferred>
## Deferred Ideas

- Real ROCm execution, hardware validation, and dataset runner integration are
  Phase 135 or later.
- Low-precision ROCm semantic implementation is Phase 134.

</deferred>
