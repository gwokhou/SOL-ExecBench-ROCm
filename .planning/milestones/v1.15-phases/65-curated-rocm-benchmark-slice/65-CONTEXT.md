# Phase 65: Curated ROCm Benchmark Slice - Context

**Gathered:** 2026-05-25
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Define and exercise a representative, bounded ROCm benchmark slice that can be
reproduced without implying full paper parity.
</domain>

<decisions>
## Implementation Decisions

- Define the slice as documentation and reproducibility contract first; do not
  add a second benchmark runner.
- Use existing examples and samples as the stable release-preview seed.
- Treat missing ROCm library dependencies as explicit unavailable states.
</decisions>

<code_context>
## Existing Code Insights

- Repository examples already cover PyTorch, Triton, HIP/C++, hipBLAS, CK,
  rocWMMA, and MIOpen paths.
- `scripts/run_dataset.py` already supports bounded execution and derived
  evidence sidecars.
</code_context>

<specifics>
## Specific Ideas

- Add `docs/curated_rocm_slice.md`.
- Include selection criteria, initial problem list, commands, optional evidence,
  and artifact expectations.
</specifics>

<deferred>
## Deferred Ideas

- Do not require full 235-problem execution.
- Do not require unavailable hardware/library paths to pass.
</deferred>

