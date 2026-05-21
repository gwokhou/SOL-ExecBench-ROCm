# Phase 04: ROCm Library and Example Migration - Context

**Gathered:** 2026-05-21
**Status:** Ready for planning
**Source:** Autonomous smart discuss

<domain>
## Phase Boundary

Phase 4 migrates public examples and sample solution metadata from CUDA/NVIDIA-specific categories to ROCm-compatible examples or explicit feasibility notes.

Owned areas:
- `examples/` solution metadata, source files, and example category directories.
- `tests/sol_execbench/samples/` sample solution metadata and embedded source strings where they exercise public example categories.
- Documentation artifact in this phase directory describing replacement choices for rocBLAS/hipBLASLt, MIOpen, Composable Kernel, rocWMMA, hipCUB, rocPRIM, and rocThrust.
- Focused tests/audits proving migrated example/sample metadata no longer uses rejected Phase 2 schema values.

Deferred:
- Full test marker overhaul and hardware matrix validation remain Phase 5.
- README/schema/profiling public docs remain Phase 6 except for Phase 4's internal replacement-decision artifact.
- Broad algorithmic benchmarking of replacement kernels is out of scope; Phase 4 only needs credible examples and documented alternatives.
</domain>

<decisions>
## Implementation Decisions

### Migration Strategy
- Use the user-selected pragmatic migration strategy.
- PyTorch, Triton, and HIP/C++ examples should be made ROCm-schema-compatible and runnable where feasible.
- CUTLASS/cuDNN/CuTe DSL/cuTile-style categories should be reimplemented only when the replacement is simple and credible in this codebase.
- When a reliable ROCm implementation is too large or would be speculative, replace or quarantine the example with explicit feasibility notes rather than hard-coding a weak substitute.

### Schema Alignment
- Use Phase 2 schema names: `hip_cpp`, `hipblas`, `miopen`, `ck`, and `rocwmma`.
- Use `hip_cflags`, `.hip`/C++ source suffixes, and ROCm dependencies.
- Do not reintroduce `cuda_cpp`, `cuda_cflags`, CUTLASS/cuDNN/cuBLAS/CuTe/cuTile schema language values, or `.cu` entry points in migrated example metadata.

### Compatibility Notes
- PyTorch ROCm still exposes device APIs through `torch.cuda`; this is allowed where the surrounding code is ROCm-compatible.
- PyTorch native extension code may still refer to `at::cuda` namespaces when that is the PyTorch ROCm API surface.
- CUDA kernel syntax in HIP sources is acceptable when compiled by HIP/hipcc through PyTorch ROCm, but NVIDIA-only headers/types must be replaced.
</decisions>

<code_context>
## Relevant Existing State

- Phase 2 changed the solution schema and packager to reject legacy CUDA/NVIDIA language and compile-option values.
- Phase 3 changed eval-driver native shared-object routing to ROCm language enums.
- `examples/` still contains legacy directories:
  - `examples/cuda_cpp/`
  - `examples/cutlass/`
  - `examples/cudnn/`
  - `examples/cute_dsl/`
  - `examples/cutile/`
- `tests/sol_execbench/samples/` still contains legacy metadata:
  - `flux_rope/solution_cuda.json`
  - `rmsnorm/solution_cuda.json`
  - `jamba_attn_proj/solution_cute_dsl.json`
  - `jamba_attn_proj/solution_cutile.json`
- PyTorch and Triton examples mostly use schema-compatible `pytorch` and `triton` languages but may still contain NVIDIA-flavored names/descriptions.
</code_context>

<validation>
## Expected Validation

- Prefer `uv run --no-sync` for tests to avoid dependency sync/downloads.
- Run focused schema parsing tests for updated examples/samples.
- Add a source/metadata audit for Phase 4 example/sample paths.
- Hardware execution of examples is Phase 5, but metadata/schema parsing and source-level migration checks should pass now.
</validation>

<deferred_ideas>
## Deferred Ideas

- Full ROCm library performance tuning.
- RDNA4/CDNA3 execution matrix.
- Public-facing long-form documentation cleanup.
</deferred_ideas>
