# Phase 108: Native Compile Option Guardrails - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase constrains native HIP/C++ compile options so user-controlled flags
cannot easily reference host paths, response files, dynamic loaders, or unsafe
link-time behavior. It preserves existing documented ROCm/HIP extension flags.

</domain>

<decisions>
## Implementation Decisions

### Validation Location
- Enforce guardrails in `CompileOptions` schema validation so both CLI package
  loading and `build_ext.py` reject dangerous options consistently.
- Keep `build_ext.py` simple; it should continue consuming validated
  `Solution` models.
- Preserve existing ROCm-native language and entry-point validation.

### Flag Policy
- Allow established simple flags such as `-O3`, `-Wall`,
  `--offload-arch=gfx1200`, and `-lrocblas`.
- Reject response-file flags such as `@file` or linker response file forms.
- Reject include/library/sysroot/plugin/dynamic-loader/rpath style flags that
  introduce host path or runtime loader control.
- Prefer clear validation errors over silent filtering.

### Testing Boundary
- Add schema tests for accepted and rejected option classes.
- Add build-template tests proving rejected flags fail before reaching
  `torch.utils.cpp_extension.load()`.
- Do not attempt a full sandbox or compiler command parser in this phase.

### the agent's Discretion
The exact denylist can be conservative as long as it preserves existing examples
and keeps error messages clear enough to diagnose rejected flags.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CompileOptions` in `src/sol_execbench/core/data/solution.py` is the central
  schema for `cflags`, `hip_cflags`, and `ld_flags`.
- `build_ext.py` already constructs a `Solution` from `solution.json`, so schema
  validation applies during native compilation.
- Existing tests cover solution schema validation and build template flag
  forwarding.

### Established Patterns
- The ROCm schema rejects legacy CUDA/NVIDIA terms with explicit migration
  guidance.
- Validation errors are raised through Pydantic model validators.

### Integration Points
- `src/sol_execbench/driver/templates/build_ext.py` forwards
  `compile_options.hip_cflags`, `cflags`, and `ld_flags` directly to PyTorch
  extension loading.
- Existing examples use simple optimization and architecture/library flags.

</code_context>

<specifics>
## Specific Ideas

Use a narrow denylist that blocks high-risk path/loader/response-file flags
while preserving existing tests and examples.

</specifics>

<deferred>
## Deferred Ideas

- Full compiler sandboxing and syscall isolation remain out of scope.
- More expressive allowlist policy can be added later if real ROCm examples need
  additional safe flags.

</deferred>
