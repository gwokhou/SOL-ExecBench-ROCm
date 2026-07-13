# Phase 16: ROCm Library Category Readiness - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Clarify which ROCm native library categories are runnable today and which are
schema-recognized candidates or compatibility examples.
</domain>

<decisions>
## Implementation Decisions

- Do not remove schema values for `hipblas`, `miopen`, `ck`, or `rocwmma`; they
  are useful replacement intent markers.
- Stop advertising candidate categories as fully supported until runnable public
  examples and tests exist.
- Preserve former NVIDIA example directories only as compatibility examples when
  their actual implementation uses PyTorch ROCm.

### the agent's Discretion
The exact documentation layout is flexible, but README and solution docs must
point users to the support-level distinction.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/user/solution.md` already lists supported language values.
- `tests/sol_execbench/test_rocm_library_examples.py` validates example metadata.

### Established Patterns
- Compatibility examples keep historical directory names while using ROCm
  schema metadata.
- Documentation tests protect public wording.

### Integration Points
- Add a dedicated docs page and link it from README and solution schema docs.
</code_context>

<specifics>
## Specific Ideas

- Define supported, candidate, and compatibility-example levels.
- Assert no public example currently uses a candidate category as if runnable.
</specifics>

<deferred>
## Deferred Ideas

- Implementing actual hipBLAS/MIOpen/CK/rocWMMA examples is future feature work
  unless selected in a later phase.
</deferred>
