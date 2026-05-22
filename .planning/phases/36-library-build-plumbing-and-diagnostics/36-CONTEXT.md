# Phase 36: Library Build Plumbing and Diagnostics - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

Make dependency detection, compile metadata, Docker docs, and native staging
ready for MIOpen, Composable Kernel, and rocWMMA.

This phase does not create full runnable MIOpen/CK/rocWMMA examples. It creates
the shared plumbing and tests needed by later phases.
</domain>

<decisions>
## Implementation Decisions

- Preserve public solution schema and trace JSONL.
- Use existing `compile_options` fields for include, HIP, and linker flags.
- Add reusable dependency diagnostics in the internal diagnostics module rather
  than adding a public CLI.
- Keep validation claims scoped to RDNA 4 for v1.8.
</decisions>

<code_context>
## Existing Code Insights

- `Solution` already accepts `miopen`, `ck`, and `rocwmma`.
- `ProblemPackager` already treats these categories as native C++ languages.
- `build_ext.py` already passes `hip_cflags`, `cflags`, and `ld_flags`.
- Existing docs identify these categories as candidates; phase 36 should make
  missing dependency reporting explicit before later phases promote support.
</code_context>

<specifics>
## Specific Ideas

- Add `RocmLibrarySpec` and readiness helpers for headers/libraries/packages.
- Extend Docker dependency tests to check hipBLAS, MIOpen, CK, and rocWMMA.
- Add staging tests proving candidate categories use native packaging.
- Document required headers, libraries, and packages.
</specifics>

<deferred>
## Deferred Ideas

- Real MIOpen, CK, and rocWMMA example implementation is deferred to phases
  37-39.
- CDNA 3 and CDNA 4 validation remains deferred beyond v1.8.
</deferred>
