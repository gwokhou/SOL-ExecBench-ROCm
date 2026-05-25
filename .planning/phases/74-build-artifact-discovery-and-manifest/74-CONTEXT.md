# Phase 74: Build Artifact Discovery And Manifest - Context

**Gathered:** 2026-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 74 implements CPU-safe static artifact discovery and persistence helpers
for the current HIP/C++ solution build. Operators must be able to archive
artifacts produced by the exact current staging/build tree before temporary
cleanup and inspect a manifest for those persisted copies.

This phase does not execute disassembly or metadata extractor tools, does not add
CLI flags, does not change canonical benchmark outputs, and does not attempt
global cache, ROCm install tree, Triton cache, or standalone artifact analysis.

</domain>

<decisions>
## Implementation Decisions

### Discovery Boundary
- Discover artifacts only from the current HIP/C++ staging or build tree.
- Start with `benchmark_kernel.so` and opportunistically include `.hsaco`, code
  object, `.o`, and relevant compiler output files found inside that same
  current-build boundary.
- Never scan global caches, ROCm installation directories, unrelated temporary
  directories, or broad filesystem locations that could mix artifacts across
  benchmark runs.
- Phase 74 performs discovery, copy/persistence, hashing, and manifest
  registration only. Static extractor execution belongs to Phase 75.

### Persistence And Manifest Shape
- Copy discovered artifacts into an output-derived static evidence directory so
  evidence survives staging cleanup.
- Use sidecar-relative persisted paths where possible; source paths are retained
  only as provenance for the current run.
- Record at least artifact kind, source path, persisted path, SHA256, size,
  producer, target architecture when known, and inspectability for each
  registered artifact.
- Preserve explicit unavailable or unsupported outcomes when the solution path
  lacks a stable v1.17 static artifact boundary.

### Implementation Location
- Extend `src/sol_execbench/core/bench/static_kernel_evidence.py` or a nearby
  helper in the same bench/static evidence area with CPU-safe discovery and
  persistence helpers.
- Cover the helpers directly with focused CPU-safe tests.
- Do not integrate the public CLI, extractor routing, or deep compile-flow return
  contracts in this phase unless a tiny local adapter is required to expose the
  current build directory.

### the agent's Discretion
Use nearby file-system, hashing, Pydantic, and sidecar patterns to choose exact
helper names, path-safety details, and artifact-kind normalization. Keep the
implementation small and explicit.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/bench/static_kernel_evidence.py` now owns the strict
  diagnostic sidecar schema, status/reason enums, artifact entries, and helper
  constructors from Phase 73.
- `src/sol_execbench/driver/problem_packager.py` owns HIP/C++ package staging
  and the `benchmark_kernel.so` build output boundary.
- `src/sol_execbench/core/bench/rocm_profiler.py` provides adjacent diagnostic
  artifact and sidecar persistence patterns.
- `src/sol_execbench/core/toolchain.py` provides v1.16 routed static artifact
  vocabulary and capability information for later extractor phases, but Phase
  74 should not run extractor tools.

### Established Patterns
- Static evidence is diagnostic-only and sidecar-only.
- Optional evidence failures, unsupported inputs, and unavailable artifacts must
  not fail correctness, timing, scoring, or default benchmark behavior.
- CPU-safe unit tests should prove path boundaries, manifest fields, hashing,
  persistence after source cleanup, and no global scans.

### Integration Points
- Helpers should accept explicit current-build and output/evidence directory
  paths rather than discovering broad environment state.
- Manifest entries should be compatible with the Phase 73 artifact model so
  Phase 76 can serialize them into the static evidence sidecar.
- Unsupported/unavailable helper outcomes should reuse Phase 73 reason-code
  semantics instead of inventing a second vocabulary.

</code_context>

<specifics>
## Specific Ideas

- Prefer deterministic artifact IDs based on normalized relative paths or stable
  counters rather than absolute temporary path strings.
- Treat `benchmark_kernel.so` as the primary artifact when present.
- Use SHA256 and file size from the persisted copy to make the manifest
  reproducible after staging cleanup.
- Keep symlink and directory traversal behavior conservative so discovery cannot
  escape the supplied build tree.

</specifics>

<deferred>
## Deferred Ideas

- Routed `llvm-objdump`, `readelf`, RGA, or `roc-objdump` execution.
- ISA, ELF metadata, resource usage, or symbol parsing.
- CLI flag and report rendering.
- Triton ROCm cache capture.
- Standalone analysis of pre-existing `.hsaco`, code object, or `.so` files.
- Live RDNA 4 / CDNA hardware validation.

</deferred>
