# Phase 74: Build Artifact Discovery And Manifest - Pattern Map

**Mapped:** 2026-05-26  
**Files analyzed:** 6  
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/bench/static_kernel_evidence.py` | utility, model | file-system discovery, transform, sidecar serialization | `src/sol_execbench/core/bench/rocm_profiler.py`; existing `static_kernel_evidence.py` | high |
| `tests/sol_execbench/test_static_kernel_evidence.py` | test | temp filesystem, schema round trip | `tests/sol_execbench/test_rocm_profiler.py`; existing static evidence tests | high |

## Pattern Assignments

### Static Evidence Module

Use the existing Phase 73 Pydantic schema and helper constructors as the public
contract surface. Add small CPU-safe functions in the same module for discovery
and persistence because they produce `StaticKernelEvidenceArtifact` entries and
status sidecars.

Follow `rocm_profiler.py` for bounded diagnostic artifact discovery:

- accept explicit directories from the caller
- classify known output files by suffix/name
- return sidecar-friendly artifact metadata
- keep failures nonfatal and diagnostic

Deviate from `rocm_profiler.py` where Phase 74 requires stronger persistence:

- copy artifacts to an evidence directory
- compute SHA256
- reject symlink escapes
- avoid subprocess and tool execution entirely

### Problem Packager Boundary

Do not change `ProblemPackager.compile()` unless implementation proves a tiny
adapter is required. Its current contract already returns `benchmark_kernel.so`
under the staging `output_dir`, and `build_ext.py` builds in that same directory.

### Tests

Extend `tests/sol_execbench/test_static_kernel_evidence.py` because Phase 74
extends the same sidecar contract module. Use `tmp_path` to build synthetic
current-build trees, output evidence directories, symlink escapes, and unrelated
outside files.

## Anti-Patterns To Avoid

- No `subprocess`, `shutil.which`, or ROCm executable routing in Phase 74.
- No scan of `/tmp`, `$HOME`, ROCm install directories, or tool caches.
- No CLI flags or report rendering.
- No binary parsing or ISA extraction.
- No mutation of canonical trace, scoring, timing, or contract version surfaces.
