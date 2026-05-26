# Plan 74-01 Summary: Build Artifact Discovery And Manifest

**Completed:** 2026-05-26  
**Status:** Complete  
**Commit:** `98fef0d #74 - Add static artifact discovery manifest`

## What Changed

- Extended `StaticKernelEvidenceArtifact` with manifest metadata for producer,
  target architecture, and inspectability.
- Added `collect_static_kernel_artifacts()` to persist current-build static
  artifacts from an explicit HIP/C++ build directory into an evidence directory.
- Added bounded artifact classification for `benchmark_kernel.so`, `.hsaco`,
  `.co`, `.o`, and compiler output files.
- Added SHA256 and size computation from persisted artifact copies.
- Added root-containment checks so symlink escapes and unrelated outside files
  are not copied.
- Returned explicit `unavailable` / `artifact_unavailable` sidecars when no
  stable primary `benchmark_kernel.so` boundary exists.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q`
  - Result: 15 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/driver/test_problem_packager.py -q`
  - Result: 42 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/bench/static_kernel_evidence.py tests/sol_execbench/test_static_kernel_evidence.py`
  - Result: All checks passed
- `rg -n "subprocess|shutil\\.which|/tmp|rocm_agent_enumerator|rocminfo" src/sol_execbench/core/bench/static_kernel_evidence.py`
  - Result: no matches

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| SKE-ARTIFACT-01 | Complete | Tests discover only current build-root artifacts. |
| SKE-ARTIFACT-02 | Complete | Tests prove persisted copies survive staging cleanup. |
| SKE-ARTIFACT-03 | Complete | Manifest entries include kind, source path, persisted path, SHA256, size, producer, target architecture, and inspectability. |
| SKE-ARTIFACT-04 | Complete | Tests skip outside files, nested evidence artifacts, and symlink escapes. |
| SKE-ARTIFACT-05 | Complete | Missing primary artifact returns unavailable sidecar with `artifact_unavailable`. |

## Deferred

- CLI flag integration and output-derived evidence directory selection remain
  Phase 76 work.
- Extractor routing and static tool execution remain Phase 75 work.
- Live ROCm validation remains Phase 77 work.
