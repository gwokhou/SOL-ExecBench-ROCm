# Phase 74: Build Artifact Discovery And Manifest - Research

**Researched:** 2026-05-26  
**Domain:** CPU-safe current-build artifact discovery, persistence, hashing, and static evidence manifest population  
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Discover artifacts only from the current HIP/C++ staging or build tree.
- Start with `benchmark_kernel.so` and opportunistically include `.hsaco`,
  code object, `.o`, and relevant compiler output files found inside that same
  current-build boundary.
- Never scan global caches, ROCm installation directories, unrelated temporary
  directories, or broad filesystem locations.
- Copy discovered artifacts into an output-derived static evidence directory so
  evidence survives staging cleanup.
- Record artifact kind, source path, persisted path, SHA256, size, producer,
  target architecture when known, and inspectability.
- Use sidecar-relative persisted paths where possible; source paths are
  provenance only.
- Add CPU-safe discovery/persistence helpers near
  `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- Do not add CLI flags, extractor execution, report rendering, or broad
  compile-flow rewrites in this phase.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SKE-ARTIFACT-01 | Operator can collect static evidence only from the current HIP/C++ staging or build tree, starting with `benchmark_kernel.so` and opportunistic code objects, HSACO files, object files, and compiler outputs. | `ProblemPackager.compile()` defines the HIP/C++ artifact as `output_dir / "benchmark_kernel.so"` and the build template compiles in `HERE`, so a helper can accept explicit `build_dir` plus primary artifact path and avoid environment scans. [VERIFIED: `src/sol_execbench/driver/problem_packager.py`] [VERIFIED: `src/sol_execbench/driver/templates/build_ext.py`] |
| SKE-ARTIFACT-02 | Operator can persist discovered static artifacts under an output-derived evidence directory after staging cleanup. | Use `Path` and `shutil.copy2` from stdlib to copy files to a supplied evidence root and compute metadata from persisted copies. No new dependency is needed. [VERIFIED: stdlib] |
| SKE-ARTIFACT-03 | Operator can inspect an artifact manifest with kind, source path, persisted path, SHA256, size, producer, target architecture when known, and inspectability. | Phase 73 `StaticKernelEvidenceArtifact` already has artifact type, status, reason, source path, persisted path, size, SHA256, and classification fields. Producer/target arch/inspectability can be represented through source references and classification in v1 schema without changing canonical outputs. [VERIFIED: `src/sol_execbench/core/bench/static_kernel_evidence.py`] |
| SKE-ARTIFACT-04 | Maintainer can verify discovery never scans global caches, ROCm install trees, or unrelated temporary directories. | Design helpers around explicit root-bound traversal with `Path.resolve()`, relative-path checks, symlink escape rejection, and tests that place tempting artifacts outside the build root. [VERIFIED: planned tests] |
| SKE-ARTIFACT-05 | Operator receives explicit unsupported or unavailable states for solution paths without a stable v1.17 static artifact boundary. | Reuse Phase 73 reason codes `unsupported_solution_type`, `unsupported_artifact_type`, and `artifact_unavailable` plus helper sidecar constructors. [VERIFIED: `src/sol_execbench/core/bench/static_kernel_evidence.py`] |

</phase_requirements>

## Summary

Phase 74 should add a small CPU-safe collection API that accepts explicit
current-build and evidence-output paths, discovers only root-contained static
artifacts, copies them to a durable evidence directory, computes SHA256 and size,
and returns Phase 73-compatible manifest entries. The implementation should not
call ROCm tools, parse binaries, search global paths, or change CLI/default
benchmark behavior.

The best implementation location is the existing static evidence module because
Phase 73 already placed the schema and helper constructors there. A lightweight
request/result model plus pure helper functions can preserve the sidecar
boundary while giving Phase 76 a ready integration point.

## Project Constraints

- Source code lives under `src/sol_execbench/`; package tests live under
  `tests/sol_execbench/`.
- Python style follows Ruff and nearby code patterns.
- This phase must be CPU-safe and avoid live ROCm validation.
- Static evidence remains opt-in, diagnostic-only, sidecar-only, and
  non-authoritative for correctness, timing, score, paper parity, and
  leaderboard claims.

## Architectural Responsibility Map

| Capability | Primary Location | Rationale |
|------------|------------------|-----------|
| Root-bound artifact discovery | `core/bench/static_kernel_evidence.py` | Keeps static evidence contract and collection helpers together while avoiding driver/CLI churn. |
| Current HIP/C++ build boundary | Caller-provided `build_dir` / `benchmark_kernel.so` | `ProblemPackager.compile()` already defines this boundary; helper should not infer broader state. |
| Durable persisted artifacts | Caller-provided evidence directory | Lets CLI/report phases choose the output-derived location later. |
| Manifest entries | `StaticKernelEvidenceArtifact` | Existing schema fields carry status, reason, paths, size, hash, and conservative classification. |
| Unsupported/unavailable outcomes | Phase 73 sidecar helpers | Reuses stable reason/status vocabulary. |

## Existing Patterns

### Current Build Artifact Boundary

`ProblemPackager.compile()` returns `["python", "build_ext.py"]` and
`str(output_dir / "benchmark_kernel.so")`. The template invokes
`torch.utils.cpp_extension.load(..., build_directory=str(HERE), verbose=True)`
and renames any platform-suffixed `benchmark_kernel*.so` to
`benchmark_kernel.so`. This gives Phase 74 a concrete current-build root:
the staging/output directory supplied to the packager.

### Diagnostic Artifact Discovery

`rocm_profiler.py` has a small artifact registration style: discover files in a
known output directory, classify them by suffix/name, and return sidecar-friendly
metadata. Phase 74 should follow the same direct, bounded style but persist
copies and compute SHA256.

### Static Evidence Schema

`StaticKernelEvidenceArtifact` currently includes:

- `artifact_id`
- `artifact_type`
- `status`
- `reason_code`
- `source_path`
- `persisted_path`
- `size_bytes`
- `sha256`
- `classification`
- `source_references`

Phase 74 should avoid schema churn unless tests prove a missing required field.
Producer, target architecture, and inspectability can be represented as stable
source references or conservative classification metadata until extractor phases
provide richer parsed values.

## Recommended Implementation

1. Add a CPU-safe request/result shape:
   - `build_directory: Path`
   - `evidence_directory: Path`
   - optional `primary_artifact_name = "benchmark_kernel.so"`
   - optional `sidecar_base_directory` for relative persisted paths.
2. Discover candidate files only under `build_directory`:
   - primary `.so`: `benchmark_kernel.so`
   - opportunistic `.hsaco`
   - opportunistic `.co`
   - opportunistic `.o`
   - bounded compiler outputs such as `.log`/`.txt` only if clearly under the
     current build tree.
3. Reject or skip anything that resolves outside `build_directory`, including
   symlink escapes.
4. Copy candidates into `evidence_directory / "artifacts"` preserving a safe
   relative layout or deterministic sanitized name.
5. Hash persisted bytes with SHA256 and record persisted-file size.
6. Return a collected sidecar with artifact entries when at least one artifact
   was persisted; return unavailable when no stable primary artifact is present
   or no candidates are found; return unsupported through existing helpers when
   the caller identifies a non-HIP/C++ boundary.

## Tests To Add

- Discovers and persists `benchmark_kernel.so`, `.hsaco`, `.co`, `.o`, and
  compiler output files from a temporary build directory.
- Computes SHA256 and size from persisted artifacts and leaves persisted copies
  readable after source cleanup.
- Emits sidecar-relative persisted paths.
- Does not include artifacts outside the build directory.
- Rejects symlink escapes.
- Returns `unavailable` / `artifact_unavailable` when no primary current-build
  artifact exists.
- Does not use subprocess or toolchain lookup in the static discovery helper.

## Open Questions

None. The user selected the conservative current-build-only boundary, durable
copy manifest shape, and local helper/API implementation depth.
