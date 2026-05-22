# Milestones

## v1.3 Non-CDNA Issue Closure (Shipped: 2026-05-22)

**Phases completed:** 5 phases, 5 plans, 0 tasks

**Key accomplishments:**

- Added a maintained parity audit against NVIDIA SOL ExecBench public
  functionality and original solution categories.
- Added `sol-execbench-baseline`, a public trace JSONL baseline-comparison
  command with text/JSON output and claim-level guardrails.
- Clarified ROCm library category readiness so `hipblas`, `miopen`, `ck`, and
  `rocwmma` are candidate categories unless runnable evidence exists.
- Updated the `hip-execbench` practice map to record accepted baseline
  comparison adaptation and rejected contract-changing imports.
- Closed non-CDNA validation debt with focused public-contract, parity,
  baseline, library-readiness, and practice-map tests.

**Known gaps:**

- Real CDNA 3 hardware validation remains deferred until a future milestone with
  access to `gfx94*` hardware evidence.

---

## v1.2 Engineering Practice Harvest and Compatibility Guardrails (Shipped: 2026-05-22)

**Phases completed:** 4 phases, 4 plans, 0 tasks

**Key accomplishments:**

- Documented accepted, rejected, and deferred `hip-execbench` engineering
  practices in an internal adaptation map.
- Added internal ROCm diagnostics for tool readiness, gfx classification, local
  gfx detection, and profiler backend fallback reasoning.
- Added pure trace reporting helpers that summarize existing trace objects
  without changing trace JSONL.
- Added SOL-Score interpretation guardrails that warn against unsupported AMD
  hardware performance claims while preserving the existing score formula.
- Added public contract tests for solution/workload/trace schemas, CLI help,
  HIP-facing example paths, and CDNA 3 validation deferral language.

**Known gaps:**

- Real CDNA 3 hardware validation remains deferred until a future milestone with
  access to `gfx94*` hardware evidence.
- Public baseline-comparison CLI and AMD-native roofline interpretation remain
  future scope.

---

## v1.1 CDNA 3 Support and Migration Closure (Shipped: 2026-05-21)

**Phases completed:** 3 phases, 3 plans, 0 tasks

**Key accomplishments:**

- Added explicit CDNA 3 schema targets (`gfx940`, `gfx941`, `gfx942`) and HIP offload flag staging for `gfx94*`.
- Added an active migration residue audit that classifies intentional CUDA/NVIDIA compatibility names and blocks unclassified residue.
- Renamed public native examples to HIP-facing paths and solution filenames.
- Reframed former NVIDIA library/DSL examples as ROCm compatibility examples and added CDNA 3 metadata where portable.
- Updated support docs to distinguish CDNA 3 code/schema support from deferred hardware validation.
- Created a CDNA 3 validation handoff with commands, evidence requirements, and acceptance criteria for the next milestone.

**Known gaps:**

- Real CDNA 3 hardware validation remains deferred. Run the full adapted suite on `gfx94*` before claiming CDNA 3 hardware validation.

---

## v1.0 ROCm Port (Shipped: 2026-05-21)

**Phases completed:** 6 phases, 21 plans, 18 tasks

**Key accomplishments:**

- Pinned ROCm Docker image, AMD GPU passthrough flags, and ROCm-aware entrypoint startup behavior
- PyTorch ROCm dependency declarations and lockfile resolved without CUDA/NVIDIA package residue
- ROCm runtime, HIP compiler, PyTorch ROCm, Triton ROCm, and selected library smoke tests
- ROCm-only Docker dependency pytest collection with superseded CUDA/NVIDIA smoke tests removed
- ROCm-native solution schema with hip_cpp, gfx1200, hip_cflags, and strict CUDA/NVIDIA migration errors
- ProblemPackager stages HIP/C++ solutions and injects AMD `--offload-arch` flags from explicit or local gfx targets
- HIP-aware build_ext.py discovers `.hip` and C/C++ sources, reads `hip_cflags`, and preserves the PyTorch extension loader contract
- Focused pytest audit guards Phase 2 schema/build paths against unallowlisted CUDA/NVIDIA residue

**Known gaps:**

- TEST-05 deferred: full adapted suite validation on CDNA 3 (`gfx94*`) was not recorded before v1.0 close. See `.planning/STATE.md` deferred items.

---
