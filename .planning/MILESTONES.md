# Milestones

## v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow (Shipped: 2026-05-22)

**Delivered:** Broader AMD SOL analyzer coverage, live `rocprofv3` collection
workflow, opt-in AMD-native score reports, and focused compatibility/claim
guardrails.

**Phases completed:** 27-30 (4 phases, 4 plans)

**Key accomplishments:**

- Broadened AMD SOL/SOLAR-like analyzer coverage for reductions,
  normalization-like calls, softmax-like calls, activations, and data movement,
  with derived coverage summaries and confidence labels.
- Added a live `rocprofv3` collection adapter with source-specific timing
  semantics and explicit fallback metadata.
- Connected canonical trace JSONL, live timing evidence refs, AMD SOL bounds,
  baseline latency, and hardware model refs into derived AMD-native score
  reports.
- Added `scripts/run_dataset.py --amd-score-report` as an opt-in suite report
  output without changing primary `sol-execbench` defaults.
- Added focused v1.6 compatibility and claim guardrails for trace/schema/CLI
  stability, CDNA3 validation deferral, and no NVIDIA B200/SOLAR/leaderboard
  equivalence claims.

**Known gaps:**

- Real CDNA 3 `gfx94*` full-suite validation remains deferred by explicit user
  instruction.
- Live profiler unit tests use mocked collection; real `rocprofv3` output
  validation remains environment-dependent.
- Full upstream SOLAR parity remains future scope.

---

## v1.5 AMD-native SOL Scoring and ROCm Profiler Timing (Shipped: 2026-05-22)

**Delivered:** AMD-native SOL bound and scoring foundations plus accuracy-first
ROCm timing policy and profiler evidence helpers.

**Phases completed:** 23-26 (4 phases, 4 plans)

**Key accomplishments:**

- Defined source-specific timing semantics for HIP native, Triton, PyTorch,
  mixed, and unknown workloads without forcing a single inaccurate timing
  interpretation.
- Added `rocprofv3` timing evidence helpers with command construction, CSV
  parsing, policy-aware default selection, and labeled fallback metadata.
- Added AMD SOL bound artifacts with graph nodes, FLOP/byte estimates, hardware
  model metadata, per-op bounds, aggregate bounds, and CDNA3 unvalidated status.
- Added derived AMD-native per-workload and suite score reports with timing and
  SOL-bound evidence references.
- Preserved public CLI, schema, trace JSONL, eval-driver correctness, and
  reward-hack contracts while documenting no NVIDIA B200/SOLAR/leaderboard
  equivalence claim.

**Known gaps:**

- Real CDNA 3 `gfx94*` full-suite validation remains deferred by explicit user
  instruction.
- AMD SOL operator coverage is a conservative foundation; broader analyzers are
  future work.

---

## v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness (Shipped: 2026-05-22)

**Phases completed:** 4 phases, 4 plans, 0 tasks

**Key accomplishments:**

- Added a source-grounded v1.4 compatibility inventory and guardrail tests for
  CLI, schemas, solution format, trace JSONL, and eval-driver semantics.

- Added a derived evidence/report model that combines trace summaries and stage
  diagnostics while labeling itself as non-canonical.

- Implemented CDNA 3 `gfx94*` validation readiness metadata with commands,
  evidence requirements, blockers, acceptance criteria, and no-claim wording.

- Recorded RDNA 4 `gfx1200` validation evidence: focused unit tests passed,
  existing E2E pytest passed, and `sol-execbench` CLI produced 3 passing trace
  JSONL records.

**Known gaps:**

- Real CDNA 3 hardware validation remains deferred until a future full-suite
  `gfx94*` run is recorded.

---

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
