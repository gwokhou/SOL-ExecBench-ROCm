# Milestones

## v1.12 Evaluator Contract Metadata and Boundary Guardrails (Shipped: 2026-05-25)

**Delivered:** GPU-free evaluator contract metadata, `sol-execbench contract
--json`, SOL-owned measured baseline registry namespace, SOL/HIP ownership
boundary guardrails, and PR #1 shipping evidence.

**Phases completed:** none — retroactive quick-task milestone

**Key accomplishments:**

- Added a versioned evaluator compatibility contract builder and public data
  package export.
- Exposed contract JSON through the primary CLI without requiring a problem
  directory or GPU evaluation.
- Preserved canonical trace JSONL by keeping contract-only fields out of trace
  payloads.
- Added guardrails that prevent SOL source from absorbing HIP-side agent/run
  semantics.
- Recorded that the work shipped through PR #1 into
  `gwokhou/SOL-ExecBench-ROCm:main`, not NVIDIA upstream.

**Known tech debt:** none identified for the scoped closure milestone.

---

## v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure (Shipped: 2026-05-23)

**Delivered:** acquisition/layout manifests, deterministic dataset inventory,
ROCm readiness classification, ready-subset manifests, bounded execution
closure, parity-gap JSON/Markdown reports, release-closure docs, and public
claim guardrails for the paper dataset surface.

**Phases completed:** 5 phases, 14 plans

**Key accomplishments:**

- Added public dataset acquisition/layout metadata with category counts,
  checksums, idempotent downloader verification, and explicit claim boundaries.
- Built deterministic paper inventory and ROCm readiness sidecars from
  canonical `Definition` and `Workload` contracts, including blocker reason
  codes and ready-subset manifests.
- Extended `scripts/run_dataset.py` to execute bounded ready subsets through
  the existing `sol-execbench` subprocess path and emit
  `execution_closure.json`.
- Added parity-gap JSON and Markdown reports that aggregate manifest,
  inventory, readiness, execution closure, and derived AMD evidence.
- Added release-closure documentation and guardrails preventing bounded
  sidecar artifacts from being presented as full paper validation,
  leaderboard parity, upstream SOLAR parity, or unsupported hardware
  validation.

**Known tech debt:**

- No single automated fixture runs the full sidecar chain end to end.
- Manifest/root/category/checksum consistency is provenance-only across
  sidecar inputs.
- `--readiness` remains optional for execution closure, so complete blocked
  denominator accounting depends on using the documented command.

---

## v1.10 Paper-Aligned SOLAR Automatic Derivation (Shipped: 2026-05-23)

**Phases completed:** 6 phases, 23 plans, 41 tasks

**Key accomplishments:**

- Added the sidecar-only v1.10 SOLAR derivation contract, fixture matrix, and
  strict parser/serializer coverage for semantic provenance.

- Built SOLAR derivation evidence from canonical `Definition` and `Workload`
  inputs without candidate execution, with deterministic supported, degraded,
  and unscored confidence classification.

- Promoted high-confidence families into formula/byte/bound-backed SOLAR
  evidence: linear projection, attention, convolution, embedding/positional,
  gather, and rotary-like memory-bound structures.

- Added conservative MoE and SSM/Mamba evidence paths with explicit degraded or
  unscored behavior when routing, recurrence, or metadata is incomplete.

- Added machine-verifiable `coverage_summary` and `aggregate_status` sidecars,
  AMD-native score guards, and strict round-trip/malformed-field tests.

- Integrated generated SOLAR sidecars into dataset-runner derived reports using
  `derived_evidence_refs`, while preserving canonical trace JSONL, public score
  `evidence_refs`, primary CLI behavior, and claim boundaries.

---

## v1.9 AMD SOL/SOLAR Bound Modeling Completion (Shipped: 2026-05-23)

**Delivered:** strict AMD hardware model artifacts, structured bound graph IR,
auditable operator FLOP/byte/movement modeling, v2 AMD SOL bound sidecars,
coverage semantics, score/dataset integration, documentation, guardrails, and
RDNA 4 validation evidence.

**Phases completed:** 6 phases, 17 plans, 21 tasks

**Key accomplishments:**

- Added strict packaged/external AMD hardware model loading with RDNA 4
  `gfx1200` defaults, validation-status metadata, and public-contract
  guardrails.

- Structured AMD bound graph IR with paper-aligned operation families and workload-bound tensor metadata
- Dynamic-trace-first bound graph extraction with AST fallback, dataflow edges, and explicit unsupported evidence
- AMD SOL compatibility facade now consumes structured bound graph evidence while preserving public contracts
- BoundGraph nodes now have a rich per-operator estimate contract with formula, byte bucket, confidence, rationale, and unsupported evidence fields.
- GEMM, batched GEMM, elementwise, and activation nodes now emit auditable FLOP formulas and node-local read/write byte evidence.
- Reduction, normalization, softmax, data movement, and dtype conversion nodes now expose conservative formula and byte evidence from graph attributes.
- Legacy AMD SOL v1 bounds now consume rich operator estimates while preserving WorkEstimate fields, artifact schema, canonical schemas, and primary CLI behavior.
- Added AMD SOL bound artifact v2 sidecars with graph, estimate, per-op bound,
  aggregate bound, hardware model, coverage, and deterministic warning
  evidence.

- Wired v2 bound artifacts into AMD-native workload and suite score reports
  without mutating canonical trace JSONL.

- Documented artifact semantics, confidence labels, unsupported/inexact
  degradation, and RDNA 4-only validation scope.

- Added claim guardrails against NVIDIA B200/SOLAR/leaderboard equivalence and
  premature CDNA 3 / MI300X or CDNA 4 validation claims.

**Known gaps:**

- CDNA 3 / MI300X real-hardware validation remains deferred.
- CDNA 4 validation remains deferred.
- Original paper model-to-subgraph extraction and broader upstream SOLAR parity
  remain future work.

---

## v1.8 ROCm Library Ecosystem Completion (Shipped: 2026-05-22)

**Delivered:** MIOpen, Composable Kernel, and rocWMMA are now scoped supported
ROCm library examples for RDNA 4, with dependency diagnostics, native staging
tests, public-contract documentation, and validation guardrails.

**Phases completed:** 36-40 (5 phases, 5 plans)

**Key accomplishments:**

- Added reusable ROCm library dependency diagnostics and Docker dependency
  checks for MIOpen, Composable Kernel, rocWMMA, and hipBLAS.

- Promoted MIOpen with a native softmax public example, metadata/source tests,
  native staging coverage, RDNA 4 E2E registration, and support docs.

- Promoted Composable Kernel with a scoped small GEMM public example, CK source
  consistency checks, native staging coverage, and support-status docs.

- Promoted rocWMMA with a matrix-core GEMM public example, fragment/MMA/store
  API coverage, native staging tests, RDNA 4 E2E registration, and CDNA
  deferral wording.

- Cleaned compatibility/support wording so former cuDNN, CUTLASS, CuTe DSL, and
  cuTile paths remain compatibility examples unless they contain native ROCm
  library solutions.

- Added Nyquist validation artifacts for all v1.8 phases and passed the
  milestone audit with 23/23 requirements satisfied.

**Known gaps:**

- CDNA 3 and CDNA 4 library validation remain deferred by explicit v1.8 scope.

- Full native E2E execution for some library examples depends on complete ROCm
  development/library headers in the local environment.

---

## v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration (Shipped: 2026-05-22)

**Delivered:** Optimized scoring baseline artifacts, source-specific ROCm
profiler timing evidence, expanded reward-hack defenses, the first supported
hipBLAS library example, and MI300X validation-readiness guardrails.

**Phases completed:** 31-35 (5 phases, 5 plans)

**Key accomplishments:**

- Added release-scoped scoring baseline artifacts and `baseline_source`
  reporting so AMD-native scores distinguish optimized baselines from
  provisional reference-latency fallback.

- Added source-specific `rocprofv3` timing evidence collection through the
  dataset workflow with auditable backend, aggregation, warmup, clock-lock,
  architecture, and fallback metadata.

- Added static reward-hack source review before submitted Python code import,
  covering hidden streams, semantic caches, unauthorized loaders/file I/O, and
  precision downgrade patterns.

- Promoted `hipblas` with a runnable SGEMM public example, native staging tests,
  docs, and candidate guardrails for MIOpen, CK, and rocWMMA.

- Added MI300X/CDNA3 validation handoff docs, FP8/NVFP4 decision records, and
  evidence gates that prevent premature hardware-validation claims.

**Known gaps:**

- Real AMD Instinct MI300X/CDNA3 full-suite validation remains deferred until
  hardware access is available.

- FP8 real validation waits for MI300X; NVFP4/MXFP4 remains deferred without a
  suitable AMD validation path.

- Original paper dataset extraction and full upstream SOLAR parity remain
  future work by explicit user decision.

---

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
