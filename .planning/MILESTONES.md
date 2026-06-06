# Milestones

## Current CDNA3 Validation Status (Updated: 2026-06-06)

CDNA3/gfx942 validation is no longer purely deferred. Real MI308X (`gfx942`)
cloud runs recorded an adapted pytest pass and exercised the full 235-problem
dataset validation path. The corrected dataset interpretation is 220 complete
passing problem traces, 15 expected Quant NVFP4/MXFP4 CDNA3 skips, and 6
timeout shards across 4 problems. Nested `eval_driver.py` timeout
classification was fixed and targeted verification confirmed timeout shards are
now recorded as `TIMEOUT` traces and summary failures.

This is CDNA3 validation-infrastructure evidence with known blockers, not a
completed full CDNA3/MI300X hardware-validation pass. MI308X and MI300X share
the `gfx942` code path, but the recorded MI308X evidence must not be reported
as MI300X hardware validation because the hardware configurations differ.
Remaining blockers are documented in
`.planning/milestones/CDNA3-VALIDATION-HANDOFF.md` and
`.planning/milestones/MI300X-VALIDATION-HANDOFF.md`.

## v1.29 Dataset Migration and Compliance (Shipped: 2026-06-04)

**Phases completed:** 5 phases, 5 plans, 0 tasks

**Key accomplishments:**

- Added machine-readable provenance and redistribution guardrails for
  NVIDIA/SOL-ExecBench, FlashInfer Trace, generated local migration artifacts,
  and project-owned ROCm code.

- Added deterministic local SOL-ExecBench and FlashInfer Trace migration
  commands with manifests, checksums, source revisions, license-boundary
  metadata, and explicit blocker states.

- Added ROCm readiness classification and ready-subset generation that preserve
  denominators, exclusion reasons, workload closure inputs, and no-claim
  boundaries.

- Added CPU-safe NVIDIA/Blackwell low-precision compatibility helpers for
  NVFP4/MXFP4-like semantics with unvalidated-CDNA4 evidence markers.

- Integrated migrated dataset manifests, readiness sidecars, and public
  no-redistribution guardrails into `scripts/run_dataset.py`, closure reports,
  and cookbook documentation.

**Explicitly deferred:**

- Real CDNA3 or CDNA4 full-suite execution without a complete hardware evidence
  chain.
- Real CDNA4 validation, performance authority, or hardware equivalence claims
  for NVFP4/Blackwell semantics.
- High-performance FlashInfer CUDA-kernel ROCm tuning and full performance
  comparison.
- Public redistribution of NVIDIA SOL-ExecBench original or migrated dataset
  payloads.
- Known deferred quick-task artifacts at close: 5 historical items recorded in
  `.planning/STATE.md`.

---

## v1.28 CDNA3 Test and Documentation Readiness (Shipped: 2026-06-04)

**Phases completed:** 127-130 (4 phases, 4 plans)

**Key accomplishments:**

- Added a concrete `requires_cdna3` hardware-gated pytest surface with
  CPU-safe marker registration, `gfx94*` detection, skip-behavior, and
  marker-use audit coverage.

- Expanded the MI300X/gfx942 evidence contract with required artifacts,
  validation result categories, blocker checks, FP8 readiness, and
  NVFP4/MXFP4 deferral.

- Preserved no-claim guardrails so CDNA3 schema/build/test readiness cannot be
  reported as completed CDNA3 or MI300X hardware validation without complete
  real-hardware evidence.

- Updated testing, ROCm support, and contributor documentation to explain how
  to run CDNA3 marker tests, interpret skips, add future CDNA3 tests, and
  archive future MI300X evidence.

**Explicitly deferred:**

- Actual MI300X/gfx942 full-suite execution and dataset validation.
- CDNA4 validation, paper-scale validation, upstream SOLAR parity, hosted
  leaderboard authority, and hard multi-tenant sandboxing.

---

## v1.22 Roadmap: SOL ExecBench ROCm Port (Backfilled: 2026-06-03)

**Note:** Synthesized from archive snapshot by `/gsd-health --backfill`. Original completion date unknown.

---

## v1.27 Copyright Provenance Cleanup (Shipped: 2026-06-02)

**Phases completed:** 4 phases, 4 plans, 0 tasks

**Key accomplishments:**

- Added a reviewable provenance policy and machine-readable
  `provenance.toml` manifest that separates upstream-retained,
  derivative-modified, independent ROCm, and generated/planning material.

- Corrected file-level SPDX attribution so upstream/derivative files retain
  NVIDIA notices with project attribution, while independent ROCm work uses
  project attribution instead of misleading NVIDIA-only headers.

- Updated compliance, README, research preview, public prerelease, and release
  draft wording to distinguish fork attribution, paper citation, file-level
  copyright ownership, and non-endorsement boundaries.

- Added provenance-aware readiness and residue guardrails that fail missing
  provenance docs, inconsistent NVIDIA notice allowlists, cleanup candidates
  with NVIDIA attribution, or missing required upstream notices.

---

## v1.26 Public Prerelease and Research Preview (Shipped: 2026-06-02)

**Phases completed:** 4 phases, 4 plans, 0 tasks

**Key accomplishments:**

- Added a versioned prerelease artifact bundle workflow with manifest,
  Markdown summary, SHA-256 checksums, command transcripts, release validation
  outputs, environment evidence, known gaps, and authority-class mappings.

- Added prerelease readiness gates that fail missing required artifacts,
  checksum drift, claim-boundary regressions, unknown authority classes, and
  unreviewed known-gap statuses.

- Published a research preview evidence package that separates canonical Trace
  JSONL from diagnostic-only, provisional, deferred, and unavailable evidence.

- Prepared public prerelease publishing materials, including a GitHub release
  draft, release-page checklist, artifact placeholders, support-boundary links,
  and known-limitations wording.

- Corrected active MI300X-as-CDNA3 wording so MI300X is consistently described as
  the concrete CDNA3 `gfx942` hardware target, not a separate architecture peer.

---

## v1.25 Engineering Prerelease (Shipped: 2026-06-01)

**Delivered:** bounded engineering prerelease materials for installing,
validating, interpreting, and packaging the ROCm port without overclaiming
research or hardware authority.

**Phases completed:** 114-118 (5 phases)

**Key accomplishments:**

- Added release-candidate validation across CPU-safe checks, optional
  ROCm/Docker smoke, and bounded dataset-slice workflows.

- Published support boundaries for RDNA 4 evidence, Docker/container
  user-space evidence, deferred MI300X full-suite validation on CDNA3, and
  unavailable CDNA4 validation.

- Added release claim guardrails and v1.25 release notes classifying canonical,
  diagnostic-only, provisional, deferred, and unavailable evidence.

- Clarified the first-run user path for install, minimal trace generation,
  trace interpretation, no-trace diagnostics, and PyTorch ROCm compatibility
  wording.

- Added prerelease checklist and README navigation for release-candidate
  materials.

---

## v1.24 Dataset Batch Run Trustworthiness (Shipped: 2026-06-01)

**Delivered:** dataset-scale reuse, provenance, closure, failure-mode, and
sharding trustworthiness improvements.

**Phases completed:** 110-113 (4 phases)

**Key accomplishments:**

- Moved dataset reuse and stale-provenance decisions into importable core
  helpers.

- Centralized selected-workload closure record assembly, including trace refs,
  summary refs, solution refs, and missing evidence states.

- Documented and protected the dataset failure-mode regression matrix.
- Added deterministic sharding and merge helpers with stable shard ids,
  per-shard trace refs, duplicate detection, and incomplete-shard reporting.

---

## v1.23 Evaluation Reliability and Security Hardening (Shipped: 2026-06-01)

**Delivered:** single-problem evaluation diagnostics, import isolation, native
compile guardrails, and eval-driver boundary hardening.

**Phases completed:** 106-109 (4 phases)

**Key accomplishments:**

- No-trace and noisy-output evaluation failures persist bounded diagnostics and
  point CLI users to evidence without requiring verbose mode.

- Python solution imports use unique file-based staged module identities to
  avoid collisions with already-imported modules.

- Native compile options reject dangerous host/path/linker behavior while
  preserving documented ROCm/HIP extension use cases.

- The generated eval driver delegates benchmark phases to tested importable
  helpers while preserving canonical contracts and reward-hack boundaries.

---

## v1.20 Cross-Report Consistency and Evaluation Stability (Shipped: 2026-05-31)

**Delivered:** local evidence-quality sidecars that make ROCm benchmark runs
internally consistent, timing-quality aware, and explicit about which stronger
claims remain blocked.

**Phases completed:** 89-93 (5 phases, 10 plans)

**Key accomplishments:**

- Added `sol_execbench.consistency_report.v1` and
  `scripts/report_consistency.py` for deterministic cross-report contradiction
  checks across closure, denominator, Matrix, runtime/static evidence, AMD
  score, AMD SOL/SOLAR, and bound sanity reports.

- Added `sol_execbench.evaluation_stability.v1` and
  `scripts/report_evaluation_stability.py` for timing-quality classification,
  dispersion metrics, clock policy, backend, and profiling-overhead risk.

- Added `sol_execbench.claim_upgrade.v1` and
  `scripts/report_claim_upgrade.py` to evaluate diagnostic-only,
  container/native-host validation, score-authority, paper-parity, and
  leaderboard prerequisites without mutating source reports.

- Added `sol_execbench.trust_summary.v1` and
  `scripts/report_trust_summary.py` to combine consistency, stability,
  claim-upgrade, and evidence completeness into bounded JSON/Markdown review
  artifacts.

- Published v1.20 evidence-quality docs and demo fixtures with CPU-safe
  guardrail tests, a full cross-script E2E chain, and negative claim-boundary
  wording.

- Passed milestone audit with 23/23 requirements, 5/5 phases, 5/5 integration
  checks, 5/5 flows, and no residual blockers or tech debt.

**Stats:**

- 76 files created/modified in the milestone range.
- 5,775 insertions and 88 deletions across planning, source, scripts, tests,
  docs, and examples.

- 5 phases, 10 plans, 23 requirements.
- Shipped on 2026-05-31.

**Git range:** `c236c55` -> `b8eca35`

**What's next:** define v1.21 from the deferred hardware-validation,
paper-parity, profiling-diagnostics, or leaderboard-readiness frontier.

---

## v1.19 Research Credibility Without New Hardware (Shipped: 2026-05-31)

**Phases completed:** 6 phases, 13 plans, 29 tasks

**Key accomplishments:**

- Strict paper denominator sidecar with deterministic JSON, bounded source refs, checksum, and Markdown claim-boundary wording
- Script-side paper denominator report generation with dataset exports and canonical contract guardrails
- Strict MatrixEntry and RocmCompatibilityMatrixReport JSON Schema exports with deterministic script-side output and diagnostic-only boundaries
- Deterministic semantic Matrix report diffs with severity-ranked JSON/Markdown output and diagnostic-only authority guardrails
- Existing passing traces are reused only when prior execution-closure provenance matches the current selected run configuration
- Runner closure records now classify nonzero CLI exits, timeouts, no-output failures, missing traces, and missing evidence with bounded sidecar diagnostics
- Strict amd_bound_sanity.v1 sidecar models with existing-evidence diagnostic rollups, bounded refs, false authority boundaries, checksum, and deterministic Markdown
- Thin AMD bound sanity report script with public contract guardrails proving the artifact stays sidecar-only and diagnostic-only
- Central v1.19 evidence guide with public entry links and CPU-safe wording guardrails for sidecar-only evidence interpretation
- Demo-only v1.19 evidence fixtures with focused tests for shape, bounded refs, checksums, and false authority boundaries

---

## v1.18 ROCm Version Matrix via Docker (Shipped: 2026-05-28)

**Phases completed:** 5 phases, 9 plans, 16 tasks

**Key accomplishments:**

- Strict ROCm compatibility Matrix Entry contract with bounded statuses, Target/observed evidence separation, artifact references, and diagnostic-only claim boundaries
- Deterministic Matrix execution guardrails that block mixed-version validation, separate Docker and native-host claims, and keep compatibility evidence sidecar-only
- Declared ROCm Docker Targets with deterministic selection, non-authoritative unknown overrides, runtime-unavailable preflight classification, and shell-consumable Matrix JSON
- Dockerfile and wrapper integration for declared ROCm Target selection, selected build args, and runtime-unavailable preflight stops
- Target-adjacent PyTorch ROCm wheel policy with Matrix Entry evidence, stack classification, and shell-consumable preflight JSON
- Docker wrapper dependency preflight blocks illegal PyTorch ROCm stacks before build/run while preserving bounded debug semantics

---

## v1.17 Static Kernel Evidence (Shipped: 2026-05-26)

**Phases completed:** 5 phases, 5 plans, 0 tasks

**Key accomplishments:**

- Added the `sol_execbench.static_kernel_evidence.v1` diagnostic sidecar
  contract with strict schema validation, explicit status vocabulary, and
  authority flags that keep static evidence separate from correctness,
  performance, timing, score, paper-parity, and leaderboard claims.

- Added current-build HIP/C++ artifact discovery and persistence for
  `benchmark_kernel.so`, object files, compiler outputs, and related
  inspectable artifacts under output-derived evidence directories.

- Routed static extraction through the v1.16 toolchain layer, with bounded
  `llvm-objdump` and `readelf` tool runs, preserved raw outputs, route
  provenance, and nonfatal unavailable/failed/partial sidecar states.

- Exposed opt-in CLI collection through `--static-evidence auto`, while
  keeping `--static-evidence none` as the default and preserving canonical
  trace JSONL, benchmark exit semantics, and scoring behavior.

- Published static evidence documentation, claim guardrails, researcher guide
  updates, routing docs, CPU-safe documentation tests, and a bounded RDNA 4
  validation artifact.

**Known tech debt:** CDNA 3/CDNA 4 live validation, Triton ROCm cache capture,
RGA-rich resource parsing, and paper-scale static coverage remain deferred.

---

## v1.16 ROCm Toolchain Research and Capability Routing (Shipped: 2026-05-25)

**Phases completed:** 5 phases, 5 plans, 0 tasks

**Key accomplishments:**

- Added a research-backed ROCm toolchain routing model with lifecycle states
  for active, migrated, deprecated, planned, rejected, and candidate tools.

- Added a machine-readable capability registry keyed by evidence level,
  artifact type, hardware generation, GPU architecture pattern, ROCm version,
  status, reason, and source references.

- Added bounded dynamic executable probes and explicit routing decisions that
  select tools only when registry status and host availability agree.

- Exposed `sol-execbench toolchain --json` for registry inspection and routing
  reports without mutating canonical trace JSONL.

- Documented ROCm toolchain availability, interpretation guidance, and claim
  boundaries while deferring Static Kernel Evidence implementation to v1.17.

---

## v1.15 Research-Grade ROCm Benchmark Release (Shipped: 2026-05-25)

**Phases completed:** 4 phases, 4 plans, 0 tasks

**Key accomplishments:**

- Added explicit public claim boundaries for ROCm-port, runtime, profiling,
  AMD-native-derived, and research-preview evidence.

- Defined a deterministic curated ROCm benchmark slice with artifact
  expectations and unsupported-scope boundaries.

- Added researcher guide and cookbook workflows for kernel, compiler/backend,
  agent, and reproducibility researchers.

- Added a v1.15 release closure checklist with result semantics, known gaps,
  and next-milestone direction.

- Added documentation guardrail tests that protect the release-preview claim
  boundary.

---

## v1.14 Optional rocprofv3 Profiling Evidence (Shipped: 2026-05-25)

**Delivered:** opt-in `rocprofv3` profiling artifacts for benchmark runs,
profile command provenance, artifact registration, diagnostic sidecars,
`profiling.evidence.v1` optional capability, and ROCm profiling documentation
while preserving canonical trace JSONL, default execution, correctness, timing,
scoring, and evaluator contract version `1.0`.

**Phases completed:** 61-63 (3 phases, 3 plans)

**Key accomplishments:**

- Added `sol-execbench --profile rocprofv3` with default `--profile none`.
- Added `sol_execbench.rocprofv3_profile.v1` profile sidecar metadata with
  command, working directory, timeout, artifact paths, return code,
  skipped/failed reasons, and stdout/stderr tails.

- Registered `rocpd`, CSV trace, counter, agent-info, JSON, and unknown
  profiler artifacts under stable output-derived paths.

- Kept profiler absence or failure nonfatal by falling back to normal benchmark
  execution.

- Advertised optional `profiling.evidence.v1` without bumping the evaluator
  contract.

**Known tech debt:** static RGA/GPUOpen ISA analysis and ROCm Compute Profiler
roofline workflows remain future work.

---

## v1.13 ROCm Runtime Evidence and Environment Diagnostics (Shipped: 2026-05-25)

**Delivered:** optional ROCm runtime environment snapshot evidence, opt-in
benchmark run sidecars, `runtime.evidence.v1` contract capability,
`sol-execbench doctor --json`, PyTorch ROCm preflight checks, and documentation
while preserving canonical trace JSONL and contract version `1.0`.

**Phases completed:** 58-60 (3 phases, 5 plans)

**Key accomplishments:**

- Added `sol_execbench.environment_snapshot.v1` evidence models and explicit
  bounded probes for `amd-smi`, `rocminfo`, `rocm_agent_enumerator`, and
  PyTorch ROCm metadata.

- Added optional benchmark run sidecar output through
  `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH`.

- Added `sol-execbench doctor --json` with structured tool, runtime, memory,
  and event-timing readiness checks.

- Advertised optional `runtime.evidence.v1` capability without bumping
  evaluator contract version or making evidence mandatory.

- Preserved canonical trace JSONL, correctness, timing, scoring, and primary
  benchmark CLI defaults.

**Known tech debt:** `rocprofv3` profiling artifacts are intentionally deferred
to v1.14; static RGA/GPUOpen ISA analysis remains future work.

---

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
  premature CDNA3-family validation claims, including MI300X, or CDNA 4 validation claims.

**Known gaps:**

- CDNA3-family real-hardware validation, including MI300X, remains deferred.
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

- Added MI300X-on-CDNA3 validation handoff docs, FP8/NVFP4 decision records, and
  evidence gates that prevent premature hardware-validation claims.

**Known gaps:**

- Real AMD Instinct MI300X full-suite validation on CDNA3 remains deferred until
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
