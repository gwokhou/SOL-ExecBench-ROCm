# SOL ExecBench ROCm Port

## What This Is

SOL ExecBench ROCm Port is a ROCm-only fork of SOL ExecBench for evaluating
LLM-generated GPU kernels on AMD GPUs. It keeps the original benchmark
semantics and public contracts where practical while replacing CUDA/NVIDIA
runtime, build, timing, environment, example, and documentation paths with ROCm
equivalents.

## Core Value

Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm
hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## Current State

**Shipped version:** v1.27 Copyright Provenance Cleanup,
completed 2026-06-02.

**Current milestone:** Awaiting next milestone.

**Queued milestone:** None defined.

**Latest milestone outcome:** v1.27 completed release-facing copyright and
provenance cleanup by classifying active files, correcting SPDX attribution,
documenting fork/paper/non-endorsement boundaries, and adding provenance-aware
readiness guardrails.

**Previous milestone outcome:** v1.26 produced the public prerelease and
research preview package with versioned artifact bundles, readiness gates,
research preview evidence, public publishing materials, and corrected
MI300X-as-CDNA3 wording.

**Next milestone goal:** None defined.

The v1.0 milestone migrated the repository to a ROCm-only runtime baseline.
Milestones v1.1-v1.6 added CDNA 3 code/schema support, maintained residue
audits, harvested selected `hip-execbench` practices, closed non-CDNA parity
gaps, added source-specific timing semantics, `rocprofv3` evidence helpers, AMD
SOL bound artifacts, derived AMD-native score reports, live profiler collection,
dataset score reporting, and compatibility/claim guardrails. The v1.7 milestone
added optimized scoring baseline artifacts, source-specific profiler evidence
collection, expanded reward-hack defenses, a runnable hipBLAS public example,
and MI300X validation-readiness guardrails.
The v1.8 milestone completed the remaining ROCm library ecosystem replacements
for RDNA 4 scope by adding MIOpen, CK, and rocWMMA public examples and claim
guardrails.
The v1.9 milestone completed the AMD SOL/SOLAR bound modeling pipeline for RDNA
4 scope by adding strict hardware model artifacts, a structured bound graph IR,
auditable operator estimates, v2 sidecars, coverage semantics, score/dataset
integration, documentation, and claim guardrails.
The v1.10 milestone completed the scoped paper-aligned automatic SOLAR
derivation system by adding fixture contracts, semantic provenance,
family-specific formula/byte/bound evidence, degraded complex-family evidence,
coverage/status sidecars, score guards, dataset-runner report integration, and
claim guardrails.
The v1.11 milestone completed the paper dataset parity inventory and bounded
execution-closure workflow by adding dataset acquisition/layout manifests,
inventory/readiness/ready-subset sidecars, execution-closure reports,
parity-gap reports, release-closure docs, and claim guardrails.
The v1.12 milestone retroactively recorded the evaluator contract metadata and
boundary guardrail work merged through PR #1, including `sol-execbench contract
--json`, SOL-owned compatibility metadata, and HIP-side consumer boundaries.
The v1.13 milestone starts from ROCm Systems/GPUOpen enhancement research and
focuses only on backward-compatible runtime environment evidence, diagnostic
commands, and architecture-aware smoke/preflight checks. `rocprofv3` artifact
collection is deferred to the planned v1.14 milestone, and static RGA/GPUOpen
ISA analysis remains later candidate work.
The v1.14 milestone built on that evidence surface by adding explicitly
enabled `rocprofv3` profiling command provenance, artifact registration,
diagnostic sidecars, and reporting. Profiling remains diagnostic evidence only
and does not change default benchmark execution or correctness/scoring
outcomes.
The v1.15 milestone packaged the port into a bounded research preview by adding
claim boundaries, a curated ROCm slice, researcher guide and cookbook workflows,
release closure, and documentation guardrail tests.
The v1.16 milestone researched ROCm's fragmented toolchain ecosystem and added
a central capability registry plus dynamic routing reports so evidence paths can
explain which tools are available, unavailable, planned, migrated, rejected, or
deprecated without overclaiming benchmark authority.
The v1.17 milestone added opt-in diagnostic Static Kernel Evidence by capturing
current-build HIP/C++ artifacts, extracting bounded ISA and ELF metadata through
routed static tools, publishing trace-adjacent sidecars, documenting claim
boundaries, and recording bounded RDNA 4 validation.
The v1.18 milestone added ROCm Docker version Matrix infrastructure, target
selection, target-specific PyTorch ROCm dependency policy, runtime evidence and
compatibility reports, and claim guardrails for container user-space validation.
The v1.19 milestone added closure contracts, paper denominator accounting,
Matrix schema/diff tooling, runner closure hardening, AMD bound sanity reports,
and public docs/examples/guardrails for research credibility without expanding
hardware validation.
The v1.20 milestone added cross-report consistency lint, evaluation stability
diagnostics, claim-upgrade prerequisite gates, trust summaries, public
evidence-quality docs/fixtures, and a full CPU-safe script-chain regression.
The v1.21 milestone reduced codebase debt and hardened execution boundaries by
extracting dataset runner, eval-driver runtime, AMD/SOLAR, and static-evidence
helpers; adding boundary regressions; and documenting which concerns remain
externally deferred.
The v1.22 milestone closed the remaining code-actionable concern-map items by
finishing dataset runner seams, eval-driver diagnostics, source-review boundary
evidence, scoring/static-evidence fixtures, dependency/closure/marker
guardrails, and concern-map stewardship while preserving explicit deferred
boundaries for hardware validation, paper parity, leaderboard readiness, and
hard sandboxing.
The v1.23 milestone hardened single-problem evaluation diagnostics, staged
Python import isolation, native compile option validation, and eval-driver
responsibility boundaries while preserving public contracts and deferred
hardware/paper/leaderboard authority boundaries.
The v1.24 milestone hardened dataset batch trustworthiness by extracting reuse
policy, centralizing closure/evidence record construction, documenting the
failure-mode regression matrix, and adding deterministic sharding plan/merge
helpers without changing default dataset CLI behavior.
The v1.25 milestone packaged the ROCm port for an engineering prerelease by
adding bounded release-candidate validation, a public support matrix, release
claim guardrails, first-run documentation, and release-candidate materials
while keeping paper validation, upstream SOLAR parity, leaderboard readiness,
hard sandboxing, MI300X full-suite validation on CDNA3, and CDNA4 validation out
of scope.
The v1.26 milestone turned that engineering prerelease into a public
prerelease and research preview package by adding versioned artifact bundles,
release-readiness gates, research-preview evidence, public publishing
materials, and corrected MI300X-as-CDNA3 wording.
The v1.27 milestone completed release-facing copyright and provenance cleanup
by adding a provenance policy and manifest, correcting SPDX file attribution,
updating public compliance and attribution wording, and wiring provenance
checks into prerelease readiness gates.

## Recently Shipped Milestone: v1.25 Engineering Prerelease

**Goal:** Turn the v1.24 ROCm port state into an engineering prerelease /
release-candidate package that external users can install, validate, and
interpret without overstated research or hardware claims.

**Target features:**
- Release-candidate validation coverage for CPU-safe tests, focused ROCm/Docker
  smoke checks, and a bounded dataset slice.
- A clear support matrix that distinguishes RDNA 4 evidence, container
  user-space evidence, deferred MI300X full-suite validation on CDNA3, and
  unavailable CDNA4 validation.
- Release documentation and claim wording that prevent accidental paper-parity,
  leaderboard, hard-sandbox, upstream SOLAR, or CDNA4 validation claims.
- A polished first-run path for installation, example execution, trace
  generation, result interpretation, and failure diagnosis.

**Explicitly deferred:**
- Full 235-problem paper-scale validation and upstream SOLAR parity.
- MI300X full-suite hardware validation on CDNA3 unless a complete evidence chain
  is produced separately.
- CDNA4 validation because suitable hardware is not currently available.
- Hosted leaderboard, remote submission operations, or hard multi-tenant
  sandboxing.
- Large PyTorch/ROCm dependency relocking or Docker privilege-model redesign
  unless release validation exposes a blocking issue.

## Recently Shipped Milestone: v1.27 Copyright Provenance Cleanup

**Goal:** Complete release-facing copyright and provenance cleanup so the
Apache-2.0 ROCm port accurately preserves upstream NVIDIA notices only where
they apply and clearly attributes independent ROCm work to this project.

**Shipped outcome:** v1.27 added `provenance.toml` and
`docs/provenance.md`, corrected file-level SPDX attribution according to
provenance, updated public compliance/release/research wording, and added
readiness/provenance tests that prevent future blanket NVIDIA header drift.

**Explicitly deferred:**
- Git history rewriting for prior blanket copyright headers.
- Relicensing away from Apache-2.0 or performing a full legal audit.
- Performance optimization, new GPU validation, benchmark semantic changes, or
  paper-scale validation.
- Full 235-problem paper-scale validation and upstream SOLAR parity.
- MI300X full-suite hardware validation on CDNA3 unless complete real-hardware
  evidence becomes available.
- CDNA4 validation because suitable hardware is not currently accessible.
- Hosted leaderboard, remote submission operations, and hard multi-tenant
  sandboxing.
- Rebranding as a stable benchmark authority release.

## Recently Shipped Milestone: v1.26 Public Prerelease and Research Preview

**Goal:** Produce a publishable engineering prerelease and research preview
package with versioned validation artifacts, release-readiness checks, and
bounded research interpretation that matches the project's actual ROCm
evidence.

**Shipped outcome:** v1.26 added a reproducible prerelease artifact bundle,
readiness gates, research-preview evidence, and public release materials while
keeping paper parity, leaderboard authority, hard sandboxing, full MI300X validation on CDNA3
validation, and CDNA4 validation out of scope.

## Recently Shipped Milestone: v1.24 Dataset Batch Run Trustworthiness

**Goal:** Make dataset-scale reuse, provenance, closure, failure reporting, and
sharding auditable after the single-evaluation boundary has been tightened.

**Shipped outcome:** Dataset reuse and stale-provenance decisions are now
package-owned helpers; closure records classify missing trace and missing
evidence states through shared helpers; failure-mode docs pin CPU-safe
regression coverage; deterministic sharding helpers define stable shard ids,
per-shard trace refs, ordered merges, duplicate detection, and incomplete-shard
reporting.

**Explicitly deferred:**
- Remote dataset cache/index support beyond checksum-preserving local
  provenance.
- Parallel dataset process scheduling and live ROCm shard execution.
- Hardware-validation, paper-scale parity, leaderboard readiness, and complete
  hard sandboxing.

## Recently Shipped Milestone: v1.23 Evaluation Reliability and Security Hardening

**Goal:** Make single-problem evaluation more trustworthy, diagnosable, and
maintainable before further dataset-scale or hardware-validation work.

**Shipped outcome:**
- Persist bounded stdout/stderr or equivalent structured diagnostics for
  no-trace and noisy-output evaluation failures.
- Load staged Python submissions through unique file-based module identities so
  submitted filenames cannot collide with already-imported modules.
- Reject dangerous native compiler/linker options while preserving documented
  ROCm/HIP extension use cases.
- Thin the generated eval driver by moving trace emission and reward-hack
  boundary behavior into tested importable helpers.

**Explicitly deferred:**
- Complete OS/container hard sandboxing for untrusted or multi-tenant
  submissions.
- CDNA3-family hardware validation, including MI300X, CDNA4, or native-host full-suite validation.
- Full 235-problem paper-scale SOLAR, upstream SOLAR, hosted leaderboard, or
  NVIDIA B200/Blackwell equivalence claims.
- Canonical Trace, Definition, Workload, Solution, correctness, timing, score,
  or evaluator contract schema changes unless separately approved.

## Recently Shipped Milestone: v1.22 Concern Closure and Execution Boundary Hardening

**Goal:** Fix the remaining code-actionable concerns in
`.planning/codebase/CONCERNS.md` while preserving benchmark contracts and
explicitly deferring hardware-validation, paper-parity, leaderboard, and hard
sandbox responsibilities that require external evidence or architecture.

**Target features:**
- Continued dataset runner closure by extracting solution wrapping, CLI
  invocation, report writing, closure assembly, and runner scheduling seams
  from `scripts/run_dataset.py`.
- Continued generated eval-driver thinning by moving reference timing,
  correctness/timing orchestration, trace emission, and stdout-framing behavior
  into importable helpers with focused tests.
- Replaced the `stream` to `strm` source rewriting workaround with token-aware
  or AST-aware handling plus regressions for comments, strings, and legitimate
  identifiers.
- Made reference-timing failures explicit in traces, logs, or status semantics
  when reference benchmarking is requested.
- Expanded source-review and reward-hack coverage toward AST/source-review
  evidence while keeping the benchmark boundary clear.
- Added family-specific golden fixtures and helper boundaries for SOLAR and AMD
  bound derivation.
- Improved static-evidence artifact manifest and dataset closure provenance
  guardrails without changing diagnostic-only authority.
- Restored `CONCERNS.md` milestone-management context so refreshed concern maps
  preserve which concerns were narrowed, still actionable, or externally
  deferred.

**Explicitly deferred:**
- CDNA3-family hardware validation, including MI300X, CDNA4, or native-host full-suite validation.
- Full 235-problem paper-scale SOLAR, upstream SOLAR, hosted leaderboard, or
  NVIDIA B200/Blackwell equivalence claims.
- Complete OS/container hard sandboxing for untrusted or multi-tenant
  submissions.
- Large PyTorch/ROCm dependency relocking or Docker privilege-model redesign.

**Shipped outcome:** v1.22 closed or narrowed the remaining code-actionable
concern-map items while preserving canonical contracts and carrying external
hardware, paper-scale, leaderboard, and hard-sandbox boundaries forward.

## Recently Shipped Milestone: v1.21 Codebase Debt Reduction and Execution Boundary Hardening

**Goal:** Fix concerns that can be resolved through code, tests, documentation,
and repeatable local verification while making benchmark responsibility
boundaries explicit for issues that require external isolation or hardware
evidence.

**Target features:**
- Dataset runner decomposition so `scripts/run_dataset.py` delegates selection,
  run-state, closure/provenance, derived evidence discovery, and output
  persistence to tested package helpers.
- Generated eval driver thinning so deterministic helper behavior moves into
  importable runtime modules with focused unit tests while the template remains
  a staging/subprocess integration shell.
- Analysis module decomposition for AMD/SOLAR scoring and static kernel
  evidence, isolating graph extraction, operation-family classification, bound
  formulas, artifact/tool parsing, and report rendering behind smaller tested
  helpers.
- Execution-boundary hardening through additional reward-hack catalog tests,
  clock/timing/static-evidence fixtures, dataset resume/closure combinations,
  and explicit security/claim guardrails.
- Documentation and public guardrails that separate benchmark-owned fixes from
  responsibilities that require hard sandboxing, real hardware validation,
  paper-scale evidence, or hosted leaderboard infrastructure.

**Explicitly deferred:**
- Complete OS/container hard sandboxing for untrusted or multi-tenant
  submissions.
- CDNA3-family hardware validation, including MI300X, CDNA4, or native-host full-suite validation.
- Full 235-problem paper-scale SOLAR, upstream SOLAR, hosted leaderboard, or
  NVIDIA B200/Blackwell equivalence claims.
- One-for-one native ROCm replacement proof for every former NVIDIA library
  category.
- Large PyTorch/ROCm dependency relocking or Docker privilege-model redesign.

**Shipped outcome:** v1.21 narrowed major codebase concerns through focused
helper extraction, boundary tests, and documentation updates. Remaining
code-actionable concerns were carried into v1.22; hardware-validation,
paper-scale parity, leaderboard, and hard sandbox work remained deferred.

## Recently Shipped Milestone: v1.20 Cross-Report Consistency and Evaluation Stability

**Goal:** Make ROCm benchmark evidence internally consistent and timing-quality
aware before expanding hardware validation.

**Target features:**
- Cross-report consistency lint across execution closure, paper denominator,
  Matrix, runtime/static evidence, AMD score, AMD SOL/SOLAR, and docs claim
  boundaries.
- `evaluation_stability.v1` diagnostic sidecar for warmup/repeat counts,
  timing distribution, clock policy, backend, variance, and profiling-overhead
  risk.
- Explicit claim-upgrade rules defining what evidence is required for
  container/native-host validation, score authority, paper-parity candidate
  status, and leaderboard readiness.
- Researcher-facing trust summary that explains whether a run is internally
  consistent, stable enough to interpret, and still diagnostic-only.
- CPU-safe guardrail tests plus focused ROCm E2E coverage for real
  timing/closure/stability paths.

**Explicitly deferred:**
- Full 235-problem paper validation, CDNA3-family validation, including MI300X, CDNA4 validation,
  native-host Matrix authority, hosted leaderboard readiness, and upstream
  SOLAR parity.

**Shipped outcome:** v1.20 delivered local JSON/Markdown sidecars and scripts
for consistency, timing stability, claim-upgrade prerequisites, and trust
summaries. It also added docs, fixtures, guardrail tests, and a full
cross-script E2E chain. Final milestone audit passed with 23/23 requirements,
5/5 integration checks, 5/5 flows, and no residual blockers or tech debt.

## Recently Shipped Milestone: v1.19 Research Credibility Without New Hardware

**Goal:** Improve the ROCm port's research credibility without expanding
hardware validation by making paper denominator status, compatibility-matrix
changes, dataset-runner closure, and AMD SOL/SOLAR sanity evidence auditable on
the existing RDNA 4 and Docker evidence surface.

**Target features:**
- Paper dataset denominator accounting for ready, blocked, unsupported,
  deferred, and evidence-missing benchmark problems.
- Compatibility Matrix diff tooling and JSON schema export for downstream CI or
  external evidence producers.
- Dataset-runner hardening for failure classification, resume/manifest
  consistency, and stable closure reporting.
- AMD SOL/SOLAR bound sanity checks and documentation that clarify provisional
  RDNA 4 model risk without adding new hardware-validation claims.
- Documentation and guardrail tests that keep curated, Docker, denominator, and
  bound-sanity evidence separate from paper parity, score authority, leaderboard
  authority, and native host validation.

**Shipped outcome:** v1.19 delivered deterministic CPU-safe sidecar/reporting
contracts and documentation for closure, denominator, Matrix, dataset-runner,
and AMD SOL/SOLAR sanity evidence. Final milestone audit passed with 28/28
requirements wired, 8/8 end-to-end flows complete, and no residual blockers or
tech debt. No new hardware validation, Docker privilege expansion, or dependency
relock was performed.

## Recently Shipped Milestone: v1.17 Static Kernel Evidence

**Goal:** Add diagnostic static kernel evidence by capturing ROCm compiler
artifacts, extracting ISA and metadata through routed static tools, and
publishing sidecars and reports that remain separate from correctness,
performance, paper-parity, and leaderboard claims.

**Target features:**
- Code-object or HSACO capture from ROCm solution builds where the current
  build path exposes stable artifacts.
- Static metadata and ISA extraction using routed tools such as RGA,
  `llvm-objdump`, `roc-objdump`, or `readelf` when available.
- A machine-readable `static_kernel_evidence.v1` sidecar with provenance,
  artifact paths, selected tool, unavailable states, and classification
  results.
- Compiler/static artifact reports that researchers can inspect alongside
  traces, profiles, and AMD-native derived score artifacts.
- Documentation and claim guardrails that define static evidence as diagnostic
  static-analysis evidence, not correctness, performance, paper-parity, or
  leaderboard authority.

**Explicitly deferred:**
- Making static evidence mandatory for every benchmark run.
- Treating static evidence as score, correctness, timing, or leaderboard
  authority.
- Full paper-scale 235-problem static artifact coverage unless the milestone
  requirements explicitly include a bounded subset.
- Full 235-problem paper parity, original 124-model extraction, hosted
  leaderboard support, and full CDNA 3 / CDNA 4 validation remain out of this
  milestone.

**Shipped outcome:** v1.17 delivered the sidecar contract, current-build
artifact discovery, routed `llvm-objdump` / `readelf` extraction, CLI
`--static-evidence auto`, public docs, claim guardrails, and a bounded RDNA 4
validation artifact. CDNA 3/CDNA 4 live validation, Triton cache capture,
RGA-rich parsing, and paper-scale static coverage remain deferred.

## Recently Shipped Milestone: v1.16 ROCm Toolchain Research and Capability Routing

**Goal:** Systematically research ROCm's fragmented toolchain ecosystem and
build a capability routing model that tells the benchmark which tools are
available for a given hardware generation, GPU architecture, ROCm version,
artifact type, and evidence level.

**Target features:**
- External research over ROCm Systems, ROCprofiler SDK, RGA/GPUOpen manuals,
  HIP compiler documentation, LLVM/object tools, and repository migration
  status.
- A tool inventory and lifecycle model covering active, deprecated, migrated,
  planned, and rejected tools.
- A capability registry schema for tool, hardware generation, GPU architecture,
  ROCm version, artifact type, evidence level, status, and reason fields.
- A routing policy that combines static registry data with dynamic probes,
  fallback decisions, and explicit unavailable or unsupported reasons.
- Documentation, cookbook guidance, and claim guardrails for ROCm toolchain
  availability without treating routing as correctness, performance, or
  leaderboard authority.

**Explicitly deferred:**
- Static artifact capture, HSACO/code-object collection, ISA extraction,
  `static_kernel_evidence.v1`, and compiler artifact reports were deferred to
  v1.17.
- Full 235-problem paper parity, original 124-model extraction, hosted
  leaderboard support, and full CDNA 3 / CDNA 4 validation remained out of this
  milestone.

## Recently Shipped Milestone: v1.15 Research-Grade ROCm Benchmark Release

**Goal:** Turn the current ROCm port into a small, complete, research-grade
release that external GPU kernel researchers can understand, reproduce, cite,
and extend.

**Target features:**
- A claim-boundary guide that states what the project can and cannot claim
  today, and what evidence is required to upgrade claims.
- A curated ROCm benchmark slice with representative kernel/problem coverage
  and stable artifact expectations.
- Researcher-facing workflows and cookbooks for kernel authors, compiler or
  backend researchers, agent optimization researchers, and benchmark
  reproducibility users.
- A reproducibility closure bundle that records commands, artifacts, known
  gaps, and release evidence for the curated slice.

## Recently Shipped Milestone: v1.14 Optional rocprofv3 Profiling Evidence

**Goal:** Add opt-in `rocprofv3` profiling artifacts as benchmark evidence so
anomalous or hardware-sensitive results can be diagnosed without changing
correctness, timing, scoring, or default execution behavior.

**Target features:**
- Add an explicit profiling option such as `--profile rocprofv3`.
- Record profiler command provenance, availability, skipped/unavailable states,
  exit status, and stdout/stderr tails.
- Persist profiler artifacts with stable paths, preferring `rocpd` where
  supported and CSV outputs where available.
- Report profiling evidence as optional diagnostics, not correctness or score
  authority.

**Explicitly deferred:**
- Making profiling mandatory for benchmark runs.
- ROCm Compute Profiler roofline analysis.
- RGA/code-object/static ISA analysis.
- Contract version bump unless profiling becomes a required consumer contract.

Validation status:

- RDNA 4 (`gfx1200`) full adapted suite passed: `462 passed, 58 skipped`.
- CDNA 3 (`gfx940`, `gfx941`, `gfx942`) is supported by schema/build/docs.
- CDNA 3 (`gfx94*`) full adapted suite validation remains deferred by user
  decision and must be completed before claiming CDNA 3 hardware validation.

## Recently Shipped Milestone: v1.10 Paper-Aligned SOLAR Automatic Derivation

**Goal:** Upgrade the AMD SOL/SOLAR derived bound pipeline into a
paper-aligned automatic SOLAR derivation system that can extract richer
workload graphs, model currently unsupported operation families, produce
auditable hardware-bound evidence, and protect score reports from partial
coverage misuse.

**Target features:**
- Extend BoundGraph/SOLAR IR so attention, MoE, convolution, SSM/Mamba,
  embedding/positional, and linear projection families have explicit extraction
  and estimation paths instead of taxonomy-only placeholders.
- Derive FLOP, byte, movement, intermediate, confidence, and rationale evidence
  from reference/workload structure with deterministic unsupported/inexact
  degradation.
- Strengthen AMD SOL v2 sidecars and AMD-native score reports so complete,
  degraded, and unscored states are machine-verifiable.
- Preserve ROCm-only scope, canonical trace JSONL, public schemas, and primary
  CLI behavior.

**Explicitly deferred:**
- Original-paper 124-model / 235-problem extraction and curation pipeline.
- MI300X real-hardware validation on CDNA 3 and CDNA 4 real-hardware validation.
- NVFP4/MXFP4 hardware validation.
- Official hosted leaderboard or submission service.

## Recently Shipped Milestone: v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure

**Goal:** Turn the paper's public benchmark dataset into a concrete,
auditable ROCm-port surface: acquired or layout-verified, counted, classified,
diagnosed, runnable in small ready subsets, and reported with explicit parity
and closure status.

**Target features:**
- Define and enforce the dataset acquisition/layout contract for the public
  SOL-ExecBench benchmark categories: `L1`, `L2`, `Quant`, and
  `FlashInfer-Bench`.
- Generate a machine-readable paper parity inventory with problem counts,
  categories, dtype coverage, forward/backward indicators, custom input and
  safetensors usage, reference availability, and solution availability.
- Classify each problem's ROCm readiness as ready, schema/input blocked, dtype
  blocked, custom-input blocked, runtime blocked, unsupported NVIDIA-only path,
  or needing hardware evidence.
- Extend the dataset runner and reports so ready problems can run in small
  batches and emit canonical traces, summary JSON, AMD-native score reports,
  AMD SOL/SOLAR sidecars, and SOLAR derivation sidecars.
- Produce a parity gap report with claim guardrails that distinguish inventory
  completion from full 235-problem ROCm validation or leaderboard equivalence.

**Explicitly deferred:**
- Recreating the original 124-model / 7,400-subgraph extraction and curation
  pipeline.
- Full upstream NVlabs/SOLAR equivalence.
- CDNA3-family real-hardware validation, including MI300X, CDNA 4, and NVFP4/MXFP4 real-hardware validation.
- Hosted leaderboard or remote submission service.
- Full 235-problem real-hardware validation unless this milestone discovers
  the environment and runtime budget are sufficient.

## Recently Shipped Milestone: v1.9 AMD SOL/SOLAR Bound Modeling Completion

**Goal:** Make the ROCm port's AMD SOL bound pipeline credible, structured,
auditable, and broad enough to support paper-style SOL scoring on the current
ROCm path, with validation scoped to RDNA 4 only.

**Target features:**
- Upgrade the current conservative AST-only estimator into a structured graph/IR
  with explicit operation coverage.
- Add auditable FLOP, byte, and memory-movement bound generation for common SOL
  ExecBench operator families.
- Externalize AMD hardware models and validation/confidence metadata instead of
  relying only on hard-coded provisional defaults.
- Integrate improved bound artifacts with AMD-native score reports and clear
  unsupported/inexact degradation behavior.
- Add golden tests, documentation, and RDNA 4 validation coverage for the new
  modeling pipeline.

**Explicitly deferred:**
- CDNA3-family real-hardware validation, including MI300X.
- CDNA 4 validation.
- NVFP4/MXFP4 hardware validation unless a suitable AMD path is separately
  established.
- Full original 124-model extraction pipeline unless it is needed only as
  reference context.

## Requirements

### Validated

- Existing CLI can load definitions, workloads, solutions, and emit trace results through `sol-execbench` - existing.
- Existing driver stages problem files, compiles native solutions, and runs evaluation in a subprocess - existing.
- Existing Pydantic schemas define public `definition.json`, `workload.jsonl`, `solution.json`, and trace contracts - existing.
- Existing benchmark helpers cover input generation, correctness checks, timing, clock checks, environment capture, and reward-hack detection - existing.
- ROCm >= 7.0 Docker/runtime baseline with HIP compiler tooling, ROCm profiling tools, PyTorch ROCm, Triton ROCm, and selected ROCm library smoke coverage - v1.0.
- ROCm-only solution metadata and schema validation with HIP/C++, AMD gfx targets, `hip_cflags`, and explicit rejection guidance for legacy CUDA/NVIDIA values - v1.0.
- HIP/C++ solution sources can be staged, compiled, and loaded through the existing driver flow with AMD offload architecture handling - v1.0.
- PyTorch ROCm, Triton ROCm, and HIP/C++ solutions run through the isolated eval driver and produce valid trace JSONL - v1.0.
- Benchmark timing no longer depends on CUPTI or CUDA-only tooling and uses ROCm-compatible PyTorch device events - v1.0.
- Environment and clock checks use AMD/ROCm tooling and report ROCm/HIP/PyTorch hardware context - v1.0.
- Public examples and sample categories are migrated to ROCm-compatible execution, documented replacements, or explicit fallback notes - v1.0.
- Adapted pytest markers and tests use ROCm/AMD semantics, with reward-hack defenses still active - v1.0.
- Documentation explains ROCm setup, solution schema, trace schema, analysis workflow, compliance, unsupported NVIDIA runtime features, and known gaps - v1.0.
- CDNA 3 `gfx94*` schema targets and HIP offload staging are supported without claiming hardware validation - v1.1.
- Active source, tests, docs, examples, scripts, Docker, README, and pyproject have maintained CUDA/NVIDIA residue audit coverage - v1.1.
- Public native examples use HIP-facing paths and solution filenames - v1.1.
- Former NVIDIA library/DSL examples use ROCm compatibility-example terminology and portable examples include CDNA 3 metadata where appropriate - v1.1.
- CDNA 3 hardware validation handoff is documented for the next milestone - v1.1.
- `hip-execbench` engineering practices have been assessed and mapped to
  accepted, rejected, and deferred SOL ExecBench ROCm adaptations - v1.2.
- Internal ROCm diagnostics and trace reporting helpers improve operator
  inspection without changing public CLI or trace contracts - v1.2.
- SOL-Score interpretation guardrails warn against unsupported AMD-native
  performance claims while preserving the existing score formula - v1.2.
- Public contract guardrail tests protect schemas, CLI help, trace behavior,
  HIP-facing examples, and CDNA 3 validation deferral language - v1.2.
- Original NVIDIA SOL ExecBench public functionality is mapped to ROCm port
  disposition with tests protecting parity documentation - v1.3.
- Public baseline comparison over existing trace JSONL is available through
  `sol-execbench-baseline` without changing trace schemas or the main benchmark
  CLI - v1.3.
- ROCm library categories distinguish supported HIP/C++ from candidate
  `hipblas`, `miopen`, `ck`, and `rocwmma` replacement directions - v1.3.
- `hip-execbench` baseline/reporting practices have been selectively adapted or
  explicitly rejected according to SOL ExecBench public-contract constraints -
  v1.3.
- Non-CDNA validation debt is closed; real CDNA 3 hardware validation remains
  the only deferred project-level item - v1.3.
- `hip-execbench` engineering practice adaptation, internal evidence/report
  modeling, CDNA 3 validation readiness metadata, and RDNA 4 validation
  evidence are implemented without changing public SOL ExecBench ROCm
  contracts - v1.4.
- Source-specific timing policy, `rocprofv3` timing evidence helpers, AMD SOL
  bound artifacts, and derived AMD-native score reports are implemented without
  mutating canonical trace JSONL or claiming NVIDIA B200/SOLAR/leaderboard
  equivalence - v1.5.
- Broader AMD SOL analyzer coverage, live `rocprofv3` collection helpers,
  dataset-runner AMD-native score reports, and v1.6 compatibility/claim
  guardrails are implemented without breaking canonical trace JSONL, public
  schemas, or primary `sol-execbench` CLI defaults - v1.6.
- Optimized scoring baseline artifacts, explicit baseline-source semantics,
  dataset integration, and baseline documentation are implemented without
  mutating canonical trace JSONL - v1.7.
- Source-specific ROCm profiler timing evidence collection, dataset integration,
  parser coverage, and auditable timing metadata are implemented - v1.7.
- Static reward-hack review now covers hidden streams, semantic caches,
  unauthorized loaders/file I/O, opaque payloads, and precision downgrade
  patterns before submitted Python source import - v1.7.
- `hipblas` has a runnable public SGEMM example and native staging tests, while
  MIOpen, CK, and rocWMMA were documented candidate categories with
  overclaiming guardrails - v1.7.
- MI300X-as-CDNA3 validation instructions, FP8/NVFP4 decision records, and
  evidence gates for validation claims are implemented - v1.7.
- MIOpen, Composable Kernel, and rocWMMA each have scoped native ROCm public
  examples, native staging tests, dependency diagnostics, RDNA 4 E2E
  registration, and support-status docs - v1.8.
- Former NVIDIA library/DSL compatibility paths are mapped to supported ROCm
  examples or kept explicitly as PyTorch compatibility examples - v1.8.
- v1.8 library validation claims are scoped to RDNA 4; CDNA 3 and CDNA 4
  validation remain deferred - v1.8.
- AMD SOL/SOLAR operator modeling now provides rich per-node formulas, formula
  inputs, read/write/intermediate/movement/total byte evidence, confidence,
  rationale, and legacy v1 `WorkEstimate` adaptation for GEMM/BMM, elementwise,
  activation, reduction, normalization, softmax, data movement, and dtype
  conversion families - v1.9 Phase 43.
- AMD SOL/SOLAR bound modeling now has strict AMD hardware model artifacts,
  structured bound graph IR extraction, rich operator estimates, v2 bound
  sidecars, coverage/warning semantics, AMD-native score and dataset
  integration, documentation, and public-contract guardrails - v1.9.
- v1.9 validation claims are scoped to RDNA 4 (`gfx1200`); CDNA3-family
  real-hardware validation including MI300X and CDNA 4 validation remain
  deferred - v1.9.
- Paper-aligned automatic SOLAR derivation now has sidecar-only fixture
  contracts, semantic provenance, formula/byte/bound evidence for scoped
  families, degraded MoE and SSM/Mamba evidence, coverage/status sidecars,
  AMD-native score guards, dataset-runner generated SOLAR sidecars,
  `derived_evidence_refs`, documentation, and public/claim guardrails - v1.10.

### Active

- Cross-report consistency lint, evaluation stability evidence, claim-upgrade
  rules, trust summary documentation, and guardrail tests for v1.20.

### Out of Scope

- Maintaining CUDA/NVIDIA runtime compatibility - ROCm is the target platform for this port.
- Preserving NVIDIA-specific dependency implementations when a ROCm replacement is required - equivalent behavior matters more than retaining original code.
- Guaranteeing one-to-one high-performance replacements for every NVIDIA DSL - some former CUTLASS/cuDNN/CuTe/cuTile categories remain documented compatibility examples or candidates.
- Claiming CDNA 3 hardware validation before a full `gfx94*` suite pass is recorded.
- Claiming NVIDIA leaderboard equivalence or B200/SOLAR equivalence for AMD-native scores.
- Reintroducing NVIDIA/CUDA runtime parity as a goal - this project remains a
  ROCm-only port.
- Recreating the original paper's 124-model extraction and curation pipeline in
  v1.7 - explicitly deferred.
- Completing full upstream SOLAR parity in v1.7 - explicitly deferred.
- Claiming NVFP4/MXFP4 hardware validation without suitable AMD hardware
  evidence - explicitly deferred.
- CDNA3-family real-hardware validation, including MI300X, in v1.9 - explicitly deferred so
  the milestone can focus on modeling correctness and RDNA 4 validation.
- CDNA 4 hardware validation in v1.9 - explicitly deferred.
- Full original 124-model extraction pipeline in v1.9 - deferred unless needed
  only as reference context for bound-modeling decisions.
- Full upstream NVlabs/SOLAR equivalence in v1.11 - deferred so the milestone
  can focus on public dataset parity inventory and ROCm execution closure.
- Full 235-problem real-hardware validation in v1.11 - deferred unless the
  environment and runtime budget are proven sufficient during the milestone.
- Full 235-problem paper parity in v1.15 - deferred so the milestone can focus
  on a small, complete, reproducible research preview.
- Original 124-model extraction and curation pipeline in v1.15 - deferred as a
  larger paper-parity effort.
- Hosted leaderboard support in v1.15 - deferred until the local evidence and
  claim model are stable enough for external submissions.
- Static RGA/code-object/GPUOpen ISA analysis in v1.15 - deferred until after
  v1.16 establishes ROCm toolchain capability routing.
- Static kernel evidence in v1.16 - explicitly deferred to v1.17 so the
  toolchain research, lifecycle model, registry, and routing policy can land
  first.
- CDNA3-family real-hardware validation, including MI300X, and CDNA 4 real-hardware validation in v1.19 - explicitly
  deferred by current user direction so the milestone can improve research
  credibility without adding hardware-validation scope.
- Native host ROCm 7.0.x / 7.1.x / 7.2.x validation in v1.19 - deferred so the
  milestone can build audit/reporting infrastructure on existing Docker
  container user-space evidence.
- Full 235-problem paper validation, official leaderboard parity, and hosted
  submission service in v1.19 - deferred until denominator accounting and
  closure reporting are stronger.
- Full 235-problem paper validation, CDNA3-family validation, including MI300X, CDNA 4 validation,
  native-host Matrix authority, hosted leaderboard readiness, and upstream
  SOLAR parity in v1.20 - deferred so this milestone can make current and
  future evidence internally auditable before adding expensive validation
  claims.

## Context

The current codebase is now ROCm-oriented. `docker/Dockerfile` uses a ROCm
development base image. Python dependency declarations point PyTorch and
torchvision at ROCm wheels. The package remains a Python CLI benchmark runner:
`src/sol_execbench/cli/main.py` loads problem JSON, `ProblemPackager` stages
files and injects AMD architecture flags, and `src/sol_execbench/driver/templates/`
contains the HIP-aware build and eval subprocess scripts.

Current high-risk areas after v1.10 are real MI300X hardware validation on CDNA3,
CDNA 4 validation, FP8/NVFP4/MXFP4 evidence, hosted leaderboard/service
behavior, and original-paper dataset extraction. v1.10 intentionally completed
the local SOLAR derivation system itself, not dataset-scale extraction or
commercial GPU validation. No CDNA3-family validation claim, including MI300X, should be made
until a full adapted suite run and required environment evidence are archived in
a later milestone.

The v1.8 milestone was scoped to RDNA 4 validation only. CDNA 3 and CDNA 4
library validation remain intentionally deferred; current artifacts preserve
schema, documentation, and no-claim guardrails for those architectures.

After v1.15, the highest leverage next step is not full paper parity or static
ISA extraction itself. The project needs a ROCm toolchain map and routing layer
because ROCm Systems, ROCprofiler SDK, RGA/GPUOpen tools, HIP compiler
metadata, LLVM object tools, and migrated/deprecated repositories expose
different capabilities depending on ROCm version, hardware generation, GPU
architecture, artifact type, and installation state.

After v1.18, the project has strong sidecar and claim-boundary infrastructure
for ROCm compatibility evidence, including Docker container user-space rows for
ROCm 7.0.x, 7.1.x, and 7.2.x on recorded RDNA 4 host devices. The next useful
step is to make research claims easier to audit without requiring new hardware:
paper denominator accounting, Matrix diff/schema tooling, dataset-runner
closure hardening, and bounded AMD SOL/SOLAR sanity evidence.

After v1.19, the project has many independent evidence sidecars and guardrail
docs. The next useful step is to verify that those reports do not contradict
each other and that real benchmark timing evidence is stable enough to interpret
before using it to support stronger hardware, score, or paper-parity claims.

After v1.20, the project has strong evidence-quality sidecars and claim gates,
but `.planning/codebase/CONCERNS.md` identifies maintainability and boundary
debt in the dataset runner, generated evaluator template, AMD/SOLAR analysis
modules, static evidence, and execution-boundary disclosures. The next useful
step is to reduce that debt while preserving benchmark contracts and clearly
deferring hard sandboxing, new hardware validation, paper-scale parity, and
leaderboard readiness.

## Constraints

- **Platform**: ROCm >= 7.0 is the supported software baseline.
- **Hardware**: RDNA 4 is validated; CDNA 3 is code/schema-supported but hardware validation remains deferred.
- **Compatibility**: Preserve SOL ExecBench benchmark semantics and public schemas unless a ROCm-specific change is unavoidable.
- **Scope**: NVIDIA/CUDA runtime support is intentionally not maintained.
- **Licensing**: All retained and replacement code must comply with the repository LICENSE and third-party dependency obligations.
- **Quality**: Migrated tests, examples, Docker checks, and end-to-end evaluation must pass under ROCm.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ROCm-only port | User explicitly chose to replace NVIDIA/CUDA paths rather than maintain dual backend compatibility. | Validated in v1.0 |
| Broad solution ecosystem target | User chose to pursue ROCm equivalents for all original solution categories, not just HIP-only or a minimal subset. | Partially validated in v1.1; some former NVIDIA DSL categories remain compatibility examples/documented candidates |
| Adapted test suite is the completion gate | User defined done as passing the migrated existing tests on ROCm, rather than adding a separate parity suite as v1 scope. | Validated on RDNA 4 in v1.0 |
| Preserve benchmark standards | The port should remain consistent with the NVIDIA SOL ExecBench paper and implementation standards wherever feasible. | Validated for public schemas and eval semantics in v1.0 |
| Defer CDNA 3 validation | No CDNA 3 hardware run was available during closure. | Accepted as v1.0 technical debt |
| Separate CDNA 3 code support from hardware validation | User requested CDNA 3 support in v1.1 while deferring real hardware validation to the next milestone. | Validated in v1.1 |
| Preserve public interfaces during practice harvest | User requested borrowing engineering experience from `hip-execbench` without changing this project's external interfaces or formats. | Validated in v1.2 |
| Close non-CDNA gaps before CDNA hardware validation | User requested v1.3 converge all issues except CDNA 3 validation, using NVIDIA SOL ExecBench parity and `hip-execbench` engineering experience as references. | Validated in v1.3 |
| Preserve compatibility during v1.4 engineering adaptation | User requested comprehensive `hip-execbench` engineering practice adaptation without breaking existing CLI, schema, trace, solution, or benchmark semantics. | Validated in v1.4 |
| Implement CDNA 3 readiness without hardware claim | User clarified v1.4 should implement CDNA 3 validation workflow readiness but not perform or claim real CDNA 3 validation. | Validated in v1.4 |
| RDNA 4 is the validation platform for v1.4 | User clarified v1.4 implementation must be validated on RDNA 4 with unit and E2E evidence. | Validated in v1.4 |
| Build a SOLAR-like AMD model | User selected a SOLAR-like pipeline over a minimal configuration-only roofline model for v1.5. | Validated in v1.5 |
| Replace default timing with profiler timing | User selected replacing the default event timing path with a profiler-backed path for v1.5. | Validated in v1.5 as policy/evidence foundation |
| Accuracy beats unified timing semantics | User clarified that Triton, HIP native, and PyTorch timing semantics must be investigated separately; if one unified timing definition is inaccurate, expose operator-source-specific timer backends. | Validated in v1.5 |
| Exclude CDNA 3 hardware validation from v1.5 | User explicitly said the next milestone does not include CDNA 3 validation. | Validated in v1.5 |
| Preserve public contracts during v1.6 workflow integration | User made canonical trace JSONL, schema, and primary CLI compatibility a hard constraint for v1.6. | Validated in v1.6 |
| Keep CDNA 3 validation out of v1.6 | User said the next milestone's forced constraint still excludes CDNA 3 validation. | Validated in v1.6 |
| Expose source-specific timer chimneys when needed | User required timing accuracy over one unified口径 for Triton, HIP, and PyTorch sources. | Validated in v1.6 |
| Treat NVIDIA/B200 parity as non-scope | User clarified the project is ROCm-only and original NVIDIA hardware should not be counted as a missing feature. | Validated in v1.7 |
| Use MI300X as commercial validation target | User selected AMD MI300X as the CDNA3 real-hardware validation example, including FP8 validation when available. | Validated in v1.7 readiness docs |
| Defer dataset extraction and full SOLAR parity | User explicitly postponed the original paper's extraction pipeline and deeper SOLAR parity work. | Deferred |
| Prioritize baseline, timing, reward-hack, and ROCm libraries | User marked scoring baseline, evaluation timing loop, reward-hack defenses, and ROCm library migration as the implementation priorities. | Validated in v1.7 |
| Defer NVFP4/MXFP4 validation | User noted MI300X can validate FP8 later, but no hardware is available for NVFP4/MXFP4 validation now. | Deferred |
| Complete ROCm library replacement ecosystem on RDNA 4 | User requested the next milestone focus on thoroughly resolving incomplete ROCm library replacements, and clarified only RDNA 4 validation is in scope. | Validated in v1.8 |
| Focus v1.9 on AMD SOL/SOLAR bound modeling | User requested the next milestone completely solve AMD SOL/SOLAR bound modeling, with validation scoped to RDNA 4. | Validated in v1.9 |
| Defer CDNA3-family and CDNA 4 validation from v1.9 | User clarified MI300X belongs to the CDNA3 validation scope rather than a separate architecture item, and CDNA 4 remains deferred. | Validated in v1.9 |
| Scope v1.10 to SOLAR derivation only | User chose to upgrade "论文级完整 SOLAR 自动推导" as a new milestone while excluding dataset extraction and real-hardware validation from the milestone boundary. | Validated in v1.10 |
| Focus v1.11 on public dataset parity and execution closure | User requested the next milestone target "论文数据集 parity + ROCm execution closure" and then narrowed it to dataset acquisition/layout, inventory, compatibility classification, small ready-subset execution, and gap reporting. | Active in v1.11 |
| Prioritize a research-grade ROCm preview before paper parity | User asked whether to push toward paper parity next; we chose a smaller complete release with claim boundaries, curated slice, researcher workflows, and reproducibility evidence before attempting full 235-problem parity. | Active in v1.15 |
| Route ROCm tools before extracting static evidence | User noted ROCm tools are more fragmented than CUDA tools and requested per-tool hardware generation/model availability routing, including prepared and historical tools. | Active in v1.16 |
| Defer ROCm minor-version by GPU-generation matrix validation | User narrowed the possible compatibility discussion to ROCm 7.0.x/7.1.x/7.2.x but decided the full matrix still has high validation cost and long cycle time. Treat it as deferred compatibility validation rather than a near-term release gate. | Deferred |
| Defer static kernel evidence to v1.17 | User explicitly moved Static Kernel Evidence out of v1.16 so v1.16 can focus on external research, tool lifecycle, capability registry, and routing policy. | Active in v1.16 |
| Avoid expanding hardware validation in v1.19 | User explicitly does not want to expand CDNA3-family validation, including MI300X, CDNA 4, or native host ROCm matrix validation right now. | Active in v1.19 |
| Improve research credibility without new hardware | User chose denominator accounting, Matrix diff/schema export, dataset-runner hardening, and AMD SOL/SOLAR sanity over additional hardware validation. | Active in v1.19 |
| Audit current evidence before expanding validation | User chose cross-report consistency and evaluation stability as the next milestone so future MI300X-on-CDNA3 or paper-scale validation can rely on cleaner evidence. | Active in v1.20 |
| Fix codebase concerns before new validation claims | User chose a v1.21 milestone focused on codebase debt reduction and execution-boundary hardening from `CONCERNS.md`, while deferring hard sandboxing, MI300X-on-CDNA3 validation, paper-scale parity, and leaderboard authority. | Active in v1.21 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-01 after v1.21 milestone start*
