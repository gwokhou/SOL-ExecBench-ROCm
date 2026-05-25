# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Active **v1.16 ROCm Toolchain Research and Capability Routing** —
  Phases 68-72. This milestone researches ROCm's fragmented tool ecosystem and
  creates a capability registry plus routing policy before static kernel
  evidence is implemented in v1.17.

- Complete **v1.15 Research-Grade ROCm Benchmark Release** —
  Phases 64-67 (shipped 2026-05-25). See
  `.planning/milestones/v1.15-ROADMAP.md`.

- Complete **v1.14 Optional rocprofv3 Profiling Evidence** —
  Phases 61-63 (shipped 2026-05-25). See
  `.planning/milestones/v1.14-ROADMAP.md`.

- Complete **v1.13 ROCm Runtime Evidence and Environment Diagnostics** —
  Phases 58-60 (shipped 2026-05-25). See
  `.planning/milestones/v1.13-ROADMAP.md`.

- Complete **v1.12 Evaluator Contract Metadata and Boundary Guardrails** —
  retroactive quick-task milestone (shipped 2026-05-25). See
  `.planning/milestones/v1.12-ROADMAP.md`.

- Complete **v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure** —
  Phases 53-57 (shipped 2026-05-23). See
  `.planning/milestones/v1.11-ROADMAP.md`.

- Complete **v1.10 Paper-Aligned SOLAR Automatic Derivation** —
  Phases 47-52 (shipped 2026-05-23). See
  `.planning/milestones/v1.10-ROADMAP.md`.

- Complete **v1.9 AMD SOL/SOLAR Bound Modeling Completion** —
  Phases 41-46 (shipped 2026-05-23). See
  `.planning/milestones/v1.9-ROADMAP.md`.

- Complete **v1.8 ROCm Library Ecosystem Completion** —
  Phases 36-40 (shipped 2026-05-22). See
  `.planning/milestones/v1.8-ROADMAP.md`.

- Complete **v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration** —
  Phases 31-35 (shipped 2026-05-22). See
  `.planning/milestones/v1.7-ROADMAP.md`.

- Complete **v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow** —
  Phases 27-30 (shipped 2026-05-22). See
  `.planning/milestones/v1.6-ROADMAP.md`.

- Complete **v1.5 AMD-native SOL Scoring and ROCm Profiler Timing** —
  Phases 23-26 (shipped 2026-05-22). See
  `.planning/milestones/v1.5-ROADMAP.md`.

- Complete **v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness** —
  shipped 2026-05-22. See `.planning/milestones/v1.4-ROADMAP.md`.

- Complete **v1.3 Non-CDNA Issue Closure** — shipped 2026-05-22. See
  `.planning/milestones/v1.3-ROADMAP.md`.

- Complete **v1.2 Engineering Practice Harvest and Compatibility Guardrails** —
  shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.

- Complete **v1.1 CDNA 3 Support and Migration Closure** — shipped
  2026-05-21. See `.planning/milestones/v1.1-ROADMAP.md`.

- Complete **v1.0 ROCm Port** — shipped 2026-05-21. See
  `.planning/milestones/v1.0-ROADMAP.md`.

## Current Position

**Active milestone:** v1.16 ROCm Toolchain Research and Capability Routing.

**Status:** defining requirements and roadmap.

**Milestone goal:** build a research-backed ROCm toolchain capability routing
foundation before implementing static kernel evidence in v1.17.

## Active Phase Roadmap

### Phase 68: External ROCm Toolchain Research

**Status:** Planned
**Milestone:** v1.16 ROCm Toolchain Research and Capability Routing
**Goal:** Gather and record primary-source research for ROCm Systems,
ROCprofiler SDK, RGA/GPUOpen tools, HIP compiler docs, LLVM/object tools, and
repository migration status.

**Scope:**
- Review manuals, docs, and relevant repositories for current and historical
  tool capabilities.
- Extract output formats, probe surfaces, lifecycle status, and repository
  migration signals.
- Separate runtime, profiling, static/future, and derived-score evidence levels.

**Requirement IDs:** RESEARCH-01, RESEARCH-02, RESEARCH-03

**Success criteria:**
1. Research sources and findings are recorded in planning artifacts.
2. Tool capabilities and lifecycle signals are explicit.
3. Future static evidence is represented as deferred scope, not hidden work.

### Phase 69: Toolchain Inventory and Lifecycle Model

**Status:** Planned
**Milestone:** v1.16 ROCm Toolchain Research and Capability Routing
**Goal:** Create a central tool inventory and lifecycle model for active,
deprecated, migrated, planned, rejected, and candidate tools.

**Scope:**
- Define tool identity, display name, lifecycle, replacement, source refs, and
  expected executable metadata.
- Preserve historical and deprecated tools as explicit aliases or entries.
- Include tools already integrated, previously considered, and planned for
  later static evidence.

**Requirement IDs:** TOOL-01, TOOL-02, TOOL-03

**Success criteria:**
1. Current and historical tools have explicit lifecycle states.
2. Migrated/deprecated tools point to replacement or source-of-truth entries.
3. Tool inventory does not imply evidence authority by itself.

### Phase 70: Capability Registry Schema

**Status:** Planned
**Milestone:** v1.16 ROCm Toolchain Research and Capability Routing
**Goal:** Define the schema for mapping tools to hardware generations, GPU
architecture patterns, ROCm versions, artifact types, evidence levels, statuses,
reasons, and source references.

**Scope:**
- Model hardware generation, GPU arch pattern, ROCm version range, artifact
  class, evidence level, status, and reason code.
- Include the complete status vocabulary for available, unavailable,
  unsupported, deprecated, migrated, planned, rejected, and failed states.
- Add parser/serialization tests or guardrails for registry records.

**Requirement IDs:** CAP-01, CAP-02, CAP-03

**Success criteria:**
1. Registry records are structured and machine-verifiable.
2. Unsupported/unavailable states carry auditable reasons.
3. Source references remain attached to capability claims.

### Phase 71: Dynamic Probe and Routing Policy

**Status:** Planned
**Milestone:** v1.16 ROCm Toolchain Research and Capability Routing
**Goal:** Define and implement routing semantics that combine static registry
facts with host dynamic probes and explicit fallback decisions.

**Scope:**
- Probe executable presence, version output, ROCm root, GPU architecture, and
  dry-run/list behavior where safe.
- Return selected tool, fallback, status, and reason without mutating canonical
  trace JSONL.
- Share routing semantics across runtime, profiling, static/future, and
  derived-score evidence while preserving separate authority boundaries.
- Keep Static Kernel Evidence implementation deferred to v1.17.

**Requirement IDs:** ROUTE-01, ROUTE-02, ROUTE-03, ROUTE-04

**Success criteria:**
1. Routing outputs explain why a tool is selected or unavailable.
2. Dynamic probes are bounded and nonfatal.
3. Static-evidence extraction remains out of v1.16 implementation scope.

### Phase 72: Toolchain Matrix Docs and Guardrails

**Status:** Planned
**Milestone:** v1.16 ROCm Toolchain Research and Capability Routing
**Goal:** Document the ROCm toolchain availability matrix, routing cookbook,
and claim boundaries for routing evidence.

**Scope:**
- Add a toolchain matrix document for hardware generation, GPU architecture,
  ROCm version, artifact type, and install/probe state.
- Add cookbook guidance for interpreting routing decisions and unavailable
  reasons.
- Update claim guardrails so routing success is not described as correctness,
  performance, paper parity, or leaderboard authority.

**Requirement IDs:** DOC-01, DOC-02, DOC-03

**Success criteria:**
1. Researchers can understand what tool should be used and why.
2. Unsupported or unavailable states are documented as first-class outcomes.
3. Tests protect claim boundaries for routing and deferred static evidence.

## Progress

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.16 ROCm Toolchain Research and Capability Routing | 68-72 | 0/5 | Active | - |
| v1.15 Research-Grade ROCm Benchmark Release | 64-67 | 4/4 | Complete | 2026-05-25 |
| v1.14 Optional rocprofv3 Profiling Evidence | 61-63 | 3/3 | Complete | 2026-05-25 |
| v1.13 ROCm Runtime Evidence and Environment Diagnostics | 58-60 | 5/5 | Complete | 2026-05-25 |
| v1.12 Evaluator Contract Metadata and Boundary Guardrails | none | quick task 260524-xb3 | Complete | 2026-05-25 |
| v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure | 53-57 | 14/14 | Complete | 2026-05-23 |
| v1.10 Paper-Aligned SOLAR Automatic Derivation | 47-52 | 23/23 | Complete | 2026-05-23 |
| v1.9 AMD SOL/SOLAR Bound Modeling Completion | 41-46 | 17/17 | Complete | 2026-05-23 |
| v1.8 ROCm Library Ecosystem Completion | 36-40 | 5/5 | Complete | 2026-05-22 |
| v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration | 31-35 | 5/5 | Complete | 2026-05-22 |
| v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow | 27-30 | 4/4 | Complete | 2026-05-22 |
| v1.5 AMD-native SOL Scoring and ROCm Profiler Timing | 23-26 | 4/4 | Complete | 2026-05-22 |
| v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness | - | - | Complete | 2026-05-22 |
| v1.3 Non-CDNA Issue Closure | - | - | Complete | 2026-05-22 |
| v1.2 Engineering Practice Harvest and Compatibility Guardrails | - | - | Complete | 2026-05-22 |
| v1.1 CDNA 3 Support and Migration Closure | - | - | Complete | 2026-05-21 |
| v1.0 ROCm Port | - | - | Complete | 2026-05-21 |

## Future Candidate Work

- v1.17 Static Kernel Evidence: code-object/HSACO capture, ISA/metadata
  extraction, extractor adapters, `static_kernel_evidence.v1`, and compiler
  artifact reports.
- Original paper-scale 124-model / 235-problem extraction and curation.
- MI300X, CDNA 3, and CDNA 4 real-hardware validation.
- NVFP4 and MXFP4 validation if a suitable AMD hardware path exists.
- Hosted leaderboard or submission service.
- NVIDIA Blackwell/B200 comparison methodology, if ever scoped as a separate
  non-ROCm claim analysis effort.
