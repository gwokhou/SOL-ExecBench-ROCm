# Roadmap: SOL ExecBench ROCm Port

## Milestones

- ✅ **v1.13 ROCm Runtime Evidence and Environment Diagnostics** —
  Phases 58-60 (shipped 2026-05-25). See
  `.planning/milestones/v1.13-ROADMAP.md`.

- 🚧 **v1.14 Optional rocprofv3 Profiling Evidence** —
  Phases 61-63 (active). See
  `.planning/milestones/v1.14-ROADMAP.md`.

- ✅ **v1.12 Evaluator Contract Metadata and Boundary Guardrails** —
  retroactive quick-task milestone (shipped 2026-05-25). See
  `.planning/milestones/v1.12-ROADMAP.md`.

- ✅ **v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure** —
  Phases 53-57 (shipped 2026-05-23). See
  `.planning/milestones/v1.11-ROADMAP.md`.

- ✅ **v1.10 Paper-Aligned SOLAR Automatic Derivation** — Phases 47-52
  (shipped 2026-05-23). See `.planning/milestones/v1.10-ROADMAP.md`.

- ✅ **v1.9 AMD SOL/SOLAR Bound Modeling Completion** — Phases 41-46
  (shipped 2026-05-23). See `.planning/milestones/v1.9-ROADMAP.md`.

- ✅ **v1.8 ROCm Library Ecosystem Completion** — Phases 36-40 (shipped
  2026-05-22). See `.planning/milestones/v1.8-ROADMAP.md`.

- ✅ **v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library
  Migration** — Phases 31-35 (shipped 2026-05-22). See
  `.planning/milestones/v1.7-ROADMAP.md`.

- ✅ **v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow** —
  Phases 27-30 (shipped 2026-05-22). See
  `.planning/milestones/v1.6-ROADMAP.md`.

- ✅ **v1.5 AMD-native SOL Scoring and ROCm Profiler Timing** — Phases 23-26
  (shipped 2026-05-22). See `.planning/milestones/v1.5-ROADMAP.md`.

- ✅ **v1.4 hip-execbench Engineering Experience Adaptation + Validation
  Workflow Readiness** — shipped 2026-05-22. See
  `.planning/milestones/v1.4-ROADMAP.md`.

- ✅ **v1.3 Non-CDNA Issue Closure** — shipped 2026-05-22. See
  `.planning/milestones/v1.3-ROADMAP.md`.

- ✅ **v1.2 Engineering Practice Harvest and Compatibility Guardrails** —
  shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.

- ✅ **v1.1 CDNA 3 Support and Migration Closure** — shipped 2026-05-21. See
  `.planning/milestones/v1.1-ROADMAP.md`.

- ✅ **v1.0 ROCm Port** — shipped 2026-05-21. See
  `.planning/milestones/v1.0-ROADMAP.md`.

## Current Position

**Active milestone:** none; v1.14 is complete.

**Status:** v1.14 shipped 2026-05-25.

**Next planned milestone:** static kernel evidence with RGA/code-object
analysis and GPUOpen ISA classification remains future candidate work.

## Active Phase Roadmap

### Phase 61: Profiling Option and Command Provenance

**Status:** Complete
**Milestone:** v1.14 Optional rocprofv3 Profiling Evidence
**Goal:** Introduce the opt-in CLI/config surface and profiler command builder.

**Scope:**
- Add `--profile rocprofv3` or equivalent profile selection.
- Validate profiler availability and output directory setup.
- Record command provenance and skipped/unavailable states.

**Requirement IDs:** PROF-01, PROF-02, PROF-03

### Phase 62: rocprofv3 Artifact Lifecycle

**Status:** Complete
**Milestone:** v1.14 Optional rocprofv3 Profiling Evidence
**Goal:** Collect and register profiler artifacts in stable result metadata.

**Scope:**
- Prefer `rocpd` output when supported.
- Register CSV trace/counter/agent-info outputs where available.
- Preserve profiler exit status and stdout/stderr tails.
- Add artifact cleanup/retention behavior consistent with existing run output.

**Requirement IDs:** ART-01, ART-02, ART-03

### Phase 63: Profiling Reports and Documentation

**Status:** Complete
**Milestone:** v1.14 Optional rocprofv3 Profiling Evidence
**Goal:** Make profiling evidence understandable and operationally usable.

**Scope:**
- Add reporting surfaces for profiler status and artifact locations.
- Document Docker/local requirements, permissions, and troubleshooting.
- Add tests for success, skipped, unavailable, and failed profile states.

**Requirement IDs:** REPORT-01, REPORT-02, REPORT-03

## Progress

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.13 ROCm Runtime Evidence and Environment Diagnostics | 58-60 | 5/5 | Complete | 2026-05-25 |
| v1.14 Optional rocprofv3 Profiling Evidence | 61-63 | 3/3 | Complete | 2026-05-25 |
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

- Static kernel evidence with RGA/code-object analysis and GPUOpen ISA
  classification.
- Original paper-scale 124-model / 235-problem extraction and curation.
- MI300X, CDNA 3, and CDNA 4 real-hardware validation.
- NVFP4 and MXFP4 validation if a suitable AMD hardware path exists.
- Hosted leaderboard or submission service.
- NVIDIA Blackwell/B200 comparison methodology, if ever scoped as a separate
  non-ROCm claim analysis effort.
