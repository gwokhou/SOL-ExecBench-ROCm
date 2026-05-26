# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Complete **v1.17 Static Kernel Evidence** —
  Phases 73-77 (shipped 2026-05-26). See
  `.planning/milestones/v1.17-ROADMAP.md`.

- Complete **v1.16 ROCm Toolchain Research and Capability Routing** —
  Phases 68-72 (shipped 2026-05-25). See
  `.planning/milestones/v1.16-ROADMAP.md`.

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

**Active milestone:** none.

**Status:** v1.17 shipped and archived. Start the next milestone with
`$gsd-new-milestone`.

## Future Candidate Work

- CDNA 3 and CDNA 4 live validation with archived hardware-specific runs.
- Triton ROCm cache capture once cache-to-solution provenance is proven.
- RGA-derived VGPR, SGPR, LDS, scratch, occupancy-like, or resource-summary
  parsing after live fixture validation.
- Static instruction-family classification, static sidecar diffs, and
  static/profile kernel-name correlation.
- Dataset-level static evidence aggregation without requiring full paper-scale
  coverage.
- Standalone static artifact analysis for existing `.hsaco`, code object,
  shared object, or ELF inputs.
- ROCm 7.0.x/7.1.x/7.2.x compatibility validation across supported GPU
  generations.
- Original paper-scale 124-model / 235-problem extraction and curation.
- MI300X, CDNA 3, and CDNA 4 real-hardware validation.
- Hosted leaderboard or submission service.
