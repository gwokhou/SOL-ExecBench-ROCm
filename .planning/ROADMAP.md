# Roadmap: SOL ExecBench ROCm Port

## Milestones

- ✅ **v1.8 ROCm Library Ecosystem Completion** — Phases 36-40 (shipped 2026-05-22). See `.planning/milestones/v1.8-ROADMAP.md`.
- ✅ **v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration** — Phases 31-35 (shipped 2026-05-22). See `.planning/milestones/v1.7-ROADMAP.md`.
- ✅ **v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow** — Phases 27-30 (shipped 2026-05-22). See `.planning/milestones/v1.6-ROADMAP.md`.
- ✅ **v1.5 AMD-native SOL Scoring and ROCm Profiler Timing** — Phases 23-26 (shipped 2026-05-22). See `.planning/milestones/v1.5-ROADMAP.md`.
- ✅ **v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness** — shipped 2026-05-22. See `.planning/milestones/v1.4-ROADMAP.md`.
- ✅ **v1.3 Non-CDNA Issue Closure** — shipped 2026-05-22. See `.planning/milestones/v1.3-ROADMAP.md`.
- ✅ **v1.2 Engineering Practice Harvest and Compatibility Guardrails** — shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.
- ✅ **v1.1 CDNA 3 Support and Migration Closure** — shipped 2026-05-21. See `.planning/milestones/v1.1-ROADMAP.md`.
- ✅ **v1.0 ROCm Port** — shipped 2026-05-21. See `.planning/milestones/v1.0-ROADMAP.md`.

## Current Position

No active milestone is currently planned. Start the next milestone with
`$gsd-new-milestone`.

## Recently Shipped

<details>
<summary>✅ v1.8 ROCm Library Ecosystem Completion (Phases 36-40) — SHIPPED 2026-05-22</summary>

- [x] Phase 36: Library Build Plumbing and Diagnostics (1/1 plan)
- [x] Phase 37: MIOpen Supported Replacement (1/1 plan)
- [x] Phase 38: Composable Kernel Supported Replacement (1/1 plan)
- [x] Phase 39: rocWMMA Supported Replacement (1/1 plan)
- [x] Phase 40: Compatibility Cleanup and RDNA 4 Validation Closure (1/1 plan)

Full archive:
- `.planning/milestones/v1.8-ROADMAP.md`
- `.planning/milestones/v1.8-REQUIREMENTS.md`
- `.planning/milestones/v1.8-MILESTONE-AUDIT.md`
- `.planning/milestones/v1.8-phases/`

</details>

## Future Candidate Work

- CDNA 3 / MI300X full adapted-suite validation.
- FP8 behavior and performance validation on MI300X.
- CDNA 3 and CDNA 4 validation for supported ROCm library examples.
- Profiler-backed performance comparison reports for supported ROCm library examples.
- Original paper model-to-subgraph extraction and curation pipeline adaptation.
- Deeper upstream SOLAR parity: graph tracing, einsum/IR conversion, lookup validation, and tighter movement bounds.
- NVFP4/MXFP4-like validation if suitable AMD hardware support and methodology become available.
