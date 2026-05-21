# Roadmap: SOL ExecBench ROCm Port

**Created:** 2026-05-21
**Last updated:** 2026-05-21 after v1.0 milestone archive

## Milestones

- [x] **v1.0 ROCm Port** - Phases 1-6 shipped 2026-05-21. Full archive: [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)

## Phases

<details>
<summary>v1.0 ROCm Port (Phases 1-6) - SHIPPED 2026-05-21</summary>

- [x] Phase 1: ROCm Environment Baseline (4/4 plans) - completed 2026-05-21
- [x] Phase 2: ROCm Schema and Native Build Layer (4/4 plans) - completed 2026-05-21
- [x] Phase 3: ROCm Evaluation, Timing, and Hardware Introspection (4/4 plans) - completed 2026-05-21
- [x] Phase 4: ROCm Library and Example Migration (3/3 plans) - completed 2026-05-21
- [x] Phase 5: ROCm Test Suite and Hardware Validation (3/3 plans) - completed 2026-05-21 with CDNA 3 validation deferred
- [x] Phase 6: Documentation, Analysis Workflow, and Compliance (3/3 plans) - completed 2026-05-21

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
| --- | --- | ---: | --- | --- |
| 1. ROCm Environment Baseline | v1.0 | 4/4 | Complete | 2026-05-21 |
| 2. ROCm Schema and Native Build Layer | v1.0 | 4/4 | Complete | 2026-05-21 |
| 3. ROCm Evaluation, Timing, and Hardware Introspection | v1.0 | 4/4 | Complete | 2026-05-21 |
| 4. ROCm Library and Example Migration | v1.0 | 3/3 | Complete | 2026-05-21 |
| 5. ROCm Test Suite and Hardware Validation | v1.0 | 3/3 | Complete with TEST-05 deferred | 2026-05-21 |
| 6. Documentation, Analysis Workflow, and Compliance | v1.0 | 3/3 | Complete | 2026-05-21 |

## Current Status

v1.0 is archived. The next milestone should start with fresh requirements via `$gsd-new-milestone`.

Known deferred follow-up:

- TEST-05: run the full adapted suite under PyTorch ROCm on CDNA 3 (`gfx94*`) before claiming CDNA 3 hardware support.
