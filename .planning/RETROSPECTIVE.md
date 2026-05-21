# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 - ROCm Port

**Shipped:** 2026-05-21
**Phases:** 6 | **Plans:** 21 | **Tasks:** 18

### What Was Built

- ROCm Docker/runtime baseline with HIP compiler tooling, ROCm profiling tools, PyTorch ROCm, Triton ROCm, and dependency smoke tests.
- ROCm-native solution schema and HIP/C++ native build flow with AMD gfx offload architecture handling.
- ROCm-compatible evaluation runtime, timing path, AMD environment snapshots, and clock tooling.
- Migrated examples, library replacement documentation, ROCm pytest markers, and RDNA 4 full-suite validation.
- User-facing README, setup, schema, trace, analysis, and compliance documentation for the ROCm-only port.

### What Worked

- Horizontal phase ordering kept hard dependencies clear: environment, schema/build, runtime/timing, examples, validation, then docs.
- Focused audit tests were useful for preventing CUDA/NVIDIA tooling regressions while still allowing PyTorch ROCm's `torch.cuda` compatibility API.
- Recording the hardware matrix separately made the RDNA 4 pass and CDNA 3 gap explicit instead of mixing hardware claims into general test status.

### What Was Inefficient

- Some summary files lacked extractable one-line metadata, so milestone accomplishment extraction depended on manual synthesis.
- CDNA 3 validation was discovered as a closure gap late in the milestone and had to be deferred instead of planned as an available hardware run.

### Patterns Established

- Keep legacy CUDA/NVIDIA strings only when they are compatibility API names, rejection tests, migration guidance, or attribution.
- Add source audits for high-risk migration surfaces instead of relying on broad text search alone.
- Treat hardware support claims as evidence-backed: schema values and docs should follow recorded full-suite runs.

### Key Lessons

1. ROCm ports should distinguish API namespace compatibility from runtime dependency residue; PyTorch ROCm still legitimately uses `torch.cuda`.
2. Hardware matrix evidence should be planned as first-class milestone work whenever multiple AMD architecture families are part of the goal.
3. Documentation should name unsupported NVIDIA runtime features directly so users do not infer dual-backend support from retained upstream attribution.

### Cost Observations

- Model mix: not recorded.
- Sessions: not recorded.
- Notable: phase archives and focused verification files now give enough structure for future milestone closure with lower context load.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | not recorded | 6 | Established ROCm-only milestone workflow with phase verification, validation artifacts, and hardware matrix evidence |

### Cumulative Quality

| Milestone | Tests | Coverage | Deferred Items |
|-----------|-------|----------|----------------|
| v1.0 | 462 passed, 58 skipped on RDNA 4 | v1 requirements 38/39 complete, 1 deferred | TEST-05 CDNA 3 full-suite validation |

### Top Lessons

1. Hardware-specific requirements need hardware-specific evidence before support claims are made.
2. Compatibility wrappers can preserve caller stability while changing the underlying ROCm implementation path.
