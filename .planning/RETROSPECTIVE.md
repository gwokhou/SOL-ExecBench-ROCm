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

## Milestone: v1.2 - Engineering Practice Harvest and Compatibility Guardrails

**Shipped:** 2026-05-22
**Phases:** 4 | **Plans:** 4 | **Tasks:** 0

### What Was Built

- Internal adaptation map for selected `hip-execbench` engineering practices.
- ROCm diagnostic helpers for tool readiness, gfx classification, local gfx
  detection, and profiler backend fallback reasoning.
- Pure trace summary helpers that preserve the existing trace JSONL contract.
- SOL-Score interpretation guardrails for unsupported AMD-native performance
  claims.
- Public contract tests for schemas, CLI help, trace behavior, examples, and
  CDNA 3 validation deferral language.

### What Worked

- Treating `hip-execbench` as a practice source instead of a port target kept
  changes narrow and avoided public interface churn.
- Pure helper modules made diagnostics/reporting easy to test without requiring
  GPU hardware in CI.
- Contract tests gave concrete protection for the user's "do not change public
  formats" constraint.

### What Was Inefficient

- The SDK autonomous runner failed immediately, so the workflow had to be
  executed inline.
- Summary extraction produced generic "Completed:" one-liners, requiring manual
  milestone accomplishment synthesis.

### Patterns Established

- Borrow practices through an explicit accept/reject/defer map before touching
  runtime code.
- Add public-contract guardrail tests whenever internal quality work risks
  drifting schemas, CLI help, examples, or trace output.
- Keep score interpretation warnings separate from score calculation so legacy
  benchmark semantics remain stable.

### Key Lessons

1. Practice-harvest milestones need compatibility tests as first-class outputs,
   not just implementation improvements.
2. ROCm diagnostic routes can be modeled as descriptive readiness metadata
   before becoming user-facing commands.
3. AMD performance claims should be guarded in code and docs until a validated
   AMD-native interpretation model exists.

### Cost Observations

- Model mix: not recorded.
- Sessions: 1 autonomous session with inline fallback after SDK runner failure.
- Notable: phase artifacts and commits were kept small enough to audit.

---

## Milestone: v1.5 - AMD-native SOL Scoring and ROCm Profiler Timing

**Shipped:** 2026-05-22
**Phases:** 4 | **Plans:** 4 | **Tasks:** 0

### What Was Built

- Source-specific timing policy for HIP native, Triton, PyTorch, mixed, and
  unknown workloads.
- `rocprofv3` timing evidence helpers with command construction, CSV parsing,
  policy-aware default selection, and fallback metadata.
- AMD SOL bound artifacts with graph nodes, FLOP/byte work estimates, hardware
  model metadata, per-op bounds, and aggregate bounds.
- Derived AMD-native score reports with per-workload scores, suite aggregation,
  evidence references, and CDNA3 no-validation guardrails.

### What Worked

- Keeping timing policy, profiler evidence, SOL bounds, and scoring reports as
  separate layers made each phase testable without GPU hardware.
- Derived artifacts preserved the canonical trace JSONL contract while still
  giving future scoring workflows auditable evidence.
- Explicit CDNA3 no-claim wording kept readiness scaffolding separate from
  hardware validation claims.

### What Was Inefficient

- Summary extraction still produced generic `Status:` entries, so milestone
  accomplishments required manual synthesis.
- The `rocprofv3` work remains an evidence-helper foundation; live benchmark
  integration is still future work.

### Patterns Established

- Treat timing source type as part of measurement semantics, not just metadata.
- Carry confidence and rationale through graph extraction, FLOP/byte estimates,
  hardware model data, and final score reports.
- Keep AMD-native performance interpretation as a derived report until the
  hardware model and validation evidence are strong enough for public claims.

### Key Lessons

1. A chimney-style timing model is preferable when one timing backend would
   conflate HIP runtime, kernel activity, Triton, and PyTorch operator costs.
2. SOL-like score reports need evidence references and claim guardrails as much
   as they need formulas.
3. CDNA3 support scaffolding should remain visibly unvalidated until a real
   `gfx94*` full-suite pass exists.

### Cost Observations

- Model mix: not recorded.
- Sessions: 1 autonomous session plus milestone completion.
- Notable: phase separation kept the final audit small; 42 milestone tests and
  focused ruff checks passed at closure.

---

## Milestone: v1.7 - Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration

**Shipped:** 2026-05-22
**Phases:** 5 | **Plans:** 5 | **Tasks:** 0

### What Was Built

- Release-scoped optimized scoring baseline artifacts with explicit
  `baseline_source` reporting and dataset-runner integration.
- Source-specific `rocprofv3` timing evidence collection with backend,
  aggregation, warmup, clock-lock, architecture, and fallback metadata.
- Static reward-hack source review before submitted Python import for stream
  hiding, semantic caches, unauthorized loaders, opaque payloads, and precision
  downgrade patterns.
- A runnable hipBLAS SGEMM public example with native staging tests and support
  docs distinguishing supported and candidate ROCm library categories.
- MI300X validation handoff docs and evidence gates that prevent premature
  CDNA3/MI300X hardware-validation claims.

### What Worked

- Keeping scoring baselines and timing evidence as derived artifacts preserved
  canonical trace JSONL while improving auditability.
- The reward-hack source review was integrated before execution, which reduced
  subprocess failure ambiguity and kept findings in existing `REWARD_HACK`
  semantics.
- Promoting only hipBLAS to supported status avoided overclaiming MIOpen, CK,
  and rocWMMA without blocking future migration.

### What Was Inefficient

- Automatic milestone accomplishment extraction still produced generic
  `Completed:` entries, requiring manual rewrite during closure.
- Formal Nyquist `*-VALIDATION.md` artifacts were absent for Phases 31-35, so
  the milestone audit recorded this as discovery-only.

### Patterns Established

- Label every AMD-native score with its baseline source.
- Record timer backend and fallback reason as first-class timing evidence.
- Separate validation readiness from validation claims through pure evidence
  gates and public-contract tests.

### Key Lessons

1. ROCm-only parity reviews should remove NVIDIA hardware expectations from the
   missing-feature list while preserving original benchmark semantics.
2. Library category support should advance one runnable, tested path at a time;
   candidate categories need tests that prevent public overclaiming.
3. MI300X/FP8 validation can be prepared before hardware access, but status
   upgrades must remain gated by full-suite evidence.

### Cost Observations

- Model mix: not recorded.
- Sessions: 1 autonomous session plus audit and milestone completion.
- Notable: the final audit aggregate covered 67 focused tests across scoring,
  profiler timing, reward-hack review, hipBLAS examples, and MI300X guardrails.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | not recorded | 6 | Established ROCm-only milestone workflow with phase verification, validation artifacts, and hardware matrix evidence |
| v1.2 | 1 | 4 | Added practice-harvest workflow with explicit public-contract guardrails |
| v1.5 | 1 | 4 | Added AMD-native timing/SOL/scoring evidence layers while preserving public contracts |
| v1.7 | 1 | 5 | Added baseline/timing/reward-hack/library/MI300X readiness hardening before real commercial GPU validation |

### Cumulative Quality

| Milestone | Tests | Coverage | Deferred Items |
|-----------|-------|----------|----------------|
| v1.0 | 462 passed, 58 skipped on RDNA 4 | v1 requirements 38/39 complete, 1 deferred | TEST-05 CDNA 3 full-suite validation |
| v1.2 | 16 focused tests passed; ruff clean | v1.2 requirements 14/14 complete | CDNA 3 real hardware validation; AMD-native scoring model |
| v1.5 | 42 focused milestone tests passed; ruff clean | v1.5 requirements 20/20 complete | Real CDNA 3 hardware validation; broader AMD SOL operator coverage |
| v1.7 | 67 focused audit tests passed | v1.7 requirements 21/21 complete | MI300X/CDNA3 full-suite validation; FP8 real-hardware validation; paper extraction; full SOLAR parity |

### Top Lessons

1. Hardware-specific requirements need hardware-specific evidence before support claims are made.
2. Compatibility wrappers can preserve caller stability while changing the underlying ROCm implementation path.
3. Engineering-practice borrowing should be protected by explicit public-contract tests.
