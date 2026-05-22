# Project Research Summary

**Project:** SOL ExecBench ROCm Port
**Domain:** AMD-native SOL scoring and ROCm profiler timing
**Researched:** 2026-05-22
**Confidence:** MEDIUM-HIGH

## Executive Summary

The v1.5 milestone should treat the SOL ExecBench paper as the semantic
baseline while adding AMD-native performance interpretation that the current
ROCm port intentionally deferred. The core missing pieces are a SOLAR-like
pipeline for AMD hardware, auditable FLOP/byte and hardware-model inputs, and
score outputs that distinguish measured performance from theoretical AMD SOL
bounds.

For timing, the current event-backed path is not enough to satisfy the new
accuracy requirement. ROCm profiler-backed timing should become the default
where it provides the most accurate device activity measurement. However,
Triton, HIP native, and PyTorch workloads should not be forced into one
semantic bucket if that reduces accuracy. The recommended design is an
accuracy-first timing policy that can expose a source-specific chimney:
operator/source type -> timer backend -> interpretation.

The main roadmap risk is producing cleaner-looking numbers without enough
evidence. The milestone should therefore build bound artifacts and timing
evidence bundles before broad aggregation or public claims. CDNA3 model
scaffolding may exist, but real CDNA3 validation remains explicitly out of
scope.

## Key Findings

### Recommended Stack

Use the existing Python package, Pydantic models, Click/Rich CLI boundary, and
pytest suite. Add narrowly scoped modules for AMD SOL analysis, hardware model
data, profiler timing policy, and evidence reports. Use ROCprofiler-SDK
`rocprofv3` for kernel/HIP/HSA activity evidence and PyTorch's profiler for
PyTorch op attribution, while documenting ROCm's CUDA-named PyTorch profiler
surface.

**Core technologies:**
- ROCprofiler-SDK `rocprofv3`: profiler-backed kernel/HIP/HSA activity timing.
- PyTorch profiler on ROCm: PyTorch operator attribution and device activity
  cross-checks.
- AMD hardware model registry: architecture, dtype/path, and confidence inputs
  for SOL bounds.
- Existing SOL ExecBench schemas and trace JSONL: public benchmark contract to
  preserve.

### Expected Features

**Must have:**
- SOLAR-like AMD graph extraction and FLOP/byte analysis.
- AMD hardware model inputs with source and confidence metadata.
- Auditable per-op and aggregate SOL bound artifacts.
- AMD-native score computation with measured time, bound, baseline, and suite
  aggregation.
- Profiler-backed default timing where accurate.
- Source-specific timing policy for Triton, HIP native, and PyTorch.
- Claim guardrails excluding real CDNA3 validation.

**Should have:**
- Timing drift comparison between old event timing and profiler-backed timing.
- Profiler evidence bundle with parsed activity rows and reproducibility
  metadata.
- Bound confidence levels for incomplete graph or hardware-model coverage.
- Documentation that explains timer interpretations by source type.

**Defer:**
- Real CDNA3 `gfx94*` full-suite validation.
- NVIDIA leaderboard or B200/SOLAR equivalence claims.
- A single universal timing interpretation if it would weaken accuracy.

### Architecture Approach

Add two cooperating internal layers around the existing benchmark contract: an
AMD-native SOL analysis layer and an accuracy-first timing layer. The SOL layer
extracts a normalized graph, estimates FLOPs/bytes, applies an AMD hardware
model, and emits bound artifacts. The timing layer classifies source types,
selects a profiler/timer backend, parses evidence, and reports measured
durations with explicit interpretation.

**Major components:**
1. Source classifier and timing policy: maps HIP, Triton, PyTorch, and mixed
   workloads to accurate timing backends.
2. rocprofv3 collector/parser: captures kernel/HIP/HSA activity and preserves
   domain semantics.
3. Graph and FLOP/byte analyzer: creates auditable inputs for SOL bounds.
4. AMD hardware model registry: stores arch/dtype/path-specific peak inputs and
   validation status.
5. Evidence/report writer: emits separate SOL and timing artifacts with claim
   guards.

### Critical Pitfalls

1. **Timing accuracy lost to unification** - avoid by exposing source-specific
   timing policies when needed.
2. **Profiler trace misinterpretation** - avoid by modeling activity domains and
   timer interpretation explicitly.
3. **SOL claims without evidence** - avoid by requiring bound artifacts before
   score aggregation.
4. **Paper-baseline semantic drift** - avoid by keeping trace/schema changes
   additive and documented.
5. **CDNA3 validation leakage** - avoid by claim-level guardrails that keep real
   CDNA3 validation out of this milestone.

## Implications for Roadmap

### Phase 1: Timing Semantics and Policy

**Rationale:** The user's highest rule is timing accuracy, and all downstream
scores depend on measured time.
**Delivers:** Source classification, timer policy table, legacy event-vs-profiler
comparison plan, and public interpretation docs.
**Addresses:** Triton/HIP/PyTorch timing semantics and chimney exposure.
**Avoids:** Unified timing abstraction that hides accuracy loss.

### Phase 2: rocprofv3 Default Timing Path

**Rationale:** The current default timing path must be replaced where ROCm
profiler activity is the more accurate source.
**Delivers:** rocprofv3 wrapper/parser, fixture tests, default timing dispatch,
fallback labeling, and timing evidence bundle.
**Uses:** ROCprofiler-SDK `rocprofv3`, existing `core/bench/timing.py`, pytest.

### Phase 3: AMD SOL Bound Foundation

**Rationale:** A SOLAR-like score needs auditable theoretical bounds before score
aggregation is meaningful.
**Delivers:** Graph extraction models, FLOP/byte calculators, AMD hardware model
registry, bound artifacts, and confidence metadata.
**Implements:** SOLAR-like AMD analysis pipeline.

### Phase 4: AMD-native Scoring and Guarded Reports

**Rationale:** Once timing and bounds are evidence-backed, the benchmark can
produce AMD-native scores and suite summaries.
**Delivers:** Per-problem score, baseline comparison, suite aggregation, report
docs, and guardrails excluding CDNA3 validation claims.
**Avoids:** Leaderboard-equivalence and unsupported hardware validation claims.

### Phase Ordering Rationale

- Timing comes first because measured runtime is an input to every score.
- rocprofv3 parser work comes before broad scoring so profiler semantics are
  proven on fixtures.
- Bound artifacts come before final scoring so SOL results are auditable.
- Guarded reports come last because they integrate timing, bounds, baselines,
  and claim levels.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Exact classification evidence for Triton-generated kernels and
  PyTorch fused/device activity.
- **Phase 2:** Current `rocprofv3` output schema on the local ROCm 7 baseline.
- **Phase 3:** AMD peak hardware model sources and dtype-specific formulas.

Phases with standard patterns:
- **Phase 4:** Existing score/report guardrail structure can be extended once
  inputs are available.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official ROCm and PyTorch docs support the recommended profiler stack. |
| Features | HIGH | They follow directly from paper gaps and user-selected features 1, 2, 6, 7. |
| Architecture | MEDIUM-HIGH | Shape is clear; exact parser and graph details need phase planning. |
| Pitfalls | HIGH | Risks are directly tied to timing semantics and benchmark claim boundaries. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- Verify exact `rocprofv3` output columns and activity names on the local ROCm
  7 environment during Phase 2.
- Decide whether graph extraction starts with workload metadata, PyTorch
  export/FX, profiler attribution, or a hybrid based on source type.
- Select authoritative AMD hardware peak inputs and represent confidence for
  RDNA4 versus unvalidated CDNA3.
- Define score output format as derived artifacts, not trace JSONL mutations.

## Sources

### Primary

- SOL ExecBench paper baseline: https://arxiv.org/abs/2603.19173
- ROCprofiler-SDK `rocprofv3` usage: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- PyTorch profiler reference: https://docs.pytorch.org/docs/2.12/profiler.html
- PyTorch HIP semantics: https://docs.pytorch.org/docs/2.12/notes/hip.html
- ROCm HIP performance guidelines: https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html
- ROCm Triton optimization guidance: https://rocm.docs.amd.com/en/docs-6.2.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html

### Project Context

- `src/sol_execbench/core/bench/timing.py` - current event-backed timing boundary.
- `src/sol_execbench/core/scoring_guardrails.py` - existing claim guardrail surface.
- `docs/original_parity.md` - current gaps around AMD-native SOL interpretation.
- `docs/analysis.md` - current ROCm timing and scoring limitations.

---
*Research completed: 2026-05-22*
*Ready for roadmap: yes*
