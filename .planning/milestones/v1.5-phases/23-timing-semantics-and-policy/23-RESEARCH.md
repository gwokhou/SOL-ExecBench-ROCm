# Phase 23: Timing Semantics and Policy - Research

**Researched:** 2026-05-22
**Domain:** ROCm benchmark timing source classification and policy modeling
**Confidence:** HIGH

<user_constraints>
## User Constraints

No `CONTEXT.md` exists for this phase. The user explicitly chose to continue
without discuss-phase context.

### Locked Decisions

- Timing accuracy is the highest rule.
- Triton, HIP native, and PyTorch source operators must be investigated
  separately.
- If a unified timing口径 would reduce timing accuracy, expose a chimney-style
  mapping: operator/source type -> timer backend -> interpretation.
- Real CDNA3 `gfx94*` full-suite validation is not included in this milestone.

### the agent's Discretion

- Exact module and enum names.
- Whether policy objects live under `core/bench/` only or share small reporting
  models elsewhere.
- Exact documentation file name for timing semantics.

### Deferred Ideas

- Invoking `rocprofv3` as the default timing path. Phase 24 owns collector,
  parser, fallback execution, and profiler evidence bundles.
- AMD SOL bound calculation. Phase 25 owns graph/FLOP/byte/hardware model work.
- AMD-native score aggregation and reports. Phase 26 owns scoring/report
  integration.
</user_constraints>

<architectural_responsibility_map>
## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Source type classification | Core benchmark library | Driver/eval template | Classification should be testable without launching benchmark subprocesses; eval driver can pass solution metadata later. |
| Timing backend policy | Core benchmark library | Docs/tests | A pure policy table should be inspectable before profiler integration. |
| Timer interpretation labels | Core benchmark library | Derived evidence/report layer | Interpretation must be data, not prose-only, so Phase 24 evidence can serialize it. |
| User-facing timing semantics | Documentation | Tests | Maintainers need stable explanation for HIP native, Triton, PyTorch, and fallback timing. |

</architectural_responsibility_map>

<research_summary>
## Summary

Phase 23 should build the semantic contract for timing before changing how
timing is collected. Current code routes all benchmark timing through
`time_runnable()` in `src/sol_execbench/core/bench/timing.py`, which uses
PyTorch ROCm's HIP-backed `torch.cuda.Event` API. The legacy
`bench_gpu_time_with_cupti()` entry point is only a compatibility wrapper and
does not call CUPTI or ROCm profiler tooling.

The existing solution schema already exposes the strongest classification input:
`BuildSpec.languages` distinguishes `pytorch`, `triton`, and ROCm native/library
categories such as `hip_cpp`, `hipblas`, `miopen`, `ck`, and `rocwmma`. Phase 23
should translate those public schema values into a smaller internal timing
source taxonomy: `pytorch`, `triton`, `hip_native`, and `mixed/unknown` as
needed. It should then provide a timing policy record that names the selected
timer backend and exactly what the duration means.

**Primary recommendation:** Implement a pure, serializable timing policy layer
under `src/sol_execbench/core/bench/` with enums/models for source type, timer
backend, activity domain, aggregation rule, fallback status, and interpretation.
Do not invoke `rocprofv3` until Phase 24.
</research_summary>

<standard_stack>
## Standard Stack

### Core

| Library/API | Version | Purpose | Why Standard |
|-------------|---------|---------|--------------|
| Python stdlib `enum` / dataclasses or Pydantic | Python 3.12+ | Policy and classification models | Matches existing project patterns and keeps Phase 23 pure. |
| Existing `SupportedLanguages` enum | Current project | Source metadata input | Public solution schema already classifies PyTorch, Triton, and native ROCm categories. |
| Existing `time_runnable()` boundary | Current project | Current timing abstraction | Phase 24 can route through this boundary after Phase 23 defines policy. |
| Existing diagnostics profiler readiness | Current project | Backend/fallback vocabulary precedent | `ProfilerReadiness` already models backend, reason, fallback, and effective level. |

### Supporting

| API | Purpose | When to Use |
|-----|---------|-------------|
| `torch.cuda.Event` on ROCm | Fallback event timing label | Keep as compatibility/fallback interpretation, not as universal kernel activity timing. |
| `torch.profiler` | PyTorch op attribution label | Phase 23 should define semantics; Phase 24+ can collect it when needed. |
| `rocprofv3` | Kernel/HIP/HSA activity backend label | Phase 23 should define backend semantics; Phase 24 owns invocation and parsing. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure policy model | Directly modify `time_runnable()` to call profiler | Higher risk; mixes semantic design with subprocess/profiler failures. |
| `BuildSpec.languages` as sole source | Source-code scanning for `triton.jit` or `torch.*` | Scanning can be added later for mixed/ambiguous cases, but schema metadata is the stable public input. |
| One enum for backend only | Separate source/backend/domain/interpretation enums | Separate dimensions are more verbose but prevent false equivalence across HIP, Triton, and PyTorch. |

</standard_stack>

<architecture_patterns>
## Architecture Patterns

### System Architecture Diagram

```
Solution metadata + optional runtime evidence
    |
    v
Source classifier
    |-- pytorch -----> timing policy: pytorch attribution + device activity label
    |-- triton ------> timing policy: generated kernel activity label
    |-- hip_native --> timing policy: HIP kernel activity label
    |-- mixed/unknown -> timing policy: fallback or explicit unsupported label
    |
    v
Timing policy record
    -> timer backend
    -> activity domain
    -> aggregation rule
    -> interpretation
    -> fallback reason
    |
    v
Phase 24 timing collector/parser
```

### Recommended Project Structure

```
src/sol_execbench/core/bench/
├── timing.py                  # Existing timing boundary; later consumes policy
├── timing_policy.py           # New source/backend/domain policy models
└── rocm_profiler.py           # Defer collector/parser to Phase 24

docs/
└── rocm_timing.md             # Timing semantics and chimney explanations

tests/sol_execbench/
├── test_timing_policy.py      # Pure policy/classifier coverage
└── test_rocm_eval_timing_audit.py  # Existing audit can gain doc/path assertions
```

### Pattern 1: Classifier from Public Schema

**What:** Convert solution language categories to internal timing source types.
**When to use:** Before selecting a timer backend or emitting timing evidence.
**Target mapping:**

| Public language | Internal source type | Notes |
|-----------------|----------------------|-------|
| `pytorch` | `pytorch` | PyTorch operator source; may dispatch many library kernels. |
| `triton` | `triton` | Triton-generated kernels; warmup/JIT semantics must be explicit later. |
| `hip_cpp`, `hipblas`, `miopen`, `ck`, `rocwmma` | `hip_native` | Native ROCm path launched through torch extension or ROCm libraries. |
| Multiple Python + native categories | invalid today | Schema rejects Python/native mixing. |
| Empty or unrecognized evidence | `unknown` | Requires labeled fallback or unsupported policy. |

### Pattern 2: Policy Record, Not Backend String

**What:** A timing policy should carry source type, backend, activity domain,
aggregation rule, interpretation, fallback status, and reason.
**When to use:** Any timing selection or report/evidence generation.
**Rationale:** Backend alone cannot distinguish "kernel activity duration" from
"PyTorch op attribution" or "fallback event elapsed time."

### Pattern 3: Explicit Fallback Policy

**What:** Event timing remains available as a labeled fallback, not a hidden
replacement for profiler-backed semantics.
**When to use:** Profiler unavailable, source type unknown, or unsupported
profiler integration.
**Rationale:** Existing reward-hack defenses and tests patch
`torch.cuda.Event.elapsed_time`; keeping fallback explicit avoids accidental
security/semantics regressions.

### Anti-Patterns to Avoid

- **One backend enum as the whole model:** hides source semantics and makes
  reports ambiguous.
- **Changing trace JSONL in Phase 23:** timing policy is preparatory; canonical
  trace compatibility must remain intact.
- **Profiler subprocess in Phase 23:** pushes Phase 24 failure modes into
  semantic planning and makes unit tests hardware/tool dependent too early.
- **Treating PyTorch `ProfilerActivity.CUDA` as NVIDIA evidence:** PyTorch ROCm
  intentionally reuses CUDA-named APIs; docs should explain compatibility
  naming.
</architecture_patterns>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Source Classification Equals Implementation Language Only

**What goes wrong:** A `pytorch` solution is treated like one PyTorch op with one
duration, even though it may launch multiple ROCm library kernels.
**Why it happens:** `BuildSpec.languages` is convenient and stable.
**How to avoid:** Use schema language as first-pass source classification, but
make policy interpretations explicit and allow Phase 24+ runtime evidence to
refine attribution.
**Warning signs:** Policy names say `pytorch_kernel_time` without explaining
operator-to-kernel attribution.

### Pitfall 2: Hidden Event-Timing Fallback

**What goes wrong:** The benchmark silently falls back to `torch.cuda.Event`
timing and reports it as profiler-backed or kernel activity timing.
**Why it happens:** Current code already works with events and has compatibility
wrappers.
**How to avoid:** Every fallback policy must include backend, fallback flag,
reason, and interpretation.
**Warning signs:** Tests assert only a numeric latency without checking selected
backend metadata.

### Pitfall 3: Triton Warmup/JIT Semantics Leak Into Policy

**What goes wrong:** Phase 23 hard-codes profiler collection behavior for Triton
before Phase 24 validates local `rocprofv3` output and Triton-generated kernel
names.
**Why it happens:** Triton-specific timing is a known problem and tempting to
solve immediately.
**How to avoid:** In Phase 23, define the interpretation and aggregation labels;
defer collector details, warmup enforcement, and kernel name matching to Phase
24.
**Warning signs:** New code invokes Triton or profiler tooling in pure policy
tests.
</common_pitfalls>

<validation_architecture>
## Validation Architecture

Phase 23 should be validated mostly with fast unit tests and documentation
assertions.

| Requirement | Validation Type | Recommended Test |
|-------------|-----------------|------------------|
| TIME-01 | Unit | `tests/sol_execbench/test_timing_policy.py` asserts `SupportedLanguages` values map to source types. |
| TIME-02 | Unit | Test policy table contains backend, activity domain, aggregation rule, and interpretation for each source type. |
| TIME-03 | Unit/docs | Test policy exposes distinct entries for `pytorch`, `triton`, and `hip_native`; docs contain the chimney mapping phrase or table. |
| TIME-04 | Docs/unit | Test docs mention kernel activity, HIP runtime/API activity, PyTorch operator attribution, and fallback event timing. |

**Quick command:** `uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py`

**Full phase command:** `uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py`
</validation_architecture>

<open_questions>
## Open Questions

1. **Should policy models be dataclasses or Pydantic models?**
   - What we know: Existing public data models use Pydantic v2; internal
     diagnostics use dataclasses and enums.
   - What's unclear: Whether Phase 24 evidence serialization will benefit from
     Pydantic validation.
   - Recommendation: Use small enums plus a frozen dataclass in Phase 23 unless
     an existing reporting/evidence API expects Pydantic.

2. **How much runtime evidence should source classification accept?**
   - What we know: `BuildSpec.languages` is stable and enough for first-pass
     classification.
   - What's unclear: Mixed PyTorch/Triton patterns or `torch.compile` may blur
     source attribution.
   - Recommendation: Keep Phase 23 classifier deterministic from schema
     metadata, with an explicit `unknown/mixed` escape hatch for future runtime
     evidence.

3. **Where should docs live?**
   - What we know: Existing docs include `docs/analysis.md`, `docs/trace.md`,
     and `docs/solution.md`.
   - What's unclear: Whether a dedicated `docs/rocm_timing.md` is preferable to
     expanding `docs/analysis.md`.
   - Recommendation: Add a dedicated `docs/rocm_timing.md` and link it from
     existing timing/analysis docs in a later phase if needed.
</open_questions>

<sources>
## Sources

### Primary

- `src/sol_execbench/core/bench/timing.py` - current event-backed timing
  boundary and compatibility wrappers.
- `src/sol_execbench/core/data/solution.py` - public language metadata used for
  source classification.
- `src/sol_execbench/core/diagnostics.py` - existing profiler backend/fallback
  vocabulary.
- `src/sol_execbench/driver/templates/eval_driver.py` - current timing call
  site and reward-hack sequencing.
- `.planning/research/SUMMARY.md` - v1.5 milestone research and phase ordering.
- SOL ExecBench paper baseline: https://arxiv.org/abs/2603.19173
- ROCprofiler-SDK `rocprofv3` usage:
  https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- PyTorch profiler reference: https://docs.pytorch.org/docs/2.12/profiler.html
- PyTorch HIP semantics: https://docs.pytorch.org/docs/2.12/notes/hip.html

### Secondary

- `docs/analysis.md` - current optional `rocprofv3` guidance and event timing
  limitations.
- `tests/sol_execbench/test_rocm_eval_timing_audit.py` - current ROCm timing
  audit expectations.
- `tests/sol_execbench/core/bench/test_timing.py` - existing timing behavior and
  legacy CUPTI compatibility tests.
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: ROCm timing source classification and policy data model.
- Ecosystem: PyTorch ROCm event/profiler naming, ROCprofiler-SDK backend labels,
  existing SOL ExecBench timing and solution schema.
- Patterns: Pure policy layer before profiler subprocess integration.
- Pitfalls: hidden fallback, source semantics collapse, PyTorch CUDA-named ROCm
  APIs, Triton JIT/warmup leakage.

**Confidence breakdown:**
- Standard stack: HIGH - Existing project code provides stable classification
  and timing boundaries.
- Architecture: HIGH - Phase 23 is intentionally pure and feeds Phase 24.
- Pitfalls: HIGH - Directly visible in current event wrapper and v1.5 research.
- Code examples: MEDIUM - Phase 23 should avoid detailed implementation code in
  research; planner can define exact files/tasks.

**Research date:** 2026-05-22
**Valid until:** 2026-06-21
</metadata>

---
*Phase: 23-timing-semantics-and-policy*
*Research completed: 2026-05-22*
*Ready for planning: yes*
