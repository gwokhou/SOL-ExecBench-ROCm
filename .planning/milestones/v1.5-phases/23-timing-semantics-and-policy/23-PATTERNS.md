# Phase 23: Timing Semantics and Policy - Pattern Map

**Mapped:** 2026-05-22
**Scope:** Source classification, timing policy models, timing semantics docs,
and focused tests.

## File Mapping

| Target File | Role | Closest Existing Analog | Pattern to Reuse |
|-------------|------|-------------------------|------------------|
| `src/sol_execbench/core/bench/timing_policy.py` | New pure timing source/backend policy module | `src/sol_execbench/core/diagnostics.py` | `str, Enum` classifications plus frozen dataclasses with explicit reason/fallback fields. |
| `src/sol_execbench/core/bench/timing.py` | Existing timing boundary, maybe imports/uses policy only lightly in Phase 23 | `src/sol_execbench/core/bench/timing.py` | Preserve current callable signatures and compatibility wrappers. |
| `docs/rocm_timing.md` | New timing semantics documentation | `docs/analysis.md`, `docs/ARCHITECTURE.md` | Explain ROCm compatibility names and separate canonical trace from derived evidence. |
| `tests/sol_execbench/test_timing_policy.py` | New policy unit tests | `tests/sol_execbench/test_rocm_diagnostics_reporting.py` | Direct enum/dataclass assertions with no GPU/tool dependency. |
| `tests/sol_execbench/test_rocm_eval_timing_audit.py` | Existing audit extension | Same file | Add allowlist/doc assertions without weakening forbidden CUDA/NVIDIA checks. |

## Existing Patterns

### Internal Enum + Frozen Dataclass

`src/sol_execbench/core/diagnostics.py` uses `str, Enum` for controlled values
and `@dataclass(frozen=True)` for small internal records:

```python
class ProfilerBackend(str, Enum):
    ROCPROFV3 = "rocprofv3"
    ROCPROFILER_COMPUTE = "rocprofiler-compute"
    OMNIPERF = "omniperf"
    SKIP = "skip"


@dataclass(frozen=True)
class ProfilerReadiness:
    backend: ProfilerBackend
    reason: str
    fallback_applied: bool
    effective_level: str
```

Phase 23 should mirror this style for `TimingSourceType`, `TimingBackend`,
`TimingActivityDomain`, and a policy record carrying `interpretation`,
`aggregation_rule`, and fallback metadata.

### Public Schema as Classification Input

`src/sol_execbench/core/data/solution.py` defines `SupportedLanguages` as the
public source metadata:

```python
class SupportedLanguages(str, Enum):
    PYTORCH = "pytorch"
    TRITON = "triton"
    HIP_CPP = "hip_cpp"
    HIPBLAS = "hipblas"
    MIOPEN = "miopen"
    CK = "ck"
    ROCWMMA = "rocwmma"
```

Phase 23 classifier should accept these enum values or `BuildSpec.languages`
and map them to internal timing source types without changing the public schema.

### Current Timing Boundary

`src/sol_execbench/core/bench/timing.py` exposes `time_runnable()` and legacy
compatibility wrappers. Phase 23 should not break signatures:

```python
def time_runnable(
    fn: Any,
    inputs: list,
    outputs: list,
    device: str,
    warmup: int = 10,
    rep: int = 100,
    return_mode: Literal["mean", "median", "all"] = "median",
    methodology: Literal["events", "cuda_events", "cupti"] = "events",
) -> Union[float, list[float]]:
```

Any policy integration should be additive or preparatory; Phase 24 owns default
profiler-backed execution changes.

### Test Style

`tests/sol_execbench/test_rocm_diagnostics_reporting.py` uses direct assertions
over internal policy-like helpers:

```python
readiness = select_profiler_backend(
    "full",
    "gfx942",
    rocprofiler_compute=False,
    omniperf=False,
    rocprofv3=True,
)
assert readiness.backend == ProfilerBackend.ROCPROFV3
assert readiness.fallback_applied is True
```

Phase 23 should use the same no-mock, pure unit style for classifier/policy
tests.

## Data Flow

```
SupportedLanguages / BuildSpec.languages
    -> classify_timing_source(...)
    -> TimingSourceType
    -> select_timing_policy(...)
    -> TimingPolicy(
         source_type,
         backend,
         activity_domain,
         aggregation_rule,
         interpretation,
         fallback_applied,
         reason
       )
    -> docs/tests now
    -> Phase 24 collector/evidence later
```

## Guardrails

- Do not add fields to canonical trace JSONL in Phase 23.
- Do not invoke `rocprofv3` subprocesses in Phase 23 policy tests.
- Do not remove legacy compatibility wrapper names in `timing.py`.
- Keep PyTorch ROCm `torch.cuda` naming explicitly allowlisted and documented.
- Keep event timing as a labeled fallback, not the claimed profiler-backed
  default.

---
*Pattern map completed: 2026-05-22*
