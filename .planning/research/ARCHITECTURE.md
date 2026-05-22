# Architecture Research

**Domain:** AMD-native SOL scoring and ROCm profiler timing for SOL ExecBench
**Researched:** 2026-05-22
**Confidence:** MEDIUM-HIGH

## Recommended Architecture

### System Overview

```
Existing benchmark contract
    sol-execbench CLI
    problem/workload/solution schemas
    eval_driver correctness and trace JSONL
        |
        v
AMD-native analysis layer
    graph extraction -> op/source classification -> FLOP/byte analysis
    AMD hardware model -> SOL bound artifact -> score aggregation
        |
        v
Accuracy-first timing layer
    source type router
      hip_native -> rocprofv3 kernel/HIP trace parser
      triton     -> rocprofv3 kernel trace + Triton metadata warmup policy
      pytorch    -> torch profiler attribution + device activity cross-check
        |
        v
Derived outputs
    timing evidence bundle
    SOL bound evidence bundle
    claim-level guarded reports
```

The public benchmark path should remain the authority for correctness, workload
execution, and trace compatibility. The new SOL and timing work should be
additive where possible, but the default timing implementation can be replaced
when profiler-backed evidence is measurably more accurate.

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Source classifier | Identify HIP native, Triton, PyTorch, and mixed workloads. | Pure Python classifier using solution metadata, trace records, generated module names, and runtime evidence. |
| Graph extractor | Produce a normalized operation graph for SOL analysis. | Start from existing workload/reference information; add PyTorch FX/export and profiler attribution only where reliable. |
| FLOP/byte analyzer | Estimate operation work and memory traffic. | Per-op calculators with explicit confidence and assumptions. |
| AMD hardware model registry | Store architecture-specific peak bandwidth/compute inputs. | Versioned data model keyed by arch target, dtype, clock/source, and confidence. |
| SOL bound engine | Compute per-op and aggregate SOL bounds. | Deterministic service returning bound, limiting resource, assumptions, and evidence references. |
| Timing router | Select the most accurate timer per operator/source type. | Policy table: source type -> profiler/timer backend -> interpretation. |
| rocprofv3 collector/parser | Capture and parse kernel/HIP/HSA activity. | External command wrapper plus CSV/JSON parser with correlation IDs and timestamps. |
| PyTorch attribution adapter | Connect Python/Torch ops to device work where possible. | `torch.profiler` records plus ROCm `ProfilerActivity.CUDA` device activity naming. |
| Evidence and report writer | Emit derived artifacts without mutating trace JSONL. | JSON/Markdown evidence files with claim guards and reproducibility metadata. |

## Recommended Project Structure

```
src/sol_execbench/core/
├── bench/
│   ├── timing.py                  # Replace default timing dispatch here
│   ├── rocm_profiler.py           # Candidate rocprofv3 wrapper/parser
│   └── timing_policy.py           # Source type -> timer backend policy
├── scoring/
│   ├── amd_hardware.py            # AMD hardware model registry
│   ├── graph.py                   # Normalized graph extraction models
│   ├── analysis.py                # FLOP/byte calculators
│   ├── sol_bounds.py              # SOL bound calculation
│   └── reports.py                 # Derived score/evidence outputs
└── scoring_guardrails.py          # Extend claim boundaries and warnings

tests/sol_execbench/
├── test_timing_policy.py
├── test_rocm_profiler_parser.py
├── test_amd_sol_bounds.py
├── test_scoring_guardrails.py
└── test_public_contract_guardrails.py

docs/
├── amd_sol_scoring.md
└── rocm_timing.md
```

### Structure Rationale

- Keep scoring internals separate from trace schemas so SOL artifacts can evolve
  without changing public benchmark records.
- Put the default timing switch at the current `core/bench/timing.py` boundary;
  downstream users already depend on that benchmark-level abstraction.
- Isolate profiler collection/parsing from timing policy. `rocprofv3` command
  behavior, file layout, and parser details should not leak into scoring logic.
- Treat PyTorch attribution as an adapter, not as the universal timing source,
  because PyTorch operator names and device activity records answer different
  questions.

## Architectural Patterns

### Pattern 1: Evidence-Carrying Bound Objects

**What:** Every SOL bound returns the numeric bound plus limiting resource,
source graph node, hardware-model entry, confidence, and assumptions.
**When to use:** All AMD-native SOL score paths.
**Trade-offs:** More verbose outputs, but avoids unverifiable "score only"
claims and keeps the ROCm port honest against the original paper baseline.

### Pattern 2: Source-Specific Timing Chimneys

**What:** Timing is exposed as a mapping from operator/source type to timer
backend and interpretation.
**When to use:** Whenever a single unified timer would collapse Triton, HIP, and
PyTorch semantics in a way that reduces accuracy.
**Trade-offs:** Less aesthetically uniform than one timer, but it preserves the
highest-priority requirement: accurate elapsed device work measurement.

### Pattern 3: Derived Report Layer

**What:** SOL and timing evidence are emitted as separate derived artifacts from
existing trace and execution data.
**When to use:** Reports, debugging, milestone validation, and score
aggregation.
**Trade-offs:** Users must inspect more files, but trace JSONL compatibility is
protected.

## Data Flow

### SOL Scoring Flow

```
Problem definition + solution metadata
    -> graph extraction
    -> source/op classification
    -> FLOP and byte estimation
    -> AMD hardware model lookup
    -> per-op SOL bound
    -> workload SOL bound
    -> measured timing import
    -> normalized score and aggregation
    -> guarded report
```

### Timing Flow

```
Solution source/runtime evidence
    -> classify source type
    -> timing policy lookup
    -> run profiler or timer backend
    -> parse activity records
    -> correlate kernels/operators
    -> compute measured duration
    -> emit timing evidence with interpretation
```

### Claim Flow

```
Score/timing evidence
    -> validate source coverage
    -> validate hardware model confidence
    -> validate timer confidence
    -> assign claim level
    -> report allowed and disallowed claims
```

## Integration Points

| Boundary | Communication | Notes |
|----------|---------------|-------|
| CLI -> scoring | Existing commands or additive report options | Keep existing solution schema stable. |
| eval driver -> timing | Existing `bench_gpu_time*` abstraction | Replace implementation without changing caller semantics where possible. |
| timing -> rocprofv3 | Subprocess and output directory | `rocprofv3` supports kernel/HIP/HSA trace options and output controls. |
| PyTorch -> profiler adapter | `torch.profiler` records | ROCm uses `torch.cuda` namespace and CUDA-named profiler activities for device work. |
| scoring -> reports | Derived JSON/Markdown | Do not mutate trace JSONL for internal evidence. |

## Anti-Patterns

### One Timer to Hide All Semantics

**What people do:** Force HIP native, Triton, and PyTorch workloads through one
timer and report the same interpretation.
**Why it is wrong:** It can measure different layers of work and create false
comparisons.
**Do this instead:** Use a policy table and expose the timer interpretation.

### Score Without Bound Evidence

**What people do:** Emit a final SOL score without the graph, FLOP/byte, and AMD
hardware-model evidence that produced it.
**Why it is wrong:** The result cannot be audited against the SOL ExecBench
paper's intent.
**Do this instead:** Make the bound artifact a required intermediate.

### CDNA3 Validation Leakage

**What people do:** Let the AMD hardware registry make `gfx94*` scoring appear
validated.
**Why it is wrong:** This milestone explicitly excludes real CDNA3 validation.
**Do this instead:** Allow model entries but gate claim wording as unvalidated.

## Sources

- SOL ExecBench paper baseline: https://arxiv.org/abs/2603.19173
- ROCprofiler-SDK `rocprofv3` usage: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- PyTorch profiler reference: https://docs.pytorch.org/docs/2.12/profiler.html
- PyTorch HIP semantics: https://docs.pytorch.org/docs/2.12/notes/hip.html
- ROCm HIP performance guidelines: https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html
- ROCm Triton optimization guidance: https://rocm.docs.amd.com/en/docs-6.2.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html

---
*Architecture research for: v1.5 AMD-native SOL scoring and ROCm profiler timing*
*Researched: 2026-05-22*
