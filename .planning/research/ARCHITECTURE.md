# Project Research: Architecture

**Project:** SOL ExecBench ROCm Port
**Milestone:** v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow
**Researched:** 2026-05-22
**Confidence:** MEDIUM-HIGH

## Integration Shape

v1.6 should be built as three additive layers around the existing benchmark
contract:

1. **Analyzer layer:** extends AMD SOL graph and work estimation coverage.
2. **Timing layer:** runs live profiler collection when the timing policy says
   that is the most accurate source-specific backend.
3. **Workflow layer:** connects trace JSONL, timing evidence, SOL bounds, and
   baseline inputs into derived score reports.

Canonical trace JSONL remains the stable substrate. Timing, bound, and score
artifacts reference it rather than modifying it.

## Data Flow

```text
problem + solution
  -> existing eval driver
  -> canonical trace JSONL
  -> source classifier
  -> timing policy
  -> live timing evidence artifact
  -> AMD SOL bound artifact
  -> AMD-native workload/suite score report
```

## Component Implications

### Analyzer Layer

Modify `amd_sol.py` by adding a small analyzer registry rather than continuing
to grow `_GraphVisitor` as a chain of string checks. Each analyzer should return
normalized nodes and work estimates with confidence and rationale. Unsupported
or partially supported operations remain first-class output.

### Timing Layer

`rocm_profiler.py` should evolve from command/parser helpers into an execution
adapter used by the benchmark or dataset path. The adapter needs controlled
output directories, version capture, CSV/rocpd parsing, and fallback evidence.

`timing_policy.py` remains the authority for whether a source uses
`rocprofv3`, PyTorch profiler, event fallback, or unsupported timing.

### Workflow Layer

The dataset runner and/or additive CLI command should accept artifact output
locations and generate a suite score report. The primary CLI defaults should
continue emitting the same canonical traces unless the user opts into derived
artifacts.

## Build Order

1. Add analyzer registry and coverage reports.
2. Add live `rocprofv3` execution adapter behind timing policy.
3. Add score-report workflow integration.
4. Add compatibility tests and docs around output boundaries.

## Sources

- SOL-ExecBench paper: https://arxiv.org/abs/2603.19173
- ROCprofiler-SDK `rocprofv3` docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.0.1/how-to/using-rocprofv3.html
- ROCprofiler-SDK quick guide: https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/quick_guide.html
