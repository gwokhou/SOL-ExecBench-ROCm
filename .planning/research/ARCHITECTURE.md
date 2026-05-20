# Architecture Research

**Domain:** ROCm port of GPU kernel benchmark framework
**Researched:** 2026-05-21
**Confidence:** MEDIUM-HIGH

## Standard Architecture

### System Overview

```text
CLI and Dataset Runner
  -> schema loading and validation
  -> ROCm-aware problem staging
  -> optional HIP/native compile phase
  -> isolated ROCm eval driver subprocess
  -> trace JSONL parsing and reporting

ROCm Runtime Layer
  -> PyTorch ROCm / Triton ROCm / HIP shared objects
  -> ROCm libraries: rocBLAS, MIOpen, CK, rocWMMA, rocPRIM
  -> profiling: rocprofiler-sdk, rocprofv3, ROCm Compute Profiler
  -> system discovery: AMD SMI, ROCm SMI, rocminfo
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| CLI | Keep user-facing invocation stable | Modify `src/sol_execbench/cli/main.py` only for ROCm-specific config/labels. |
| Schemas | Preserve public problem/trace contracts | Extend `SupportedLanguages` and hardware enums without unnecessary schema churn. |
| Packager | Stage files and compile native code | Replace SM/gencode injection with AMD gfx target handling. |
| Build template | Compile HIP/native sources | Use hipcc/amdclang++/torch extension ROCm path as appropriate. |
| Eval driver | Import user code, run workloads, emit traces | Replace CUDA-only guards and timing hooks with ROCm-aware logic. |
| Benchmark helpers | Generate inputs, compare outputs, time kernels | Preserve correctness semantics; port timing and clock control. |
| Tests/examples | Prove behavior on ROCm | Migrate sample solutions and hardware markers. |

## Recommended Project Structure

Keep the current package shape. Add ROCm-specific code by replacing backend
modules where NVIDIA assumptions are embedded, rather than creating a parallel
CUDA/ROCm abstraction tree.

```text
src/sol_execbench/
├── cli/                 # stable CLI
├── driver/              # ROCm-aware staging and compile commands
│   └── templates/       # HIP build and ROCm eval subprocess scripts
├── core/
│   ├── data/            # schemas and supported language/hardware enums
│   └── bench/           # IO, correctness, ROCm timing, SMI/clock logic
└── sol_score.py         # scoring logic, unchanged unless SOL constants change

tests/
├── docker/dependencies/ # ROCm dependency checks
├── sol_execbench/       # unit, driver, e2e tests
└── examples/            # migrated ROCm examples
```

## Architectural Patterns

### Pattern 1: Preserve CLI and Schema Contracts

**What:** Keep `definition.json`, `workload.jsonl`, `solution.json`, and trace
formats stable while changing backend implementation.
**Trade-off:** Some language names may need migration aliases.

### Pattern 2: Backend Replacement by Subsystem

**What:** Port Docker, dependencies, compile, eval, timing, examples, and tests
as separate layers.
**Trade-off:** Slower than search/replace, but prevents silent benchmark drift.

### Pattern 3: Hardware-Gated Tests

**What:** Replace `sm_100` and CUDA availability markers with ROCm architecture
checks for RDNA 4 and CDNA 3.
**Trade-off:** Requires clear skip messages and access to both hardware classes.

## Key Data Flows

1. **Python solution:** CLI -> staging -> `eval_driver.py` -> PyTorch/Triton ROCm
   import -> correctness -> ROCm timing -> trace JSON.
2. **HIP/native solution:** CLI -> staging -> HIP build -> shared object import
   -> correctness -> ROCm timing -> trace JSON.
3. **Dataset run:** `scripts/run_dataset.py` -> per-problem solution wrapping ->
   CLI invocation -> trace summary output.

## Integration Points

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Python to native build | Staging files and shared object | Existing `benchmark_kernel.so` contract can remain if build changes underneath. |
| Eval driver to profiler | Python subprocess invoking ROCm APIs/tools | Need deterministic JSON stdout and library noise isolation. |
| Test suite to hardware | pytest markers and env probes | Must distinguish no ROCm, unsupported arch, and real failures. |
| Docker to host GPU | ROCm device mounts and groups | Replace NVIDIA Docker flags with ROCm device access requirements. |

## Phase Ordering Implications

1. Establish ROCm environment and dependency baseline first.
2. Port schemas/build language names and HIP compile path before examples.
3. Port eval driver and timing before claiming benchmark validity.
4. Migrate examples/tests after the runtime paths exist.
5. Validate on RDNA 4 and CDNA 3 last, then tighten docs and license notes.

## Sources

- Current codebase map: `.planning/codebase/ARCHITECTURE.md`
- AMD HIP porting guide
- ROCm PyTorch installation docs
- rocprofv3 documentation
- ROCm library documentation for rocBLAS, MIOpen, CK, rocWMMA, and primitives

---
*Architecture research for: ROCm port of SOL ExecBench*
*Researched: 2026-05-21*
