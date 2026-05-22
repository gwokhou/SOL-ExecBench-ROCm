# Stack Research

**Domain:** AMD-native SOL scoring and ROCm profiler timing for SOL ExecBench ROCm
**Researched:** 2026-05-22
**Confidence:** MEDIUM

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| ROCprofiler-SDK / `rocprofv3` | ROCm 7.x family | Kernel dispatch tracing, HIP/HSA/runtime API tracing, and optional counter collection. | It is the official ROCm path for kernel trace rows with dispatch IDs, stream IDs, kernel names, and start/end timestamps. This is the closest ROCm equivalent to the paper's device-activity timing requirement. |
| PyTorch ROCm | Project lock: `torch==2.10.0+rocm7.1` | Reference execution, PyTorch source-operator timing category, graph capture inputs, and existing benchmark runtime. | The existing harness already uses PyTorch ROCm through the historical `torch.cuda` API. PyTorch docs confirm HIP intentionally reuses `torch.cuda` interfaces. |
| Triton ROCm | Project lock: `triton-rocm==3.6.0` | Triton source-operator timing category and generated-kernel analysis. | Triton kernels on AMD are optimized similarly to HIP/CUDA kernels and require source-specific profiling because JIT/autotune behavior can add non-kernel work. |
| Existing SOL ExecBench schemas | Local project | Definition, workload, solution, and trace contracts. | The AMD scoring system should extend around these contracts rather than replace them, so existing benchmark semantics remain stable. |
| Python standard parsers (`csv`, `json`, `sqlite3`) | Python 3.12+ | Parse profiler CSV/rocpd outputs and scoring artifacts. | Avoid extra dependencies for early implementation; ROCprofiler output can be CSV or rocpd/SQLite. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.fx` / `torch.export` candidate APIs | PyTorch-provided | Graph extraction experiments for PyTorch reference functions. | Use when tracing reference functions into analyzable op graphs. Keep fallback to explicit op analyzers because dynamic inputs and custom input generation may break generic tracing. |
| `torch.profiler` | PyTorch-provided | PyTorch operator-level attribution and optional cross-check for Torch-op categories. | Use for PyTorch source-operator attribution, not as the sole kernel timing authority. PyTorch profiler aggregates Torch ops and CUDA/HIP activity under `ProfilerActivity.CUDA`. |
| `rocprofiler-compute` / ROCm Compute Profiler | ROCm toolchain | Guided counter analysis and architecture-specific profiling. | Use for validation and diagnostics, not the first default timing backend, unless `rocprofv3` kernel trace cannot provide enough signal. |
| ROCTx markers | ROCm toolchain | Optional region markers to isolate timed iterations. | Use if kernel trace parsing needs robust boundaries between correctness, warmup, and timed sections. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `rocprofv3 --kernel-trace --output-format csv` | Collect kernel dispatch timing. | Output rows include `Dispatch_Id`, `Kernel_Name`, `Correlation_Id`, `Start_Timestamp`, and `End_Timestamp`. |
| `rocprofv3 --hip-runtime-trace --output-format csv` | Connect host HIP runtime calls to kernel activity. | Useful for HIP native and PyTorch/Triton launch correlation. |
| `rocprofv3 --runtime-trace` or `--sys-trace` | Broader trace bundle. | Generates kernel, HIP API, memory copy/allocation, scratch memory, and marker traces; useful when simple kernel trace is insufficient. |
| `rocprofv3 --output-directory` / `--output-file` | Stable artifact locations. | Required for deterministic parser integration in subprocess evaluation. |
| Existing `sol-execbench` staging directory | Profile target boundary. | Keep profiler outputs under staging or a declared artifact directory so trace JSONL remains clean. |

## Installation

No new package dependency is required for the first milestone slice. The required tools should come from ROCm and the existing Python environment.

```bash
uv sync --all-groups
which rocprofv3
uv run python - <<'PY'
import torch
print(torch.__version__, torch.version.hip, torch.cuda.is_available())
PY
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `rocprofv3` kernel trace as default timing source | PyTorch HIP-backed events | Use as fallback when profiler is unavailable or when profiler overhead/trace collection cannot be isolated. |
| Source-specific timing semantics | One unified timing aggregate | Use a unified aggregate only after evidence shows Triton, HIP native, and PyTorch workloads are measured accurately with the same backend and aggregation rule. |
| SOLAR-like staged analysis | Static config-only roofline table | Use config-only bounds for unsupported ops or bootstrap cases, but keep them explicitly marked as manual/low-confidence. |
| CSV parser first | rocpd/SQLite parser first | Use rocpd when CSV precision or schema stability is inadequate, or when trace conversion becomes required for richer correlation. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Treating PyTorch event timing as equivalent to kernel activity trace | Events can measure a timed region but do not identify all underlying kernels or operator attribution. | `rocprofv3` kernel trace with source-specific aggregation. |
| Presenting AMD scores as B200/SOLAR-equivalent | The paper's interpretation is anchored to NVIDIA B200 and SOLAR. | AMD-native claim levels with documented hardware model, clock policy, and evidence. |
| Forcing one timing口径 across Triton, HIP native, and PyTorch | It may hide JIT/autotune/library dispatch differences and reduce accuracy. | Expose `source_type -> timing_backend -> aggregation_rule`. |
| CDNA 3 hardware-validation claims | User excluded real CDNA 3 validation from v1.5. | Keep readiness/deferred wording until a full `gfx94*` run exists. |

## Stack Patterns by Variant

**If source type is `hip_native`:**
- Use kernel dispatch rows as primary timing.
- Aggregate kernels launched by the solution call within timed iteration boundaries.
- Use HIP runtime trace only for correlation/debugging unless kernel trace ambiguity exists.

**If source type is `triton`:**
- Separate compile/autotune/warmup from measured iterations.
- Prefer kernel dispatch rows for final timing, with name/correlation matching that tolerates generated kernel names.
- Record the chosen Triton timing semantics explicitly.

**If source type is `pytorch`:**
- Treat PyTorch as an operator source, not a single kernel source.
- Use kernel dispatch rows for elapsed device work and PyTorch profiler only for operator attribution/cross-check.
- Decide whether to score full operator region or selected kernel set per requirement.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| PyTorch ROCm | `torch.cuda` compatibility namespace | PyTorch HIP docs state ROCm builds intentionally reuse `torch.cuda` interfaces; naming must not imply CUDA runtime presence. |
| ROCprofiler-SDK | ROCm 7.x toolchain | `rocprofv3` supports kernel trace, HIP/HSA traces, output file/directory controls, and CSV output used by this milestone. |
| Triton ROCm | PyTorch ROCm wheel/source stack | Triton ROCm kernel profiling should be validated locally because generated kernel naming and JIT behavior can vary by version. |

## Sources

- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html - verified kernel trace, HIP/HSA traces, output controls, and CSV columns.
- https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html - verified ROCm profiling capability categories and Perfetto-style timeline use.
- https://docs.pytorch.org/docs/2.12/notes/hip.html - verified PyTorch ROCm reuse of `torch.cuda` interfaces and HIP-specific checks.
- https://docs.pytorch.org/docs/2.12/profiler.html - verified PyTorch profiler activity groups, CUDA/HIP activity naming surface, and operator/device activity knobs.
- https://rocm.docs.amd.com/en/docs-6.2.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html - verified AMD Triton optimization context and need for Triton-specific analysis.

---
*Stack research for: AMD-native SOL scoring and ROCm profiler timing*
*Researched: 2026-05-22*
