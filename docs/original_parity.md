# Original SOL ExecBench Parity

This document compares the ROCm port against NVIDIA SOL ExecBench public
functionality. It is a closure artifact for v1.3: every original public surface
is classified as ported, intentionally replaced, compatibility-only, or out of
scope for this ROCm-only fork.

## Public Surface Matrix

| Original capability | NVIDIA implementation | ROCm disposition | Notes |
| --- | --- | --- | --- |
| Single-problem CLI | `sol-execbench <problem_dir> --solution <solution-path>` and explicit `--definition/--workload/--solution` mode | Ported | CLI shape is preserved; compile wording and runtime behavior are HIP/ROCm-specific. |
| CLI output controls | `--json`, `-o/--output`, `--verbose`, `--keep-staging`, timeouts | Ported | Trace JSONL remains the machine-readable contract. |
| Clock locking | `--lock-clocks` via NVIDIA tooling | Replaced | ROCm lock/unlock operations use `amd-smi`, while `rocm-smi` is used for clock verification; failure remains explicit when requested locking is unavailable. |
| Dataset download | `scripts/download_data.sh` downloads SOL-ExecBench and FlashInfer Trace datasets | Ported | Dataset origin remains NVIDIA/Hugging Face; runtime support is ROCm-only. |
| Dataset runner | `scripts/run_dataset.py` discovers L1, L2, FlashInfer-Bench, and Quant problems | Ported | Runner preserves reference/custom solution wrapping and JSONL trace parsing. |
| Definition schema | Benchmark definition Pydantic contract | Ported | Schema semantics are preserved for workload/reference evaluation. |
| Workload schema | Workload JSONL Pydantic contract | Ported | UUID, axes, and input descriptor semantics are preserved. |
| Solution schema | NVIDIA language/hardware metadata | Replaced | Public shape is preserved, but language/hardware enum values are ROCm-only. |
| Trace schema | JSONL trace with workload, solution, evaluation, correctness, performance, environment | Ported | Environment fields report AMD/ROCm context. |
| Reward-hack checks | Monkey-patch, thread injection, lazy output, and stream-style defenses | Ported | Defenses remain active under ROCm. |
| Correctness evaluation | Reference comparison with shape, dtype, numerical, and non-finite checks | Ported | Evaluation semantics are retained. |
| Timing | CUDA/CUPTI-era timing and device synchronization | Replaced | ROCm port uses HIP-backed PyTorch device events and ROCm clock tooling. |
| SOL-Score formula | B200/SOLAR anchored formula | Partially ported | Formula is retained, but AMD-native interpretation requires a separate ROCm model. |
| Leaderboard semantics | NVIDIA B200 leaderboard context | Out of scope | This fork does not claim compatibility with NVIDIA leaderboard hardware results. |

## Solution Category Disposition

| Original category | Original value | ROCm disposition | ROCm replacement or status |
| --- | --- | --- | --- |
| PyTorch | `pytorch` | Ported | PyTorch ROCm through the historical `torch.cuda` compatibility namespace. |
| Triton | `triton` | Ported | Triton ROCm environment. |
| CUDA C++ | `cuda_cpp` | Replaced | `hip_cpp` native extension path. |
| cuBLAS | `cublas` | Replaced | `hipblas` is supported with runnable SGEMM example coverage; hipBLASLt remains a future replacement direction. |
| cuDNN C++ | `cudnn` | Candidate replacement | `miopen` or HIP/Triton fallback. |
| cuDNN frontend | `cudnn_frontend` | Candidate replacement | `miopen` or HIP/Triton fallback; no direct Python frontend parity is claimed. |
| CUTLASS | `cutlass` | Candidate replacement | `ck`, `rocwmma`, HIP, or Triton ROCm. |
| CuTe DSL | `cute_dsl` | Compatibility-only | No direct ROCm runtime in this port; examples are PyTorch compatibility examples unless replaced. |
| cuTile | `cutile` | Compatibility-only | No direct ROCm runtime in this port; examples are PyTorch compatibility examples unless replaced. |

## Intentional ROCm Substitutions

- NVIDIA container runtime requirements are replaced by AMD GPU device
  passthrough for `/dev/kfd` and `/dev/dri`.
- CUDA wheel indexes and NVIDIA Python packages are replaced by PyTorch ROCm and
  ROCm-compatible dependencies.
- CUPTI timing is replaced by HIP-backed PyTorch events and optional `rocprofv3`
  profiling.
- B200 hardware metadata is replaced by AMD gfx targets: `gfx1200`, `gfx940`,
  `gfx941`, `gfx942`, and `LOCAL`.

## Remaining Non-CDNA Gaps To Close

- AMD-native scoring and roofline interpretation are implemented as guarded
  derived artifacts; remaining claims still require appropriate evidence and
  validation scope.
- AMD-native scoring or roofline interpretation is not a claim of NVIDIA B200
  or leaderboard equivalence.
- Public baseline comparison must consume existing trace outputs without adding
  fields to trace JSONL.
- ROCm library categories must be audited so documentation matches actual
  runnable example and build coverage.

## Explicitly Out Of Scope

- Restoring CUDA/NVIDIA runtime compatibility.
- Claiming NVIDIA leaderboard equivalence.
- Claiming CDNA 3 `gfx94*` hardware validation before a full adapted-suite run
  is recorded.
