# Project Research: Stack

**Project:** SOL ExecBench ROCm Port
**Milestone:** v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow
**Researched:** 2026-05-22
**Confidence:** HIGH

## Scope

This milestone should extend the v1.5 foundation without changing canonical
trace JSONL, public schemas, or the primary `sol-execbench` CLI contract.

## Recommended Stack Additions

| Area | Recommendation | Rationale |
|------|----------------|-----------|
| Live profiling | Use `rocprofv3` runtime/kernel trace as the live timing evidence collector for HIP native and Triton-generated kernels. | ROCm 7 documentation supports runtime trace, granular `--hip-trace --kernel-trace`, CSV/rocpd outputs, kernel filters, and output directory/file controls. |
| PyTorch attribution | Keep PyTorch source timing as a separate PyTorch-operator attribution path, with ROCm device activity as evidence where available. | PyTorch operators can dispatch multiple HIP/library kernels, so raw kernel activity alone can lose operator semantics. |
| SOL analysis | Extend `src/sol_execbench/core/scoring/amd_sol.py` instead of adding a separate scoring stack. | v1.5 already defines graph nodes, work estimates, hardware models, per-op bounds, confidence, and derived artifacts. |
| Score workflow | Add additive report generation around `src/sol_execbench/core/scoring/amd_score.py`, dataset runner, and optional CLI output flags. | Existing AMD-native scores are derived artifacts and already preserve evidence references. |
| Contract tests | Reuse existing public contract guardrail tests and add negative tests for trace/schema/CLI stability. | User constraint requires compatibility to be an implementation gate. |

## Current Project Hooks

- `timing_policy.py` already models `source_type -> backend -> activity_domain`.
- `rocm_profiler.py` already builds `rocprofv3` commands and parses representative CSV.
- `amd_sol.py` currently recognizes matmul and broad elementwise/reduction-like calls, with unsupported operations surfaced explicitly.
- `amd_score.py` already builds guarded workload and suite reports from timing, baseline, and SOL-bound evidence.
- `scripts/run_dataset.py` is the natural integration point for suite-level derived score outputs.

## Stack Constraints

- Do not make `rocprofv3` trace output part of canonical benchmark stdout.
- Do not force HIP, Triton, and PyTorch into one timer if it weakens timing accuracy.
- Do not promote CDNA3 hardware validation claims in v1.6.
- Prefer derived JSON artifacts and additive options over schema changes.

## Sources

- SOL-ExecBench paper: https://arxiv.org/abs/2603.19173
- ROCprofiler-SDK `rocprofv3` docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.0.1/how-to/using-rocprofv3.html
- ROCprofiler-SDK quick guide: https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/quick_guide.html
