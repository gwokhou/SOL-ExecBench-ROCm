# Project Research Summary

**Project:** SOL ExecBench ROCm Port
**Domain:** AMD SOLAR coverage, live ROCm profiler timing, and scoring workflow
**Researched:** 2026-05-22
**Confidence:** HIGH

## Executive Summary

v1.6 should turn the v1.5 AMD-native SOL/timing/scoring foundation into an
end-to-end workflow. The original paper baseline is still SOLAR-derived
hardware SOL bounds plus SOL Score as gap closing from a scoring baseline to
the SOL bound. For the ROCm port, the equivalent must remain AMD-native and
derived, not a NVIDIA B200/SOLAR leaderboard claim.

The highest technical risk is timing accuracy. `rocprofv3` is the right live
collector for HIP native and Triton kernel-activity evidence, but PyTorch needs
operator attribution semantics, and mixed workloads may need explicit fallback
or split evidence. The user requirement to expose chimney-style
`operator/source type -> timer type` outputs is technically justified.

The strongest roadmap is:

1. broaden AMD SOL analyzer coverage with explicit coverage/confidence reports;
2. integrate live `rocprofv3` collection behind the existing timing policy;
3. connect derived AMD score reports to dataset/CLI workflows;
4. enforce compatibility tests around canonical trace JSONL, schemas, and
   primary CLI defaults.

## Key Findings

### Stack Additions

- Extend existing `amd_sol.py`, `timing_policy.py`, `rocm_profiler.py`, and
  `amd_score.py`; do not create a parallel benchmark stack.
- Use `rocprofv3 --kernel-trace` or runtime/granular trace with controlled
  output files for live timing evidence.
- Keep PyTorch attribution distinct from HIP/Triton kernel activity.
- Emit derived JSON artifacts for timing, SOL bounds, and scores.

### Feature Table Stakes

- Analyzer registry or equivalent structure for reductions, normalization,
  softmax/attention-like patterns, shape/view/broadcast/data-movement nodes,
  and clearer elementwise activations.
- Coverage summaries that state supported, inexact, and unsupported operations.
- Live timing adapter that invokes benchmark execution through `rocprofv3`,
  parses output, records tool/GPU metadata, and labels fallbacks.
- Dataset runner or additive CLI workflow that writes AMD-native score reports
  from trace JSONL, timing evidence, SOL bounds, and baseline inputs.
- Contract guardrails proving canonical traces, public schemas, and primary CLI
  behavior remain compatible.

### Watch Out For

- Do not unify HIP, Triton, and PyTorch timing if that makes the numbers less
  accurate.
- Do not let compile/autotune/warmup or unrelated profiler rows enter measured
  timing silently.
- Do not hide unsupported SOL operators behind aggregate scores.
- Do not claim CDNA3 full-suite validation or NVIDIA B200/SOLAR equivalence.

## Implications for Requirements

Recommended requirement categories:

- `SOLCOV`: AMD SOL analyzer coverage and coverage reporting.
- `PROF`: live source-specific timing evidence collection.
- `SCORE`: derived score workflow integration.
- `COMPAT`: public contract preservation.
- `CLAIM`: explicit claim boundaries, especially CDNA3 and NVIDIA equivalence.

## Implications for Roadmap

Recommended phase structure:

1. AMD SOL Analyzer Coverage
2. Live rocprofv3 Timing Integration
3. Derived AMD Scoring Workflow
4. Compatibility, Documentation, and Claim Guardrails

The phases can start at 27 because v1.5 ended at Phase 26.

## Sources

- SOL-ExecBench paper: https://arxiv.org/abs/2603.19173
- ROCprofiler-SDK `rocprofv3` docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.0.1/how-to/using-rocprofv3.html
- ROCprofiler-SDK quick guide: https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/quick_guide.html

---
*Research completed: 2026-05-22*
*Ready for requirements: yes*
