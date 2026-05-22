# Project Research: Pitfalls

**Project:** SOL ExecBench ROCm Port
**Milestone:** v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow
**Researched:** 2026-05-22
**Confidence:** HIGH

## Critical Pitfalls

| Pitfall | Why It Matters | Prevention |
|---------|----------------|------------|
| Unified timing hides source semantics | HIP native, Triton, and PyTorch do not naturally share one accurate timing interpretation. | Keep `source_type -> timer_backend -> interpretation` as an explicit contract and allow chimney-style outputs. |
| Profiler rows include unmeasured work | Compile, autotune, memory copies, unrelated kernels, or warmup can pollute measured duration. | Require timing-region evidence, post-warmup aggregation rules, kernel filters or markers where available, and fallback labels when evidence is ambiguous. |
| PyTorch kernel activity loses operator attribution | One PyTorch op can dispatch many kernels or library calls. | Treat PyTorch as operator attribution first, with device rows as supporting evidence. |
| SOL coverage appears complete when unsupported ops remain | Score reports become misleading if unsupported nodes quietly contribute zero or coarse estimates. | Surface unsupported/inexact counts and block or warn on complete-score claims. |
| Hardware model confidence is overstated | RDNA4 and CDNA3 peak values may differ by dtype/path and validation state. | Carry source, confidence, and validation status in every bound artifact. |
| Derived artifacts mutate public contracts | Changing trace JSONL or default CLI behavior would violate the milestone's hard constraint. | Add contract tests for trace fields, schemas, and primary CLI behavior before integration. |
| CDNA3 validation leaks into claims | User excluded CDNA3 validation from v1.6. | Keep CDNA3 warnings and no-claim docs active in all score outputs. |

## Phase-Level Risk Guidance

- Analyzer work should prioritize correctness of confidence labels over broad but
  vague coverage.
- Live timing should first prove the adapter can isolate measured regions and
  produce auditable evidence before becoming a report input.
- Scoring workflow should treat missing timing, missing baseline, missing bound,
  unsupported ops, or unvalidated hardware as explicit report states.
- Compatibility tests should be part of the roadmap, not a final afterthought.

## Sources

- SOL-ExecBench paper: https://arxiv.org/abs/2603.19173
- ROCprofiler-SDK `rocprofv3` docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.0.1/how-to/using-rocprofv3.html
