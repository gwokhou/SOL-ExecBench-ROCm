# Pitfalls Research

**Domain:** AMD-native SOL scoring and ROCm profiler timing for SOL ExecBench
**Researched:** 2026-05-22
**Confidence:** MEDIUM-HIGH

## Critical Pitfalls

### Pitfall 1: Timing Accuracy Lost to a Unified Abstraction

**What goes wrong:**
HIP native kernels, Triton-generated kernels, and PyTorch operators are all
reported through one timing path even when the path measures different layers of
work or loses attribution.

**Why it happens:**
A single default timer is simpler to expose and easier to test, but source types
have different runtime boundaries and metadata.

**How to avoid:**
Make timing accuracy the highest rule. Use a source-specific timing policy when
accuracy requires it, and expose source type, timer backend, and interpretation.

**Warning signs:**
Reports say "kernel time" for PyTorch op-level rows, or Triton autotune/JIT
activity is included in steady-state measurements without being labeled.

**Phase to address:**
Profiler timing semantics and policy phase.

---

### Pitfall 2: Profiler Trace Time Is Misinterpreted

**What goes wrong:**
Profiler timestamps, HIP runtime calls, kernel dispatch intervals, and
host-observed wall time are mixed as if they are interchangeable.

**Why it happens:**
ROCm tooling can emit multiple activity domains. They are all useful, but they
answer different measurement questions.

**How to avoid:**
Parse activity domain explicitly. Record whether the duration came from kernel
activity, HIP API/runtime activity, HSA activity, PyTorch op attribution, or a
fallback event timer.

**Warning signs:**
Timing code sums all profiler rows without filtering domain/type, or reports a
single elapsed time without a backend label.

**Phase to address:**
rocprofv3 collector/parser phase.

---

### Pitfall 3: SOL Bound Claims Without Auditable Inputs

**What goes wrong:**
The benchmark reports AMD-native SOL scores, but the graph extraction,
FLOP/byte estimates, and hardware model values cannot be inspected.

**Why it happens:**
Score formulas are easy to implement after a few constants exist; evidence
artifacts take more design effort.

**How to avoid:**
Require a bound artifact before aggregation. Include operation graph,
per-operation work estimates, hardware model entry, limiting resource,
confidence, and assumptions.

**Warning signs:**
Only final score and measured time are present in outputs, or hardware peak
values appear as unversioned constants inside calculation code.

**Phase to address:**
AMD SOL bound artifact phase.

---

### Pitfall 4: Paper-Baseline Semantics Drift

**What goes wrong:**
The ROCm implementation becomes easier to score but less comparable to the SOL
ExecBench paper baseline.

**Why it happens:**
ROCm-specific engineering pressure can make it tempting to change schemas,
trace semantics, correctness loops, or scoring interpretation.

**How to avoid:**
Treat the original paper as the benchmark baseline. Keep public contract changes
additive and documented; any unavoidable ROCm-specific deviation must be
explicit in reports.

**Warning signs:**
Trace JSONL changes to carry internal SOL/timing data, or documentation makes
leaderboard-equivalence claims.

**Phase to address:**
Compatibility guardrails across all phases.

---

### Pitfall 5: CDNA3 Model Entries Become Validation Claims

**What goes wrong:**
Adding CDNA3 architecture data makes reports appear to validate CDNA3 behavior.

**Why it happens:**
Hardware model support and real hardware validation are often conflated in
performance tooling.

**How to avoid:**
Keep CDNA3 validation explicitly out of scope. Allow model/data scaffolding only
with claim guards that mark CDNA3 as unvalidated unless real `gfx94*` evidence
exists in a future milestone.

**Warning signs:**
Docs say "CDNA3 supported" without qualifying whether support means model
presence, command readiness, or real benchmark validation.

**Phase to address:**
Scoring guardrail and report phase.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hard-code AMD peak constants in scoring formulas | Fast first score | Untraceable claims and hard updates | Only in tests with fake fixtures |
| Treat PyTorch `ProfilerActivity.CUDA` naming as NVIDIA-specific | Avoids confusing labels | Breaks ROCm PyTorch profiling | Never; document HIP semantics instead |
| Use event timing as hidden fallback | Keeps runs passing | Users cannot compare timing confidence | Acceptable only if labeled in evidence |
| Ignore Triton JIT/autotune warmup | Simpler benchmark run | Inflated or unstable timing | Never for final timing claims |
| Store SOL evidence in trace JSONL | Easy plumbing | Public contract drift | Never in this milestone |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| rocprofv3 | Parse all output rows as kernels | Filter by activity domain and record domain. |
| HIP native timing | Measure HIP API duration as kernel duration | Prefer kernel activity records; label API timing separately. |
| Triton timing | Include compile/autotune/setup in steady-state time | Warm up explicitly and record generated kernel mapping. |
| PyTorch timing | Treat Python op duration as GPU kernel duration | Use profiler attribution plus device activity; label interpretation. |
| Hardware model | Use one peak value for all dtypes/archs | Key by architecture, dtype/path, and confidence source. |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Excessive full-profiler runs | Benchmark becomes slow and noisy | Use profiler-backed default only around measured kernels and cache parsed evidence where valid | Dataset-scale runs |
| Unbounded trace output | CI/local runs fill output directories | Use controlled output paths, cleanup policy, and small parser fixtures | Repeated batch runs |
| Over-broad PyTorch graph capture | Missing or misleading op mapping | Start with explicit source classification and fallback confidence levels | Dynamic models or fused ops |

## Security and Reproducibility Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Storing raw proprietary generated kernels in evidence | IP leakage | Evidence should reference hashes/metadata unless source is already public. |
| Running profiler subprocesses without controlled output directories | Accidental overwrite or data leakage | Use per-run temp/evidence directories. |
| Omitting ROCm/tool versions | Irreproducible timing | Record ROCm, PyTorch, Triton, GPU arch, clock/source metadata. |

## "Looks Done But Isn't" Checklist

- [ ] **Timing replacement:** Default timing path is changed, but evidence shows timer backend and interpretation for HIP, Triton, and PyTorch cases.
- [ ] **rocprofv3 parser:** Parser handles kernel/HIP domains separately and has fixtures from representative outputs.
- [ ] **SOL scoring:** Final scores include bound artifacts, not just formulas.
- [ ] **Hardware model:** Architecture entries include source, dtype/path, and validation status.
- [ ] **Paper parity:** Public schemas and trace JSONL remain compatible unless an additive documented output is introduced.
- [ ] **CDNA3:** Reports explicitly say CDNA3 real validation is excluded from this milestone.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Unified timing abstraction is wrong | MEDIUM | Split policy by source type, preserve old result labels, add migration docs. |
| Profiler parser conflates domains | MEDIUM | Add domain model, reparse fixtures, invalidate affected timing claims. |
| SOL score lacks evidence | HIGH | Introduce bound artifact, regenerate reports, downgrade prior claims. |
| Public contract drift | HIGH | Move fields to derived artifacts, add guardrail tests, document compatibility. |
| CDNA3 claim inflation | LOW-MEDIUM | Patch docs/reports and add claim-level tests. |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Timing accuracy lost | Timing policy phase | Tests assert source type -> timer backend -> interpretation. |
| Profiler misinterpretation | rocprofv3 parser phase | Fixture tests for kernel, HIP, and fallback cases. |
| SOL claims without evidence | SOL bound phase | Score tests require bound artifacts and confidence metadata. |
| Paper semantic drift | Guardrail phase | Public contract and trace compatibility tests. |
| CDNA3 leakage | Report/guardrail phase | Tests assert CDNA3 validation claims remain absent. |

## Sources

- SOL ExecBench paper baseline: https://arxiv.org/abs/2603.19173
- ROCprofiler-SDK `rocprofv3` usage: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- PyTorch profiler reference: https://docs.pytorch.org/docs/2.12/profiler.html
- PyTorch HIP semantics: https://docs.pytorch.org/docs/2.12/notes/hip.html
- ROCm HIP performance guidelines: https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html
- ROCm Triton optimization guidance: https://rocm.docs.amd.com/en/docs-6.2.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html

---
*Pitfalls research for: v1.5 AMD-native SOL scoring and ROCm profiler timing*
*Researched: 2026-05-22*
