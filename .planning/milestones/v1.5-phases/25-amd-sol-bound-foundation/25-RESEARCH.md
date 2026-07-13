# Phase 25: AMD SOL Bound Foundation - Research

**Researched:** 2026-05-22
**Domain:** AMD speed-of-light bound artifacts for benchmark workloads
**Confidence:** MEDIUM

<research_summary>
## Summary

The current project preserves the original SOL score formula but intentionally
does not provide an AMD-native bound model. Phase 25 should add a foundational
AMD SOL artifact layer, not final scoring. The right minimum is a pure module
that takes `Definition` plus workload axes, extracts a normalized operation
graph from the reference AST, estimates FLOPs/bytes with confidence metadata,
looks up an AMD hardware model entry, and emits an auditable bound artifact.

**Primary recommendation:** Implement a conservative `amd_sol.py` module with
matmul and elementwise support, explicit unsupported entries, a small hardware
model registry, and tests proving evidence is required before scoring.
</research_summary>

<validation_architecture>
## Validation Architecture

| Requirement | Test |
|-------------|------|
| SOL-01 | Graph extraction returns matmul/elementwise/unsupported nodes from a reference function. |
| SOL-02 | FLOP/byte analysis emits confidence and rationale. |
| SOL-03 | Hardware model entries include arch, dtype/path, peak source, confidence, and validation status. |
| SOL-04 | Bound artifact contains per-op bounds and aggregate bound before scoring. |

**Command:** `uv run pytest tests/sol_execbench/test_amd_sol_bounds.py`
</validation_architecture>

<sources>
## Sources

- `src/sol_execbench/core/data/definition.py`
- `src/sol_execbench/core/scoring_guardrails.py`
- `docs/internal/analysis.md`
- SOL ExecBench paper baseline: https://arxiv.org/abs/2603.19173
</sources>

---
*Phase: 25-amd-sol-bound-foundation*
*Research completed: 2026-05-22*
*Ready for planning: yes*
