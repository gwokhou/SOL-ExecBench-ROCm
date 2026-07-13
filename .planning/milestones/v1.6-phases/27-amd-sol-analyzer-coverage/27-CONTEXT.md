# Phase 27: AMD SOL Analyzer Coverage - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 27 broadens AMD SOL/SOLAR-like analyzer coverage beyond the v1.5 matmul
and broad elementwise foundation. It must make coverage confidence visible
before scoring while keeping all SOL evidence as derived artifacts outside
canonical trace JSONL.

</domain>

<decisions>
## Implementation Decisions

### Analyzer Scope
- Extend first-pass coverage to reductions, normalization, softmax or
  attention-like patterns, shape/view/broadcast/data-movement nodes, and clearer
  activation families.
- Use a registry-style analyzer structure, or an equivalently explicit
  dispatch pattern, instead of continuing to grow `_GraphVisitor` as a long
  string-check chain.
- Treat unsupported operations as first-class coverage output visible before
  scoring; do not silently hide them inside aggregate score warnings.
- Emit coverage summary as a derived artifact/helper and keep it out of
  canonical trace JSONL.

### Estimate Semantics
- Model shape, view, transpose, and broadcast operations as zero- or low-FLOP
  data-movement nodes with explicit rationale.
- Use conservative formulas and `INEXACT` confidence for reduction and
  normalization estimates unless the exact pattern is known.
- Represent softmax or attention-like patterns as split reduction, elementwise,
  and memory-estimate components where practical; default to `INEXACT` rather
  than overstating support.
- Do not promote hardware validation state in this phase. Preserve source,
  confidence, and validation status for all hardware model entries.

### Test Boundary
- Add focused unit tests for analyzer coverage, confidence labels, coverage
  summaries, and artifact stability.
- Use small synthetic definitions plus existing RMSNorm or attention-like
  samples where practical.
- Add a targeted compatibility check that SOL artifacts do not mutate trace
  models.
- Add concise documentation for coverage semantics.

### the agent's Discretion
All implementation details not fixed above are at the agent's discretion,
provided they preserve public trace/schema/primary CLI compatibility.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/scoring/amd_sol.py` already defines graph nodes, work
  estimates, hardware models, per-op bounds, and bound artifacts.
- `tests/sol_execbench/test_amd_sol_bounds.py` covers the existing v1.5
  foundation and is the natural test home for Phase 27 additions.
- `docs/user/rocm_timing.md` and score/guardrail docs show the existing pattern for
  derived methodology artifacts and compatibility wording.

### Established Patterns
- Data-facing artifacts use dataclasses with `to_dict()` methods and explicit
  schema/version constants.
- Public contract stability is protected with focused pytest tests under
  `tests/sol_execbench/`.
- Unsupported or unvalidated evidence is preserved as warnings or confidence
  metadata rather than hidden.

### Integration Points
- Extend `extract_graph()`, `estimate_work()`, and `build_amd_sol_bound_artifact()`
  without changing public trace models.
- Add coverage summary helpers near the AMD SOL artifact code.
- Update documentation in `docs/internal/analysis.md` or a focused internal/public doc
  section without changing CLI defaults.

</code_context>

<specifics>
## Specific Ideas

The original SOL-ExecBench paper remains the semantic baseline: AMD SOL bounds
should approximate a SOLAR-like hardware bound pipeline, but this ROCm port must
label AMD-native confidence and must not claim upstream NVIDIA SOLAR parity.

</specifics>

<deferred>
## Deferred Ideas

- Real CDNA3 `gfx94*` full-suite validation.
- Hardware model promotion from provisional or unvalidated to validated.
- Full upstream SOLAR parity for all benchmark categories.

</deferred>
