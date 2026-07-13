# Phase 25: AMD SOL Bound Foundation - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous smart discuss, infrastructure path

<domain>
## Phase Boundary

Build a SOLAR-like AMD bound foundation that extracts an auditable operation
graph from benchmark definitions, estimates FLOPs and bytes with confidence,
looks up AMD hardware model inputs, and emits per-op plus aggregate bound
artifacts before AMD-native scoring is allowed.

</domain>

<decisions>
## Implementation Decisions

### Evidence First
- SOL bounds must be evidence-carrying objects, not bare formulas.
- FLOP/byte estimates must record supported, inexact, and unsupported states.
- Hardware model entries must include architecture, dtype/path, source,
  confidence, and validation status.

### Scope Control
- Start with a foundation that supports common matmul and elementwise patterns.
- Unsupported operations should be represented explicitly rather than silently
  fabricated.
- Do not integrate final AMD-native scoring in this phase; Phase 26 owns score
  aggregation and reports.

### Hardware Claims
- RDNA4 model entries may be marked as supported or provisional according to
  local evidence available in this repository.
- CDNA3 entries are allowed only as unvalidated scaffolding.
- Do not claim CDNA3 hardware validation.

### the agent's Discretion
- Exact module split under `src/sol_execbench/core/scoring/`.
- Exact initial hardware constants, provided they are clearly source-labeled and
  confidence-labeled.
- Exact bound artifact field names, provided they are serializable and
  auditable.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Definition.get_input_shapes()` and `Definition.get_output_shapes()` resolve
  symbolic shapes with workload axes.
- `DType` in `definition.py` provides dtype identifiers for byte estimates.
- `scoring_guardrails.py` already carries AMD-claim warning patterns.

### Established Patterns
- Internal data helpers use frozen dataclasses with `to_dict()`.
- Derived evidence must remain separate from canonical trace JSONL.
- Tests use small synthetic definitions and workload objects for pure logic.

### Integration Points
- Phase 25 artifacts feed Phase 26 AMD-native scoring.
- `docs/internal/analysis.md` should explain AMD SOL bound artifacts as a prerequisite
  for AMD-native claims.

</code_context>

<specifics>
## Specific Ideas

- Implement `src/sol_execbench/core/scoring/amd_sol.py`.
- Add tests in `tests/sol_execbench/test_amd_sol_bounds.py`.
- Keep all code pure and CPU-only for fast validation.

</specifics>

<deferred>
## Deferred Ideas

- Real CDNA3 validation.
- Full SOLAR parity with every upstream operator category.
- Final score aggregation and CLI exposure.

</deferred>
