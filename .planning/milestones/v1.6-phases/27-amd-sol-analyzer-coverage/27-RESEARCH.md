# Phase 27 Research: AMD SOL Analyzer Coverage

**Phase:** 27 - AMD SOL Analyzer Coverage
**Researched:** 2026-05-22
**Status:** Ready for planning

## Current Implementation

`src/sol_execbench/core/scoring/amd_sol.py` already provides the v1.5
foundation:

- `GraphNode`, `WorkEstimate`, `OpSolBound`, and `AmdSolBoundArtifact`
  dataclasses with `to_dict()` serialization.
- `EstimateConfidence` values: `SUPPORTED`, `INEXACT`, and `UNSUPPORTED`.
- A conservative AST visitor that recognizes matmul and broad elementwise
  operations.
- Work estimation based on resolved workload axes, input/output tensor bytes,
  output elements, and a simple reduction-dimension heuristic.
- Hardware model metadata with confidence and validation status.

## Relevant Tests

`tests/sol_execbench/test_amd_sol_bounds.py` currently verifies:

- matmul graph/work/bound artifact generation,
- elementwise estimates are inexact and auditable,
- unsupported operations remain visible,
- analysis documentation requires AMD SOL bound artifacts before AMD-native
  score reporting.

## Recommended Approach

Use a small internal analyzer registry. Keep public functions stable:
`extract_graph()`, `estimate_work()`, and `build_amd_sol_bound_artifact()`.
Add new internal helpers without changing canonical trace models or public
schemas.

Recommended operation families:

- reductions: `sum`, `mean`, `amax`, `max`, `min`;
- normalization-like: `rsqrt`, variance/mean patterns, RMSNorm-style code;
- softmax/attention-like: `softmax`, `exp`, `masked_fill` as inexact
  components where possible;
- data movement: `view`, `reshape`, `transpose`, `permute`, `contiguous`,
  `unsqueeze`, `squeeze`, `expand`;
- activations: `relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `exp`.

## Guardrails

- Do not mutate canonical trace JSONL or trace schema.
- Do not promote hardware model validation status.
- Prefer `INEXACT` with rationale over unsupported silent scoring or overstated
  `SUPPORTED` confidence.
- Add coverage summary as a derived helper/artifact.

## Plan Implications

The implementation should be small and test-driven:

1. Add coverage summary model/helper.
2. Refactor graph classification into explicit analyzer metadata.
3. Extend work estimates for reductions, data movement, activations, and
   softmax-like operations.
4. Add tests for confidence labels, summaries, and trace immutability.
5. Update docs with coverage semantics.
