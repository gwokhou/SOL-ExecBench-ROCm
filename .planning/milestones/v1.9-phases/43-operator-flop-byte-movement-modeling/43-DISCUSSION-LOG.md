# Phase 43: Operator FLOP/Byte/Movement Modeling - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-23
**Phase:** 43-Operator FLOP/Byte/Movement Modeling
**Areas discussed:** Formula Evidence Structure, Byte Accounting Semantics,
Operator Coverage Depth, Paper Alignment Boundary, Confidence Downgrade Rules,
Axis And Shape Evidence, Compatibility And Test Surface

---

## Formula Evidence Structure

| Option | Description | Selected |
|--------|-------------|----------|
| New rich estimate type with legacy adapter | Add a rich operator estimate type and derive legacy `WorkEstimate` from it. | ✓ |
| Replace or expand legacy `WorkEstimate` directly | Put new semantics into the old compatibility type. | |
| Add optional fields to legacy `WorkEstimate` only | Minimal change, but weaker v2 artifact contract. | |

**User's choice:** New rich estimate type with legacy adapter.
**Notes:** The new type must carry formula, formula inputs, byte buckets,
confidence, and rationale. Legacy `WorkEstimate` remains a compatibility view.

| Option | Description | Selected |
|--------|-------------|----------|
| Structured formula plus human-readable string | Include `formula_kind`, `formula`, and `formula_inputs`. | ✓ |
| Human-readable string only | Easier to inspect, harder to verify. | |
| Structured kind and inputs only | Machine-friendly, less auditable for humans. | |

**User's choice:** Structured formula plus human-readable string.

| Option | Description | Selected |
|--------|-------------|----------|
| One primary estimate per graph node | Keep the Phase 42 node boundary stable. | ✓ |
| Split complex operators into sub-estimates | More detail, but implies a subgraph IR. | |
| Mark complex operators unsupported | Too conservative for Phase 43 criteria. | |

**User's choice:** One primary estimate per `BoundGraphNode`.

---

## Byte Accounting Semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Sum read/write/intermediate/movement bytes | `total_bytes` is the explicit sum of all buckets. | ✓ |
| Use max(read+write, movement) | More roofline-like but less auditable. | |
| Use read/write only | Does not satisfy movement evidence requirements. | |

**User's choice:** Sum all byte buckets.

| Option | Description | Selected |
|--------|-------------|----------|
| Logical views are zero movement unless materialized | Preserve view evidence but avoid fake traffic. | ✓ |
| All movement ops count tensor traffic | Conservative but overcounts pure views. | |
| Ignore pure views | Loses required evidence. | |

**User's choice:** Logical views and broadcasts are zero movement by default;
materialized paths count movement.

| Option | Description | Selected |
|--------|-------------|----------|
| Node-local tensor metadata | Use `BoundGraphNode` tensor IDs and `BoundTensor` shape/dtype. | ✓ |
| Whole-definition bytes per node | Old behavior, insufficient per-node evidence. | |
| Node-local only for GEMM | Mixed semantics. | |

**User's choice:** Node-local tensor metadata.

---

## Operator Coverage Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Strict roadmap coverage only | Cover Phase 43 success criteria families and keep others unsupported. | ✓ |
| Add rough attention/convolution estimates | Broader but outside Phase 43 and weakly justified. | |
| Only GEMM and elementwise | Too narrow for Phase 43 criteria. | |

**User's choice:** Strict roadmap coverage only.

| Option | Description | Selected |
|--------|-------------|----------|
| Conservative pass-count formulas | Use explicit inexact formulas for reduction/norm/softmax. | ✓ |
| Formula inputs without FLOP numbers | Too weak for FLOP evidence. | |
| Finer theoretical lower bounds | More proof work than this phase needs. | |

**User's choice:** Conservative pass-count formulas labeled `inexact`.

| Option | Description | Selected |
|--------|-------------|----------|
| Per-node estimates | Estimate elementwise/activation chains node by node. | ✓ |
| Fuse chains into one estimate | Requires fusion/region inference. | |
| Estimate only the final activation | Hides work. | |

**User's choice:** Per-node estimates; no fusion inference in Phase 43.

---

## Paper Alignment Boundary

| Option | Description | Selected |
|--------|-------------|----------|
| No standalone EinsumGraph, but SOLAR/einsum-style fields | Keep paper alignment without adding a new IR. | ✓ |
| Introduce formal EinsumGraph/EinsumExpression IR | Larger scope than Phase 43. | |
| Keep legacy roofline WorkEstimate only | Conflicts with paper-alignment goal. | |

**User's choice:** No standalone `EinsumGraph`; use SOLAR/einsum-style rich
estimate fields.

| Option | Description | Selected |
|--------|-------------|----------|
| New `estimate_bound_work(graph)` plus legacy adapter | Clean new API and old compatibility. | ✓ |
| One `estimate_work()` supporting both old and new inputs | Type and semantic overload risk. | |
| Only modify legacy `estimate_work()` | Leaves Phase 44 without a clean entry point. | |

**User's choice:** Add `estimate_bound_work(graph)` as primary API.

---

## Confidence Downgrade Rules

| Option | Description | Selected |
|--------|-------------|----------|
| Known semantics with incomplete evidence are inexact; unknown semantics/effects are unsupported | Clear guardrail without over-dropping coverage. | ✓ |
| Any missing shape/dtype/axis is unsupported | Too strict for conservative modeling. | |
| Prefer inexact except unknown operators | Risks hiding missing byte evidence. | |

**User's choice:** Known semantics plus incomplete evidence become `inexact`;
unknown semantics or tensor effects become `unsupported`.

| Option | Description | Selected |
|--------|-------------|----------|
| Unsupported estimate with 0 FLOPs/bytes plus explicit evidence | Maintains one estimate per node without fake work. | ✓ |
| Conservative whole-tensor bytes for unsupported ops | Pollutes evidence with guesses. | |
| Do not generate estimates for unsupported ops | Breaks per-node estimate coverage. | |

**User's choice:** Unsupported nodes still get zero-valued unsupported
estimates that later phases must not hide.

---

## Axis And Shape Evidence

| Option | Description | Selected |
|--------|-------------|----------|
| Infer GEMM dimensions from input/output tensor shapes | Best source for concrete workload evidence. | ✓ |
| Parse from source expression or AST | Fragile for traced graphs and aliases. | |
| Resolve primarily from definition axes | Weak for intermediate tensors. | |

**User's choice:** Infer GEMM and BMM formula inputs from tensor shapes.

| Option | Description | Selected |
|--------|-------------|----------|
| Prefer node attributes; missing axis falls back to all-elements inexact estimate | Practical with current extractor. | ✓ |
| Missing axis is unsupported | Too strict for existing graph metadata. | |
| Parse axis from source strings | Useful later but not a primary Phase 43 rule. | |

**User's choice:** Prefer node attributes for `dim`/axis; missing axis becomes
all-elements inexact evidence.

| Option | Description | Selected |
|--------|-------------|----------|
| Zero affected bucket and downgrade confidence with rationale | Avoids fake bytes while preserving estimate. | ✓ |
| Use output shape/dtype as fallback | Can mix wrong metadata into evidence. | |
| Raise an error | Violates evidence-over-failure semantics. | |

**User's choice:** Missing bucket metadata zeros that bucket and downgrades
confidence; all key tensors missing becomes unsupported.

---

## Compatibility And Test Surface

| Option | Description | Selected |
|--------|-------------|----------|
| New API primary, old API compatibility secondary | Tests the rich contract while guarding old callers. | ✓ |
| New and old APIs tested equally strictly | Over-locks the compatibility facade. | |
| Only test the new API | Risks compatibility regressions. | |

**User's choice:** New API primary; legacy compatibility secondary.

| Option | Description | Selected |
|--------|-------------|----------|
| Golden fixture precise assertions | Lock formulas, bytes, inputs, confidence, and rationale. | ✓ |
| Category and nonzero/zero assertions only | Too weak for auditable evidence. | |
| Broad fuzz-style parameterized tests | Less explanatory and more fragile. | |

**User's choice:** Golden fixtures with precise assertions.

| Option | Description | Selected |
|--------|-------------|----------|
| Small extractor metadata additions for estimator evidence | Add only metadata needed by Phase 43. | ✓ |
| Do not touch extractor | Weakens axis/movement evidence. | |
| Refactor extractor for complete attributes | Too broad for Phase 43. | |

**User's choice:** Small extractor metadata additions are allowed, but no graph
contract or extractor architecture rewrite.

---

## the agent's Discretion

- Exact rich estimate dataclass/module names.
- Exact formula-kind string names.
- Test fixture organization.
- Whether new rich estimate exports are public immediately or kept internal
  until Phase 44, provided downstream planned use remains clear.

## Deferred Ideas

- Attention, MoE, SSM/Mamba, convolution, embedding/positional, and broader
  paper-scale operator modeling.
- Standalone extended-einsum IR.
- Bound artifact v2 sidecar serialization.
- Score and dataset integration.
- User-facing documentation and RDNA 4 validation closure.
