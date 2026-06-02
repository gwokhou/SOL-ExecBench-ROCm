# Phase 42: Structured Bound Graph IR - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-23
**Phase:** 42-Structured Bound Graph IR
**Areas discussed:** IR node granularity, extraction strategy, SOLAR pipeline boundary, tracer execution, unsupported semantics, coverage debt, module boundary, IR layers, operation taxonomy

---

## IR Node Granularity

| Option | Description | Selected |
|--------|-------------|----------|
| Operation expression | One recognized operation/expression per node; closest to the current `extract_graph()` behavior. | |
| AST statement | One source statement per node; preserves assignment context but can mix multiple operations. | |
| Tensor dataflow / operator graph | Explicit operator and tensor/value dataflow graph aligned with SOLAR. | ✓ |

**User's choice:** "I want to stay consistent with the paper."
**Notes:** Interpreted as paper-aligned operator/dataflow graph rather than a shallow AST expression list.

---

## Extraction Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Dynamic trace first | Execute/instrument the reference workload path to capture actual eager execution metadata. | ✓ |
| Static AST with dataflow enrichment | Infer graph statically from Python source and schema shapes. | |
| Hybrid minimal | Define the IR schema now and keep current AST extraction as the main filler. | |

**User's choice:** "I want to stay consistent with the paper."
**Notes:** SOLAR's extractor captures eager execution path metadata, so Phase 42 should be dynamic-trace-first with AST fallback.

---

## SOLAR Pipeline Boundary

| Option | Description | Selected |
|--------|-------------|----------|
| IR prepares einsum conversion | Build an operator/dataflow IR that is ready for later extended-einsum/formula conversion. | ✓ |
| Include minimal einsum nodes now | Generate preliminary einsum-like expressions during Phase 42. | |
| Defer einsum entirely | Build only an operator graph with no conversion-oriented metadata. | |

**User's choice:** "I want to stay consistent with the paper."
**Notes:** Phase 42 maps to the SOLAR Graph Extractor layer. Formal extended-einsum conversion remains later scope, but the IR must be conversion-ready.

---

## Tracer Execution

| Option | Description | Selected |
|--------|-------------|----------|
| Instrument reference in isolated tracer | Execute reference code in a separate IR extraction path to collect metadata. | ✓ |
| Use generated fake/meta tensors | Avoid real tensor math where possible and rely on shape-only propagation. | |
| No execution in Phase 42 | Define schema and static fallback only. | |

**User's choice:** "I want to stay consistent with the paper."
**Notes:** The tracer may execute reference code only for metadata extraction and must remain isolated from benchmark timing, correctness, and Trace JSONL generation.

---

## Unsupported Semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit unsupported nodes | Preserve unsupported behavior as graph evidence with source, reason, and confidence. | ✓ |
| Inexact graph gaps | Mark partially understood behavior as inexact evidence. | ✓ |
| Fail graph extraction | Fail the whole graph on the first unsupported semantic construct. | |

**User's choice:** "I want to stay consistent with the paper."
**Notes:** Locked as explicit unsupported nodes plus inexact partial evidence. Unknown behavior must not be silently dropped.

---

## Coverage Debt

| Option | Description | Selected |
|--------|-------------|----------|
| Propagate coverage debt | Graph still exists, but unsupported/inexact coverage is exposed and later bounds cannot treat it as zero cost. | ✓ |
| Block bound generation | Any unsupported node prevents bound artifact generation. | |
| Best-effort bound | Score supported nodes and only warn about unsupported nodes. | |

**User's choice:** "I want to stay consistent with the paper."
**Notes:** SOL bounds require auditable operator/einsum/analyzer evidence; unsupported or missing evidence cannot improve aggregate bounds.

---

## Module Boundary

| Option | Description | Selected |
|--------|-------------|----------|
| New IR module + compatibility facade | Add a dedicated IR module; keep `amd_sol.py` as public compatibility facade. | ✓ |
| In-place rewrite `amd_sol.py` | Upgrade all graph logic inside the current module. | |
| Breaking rename | Replace old public APIs with new names. | |

**User's choice:** "I want to stay consistent with the paper."
**Notes:** SOLAR is a staged pipeline, so a dedicated IR module better preserves Graph Extractor / converter / analyzer boundaries while satisfying IR-04 compatibility.

---

## IR Layers

| Option | Description | Selected |
|--------|-------------|----------|
| Operator graph only | Expose BoundGraph/BoundGraphNode/BoundTensor and reserve conversion metadata for later. | ✓ |
| Operator graph + preliminary einsum graph | Expose both graph layers in Phase 42. | |
| One unified graph | Mix operator, tensor, einsum, and formula fields in a single structure. | |

**User's choice:** "I want to stay consistent with the paper."
**Notes:** Phase 42 should correspond to SOLAR's Graph Extractor layer. Formal `EinsumGraph` belongs to later phases.

---

## Operation Taxonomy

| Option | Description | Selected |
|--------|-------------|----------|
| Paper-aligned families | Use paper-style operation families while retaining low-level operator identity. | ✓ |
| Current local families | Keep the existing local categories such as matmul, elementwise, activation, reduction. | |
| Low-level PyTorch op names | Store only callable/operator names without abstract family grouping. | |

**User's choice:** "I want to stay consistent with the paper."
**Notes:** The IR should retain both low-level op identity and a paper-aligned `op_family` suitable for coverage and later modeling.

---

## the agent's Discretion

- The agent may choose module and dataclass names.
- The agent may choose the concrete tracer mechanism and fallback strategy.
- The agent may decide which initial common PyTorch patterns are feasible for Phase 42 tests, as long as the IR remains paper-aligned and dynamic-trace-first.

## Deferred Ideas

- Full agentic extended-einsum conversion.
- Full SOL Analyzer and Orojenesis-style data movement modeling.
- Dataset and score-report integration.
- MI300X-on-CDNA3 and CDNA 4 validation.
