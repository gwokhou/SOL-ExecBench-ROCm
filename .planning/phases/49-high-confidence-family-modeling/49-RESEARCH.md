# Phase 49: High-Confidence Family Modeling - Research

**Researched:** 2026-05-23
**Domain:** Internal SOLAR derivation evidence, AMD bound estimates, family-specific formula/byte modeling
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### High-Confidence Family Scope
- Implement only the high-confidence families assigned to Phase 49:
  structurally visible attention, 1D/2D/3D convolution, embedding,
  positional/gather/rotary-like memory-bound structures, and linear projection.
- Attention evidence should cover Q/K/V projections, QK score computation,
  scale or mask handling, softmax, PV aggregation, and output projection when
  axes and mask semantics are visible.
- Convolution evidence should cover dimensionality, grouped or depthwise
  metadata, stride, padding, dilation, and output spatial dimensions.
- Linear projection should become a first-class semantic family while reusing
  GEMM-compatible formula logic when dimensions are explicit.
- MoE, SSM/Mamba, ambiguous attention, and families with dynamic or incomplete
  semantics remain degraded or unsupported and are deferred to Phase 50.

### Formula And Byte Evidence
- Each promoted family must emit a family-specific formula kind, formula text,
  and formula input map.
- Each promoted family must emit dtype-aware read, write, intermediate,
  movement, and total byte evidence.
- Formula and byte evidence must carry provenance through the Phase 48 tensor,
  semantic-axis, source, confidence, missing-evidence, and warning machinery.
- Do not fabricate formulas when required dimensions, dtypes, axes, index
  semantics, mask semantics, grouping metadata, or memory behavior are absent;
  emit deterministic degraded or unsupported evidence instead.

### Extraction Strategy
- Reuse Phase 48 `SolarDerivationEvidence`, semantic groups, source
  boundaries, confidence states, and strict parser/serializer contracts.
- Reuse existing `BoundGraph`, `BoundGraphNode`, `BoundTensor`,
  `OperatorWorkEstimate`, and AMD bound estimate infrastructure where possible
  instead of introducing a separate graph stack.
- Derive evidence from `Definition`, `Workload`, reference-visible FX or AST
  structure, existing shape/dtype resolution, and existing work estimates only.
- Do not read, parse, compile, execute, or infer from submitted candidate
  solution code.
- Avoid new framework dependencies; keep implementation inside the existing
  Python, FX, AST, and scoring stack.

### Integration Boundary
- Phase 49 should produce parseable internal formula-backed evidence usable by
  later sidecar and reporting phases.
- Do not change AMD-native score eligibility, primary `sol-execbench` output,
  canonical trace JSONL, public Pydantic benchmark schemas, or public CLI
  options in this phase.
- Score eligibility, coverage aggregation, and derived report integration are
  deferred to Phase 51.
- Public documentation and claim guardrails for paper parity, hardware parity,
  and hosted leaderboard readiness remain Phase 52 concerns unless local tests
  need targeted guardrails earlier.

### the agent's Discretion
- Exact dataclass names, helper function names, and formula-kind string values
  are at the agent's discretion as long as they are deterministic,
  machine-verifiable, and consistent with existing scoring conventions.
- The planner may split work by family, by shared formula/byte model, or by
  evidence integration layer, provided dependencies remain explicit and tests
  cover each promoted family.

### Deferred Ideas (OUT OF SCOPE)
- Conservative MoE and SSM/Mamba modeling remains deferred to Phase 50.
- Sidecar coverage aggregation, AMD-native score eligibility changes, and score
  guard integration remain deferred to Phase 51.
- Dataset-runner reporting closure, public docs, and benchmark claim guardrails
  remain deferred to Phase 52.
- Paper-scale 124-model / 235-problem extraction, MI300X/CDNA validation,
  hosted leaderboard readiness, and NVIDIA Blackwell/B200 equivalence claims
  remain out of scope for v1.10.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DERIVE-01 | Recognize structurally visible attention patterns including Q/K/V projections, QK scores, scale or mask handling, softmax, PV aggregation, and output projection with degradation when axes or mask semantics are incomplete. | Use `OpFamily.ATTENTION`, Phase 47 attention fixtures, and Phase 48 group/subrole/status machinery; require explicit q/k/v, score, softmax axis, pv, output projection, and mask/scale evidence before `supported`. [VERIFIED: `.planning/REQUIREMENTS.md`, `tests/sol_execbench/fixtures/solar_derivation/attention_positive.json`, `src/sol_execbench/core/scoring/solar_derivation.py`] |
| DERIVE-03 | Recognize 1D/2D/3D convolution with grouped/depthwise metadata, stride, padding, dilation, and output spatial dimensions. | Extend graph classification and estimate logic for `conv1d`, `conv2d`, `conv3d`; require static input/weight/output shapes plus stride/padding/dilation/groups attributes before `supported`. [VERIFIED: `.planning/REQUIREMENTS.md`, `tests/sol_execbench/fixtures/solar_derivation/convolution_positive.json`, `src/sol_execbench/core/scoring/amd_bound_graph.py`] |
| DERIVE-05 | Recognize embedding, positional, gather, and rotary-like memory-bound structures with index and output-shape evidence. | Model these as `embedding_positional` memory-bound formula evidence with explicit index/table/output tensors, index dtype, and positional/rotary transform subroles; degrade when index semantics or table/output shape is incomplete. [VERIFIED: `.planning/REQUIREMENTS.md`, `tests/sol_execbench/fixtures/solar_derivation/embedding_positional_positive.json`] |
| DERIVE-06 | Treat linear projection as a first-class semantic family while reusing GEMM-compatible formulas when dimensions are explicit. | Existing `OpFamily.LINEAR_PROJECTION` classification and `_gemm_estimate()` reuse are already present; Phase 49 should preserve family identity while using GEMM dimension inference and bytes. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`] |
| MODEL-01 | Each newly promoted family emits family-specific formula kind, formula text, and formula input map. | Extend internal parseable evidence with per-node or per-group formula evidence instead of relying only on `SolarSemanticGroupEvidence.source.detail`; populate from `OperatorWorkEstimate.formula_kind`, `formula`, and `formula_inputs`. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/solar_derivation.py`] |
| MODEL-02 | Each newly promoted family emits dtype-aware read, write, intermediate, movement, and total byte evidence. | Reuse `OperatorWorkEstimate.read_bytes`, `write_bytes`, `intermediate_bytes`, `movement_bytes`, and `total_bytes`; add strict sidecar serialization for these fields with dtype provenance from `SolarTensorEvidence`. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py`] |
| MODEL-05 | Family estimates convert into per-operation compute bound, memory bound, limiting resource, and SOL-bound evidence. | Reuse the `_bound_for_estimate()` math in `amd_sol_v2.py`; Phase 49 can add internal evidence references to these bound values without changing AMD-native score eligibility. [VERIFIED: `src/sol_execbench/core/scoring/amd_sol_v2.py`, `tests/sol_execbench/test_amd_sol_v2.py`] |
</phase_requirements>

## Summary

Phase 49 should be implemented as an extension of the existing internal derivation and AMD bound-estimate stack, not as a new extractor or public schema. Phase 48 already introduced `SolarDerivationEvidence`, tensor provenance, semantic groups, strict parser/serializer helpers, source boundaries, and deterministic `supported`/`inexact`/`unsupported` to `scored`/`degraded`/`unscored` mapping. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py`, `.planning/phases/48-extraction-pipeline-and-semantic-provenance/48-VERIFICATION.md`]

The existing `OperatorWorkEstimate` is the right low-level carrier for formula kind, formula text, formula inputs, FLOPs, dtype-aware byte buckets, movement kind, axis source, confidence, rationale, and warnings. `amd_sol_v2` already converts those estimates into compute bound, memory bound, limiting resource, and SOL bound via hardware model peak throughput and memory bandwidth. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`]

The implementation gap is family-specific, parseable SOLAR evidence. Today `SolarSemanticGroupEvidence` records group/subrole/source/required/missing evidence, but formula and byte evidence are only implied through `required_evidence` strings and group `source.detail`. Phase 49 should add sidecar-only frozen dataclasses or nested fields for formula evidence, byte evidence, and per-op bound evidence, then populate them only for high-confidence families with complete metadata. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py`]

**Primary recommendation:** Extend `solar_derivation.py`, `amd_bound_graph.py`, and `amd_bound_estimates.py` in place: add parseable family formula/byte/bound evidence to internal `SolarDerivationEvidence`, implement high-confidence family estimates through `OperatorWorkEstimate`, and preserve degraded/unscored behavior when required dimensions, dtypes, axes, mask/index/grouping metadata, or memory behavior are missing. [VERIFIED: codebase grep]

## Project Constraints (from AGENTS.md)

- Source code lives under `src/sol_execbench/`; scoring utilities are under `src/sol_execbench/core/scoring/`. [VERIFIED: `AGENTS.md`]
- Tests belong under `tests/sol_execbench/` for package behavior and should use pytest. [VERIFIED: `AGENTS.md`]
- Use Python 3.12+, Ruff style, `snake_case` for functions/variables/modules, `PascalCase` for classes/Pydantic models, and descriptive `test_*` names. [VERIFIED: `AGENTS.md`]
- Prefer small unit tests for schema and driver logic; add integration coverage when changing subprocess evaluation or GPU execution behavior. [VERIFIED: `AGENTS.md`]
- Do not commit local cache, build artifacts, downloaded benchmark output, proprietary kernels, credentials, Hugging Face tokens, or downloaded datasets. [VERIFIED: `AGENTS.md`]
- GPU evaluation may require Docker, ROCm-capable AMD hardware, ROCm drivers, and `/dev/kfd` plus `/dev/dri`, but Phase 49 research and implementation should not require hardware execution. [VERIFIED: `AGENTS.md`, `.planning/phases/49-high-confidence-family-modeling/49-CONTEXT.md`]
- Commits should be DCO signed; the user specifically requested an English DCO-signed research commit message. [VERIFIED: `AGENTS.md`, user request]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Family recognition from reference/workload structure | Scoring / Derivation backend | Python FX/AST extraction | `build_bound_graph()` already owns reference-visible graph extraction from `Definition` and `Workload`; Phase 49 should extend that boundary. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py`] |
| Family formula and byte modeling | Scoring / Bound estimate backend | Internal derivation sidecar | `estimate_bound_work()` already returns per-node formula and byte buckets; SOLAR evidence should reuse these values. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py`] |
| Confidence and degradation semantics | Internal SOLAR derivation sidecar | Bound estimates | `classify_solar_confidence()` currently maps metadata completeness and estimate confidence into sidecar states. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py`] |
| Compute/memory/SOL bound conversion | AMD SOL v2 sidecar backend | Internal derivation sidecar | `amd_sol_v2._bound_for_estimate()` already implements per-op bound math over rich estimates. [VERIFIED: `src/sol_execbench/core/scoring/amd_sol_v2.py`] |
| Public CLI, canonical schemas, trace JSONL | Existing public benchmark layers | None | Phase 49 must not modify canonical data models, primary CLI, score eligibility, or trace JSONL. [VERIFIED: `.planning/phases/49-high-confidence-family-modeling/49-CONTEXT.md`, `tests/sol_execbench/test_public_contract_guardrails.py`] |

## Standard Stack

### Core

| Library / Module | Version | Purpose | Why Standard |
|------------------|---------|---------|--------------|
| Python dataclasses | stdlib | Frozen internal evidence records and JSON-safe `to_dict()` methods | Existing Phase 48 and AMD SOL v2 sidecars use frozen dataclasses for strict internal artifacts. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`] |
| `torch.fx` | existing project dependency | Reference-visible tracing and shape propagation where available | `build_bound_graph()` already attempts `symbolic_trace()` and `ShapeProp` before AST fallback. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py`] |
| Python `ast` | stdlib | Fallback extraction for reference snippets and literal attributes | `_AstBoundGraphExtractor` already extracts calls, binops, assignments, outputs, and unsupported control flow. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py`] |
| `OperatorWorkEstimate` | local | Formula kind/text/input map, FLOPs, byte buckets, confidence, rationale, warnings | Existing rich estimate model exactly matches Phase 49 formula/byte requirements. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py`] |
| `AmdSolV2OpBound` / `_bound_for_estimate()` | local | Per-op compute, memory, limiting resource, and SOL bounds | Existing v2 sidecar math satisfies MODEL-05 without new public score eligibility changes. [VERIFIED: `src/sol_execbench/core/scoring/amd_sol_v2.py`] |

### Supporting

| Library / Module | Version | Purpose | When to Use |
|------------------|---------|---------|-------------|
| `EstimateConfidence` | local enum | `supported`, `inexact`, `unsupported` confidence vocabulary | Use for every family estimate and sidecar evidence item. [VERIFIED: `src/sol_execbench/core/scoring/amd_hardware_models.py`, `src/sol_execbench/core/scoring/solar_derivation.py`] |
| `SolarTensorEvidence` | local | Shape, dtype, semantic axes, source, producer, missing evidence | Use as the provenance anchor for formula and byte evidence. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py`] |
| Phase 47 fixture loader | local test helper | Golden family/subrole/status expectations | Use for fixture-driven tests without executing fixture references. [VERIFIED: `tests/sol_execbench/solar_derivation_fixtures.py`, `tests/sol_execbench/test_solar_derivation_contract.py`] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Existing FX/AST/local scoring stack | ONNX, MLIR, Dynamo, SymPy, NetworkX | Explicitly out of scope for v1.10; adding dependencies would violate milestone constraints. [VERIFIED: `.planning/REQUIREMENTS.md`] |
| Extending `OperatorWorkEstimate` | Separate family estimate graph | Duplicate graph/estimate state and risks disagreement with AMD SOL v2 bound conversion. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`] |
| Public Pydantic schema additions | Canonical fields on `Definition`, `Workload`, or `Trace` | Violates sidecar-only contract and existing guardrails. [VERIFIED: `docs/internal/solar_derivation_contract.md`, `tests/sol_execbench/test_public_contract_guardrails.py`] |

**Installation:**

```bash
# No new dependencies for Phase 49.
```

**Version verification:** No external package versions need verification because Phase 49 must use the existing local Python, FX, AST, and scoring stack only. [VERIFIED: `.planning/REQUIREMENTS.md`, `.planning/phases/49-high-confidence-family-modeling/49-CONTEXT.md`]

## Package Legitimacy Audit

No package legitimacy gate is required because Phase 49 installs no new external packages. [VERIFIED: `.planning/REQUIREMENTS.md`, `.planning/phases/49-high-confidence-family-modeling/49-CONTEXT.md`]

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| None | n/a | n/a | n/a | n/a | n/a | No install |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```text
Definition + Workload
        |
        v
build_bound_graph()
  |-- torch.fx trace + ShapeProp when available
  |-- AST fallback when trace fails
        |
        v
BoundGraph nodes/tensors/edges
        |
        v
estimate_bound_work()
  |-- high-confidence family formulas and bytes
  |-- confidence + warnings for missing metadata
        |
        +----------------------+
        |                      |
        v                      v
derive_solar_derivation_evidence()     amd_sol_v2 bound math helper
  |-- semantic groups                  |-- compute_bound_ms
  |-- subroles                         |-- memory_bound_ms
  |-- tensor provenance                |-- limiting_resource
  |-- formula/byte/bound evidence      |-- sol_bound_ms
  |-- scored/degraded/unscored
        |
        v
Internal sidecar evidence only
        |
        v
Canonical Definition/Workload/Trace and primary CLI remain unchanged
```

### Recommended Project Structure

```text
src/sol_execbench/core/scoring/
├── amd_bound_graph.py        # add family classifiers and static attributes
├── amd_bound_estimates.py    # add high-confidence family estimates
├── amd_sol_v2.py             # reuse bound math; avoid public score behavior changes
└── solar_derivation.py       # add strict internal formula/byte/bound evidence

tests/sol_execbench/
├── test_amd_bound_graph.py              # graph family recognition and attributes
├── test_amd_bound_estimates.py          # formula and byte bucket exactness
├── test_solar_derivation_evidence.py    # parser, sidecar, degradation, provenance
├── test_solar_derivation_family_modeling.py  # recommended new focused file
└── test_public_contract_guardrails.py   # canonical schema and primary CLI guardrails
```

### Pattern 1: Keep Rich Work Estimates As The Numeric Source Of Truth

**What:** Family-specific SOLAR evidence should be derived from `OperatorWorkEstimate` values rather than recalculating formula and byte data in `solar_derivation.py`. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py`]

**When to use:** Use for all promoted families once graph nodes and tensor metadata are available. [VERIFIED: `.planning/phases/49-high-confidence-family-modeling/49-CONTEXT.md`]

**Example:**

```python
# Source: src/sol_execbench/core/scoring/amd_bound_estimates.py
OperatorWorkEstimate(
    node_id=node.node_id,
    op_family=node.op_family,
    op_name=node.op_name,
    formula_kind="family_specific_kind",
    formula="symbolic_formula",
    formula_inputs={"dimension": value},
    flops=flops,
    read_bytes=read_bytes,
    write_bytes=write_bytes,
    intermediate_bytes=intermediate_bytes,
    movement_bytes=movement_bytes,
    total_bytes=read_bytes + write_bytes + intermediate_bytes + movement_bytes,
    confidence=confidence,
    rationale=rationale,
    axis_source=axis_source,
    warnings=tuple(warnings),
)
```

### Pattern 2: Add Parseable Evidence, Not String-Only Evidence

**What:** Add internal evidence records such as `SolarFormulaEvidence`, `SolarByteEvidence`, and optionally `SolarOpBoundEvidence` to `SolarSemanticGroupEvidence` or `SolarDerivationEvidence`, with strict parser support. [VERIFIED: Phase 48 parser uses exact keys in `src/sol_execbench/core/scoring/solar_derivation.py`]

**When to use:** Use whenever a high-confidence family emits formula/byte/bound values required by MODEL-01, MODEL-02, and MODEL-05. [VERIFIED: `.planning/REQUIREMENTS.md`]

**Recommended fields:**

```text
formula_evidence:
  node_id, family, formula_kind, formula, formula_inputs, source, confidence, rationale
byte_evidence:
  node_id, family, read_bytes, write_bytes, intermediate_bytes, movement_bytes,
  total_bytes, dtype_inputs, tensor_ids, source, confidence, rationale
bound_evidence:
  node_id, family, compute_bound_ms, memory_bound_ms, limiting_resource,
  sol_bound_ms, source, confidence, rationale
```

These records should stay internal to the SOLAR derivation sidecar and must not be added to `Definition`, `Workload`, `Trace`, primary CLI output, or score eligibility. [VERIFIED: `docs/internal/solar_derivation_contract.md`, `tests/sol_execbench/test_public_contract_guardrails.py`]

### Pattern 3: Family-Specific Supported Criteria

**What:** Keep `supported` narrow and deterministic; degrade when visible family semantics are incomplete. [VERIFIED: `classify_solar_confidence()` in `src/sol_execbench/core/scoring/solar_derivation.py`]

**When to use:** Use for every high-confidence family rather than returning unsupported for partial but recognizable structures. [VERIFIED: `.planning/phases/49-high-confidence-family-modeling/49-CONTEXT.md`]

**Recommended family criteria:**

| Family | Supported Only When | Degrade When |
|--------|---------------------|--------------|
| `attention` | Q/K/V projections or explicit Q/K/V inputs, QK matmul, scale/mask status, softmax axis, PV matmul, output projection, dtype and axes are visible. [VERIFIED: fixture contract] | Mask semantics, softmax axis, q/k sequence axes, head dim, or projection metadata are missing. [VERIFIED: `attention_degraded_partial_mask.json`] |
| `convolution` | Conv dimension, input/weight/output shapes, stride, padding, dilation, groups, and output spatial dimensions are static. [VERIFIED: fixture contract] | Padding/layout/groups or output spatial evidence is incomplete. [VERIFIED: `convolution_degraded_missing_padding.json`] |
| `embedding_positional` | Index tensor, table shape, output shape, index dtype, and positional/gather/rotary subrole are explicit. [VERIFIED: fixture contract] | Dynamic indices or partial table/output metadata are present. [VERIFIED: `embedding_positional_degraded_dynamic_indices.json`] |
| `linear_projection` | Input, weight, output shapes and dtype are explicit; bias if present is visible; GEMM dimensions infer successfully. [VERIFIED: existing tests] | Shape/rank/bias metadata is incomplete but projection family is visible. [VERIFIED: `linear_projection_degraded_missing_shape.json`] |

### Anti-Patterns to Avoid

- **Duplicating AMD SOL v2 bound math:** Reuse or factor `_bound_for_estimate()` semantics; duplicated compute/memory math can drift. [VERIFIED: `src/sol_execbench/core/scoring/amd_sol_v2.py`]
- **Treating formula text as provenance:** Formula text is useful but insufficient; store formula inputs and tensor/dtype/source evidence as separate parseable data. [VERIFIED: MODEL-01/MODEL-02 in `.planning/REQUIREMENTS.md`]
- **Marking dynamic partial evidence as supported:** Missing axes, mask semantics, index semantics, grouping metadata, or dtype must degrade or unscore. [VERIFIED: `.planning/phases/49-high-confidence-family-modeling/49-CONTEXT.md`]
- **Executing candidate solutions for derivation:** Phase 48 builder signatures accept only `Definition` and `Workload`; this boundary must hold. [VERIFIED: `tests/sol_execbench/test_solar_derivation_evidence.py`]
- **Changing public CLI or canonical schemas:** Existing guardrails assert v1.10 derivation fields stay noncanonical and primary CLI has no SOLAR derivation options. [VERIFIED: `tests/sol_execbench/test_public_contract_guardrails.py`]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Graph extraction | A second graph IR | `BoundGraph`, `BoundGraphNode`, `BoundTensor`, FX/AST extractor | Existing code already tracks family, inputs, outputs, tensor metadata, attributes, confidence, and source. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py`] |
| Formula/byte carrier | New numeric estimate model | `OperatorWorkEstimate` | It already has formula kind, formula text, inputs, FLOPs, read/write/intermediate/movement/total bytes, confidence, and warnings. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py`] |
| Bound conversion | Fresh compute/memory/SOL math | `amd_sol_v2._bound_for_estimate()` semantics or a shared local helper | Existing v2 artifacts already test rich estimate to bound conversion. [VERIFIED: `src/sol_execbench/core/scoring/amd_sol_v2.py`, `tests/sol_execbench/test_amd_sol_v2.py`] |
| Parser validation | Loose dict passthrough | Phase 48 exact-key parser style | Existing sidecar parsers reject unknown/missing fields and invalid enums. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py`] |
| Dtype byte widths | New dtype table | `_dtype_bytes()` in `amd_bound_estimates.py` | Existing table covers public dtype contract, including fp8/fp4 and integer/bool widths. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `tests/sol_execbench/test_amd_bound_estimates.py`] |

**Key insight:** The hard part is not numeric storage; it is preserving semantic confidence. Use existing estimates for values, and add family-specific completeness gates so partial patterns become degraded instead of silently scored. [VERIFIED: `.planning/phases/49-high-confidence-family-modeling/49-CONTEXT.md`, `src/sol_execbench/core/scoring/solar_derivation.py`]

## Common Pitfalls

### Pitfall 1: Formula Evidence Hidden In `required_evidence`

**What goes wrong:** Tests can only assert strings like `formula:op_1` instead of checking `formula_kind`, `formula`, and `formula_inputs`. [VERIFIED: `solar_derivation.py` currently only emits required evidence strings for formula/bytes]
**Why it happens:** Phase 48 intentionally stopped at semantic provenance. [VERIFIED: Phase 48 verification]
**How to avoid:** Add parseable formula and byte evidence records and parser tests. [VERIFIED: MODEL-01/MODEL-02]
**Warning signs:** New tests assert only `required_evidence` strings or `group.source.detail`. [VERIFIED: codebase grep]

### Pitfall 2: Attention Gets Scored Without Mask Or Axis Semantics

**What goes wrong:** A QK/PV matmul chain could be marked `supported` while mask semantics, scale, softmax dimension, sequence axes, or head dimension are missing. [VERIFIED: DERIVE-01 and context]
**Why it happens:** Existing matmul/GEMM estimates can be `SUPPORTED` based on dimensions alone. [VERIFIED: `_gemm_estimate()`]
**How to avoid:** Attention group classification must require family-specific subroles and metadata beyond per-node GEMM support. [VERIFIED: fixture contract]
**Warning signs:** `attention` groups have `status="scored"` with no softmax axis or mask/scale evidence. [VERIFIED: `attention_degraded_partial_mask.json`]

### Pitfall 3: Convolution Shape Math Ignores Groups Or Dilation

**What goes wrong:** FLOPs and bytes look plausible but are wrong for grouped, depthwise, padded, strided, or dilated convolution. [ASSUMED]
**Why it happens:** Generic tensor byte counting does not prove convolution channel/group formulas. [VERIFIED: current code lacks convolution estimates]
**How to avoid:** Require explicit `groups`, stride, padding, dilation, kernel shape, input/output spatial dimensions, and channel divisibility checks before supported evidence. [VERIFIED: convolution fixture contract]
**Warning signs:** `conv2d` support passes with missing padding or dynamic kernel metadata. [VERIFIED: degradation/unsupported fixtures]

### Pitfall 4: Embedding And Gather Reads Are Treated Like Dense Table Reads

**What goes wrong:** Memory bytes become `table_numel*dtype_bytes` instead of indexed rows/elements read, overstating movement by orders of magnitude. [ASSUMED]
**Why it happens:** `_sum_tensor_bytes()` sums full input tensors; embedding/gather needs family-specific read semantics. [VERIFIED: `amd_bound_estimates.py`]
**How to avoid:** For lookup/gather, compute index bytes plus selected output/table element bytes, not full table bytes, when index/output shapes are explicit. [ASSUMED]
**Warning signs:** `read_bytes` for a static embedding equals full vocabulary table bytes for every workload. [ASSUMED]

### Pitfall 5: Sidecar Schema Drift Breaks Old Parser Tests

**What goes wrong:** Adding fields to `SolarSemanticGroupEvidence` without parser updates breaks round-trip or exact-key rejection behavior. [VERIFIED: existing parser exact-key tests]
**Why it happens:** Phase 48 parser requires exact top-level and nested keys. [VERIFIED: `solar_derivation_from_dict()`]
**How to avoid:** Add new fields deliberately with parser support and update all contract payload helpers. [VERIFIED: `tests/sol_execbench/test_solar_derivation_evidence.py`]
**Warning signs:** Existing `test_solar_derivation_parser_rejects_unknown_schema_fields` or round-trip tests fail. [VERIFIED: test file]

## Code Examples

### Existing Estimate Carrier To Reuse

```python
# Source: src/sol_execbench/core/scoring/amd_bound_estimates.py
@dataclass(frozen=True)
class OperatorWorkEstimate:
    node_id: str
    op_family: OpFamily
    op_name: str
    formula_kind: str
    formula: str
    formula_inputs: dict[str, object]
    flops: float
    read_bytes: float
    write_bytes: float
    intermediate_bytes: float
    movement_bytes: float
    total_bytes: float
    confidence: EstimateConfidence
    rationale: str
```

### Existing Bound Conversion Semantics To Reuse

```python
# Source: src/sol_execbench/core/scoring/amd_sol_v2.py
compute_bound_ms = (
    estimate.flops / (hardware_model.peak_tflops * 1_000_000_000_000.0) * 1000.0
    if hardware_model.peak_tflops > 0.0
    else 0.0
)
memory_bound_ms = (
    estimate.total_bytes / (hardware_model.memory_bandwidth_gbps * 1_000_000_000.0) * 1000.0
    if hardware_model.memory_bandwidth_gbps > 0.0
    else 0.0
)
```

### Existing Confidence Gate To Extend

```python
# Source: src/sol_execbench/core/scoring/solar_derivation.py
if estimate.confidence == EstimateConfidence.UNSUPPORTED:
    missing.append(f"estimate:{estimate.node_id}")
if not estimate.formula_inputs:
    missing.append(f"formula_inputs:{estimate.node_id}")
if estimate.total_bytes <= 0.0:
    missing.append(f"bytes:{estimate.node_id}")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Unsupported placeholders for attention/convolution/embedding families | Phase 49 should emit high-confidence family estimates for complete visible metadata | Phase 49 planned | Planner should replace `test_out_of_scope_families_are_explicit_unsupported_estimates` expectations for Phase 49 families only. [VERIFIED: `tests/sol_execbench/test_amd_bound_estimates.py`] |
| Formula and byte data only in AMD SOL v2 estimates | Phase 49 should surface parseable formula/byte evidence in SOLAR derivation sidecars | Phase 49 planned | Enables MODEL-01/MODEL-02 without canonical schema changes. [VERIFIED: requirements and code] |
| Generic GEMM support for linear projection | First-class `linear_projection` semantic family with GEMM-compatible formulas | Partially present before Phase 49 | Preserve `op_family="linear_projection"` while using `gemm_flops` or family-specific `linear_projection_flops` deterministically. [VERIFIED: `amd_bound_graph.py`, `amd_bound_estimates.py`] |
| Score eligibility handled by AMD SOL v1/v2 artifacts | Phase 49 internal evidence remains sidecar-only; eligibility changes are Phase 51 | Phase 49 boundary | Do not wire SOLAR evidence into AMD-native scoring yet. [VERIFIED: `49-CONTEXT.md`] |

**Deprecated/outdated:**
- Treating Phase 49 families as always unsupported is outdated for attention, convolution, embedding/positional/gather/rotary-like, and linear projection once metadata is complete. [VERIFIED: Phase 49 scope]
- Adding public CLI switches for SOLAR derivation remains out of scope and should stay forbidden. [VERIFIED: `test_primary_cli_does_not_expose_v1_10_solar_derivation_options`]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Convolution group/dilation mistakes are a common source of wrong FLOP estimates. | Common Pitfalls | Planner may under-prioritize group/depthwise/dilation tests. |
| A2 | Embedding/gather memory should count selected/indexed rows rather than the full table. | Common Pitfalls | Byte evidence could overstate memory traffic and degrade SOL bound quality. |

## Open Questions

1. **Should linear projection formula kind remain `gemm_flops` or become `linear_projection_flops`?**
   - What we know: Existing `_gemm_estimate()` returns `gemm_flops` for both `GEMM` and `LINEAR_PROJECTION`. [VERIFIED: `amd_bound_estimates.py`]
   - What's unclear: MODEL-01 asks for family-specific formula kind; using `gemm_flops` preserves reuse but may be less family-specific. [VERIFIED: `.planning/REQUIREMENTS.md`]
   - Recommendation: Use `linear_projection_flops` only in sidecar formula evidence while keeping numeric inference shared with GEMM, or include `formula_family="linear_projection"` with `formula_kind="gemm_flops"`. [ASSUMED]

2. **Where should per-op bound evidence live inside `SolarDerivationEvidence`?**
   - What we know: `amd_sol_v2` op bounds are sidecar dataclasses, while Phase 48 groups are semantic sidecar records. [VERIFIED: `amd_sol_v2.py`, `solar_derivation.py`]
   - What's unclear: Whether planner prefers group-level bound evidence or top-level per-node bound evidence.
   - Recommendation: Prefer top-level tuple keyed by `node_id` to avoid duplicating bound records across family groups. [ASSUMED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python / uv | Tests and local package execution | Not probed in research; existing project commands use `uv run` | n/a | Planner can use existing repo commands. [VERIFIED: `AGENTS.md`] |
| PyTorch | Optional FX trace path | Existing project dependency, but not required for AST fallback | n/a | `build_bound_graph()` returns `None` from FX path and falls back to AST if import/trace fails. [VERIFIED: `amd_bound_graph.py`] |
| ROCm hardware | None for Phase 49 implementation tests | Not required | n/a | Use unit and sidecar tests only. [VERIFIED: `49-CONTEXT.md`] |
| New package dependencies | None | n/a | n/a | No new dependencies allowed. [VERIFIED: `.planning/REQUIREMENTS.md`] |

**Missing dependencies with no fallback:** none identified for the implementation plan. [VERIFIED: Phase 49 scope]

**Missing dependencies with fallback:** PyTorch FX can fall back to AST extraction in `build_bound_graph()`. [VERIFIED: `amd_bound_graph.py`]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest [VERIFIED: `AGENTS.md`] |
| Config file | existing repo pytest configuration, not specifically inspected in this research [ASSUMED] |
| Quick run command | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py -n 0` |
| Full suite command | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DERIVE-01 | Explicit attention recognizes Q/K/V, QK, scale or mask status, softmax axis, PV, output projection; partial mask/axis degrades. | unit/integration | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py -k attention -n 0` | ❌ Wave 0 |
| DERIVE-03 | Conv1d/2d/3d graph nodes capture stride, padding, dilation, groups, output spatial dims; missing padding degrades; dynamic kernel unscored. | unit | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py -k conv -n 0` | ⚠️ extend existing |
| DERIVE-05 | Embedding/gather/positional/rotary-like evidence records index/table/output shape and memory-bound bytes; dynamic indices degrade. | unit/integration | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py -k "embedding or gather or rotary" -n 0` | ❌ Wave 0 |
| DERIVE-06 | Linear projection remains `op_family=linear_projection`, infers GEMM dimensions, emits formula/byte evidence, and degrades missing shape. | unit | `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py -k linear -n 0` | ⚠️ partial |
| MODEL-01 | Sidecar parser round-trips formula kind/text/input map for every promoted family and rejects malformed formula payloads. | schema/unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k formula -n 0` | ❌ Wave 0 |
| MODEL-02 | Sidecar parser round-trips read/write/intermediate/movement/total bytes and dtype map for every promoted family. | schema/unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k bytes -n 0` | ❌ Wave 0 |
| MODEL-05 | Per-op bound evidence matches AMD SOL v2 compute/memory/limiting/SOL math and stays sidecar-only. | unit/regression | `uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_solar_derivation_family_modeling.py -k bound -n 0` | ❌ Wave 0 |

### Exact Tests Needed

- Add `test_solar_derivation_formula_and_byte_evidence_round_trip_preserves_all_buckets`: construct a sidecar with formula and byte evidence for one high-confidence group; assert `solar_derivation_from_dict(evidence.to_dict()).to_dict()` preserves `formula_kind`, `formula`, `formula_inputs`, `read_bytes`, `write_bytes`, `intermediate_bytes`, `movement_bytes`, and `total_bytes`. [VERIFIED: parser pattern]
- Add malformed parser tests for missing formula fields, unknown formula fields, negative byte values if disallowed, non-numeric bytes, and invalid bound `limiting_resource`. [VERIFIED: parser pattern]
- Add `test_linear_projection_reuses_gemm_dims_but_preserves_family`: `torch.nn.functional.linear(x, w, b)` or `x @ w` classified as `linear_projection` where appropriate, formula inputs contain `M/N/K`, bytes are dtype-aware, and group subroles include input/weight/bias/output. [VERIFIED: fixture and existing linear subroles]
- Update `test_out_of_scope_families_are_explicit_unsupported_estimates`: remove Phase 49 families from the unsupported loop while keeping MoE and SSM/Mamba unsupported/deferred. [VERIFIED: current test]
- Add attention positive/degraded/unsupported tests matching Phase 47 fixtures: positive dense QKV scored; partial mask degraded with `mask:semantics`; dynamic axes unscored. [VERIFIED: fixtures]
- Add convolution tests for `conv1d`, `conv2d`, `conv3d`, grouped conv, and depthwise conv; assert formula inputs include batch, channels, output spatial dims, kernel dims, groups, stride, padding, dilation, and dtype byte buckets. [VERIFIED: requirement and fixture]
- Add embedding/gather/rotary tests: static embedding lookup scored with index/table/output evidence; dynamic indices degraded; missing metadata unscored; rotary-like transform records positional subrole and movement/intermediate bytes if materialized. [VERIFIED: fixtures]
- Add public contract guardrail extension: assert new sidecar-only field names do not appear in `Definition`, `Workload`, `Trace`, primary CLI help, or AMD-native score evidence refs. [VERIFIED: existing guardrail style]
- Add deterministic ordering tests: family formula/byte/bound evidence should serialize in stable node/group order even if estimates are provided reversed. [VERIFIED: existing deterministic group test]

### Sampling Rate

- **Per task commit:** `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py -n 0`
- **Per wave merge:** `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0`
- **Phase gate:** Full suite green before `$gsd-verify-work`, plus Ruff on touched scoring and test files. [VERIFIED: AGENTS.md test/lint commands]

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_solar_derivation_family_modeling.py` - focused high-confidence family modeling coverage for attention, convolution, embedding/positional/gather/rotary-like, and linear projection.
- [ ] `tests/sol_execbench/test_solar_derivation_evidence.py` parser extensions - formula/byte/bound evidence round-trip and malformed payload rejection.
- [ ] `tests/sol_execbench/test_public_contract_guardrails.py` forbidden-field list update for new internal sidecar field names.
- [ ] Existing `tests/sol_execbench/test_amd_bound_estimates.py::test_out_of_scope_families_are_explicit_unsupported_estimates` must be updated so only Phase 50 families remain unsupported.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No auth/session surface in Phase 49. [VERIFIED: phase scope] |
| V3 Session Management | no | No session state. [VERIFIED: phase scope] |
| V4 Access Control | no | Local internal scoring evidence only. [VERIFIED: phase scope] |
| V5 Input Validation | yes | Strict parser validation for internal sidecar payloads; exact-key rejection and typed fields. [VERIFIED: `solar_derivation.py`] |
| V6 Cryptography | no | No cryptographic operations. [VERIFIED: phase scope] |

### Known Threat Patterns for Internal Sidecar Evidence

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Candidate solution execution during derivation | Elevation of privilege / Tampering | Keep builder signatures limited to `Definition` and `Workload`; tests must reject candidate inputs. [VERIFIED: `test_builder_does_not_accept_or_execute_candidate_solution_code`] |
| Public schema contamination with internal evidence | Information disclosure / Tampering | Guardrail tests over `Definition`, `Workload`, `Trace`, primary CLI help, and score artifacts. [VERIFIED: `test_public_contract_guardrails.py`] |
| Malformed sidecar payload accepted | Tampering | Exact-key parser and enum/type/value validation for every new formula/byte/bound field. [VERIFIED: Phase 48 parser pattern] |
| Unsupported/degraded evidence scored as supported | Tampering | Deterministic confidence gates and warning prefixes for missing metadata. [VERIFIED: `classify_solar_confidence()`] |

## Sources

### Primary (HIGH confidence)

- `.planning/ROADMAP.md` - Phase 49 scope, dependencies, and success criteria.
- `.planning/REQUIREMENTS.md` - DERIVE-01, DERIVE-03, DERIVE-05, DERIVE-06, MODEL-01, MODEL-02, MODEL-05, and out-of-scope constraints.
- `.planning/phases/49-high-confidence-family-modeling/49-CONTEXT.md` - locked implementation decisions, integration boundary, and deferred ideas.
- `.planning/phases/48-extraction-pipeline-and-semantic-provenance/48-VERIFICATION.md` - verified Phase 48 evidence model and public boundary behavior.
- `docs/internal/solar_derivation_contract.md` - sidecar-only rule, family/state vocabulary, fixture matrix, and claim boundaries.
- `src/sol_execbench/core/scoring/solar_derivation.py` - Phase 48 internal evidence model, parser, grouping, confidence, and source boundaries.
- `src/sol_execbench/core/scoring/amd_bound_graph.py` - `BoundGraph`, `BoundGraphNode`, `BoundTensor`, `OpFamily`, FX/AST extraction, and classifiers.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - `OperatorWorkEstimate`, dtype bytes, GEMM/pointwise/reduction/softmax/movement estimates.
- `src/sol_execbench/core/scoring/amd_sol_v2.py` - rich estimate to per-op bound conversion, aggregate semantics, coverage, parser.
- `tests/sol_execbench/test_solar_derivation_evidence.py` - existing parser, confidence, deterministic ordering, candidate boundary tests.
- `tests/sol_execbench/test_amd_bound_estimates.py` - existing formula/byte tests and unsupported family expectations.
- `tests/sol_execbench/test_amd_bound_graph.py` - graph taxonomy, extraction, unsupported and data-movement tests.
- `tests/sol_execbench/test_public_contract_guardrails.py` - canonical schema, CLI, and score eligibility guardrails.
- `tests/sol_execbench/fixtures/solar_derivation/*.json` - Phase 47 family fixture expectations.

### Secondary (MEDIUM confidence)

- None. Research did not require external web or community sources because the implementation target is a local internal scoring system with locked project decisions. [VERIFIED: source usage]

### Tertiary (LOW confidence)

- General modeling cautions for convolution grouping/dilation and embedding/gather selected-row memory are tagged `[ASSUMED]` where used.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Locked by project constraints and existing local implementation.
- Architecture: HIGH - Phase 48 and AMD SOL v2 data flow is verified in source and tests.
- Pitfalls: MEDIUM - Parser/public-boundary risks are verified; convolution and embedding byte-modeling cautions include assumptions that planner should test explicitly.

**Research date:** 2026-05-23
**Valid until:** 2026-06-22 for local architecture; re-check sooner if Phase 48/49 scoring modules change.
