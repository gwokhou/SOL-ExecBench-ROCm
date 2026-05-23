# Phase 50: Degraded Complex Family Modeling - Pattern Map

**Mapped:** 2026-05-23
**Files analyzed:** 8 implementation/test targets plus 6 MoE/SSM fixtures
**Analogs found:** 8 / 8

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | utility / graph extractor | transform | existing attention and embedding family annotation in same file | exact |
| `src/sol_execbench/core/scoring/amd_bound_estimates.py` | utility / estimator | transform | existing convolution and embedding degraded estimates in same file | exact |
| `src/sol_execbench/core/scoring/solar_derivation.py` | service / sidecar derivation | transform | existing semantic-group confidence, subrole, strict parser patterns in same file | exact |
| `tests/sol_execbench/test_solar_derivation_family_modeling.py` | test | request-response style unit construction | attention/convolution/embedding family tests in same file | exact |
| `tests/sol_execbench/test_amd_bound_graph.py` | test | transform | convolution and embedding graph metadata tests in same file | exact |
| `tests/sol_execbench/test_amd_bound_estimates.py` | test | transform | out-of-scope family, incomplete convolution, embedding byte tests in same file | exact |
| `tests/sol_execbench/fixtures/solar_derivation/moe_*.json` | test fixture / contract | file-I/O | existing Phase 50 MoE fixture contract files | exact |
| `tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_*.json` | test fixture / contract | file-I/O | existing Phase 50 SSM fixture contract files | exact |

## Pattern Assignments

### `src/sol_execbench/core/scoring/amd_bound_graph.py` (utility, transform)

**Analog:** existing attention graph annotation and embedding lookup annotation in `src/sol_execbench/core/scoring/amd_bound_graph.py`.

**Imports / enum pattern** (lines 8-17, 27-44):
```python
import ast
import operator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from sol_execbench.core.data.definition import DType, Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence

class OpFamily(str, Enum):
    ATTENTION = "attention"
    MOE = "moe"
    SSM_MAMBA = "ssm_mamba"
    UNSUPPORTED = "unsupported"
```

**Classifier extension pattern** (lines 155-244, 1237-1242):
```python
_CALL_CLASSIFIERS: tuple[tuple[set[str], _CallClassification], ...] = (
    (
        {"linear"},
        _CallClassification(
            OpFamily.LINEAR_PROJECTION,
            EstimateConfidence.SUPPORTED,
            "recognized linear projection",
        ),
    ),
    # add deterministic MoE/SSM visible primitive names here, not opaque taxonomy labels
)

def _classify_call(func_name: str) -> _CallClassification | None:
    leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
    for names, classification in _CALL_CLASSIFIERS:
        if leaf_name in names or func_name in names:
            return classification
    return None
```

**Family promotion pattern** (lines 605-607, 638-769):
```python
def _annotate_family_graph(graph: BoundGraph) -> BoundGraph:
    return _annotate_memory_bound_graph(_annotate_attention_graph(graph))

def _annotate_attention_graph(graph: BoundGraph) -> BoundGraph:
    nodes = list(graph.nodes)
    warnings = list(graph.warnings)
    for qk_index, qk_node in enumerate(nodes):
        qk_dims = _attention_qk_dims(graph, qk_node)
        if qk_dims is None:
            continue
        nodes[qk_index] = replace(
            qk_node,
            op_family=OpFamily.ATTENTION,
            attributes={**qk_node.attributes, **qk_dims, "subrole": "qk_scores"},
            confidence=EstimateConfidence.SUPPORTED,
            rationale="recognized attention QK score matmul",
        )
    return replace(graph, nodes=tuple(nodes), warnings=tuple(dict.fromkeys(warnings)))
```

**Use for Phase 50:** add `_annotate_moe_graph` and `_annotate_ssm_mamba_graph`, then thread them through `_annotate_family_graph`. Prefer promotion of visible primitive chains over adding public schema fields. Recommended helpers:

- `_annotate_moe_graph(graph: BoundGraph) -> BoundGraph`
- `_moe_route_evidence(graph, nodes) -> BoundGraphNode | None`
- `_moe_missing_routing_evidence(node) -> tuple[str, ...]`
- `_annotate_ssm_mamba_graph(graph: BoundGraph) -> BoundGraph`
- `_ssm_state_evidence(graph, nodes) -> BoundGraphNode | None`
- `_ssm_missing_recurrence_evidence(node) -> tuple[str, ...]`

**Missing / unsupported evidence pattern** (lines 966-995, 1053-1065, 1121-1133):
```python
def _dynamic_attention_evidence(graph: BoundGraph, nodes: list[BoundGraphNode]) -> BoundGraphNode | None:
    has_dynamic = any(node.op_family == OpFamily.UNSUPPORTED for node in nodes)
    names = {tensor.name for tensor in graph.tensors.values()}
    has_qkv = {"q", "k", "v"} <= names
    if not has_dynamic or not has_qkv:
        return None
    return BoundGraphNode(
        op_family=OpFamily.ATTENTION,
        op_name="dynamic_attention_axes",
        attributes={"subrole": "dynamic_attention_axes", "axis_source": "missing", "dynamic_axes": True},
        confidence=EstimateConfidence.UNSUPPORTED,
        rationale="unsupported dynamic attention axes prevent static sequence modeling",
        ...
    )

self.warnings.append(f"unsupported_operator:{func_name or '<unknown>'}")
```

**Apply to Phase 50:** dynamic MoE routing should become explicit graph evidence with `confidence=INEXACT` when route structure is visible but cardinality/top-k is not static. Taxonomy-only MoE and opaque custom scan should become `confidence=UNSUPPORTED` with no fabricated subroles.

### `src/sol_execbench/core/scoring/amd_bound_estimates.py` (utility, transform)

**Analog:** convolution and embedding estimates.

**Dispatch pattern** (lines 66-94):
```python
def estimate_bound_work(graph: BoundGraph) -> tuple[OperatorWorkEstimate, ...]:
    return tuple(_estimate_node(graph, node) for node in graph.nodes)

def _estimate_node(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    if node.op_family == OpFamily.ATTENTION:
        return _attention_estimate(graph, node)
    if node.op_family == OpFamily.CONVOLUTION:
        return _convolution_estimate(graph, node)
    return _unsupported_estimate(node)
```

**Use for Phase 50:** add MoE/SSM dispatch above fallback:
```python
if node.op_family == OpFamily.MOE:
    return _moe_estimate(graph, node)
if node.op_family == OpFamily.SSM_MAMBA:
    return _ssm_mamba_estimate(graph, node)
```

**Degraded estimate pattern** (lines 367-425):
```python
dims, missing = _convolution_dims(input_tensors, output_tensors, node)
warnings.extend(f"inexact_operator:convolution_missing_{item}" for item in missing)
if dims is None:
    return OperatorWorkEstimate(
        formula_kind="convolution_flops",
        formula="2*N*C_out*output_spatial_elements*(C_in/groups)*kernel_elements",
        formula_inputs={},
        flops=0.0,
        confidence=EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "convolution semantics recognized but static metadata is incomplete",
            rationale_parts,
        ),
        warnings=tuple(dict.fromkeys(warnings)),
        ...
    )
```

**Use for Phase 50:** if MoE routing or SSM recurrence is visible but incomplete, return `INEXACT`, keep formula names stable, leave `formula_inputs={}` or include only directly observed inputs, set flops to `0.0` when a numeric formula would require guessing.

**Byte/missing-evidence pattern** (lines 438-527, 900-950):
```python
missing: list[str] = []
if index_tensor is None:
    missing.append("index_tensor")
if selected_elements is None:
    missing.append("selected_elements")
warnings.extend(f"inexact_operator:embedding_positional_missing_{item}" for item in missing)

def _sum_tensor_bytes(...):
    if tensor_bytes is None:
        warnings.append(f"inexact_bytes:missing_shape:{tensor.tensor_id}")
        warnings.append(f"inexact_bytes:missing_dtype:{tensor.tensor_id}")
```

**Unsupported fallback pattern** (lines 828-860):
```python
warning_kind = (
    "unsupported_operator"
    if node.op_family == OpFamily.UNSUPPORTED
    else "unsupported_family"
)
warning = f"{warning_kind}:{node.op_name or node.op_family.value}"
return OperatorWorkEstimate(
    formula_kind="unsupported",
    formula="0",
    formula_inputs={},
    confidence=EstimateConfidence.UNSUPPORTED,
    warnings=estimate_warnings,
    ...
)
```

**Important existing contract to update:** `tests/sol_execbench/test_amd_bound_estimates.py` currently asserts MoE and SSM/Mamba are out-of-scope unsupported (lines 635-648). Phase 50 should replace or narrow that test so unsupported remains only for opaque / taxonomy-only cases.

### `src/sol_execbench/core/scoring/solar_derivation.py` (service, transform)

**Analog:** internal sidecar dataclasses, semantic grouping, strict parser, confidence classification.

**Sidecar-only dataclass pattern** (lines 201-238):
```python
@dataclass(frozen=True)
class SolarSemanticGroupEvidence:
    family: str
    group_id: str
    node_ids: tuple[str, ...]
    subroles: tuple[SolarSubroleEvidence, ...]
    confidence: EstimateConfidence | str
    status: str
    required_evidence: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    warning_prefixes: tuple[str, ...]
    source: SolarEvidenceSource
    rationale: str
    formula_evidence: tuple[SolarFormulaEvidence, ...] = ()
    byte_evidence: tuple[SolarByteEvidence, ...] = ()
    bound_evidence: tuple[SolarBoundEvidence, ...] = ()
```

**Do not add new fields** for MoE/SSM. Use `subroles`, `required_evidence`, `missing_evidence`, and `warning_prefixes`.

**Group construction pattern** (lines 809-881):
```python
subroles = _subroles_for_group(family, nodes, tensor_evidence_by_id)
classification = classify_solar_confidence(
    family=family,
    nodes=nodes,
    tensors=related_tensors,
    estimates=ordered_estimates,
    subrole_names=tuple(subrole.name for subrole in subroles),
)
formula_evidence = _formula_evidence_for_estimates(ordered_estimates)
byte_evidence = _byte_evidence_for_estimates(...)
bound_evidence = _bound_evidence_for_estimates(ordered_estimates)
```

**Use for Phase 50:** extend `_subroles_for_group` with `OpFamily.MOE.value` and `OpFamily.SSM_MAMBA.value`, then add `_moe_subroles`, `_ssm_mamba_subroles`, `_moe_confidence_evidence`, and `_ssm_mamba_confidence_evidence`.

**Family-specific confidence pattern** (lines 318-427, 1151-1203, 1206-1266):
```python
if family_value == OpFamily.ATTENTION.value:
    attention_missing, attention_warnings = _attention_confidence_evidence(...)
    missing.extend(attention_missing)
    warning_prefixes.extend(attention_warnings)

if missing:
    confidence = _worse_confidence(confidence, EstimateConfidence.INEXACT)

if confidence == EstimateConfidence.INEXACT:
    warning_prefixes.append(f"inexact_operator:{family_value}")
    warning_prefixes.append("aggregate_degraded:incomplete semantic evidence")
elif confidence == EstimateConfidence.UNSUPPORTED:
    warning_prefixes.append(f"unsupported_operator:{family_value}")
    warning_prefixes.append("aggregate_unscored:unsupported semantic evidence")
```

**Recommended Phase 50 missing-evidence names:**

- MoE: `subrole:router`, `subrole:top_k`, `subrole:dispatch`, `subrole:expert_projection`, `subrole:combine`, `route:top_k`, `route:static_cardinality`, `shape:experts`, `shape:tokens`.
- SSM/Mamba: `subrole:input_projection`, `subrole:depthwise_convolution`, `subrole:scan`, `subrole:state_update`, `subrole:gating`, `subrole:output_projection`, `subrole:recognized_scan`, `shape:sequence`, `shape:hidden`, `shape:state`, `recurrence:update_formula`.

**Strict parse / public contract pattern** (lines 430-488, 491-555, 560-765, 1602-1781):
```python
_require_exact_keys(payload, {"schema_version", "derived", "definition", ...}, source="SOLAR derivation evidence")
if schema_version != SOLAR_DERIVATION_SCHEMA_VERSION:
    raise ValueError(...)

def _default_source_boundary() -> dict[str, bool]:
    return {
        "canonical_trace_jsonl": False,
        "public_schema": False,
        "candidate_solution_execution": False,
    }
```

**Guardrail:** keep `SOLAR_DERIVATION_SCHEMA_VERSION` unchanged unless a schema migration is explicitly required. Phase 50 can change sidecar contents, not public `Definition`, `Workload`, `Trace`, canonical trace JSONL, or CLI behavior.

### `tests/sol_execbench/test_solar_derivation_family_modeling.py` (test, transform)

**Analog:** attention positive/degraded/unsupported tests and convolution/embedding tests.

**Positive sidecar pattern** (lines 73-114, 202-254, 297-348):
```python
evidence = build_solar_derivation_evidence(definition, workload)
payload = solar_derivation_from_dict(evidence.to_dict()).to_dict()
group = next(group for group in payload["groups"] if group["family"] == "attention")

assert group["status"] == "scored"
assert group["confidence"] == "supported"
assert group["missing_evidence"] == []
assert len(group["byte_evidence"]) == len(group["formula_evidence"])
assert len(group["bound_evidence"]) == len(group["formula_evidence"])
```

**Degraded pattern** (lines 117-132, 257-294, 351-381):
```python
group = next(group for group in evidence.groups if group.family == "convolution")

assert group.status == "degraded"
assert group.confidence.value == "inexact"
assert "convolution:padding" in group.missing_evidence
assert any(
    warning.startswith("inexact_operator:convolution_missing_padding")
    for warning in group.warning_prefixes
)
```

**Unsupported pattern** (lines 135-177):
```python
assert group.status == "unscored"
assert group.confidence.value == "unsupported"
assert "axis:static_sequence" in group.missing_evidence
assert not any(subrole.name == "q_projection" for subrole in group.subroles)
```

**Use for Phase 50:** add six tests matching the MoE/SSM fixtures: positive, degraded, unsupported for each family. Assert no fabricated subroles for taxonomy-only MoE and no supported state-update evidence for opaque custom scan.

### `tests/sol_execbench/test_amd_bound_graph.py` (test, transform)

**Analog:** taxonomy, unsupported visibility, dynamic control flow, convolution attributes, embedding lookup metadata.

**Taxonomy already includes target families** (lines 56-75):
```python
assert {
    "attention",
    "moe",
    "ssm_mamba",
    "unsupported",
} <= values
```

**Unsupported visibility pattern** (lines 120-136, 162-182):
```python
graph = build_bound_graph(definition, workload)
assert graph.nodes[0].op_family == OpFamily.UNSUPPORTED
assert graph.nodes[0].confidence == EstimateConfidence.UNSUPPORTED
assert "unsupported_operator:torch.linalg.inv" in graph.warnings
```

**Metadata assertion pattern** (lines 410-420, 458-475):
```python
conv = next(node for node in graph.nodes if node.op_family == OpFamily.CONVOLUTION)
assert conv.attributes["dimensionality"] == 2
assert conv.attributes["padding"] == (1, 1)

lookup_nodes = [node for node in graph.nodes if node.op_family == OpFamily.EMBEDDING_POSITIONAL]
assert {node.attributes.get("memory_subrole") for node in lookup_nodes} >= {
    "embedding_lookup",
    "gather_lookup",
    "positional_add",
}
```

**Use for Phase 50:** assert `OpFamily.MOE` nodes expose stable attributes such as `subrole`, `route_top_k`, `route_cardinality_source`, and `expert_count` only when observed. Assert `OpFamily.SSM_MAMBA` nodes expose `subrole`, `state_shape`, `sequence_axis_source`, and `recurrence_source` only when observed.

### `tests/sol_execbench/test_amd_bound_estimates.py` (test, transform)

**Analog:** unsupported family fallback, incomplete metadata degradation, public schema non-mutation.

**Existing out-of-scope test to replace** (lines 635-648):
```python
for family in (
    OpFamily.MOE,
    OpFamily.SSM_MAMBA,
):
    graph = _single_node_graph(_unsupported_node(family))
    estimate = estimate_bound_work(graph)[0]
    assert estimate.confidence == EstimateConfidence.UNSUPPORTED
    assert estimate.formula_kind == "unsupported"
    assert estimate.warnings == ("unsupported_family:torch.linalg.inv",)
```

**Incomplete metadata pattern** (lines 735-798):
```python
estimate = estimate_bound_work(graph)[0]

assert estimate.formula_kind == "convolution_flops"
assert estimate.formula_inputs == {}
assert estimate.flops == 0.0
assert estimate.axis_source is None
assert estimate.confidence == EstimateConfidence.INEXACT
assert "inexact_operator:convolution_missing_padding" in estimate.warnings
```

**Public schema guardrail pattern** (lines 942-980):
```python
_ = estimate_bound_work(graph)

for payload in (
    definition.model_dump(mode="json"),
    workload.model_dump(mode="json"),
    trace.model_dump(mode="json"),
):
    assert "formula_kind" not in payload
    assert "read_bytes" not in payload
    assert "movement_bytes" not in payload
```

**Use for Phase 50:** add MoE/SSM estimate unit tests for supported visible formulas, degraded missing routing/state metadata, unsupported opaque evidence, and unchanged public schema payloads.

### `tests/sol_execbench/fixtures/solar_derivation/moe_*.json` (fixture, file-I/O)

**Analog:** existing MoE contract fixtures.

**Positive contract:** `moe_positive.json`
```json
"expected_subroles": ["router", "top_k", "dispatch", "expert_projection", "combine"],
"expected_confidence": "supported",
"expected_status": "scored",
"required_evidence": ["shape:tokens", "shape:hidden", "shape:experts", "route:top_k"],
"missing_evidence": [],
"warning_prefixes": []
```

**Degraded contract:** `moe_degraded_dynamic_routing.json`
```json
"expected_confidence": "inexact",
"expected_status": "degraded",
"missing_evidence": ["route:top_k", "route:static_cardinality"],
"warning_prefixes": ["inexact_operator:moe_dynamic_routing", "aggregate_degraded:moe"]
```

**Unsupported contract:** `moe_unsupported_taxonomy_only.json`
```json
"expected_subroles": [],
"expected_confidence": "unsupported",
"expected_status": "unscored",
"missing_evidence": ["subrole:router", "subrole:expert_projection", "subrole:dispatch", "subrole:combine"],
"warning_prefixes": ["unsupported_operator:moe_taxonomy_only", "aggregate_unscored:moe"]
```

### `tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_*.json` (fixture, file-I/O)

**Analog:** existing SSM/Mamba contract fixtures.

**Positive contract:** `ssm_mamba_positive.json`
```json
"expected_subroles": ["input_projection", "depthwise_convolution", "scan", "state_update", "gating", "output_projection"],
"expected_confidence": "supported",
"expected_status": "scored",
"required_evidence": ["shape:sequence", "shape:hidden", "shape:state", "subrole:scan"],
"missing_evidence": []
```

**Degraded contract:** `ssm_mamba_degraded_missing_recurrence.json`
```json
"expected_subroles": ["input_projection", "depthwise_convolution", "scan"],
"expected_confidence": "inexact",
"expected_status": "degraded",
"missing_evidence": ["shape:state", "recurrence:update_formula"],
"warning_prefixes": ["inexact_operator:ssm_missing_recurrence", "aggregate_degraded:ssm_mamba"]
```

**Unsupported contract:** `ssm_mamba_unsupported_custom_scan.json`
```json
"expected_subroles": ["scan"],
"expected_confidence": "unsupported",
"expected_status": "unscored",
"missing_evidence": ["subrole:recognized_scan", "shape:state", "recurrence:update_formula"],
"warning_prefixes": ["unsupported_operator:ssm_custom_scan", "aggregate_unscored:ssm_mamba"]
```

## Shared Patterns

### Naming

**Source:** `amd_bound_graph.py` lines 27-44 and `solar_derivation.py` lines 1069-1092.

Use enum values as serialized family names: `moe`, `ssm_mamba`. Use snake_case subroles and warning suffixes. Do not use display names like `Mamba`, `SSM`, or `MixtureOfExperts` in serialized evidence.

Recommended subroles:

- MoE: `router`, `top_k`, `dispatch`, `expert_projection`, `combine`.
- SSM/Mamba: `input_projection`, `depthwise_convolution`, `scan`, `state_update`, `gating`, `output_projection`.

Recommended formula kinds:

- `moe_router_flops`, `moe_expert_projection_flops`, `moe_dispatch_bytes`, `moe_combine_bytes`.
- `ssm_projection_flops`, `ssm_depthwise_convolution_flops`, `ssm_scan_flops`, `ssm_state_update_flops`, `ssm_gating_flops`.

### Warning Prefixes

**Source:** `amd_bound_estimates.py` lines 150, 231, 373, 473, 795-801, 828-860 and `solar_derivation.py` lines 397-410.

Use these deterministic prefixes:

- `inexact_operator:moe_dynamic_routing`
- `inexact_operator:moe_missing_top_k`
- `inexact_operator:moe_missing_static_cardinality`
- `unsupported_operator:moe_taxonomy_only`
- `aggregate_degraded:moe`
- `aggregate_unscored:moe`
- `inexact_operator:ssm_missing_recurrence`
- `inexact_operator:ssm_missing_state_shape`
- `unsupported_operator:ssm_custom_scan`
- `aggregate_degraded:ssm_mamba`
- `aggregate_unscored:ssm_mamba`

Keep the generic existing aggregate warnings too: `aggregate_degraded:incomplete semantic evidence` and `aggregate_unscored:unsupported semantic evidence`.

### Missing Evidence

**Source:** `solar_derivation.py` lines 1151-1266 and `amd_bound_estimates.py` lines 367-425.

MoE missing evidence should describe routing semantics, not just generic tensor shape gaps:

- `route:top_k`
- `route:static_cardinality`
- `shape:experts`
- `shape:tokens`
- `subrole:router`
- `subrole:expert_projection`
- `subrole:dispatch`
- `subrole:combine`

SSM/Mamba missing evidence should keep state/recurrence separate from convolution/projection:

- `shape:sequence`
- `shape:hidden`
- `shape:state`
- `recurrence:update_formula`
- `subrole:recognized_scan`
- `subrole:state_update`
- `subrole:gating`

### Public Contract Guardrails

**Source:** `solar_derivation.py` lines 32-42, 278-315, 430-488, 1602-1607; `test_amd_bound_estimates.py` lines 942-980.

Preserve these contracts:

- Evidence is internal and sidecar-only.
- `source_boundary` remains false for `canonical_trace_jsonl`, `public_schema`, and `candidate_solution_execution`.
- Do not mutate or extend `Definition`, `Workload`, `Trace`, canonical trace JSONL, primary `sol-execbench` CLI behavior, or AMD-native score eligibility.
- Do not execute submitted candidate solution code.
- Do not add framework dependencies.
- Keep dataclass serialization explicit through `to_dict()`.
- Keep parser strict with `_require_exact_keys`; do not silently accept new public payload fields.

### Provenance and Ordering

**Source:** `solar_derivation.py` lines 917-940, 942-980, 983-1025, 1598-1599.

Copy the existing source/evidence convention:

```python
source=SolarEvidenceSource(
    kind="estimate",
    detail=f"{estimate.formula_kind}:{estimate.formula}",
    node_id=estimate.node_id,
    tensor_id=None,
)
return tuple(sorted(evidence, key=lambda item: item.node_id))
```

Use `tuple(dict.fromkeys(...))` where preserving discovery order matters and `_unique_sorted(...)` where stable sorted public sidecar output is already expected.

## Anti-Patterns to Avoid

- Do not infer expert count, top-k, token-to-expert assignment, static route cardinality, state size, scan order, or recurrence update formula from names alone.
- Do not classify taxonomy-only strings as supported MoE/SSM evidence.
- Do not turn opaque custom scans into scored SSM/Mamba recurrence evidence.
- Do not collapse SSM/Mamba state update evidence into ordinary convolution or linear projection evidence.
- Do not emit formula inputs when inputs are guesses; use empty `formula_inputs` and degraded/unsupported confidence.
- Do not drop unsupported dynamic routing/control-flow evidence silently; preserve visible unsupported nodes and warnings.
- Do not broaden CLI, schema, trace, or scoring eligibility behavior while adding sidecar evidence.
- Do not add non-deterministic warning strings or warnings that depend on dict iteration order.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| N/A | N/A | N/A | All Phase 50 target changes have close analogs in Phase 49 family modeling code and tests. |

## Metadata

**Analog search scope:** `src/sol_execbench/core/scoring/`, `tests/sol_execbench/test_*`, `tests/sol_execbench/fixtures/solar_derivation/`
**Files scanned:** 8 required files plus fixture directory listing
**Pattern extraction date:** 2026-05-23
