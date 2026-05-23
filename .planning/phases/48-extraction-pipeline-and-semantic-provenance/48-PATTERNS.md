# Phase 48: Extraction Pipeline And Semantic Provenance - Pattern Map

**Mapped:** 2026-05-23
**Files analyzed:** 4 likely new/modified files
**Analogs found:** 4 / 4

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/scoring/solar_derivation.py` | model, service, utility | transform, request-response-like builder | `src/sol_execbench/core/scoring/amd_sol_v2.py` + `src/sol_execbench/core/scoring/amd_bound_graph.py` | exact |
| `src/sol_execbench/core/scoring/__init__.py` | config | import/export surface | `src/sol_execbench/core/scoring/__init__.py` | exact |
| `tests/sol_execbench/test_solar_derivation_evidence.py` | test | fixture-driven transform, parser round-trip | `tests/sol_execbench/test_amd_sol_v2.py` + `tests/sol_execbench/test_amd_bound_graph.py` | exact |
| `tests/sol_execbench/test_public_contract_guardrails.py` | test | public contract guardrail | `tests/sol_execbench/test_public_contract_guardrails.py` | exact |

## Pattern Assignments

### `src/sol_execbench/core/scoring/solar_derivation.py` (model/service/utility, transform)

**Analog:** `src/sol_execbench/core/scoring/amd_sol_v2.py`

**Imports pattern** (lines 6-23):
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimates import (
    OperatorWorkEstimate,
    estimate_bound_work,
)
from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph
from sol_execbench.core.scoring.amd_hardware_models import (
    AmdHardwareModel,
    EstimateConfidence,
    HardwareValidationStatus,
    amd_hardware_model_from_dict,
)
```

Apply this style with imports from `Definition`, `Workload`, `BoundGraph`, `BoundGraphNode`, `BoundTensor`, `OpFamily`, `OperatorWorkEstimate`, `build_bound_graph`, `estimate_bound_work`, and `EstimateConfidence`. Do not import CLI, trace, solution, or driver code.

**Frozen sidecar dataclass pattern** (lines 107-140):
```python
@dataclass(frozen=True)
class AmdSolBoundV2Artifact:
    """Stable AMD SOL bound artifact v2 sidecar."""

    definition: str
    workload_uuid: str
    hardware_model_ref: str | None
    hardware_model: AmdHardwareModel
    bound_graph: dict[str, object]
    operator_work_estimates: tuple[dict[str, object], ...]
    op_bounds: tuple[AmdSolV2OpBound, ...]
    aggregate_bound: AmdSolV2AggregateBound
    warnings: tuple[str, ...]
    coverage_summary: AmdSolV2CoverageSummary
    schema_version: str = AMD_SOL_V2_SCHEMA_VERSION
    derived: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "hardware_model_ref": self.hardware_model_ref,
            "hardware_model": self.hardware_model.to_dict(),
            "bound_graph": dict(self.bound_graph),
            "operator_work_estimates": [
                dict(estimate) for estimate in self.operator_work_estimates
            ],
            "op_bounds": [bound.to_dict() for bound in self.op_bounds],
            "aggregate_bound": self.aggregate_bound.to_dict(),
            "warnings": list(self.warnings),
            "coverage_summary": self.coverage_summary.to_dict(),
        }
```

Copy this shape for internal records such as `SolarEvidenceSource`, `SolarTensorEvidence`, `SolarSubroleEvidence`, `SolarSemanticGroupEvidence`, and `SolarDerivationEvidence`. Keep tuples internally and emit JSON-safe lists in `to_dict()`. Include `schema_version`, `derived`, `definition`, `workload_uuid`, `groups`, `tensors`, `warnings`, and a `source_boundary` object with public/candidate execution flags.

**Builder pattern** (lines 143-168):
```python
def build_amd_sol_bound_v2_artifact(
    definition: Definition,
    workload: Workload,
    hardware_model: AmdHardwareModel,
    *,
    hardware_model_ref: str | None = None,
) -> AmdSolBoundV2Artifact:
    """Build an AMD SOL bound artifact v2 sidecar."""
    graph = build_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)
    op_bounds = tuple(_bound_for_estimate(estimate, hardware_model) for estimate in estimates)
    coverage = _coverage_for_estimates(estimates)
    aggregate = _aggregate_for_bounds(op_bounds, hardware_model)
    warnings = _warnings_for_artifact(graph.warnings, estimates, aggregate, hardware_model)
    return AmdSolBoundV2Artifact(
        definition=definition.name,
        workload_uuid=workload.uuid,
        hardware_model_ref=hardware_model_ref,
        hardware_model=hardware_model,
        bound_graph=graph.to_dict(),
        operator_work_estimates=tuple(estimate.to_dict() for estimate in estimates),
        op_bounds=op_bounds,
        aggregate_bound=aggregate,
        warnings=warnings,
        coverage_summary=coverage,
    )
```

For Phase 48, mirror this as `build_solar_derivation_evidence(definition, workload)` and a lower-level `derive_solar_derivation_evidence(definition, workload, graph, estimates)` so tests can inject prebuilt graph/estimate evidence. The builder must use `Definition.reference`, `Workload.axes`, `Workload.inputs`, `build_bound_graph()`, and `estimate_bound_work()` only; it must not execute candidate solutions.

**Strict parser pattern** (lines 171-190 and 403-420):
```python
def amd_sol_bound_v2_from_dict(payload: dict[str, Any]) -> AmdSolBoundV2Artifact:
    """Parse an AMD SOL bound artifact v2 sidecar payload."""
    if not isinstance(payload, dict):
        raise ValueError("AMD SOL v2 artifact payload must be an object")
    _require_keys(
        payload,
        {
            "schema_version",
            "derived",
            "definition",
            "workload_uuid",
            "hardware_model_ref",
            "hardware_model",
            "bound_graph",
            "operator_work_estimates",
            "op_bounds",
            "aggregate_bound",
            "warnings",
            "coverage_summary",
        },
```

```python
def _op_bound_from_dict(payload: Any, index: int) -> AmdSolV2OpBound:
    raw = _ensure_dict(payload, source=f"op_bounds[{index}]")
    _require_keys(
        raw,
        {
            "node_id",
            "op_family",
            "op_name",
            "compute_bound_ms",
            "memory_bound_ms",
            "sol_bound_ms",
            "limiting_resource",
            "confidence",
            "rationale",
            "estimate_warnings",
        },
        source=f"op_bounds[{index}]",
    )
```

Use a `solar_derivation_from_dict()` parser with required-field checks for every top-level and nested object. Reject invalid `schema_version`, non-list `groups`/`tensors`/`warnings`, invalid confidence/status values, empty IDs, and missing evidence objects. Keep errors deterministic and similar to `"{source} missing required field: {key}"`.

**Graph/tensor evidence pattern** from `amd_bound_graph.py` (lines 47-145):
```python
@dataclass(frozen=True)
class BoundTensor:
    """Tensor metadata bound to a concrete workload."""

    tensor_id: str
    name: str
    role: BoundTensorRole
    shape: tuple[int, ...] | None
    dtype: str
    producer_node_id: str | None
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_id": self.tensor_id,
            "name": self.name,
            "role": self.role.value,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "producer_node_id": self.producer_node_id,
            "source": self.source,
        }
```

```python
@dataclass(frozen=True)
class BoundGraphNode:
    """Operation node in a structured AMD bound graph."""

    node_id: str
    op_family: OpFamily
    op_name: str
    source_expression: str
    input_tensor_ids: tuple[str, ...]
    output_tensor_ids: tuple[str, ...]
    attributes: dict[str, object]
    confidence: EstimateConfidence
    rationale: str
```

The semantic sidecar should reference `node_id`, `input_tensor_ids`, and `output_tensor_ids`; do not rewrite `BoundGraph` or add public schema fields. Use existing `BoundTensor.shape`, `BoundTensor.dtype`, `BoundTensor.source`, `BoundGraphNode.attributes`, and `BoundGraphNode.source_expression` as evidence inputs.

**FX-first / AST-fallback extraction boundary** from `amd_bound_graph.py` (lines 253-287 and 290-320):
```python
def build_bound_graph(definition: Definition, workload: Workload) -> BoundGraph:
    """Build a structured bound graph for a concrete definition/workload pair."""
    input_shapes = definition.get_input_shapes(workload.axes)
    output_shapes = definition.get_output_shapes(workload.axes)
    tensors = _declared_tensors(definition, input_shapes, output_shapes)
    warnings: list[str] = []

    fx_graph = _try_fx_bound_graph(definition, workload, input_shapes, output_shapes, tensors)
    if fx_graph is not None:
        return fx_graph

    warnings.append("dynamic_trace_failed")

    try:
        tree = ast.parse(definition.reference, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Reference must be valid Python code for graph extraction: {exc}") from exc
```

```python
def _try_fx_bound_graph(
    definition: Definition,
    workload: Workload,
    input_shapes: dict[str, tuple[int, ...] | None],
    output_shapes: dict[str, tuple[int, ...] | None],
    declared_tensors: dict[str, BoundTensor],
) -> BoundGraph | None:
    """Trace the reference with torch.fx and convert common nodes to BoundGraph."""
    try:
        import torch
        from torch.fx import Node, symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp
    except Exception:
        return None
```

Phase 48 should consume this existing extractor instead of duplicating FX/AST logic. Tag `SolarEvidenceSource.kind` with values derived from `BoundTensor.source`, node `attributes["trace_source"]`, graph warnings, `workload`, `definition`, or `estimate`.

**Confidence and warning pattern** from `amd_sol_v2.py` (lines 271-400):
```python
if not op_bounds:
    return AmdSolV2AggregateBound(
        status="unscored",
        scored=False,
        sol_bound_ms=sol_bound_ms,
        reason="missing operation bound evidence",
        node_ids=node_ids,
    )
if any(bound.confidence == EstimateConfidence.UNSUPPORTED for bound in op_bounds):
    return AmdSolV2AggregateBound(
        status="unscored",
        scored=False,
        sol_bound_ms=sol_bound_ms,
        reason="unsupported operation evidence present",
        node_ids=node_ids,
    )
```

```python
for estimate in estimates:
    for warning in estimate.warnings:
        warnings.append(f"estimate_warning:{estimate.node_id}:{warning}")
    if estimate.confidence == EstimateConfidence.INEXACT:
        warnings.append(f"inexact_operator:{estimate.node_id}:{estimate.op_family.value}")
    elif estimate.confidence == EstimateConfidence.UNSUPPORTED:
        warnings.append(
            f"unsupported_operator:{estimate.node_id}:{estimate.op_family.value}"
        )
if aggregate.status == "degraded":
    warnings.append(f"aggregate_degraded:{aggregate.reason}")
elif aggregate.status == "unscored":
    warnings.append(f"aggregate_unscored:{aggregate.reason}")
return _unique(warnings)
```

Use one pure deterministic confidence helper. Map `supported -> scored`, `inexact -> degraded`, and `unsupported -> unscored`. Require visible family, subroles, shape, dtype, semantic axes, and source/formula provenance for `supported`. Emit structured `missing_evidence` plus stable warning prefixes for degraded/unsupported evidence.

**Estimate evidence pattern** from `amd_bound_estimates.py` (lines 21-68 and 91-127):
```python
@dataclass(frozen=True)
class OperatorWorkEstimate:
    """Auditable work estimate for one BoundGraph operation node."""

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
    axis_source: str | None = None
    movement_kind: str | None = None
    warnings: tuple[str, ...] = ()
```

```python
dims = _infer_gemm_dims(input_tensors, output_tensors)
if dims is None:
    if not input_tensors and not output_tensors:
        return _unsupported_estimate(
            node,
            rationale="unsupported GEMM estimate: all key tensors are unresolved",
            warnings=("unsupported_operator:gemm_missing_tensors",),
        )
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="gemm_flops",
        formula="2*M*N*K",
        formula_inputs={},
        flops=0.0,
        ...
        confidence=EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "GEMM semantics recognized but missing shape evidence for M/N/K",
            rationale_parts,
        ),
        warnings=tuple(warnings or ("inexact_operator:gemm_missing_shape",)),
    )
```

Reuse estimate records as provenance inputs, but do not add family-specific formulas in Phase 48. For recognized-but-incomplete compound families, emit semantic groups with `confidence=INEXACT`, `status="degraded"`, and missing evidence instead of inventing dimensions.

---

### `src/sol_execbench/core/scoring/__init__.py` (config, import/export surface)

**Analog:** `src/sol_execbench/core/scoring/__init__.py`

**Current export style** (lines 20-41 and 71-118):
```python
from .amd_sol_v2 import (
    AMD_SOL_V2_SCHEMA_VERSION,
    AmdSolBoundV2Artifact,
    AmdSolV2AggregateBound,
    AmdSolV2CoverageSummary,
    AmdSolV2OpBound,
    amd_sol_bound_v2_from_dict,
    build_amd_sol_bound_v2_artifact,
)
from .amd_bound_graph import (
    BoundEdge,
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
    build_bound_graph,
)
from .amd_bound_estimates import (
    OperatorWorkEstimate,
    estimate_bound_work,
)
```

```python
__all__ = [
    "AMD_SCORE_CLAIM_LEVEL",
    "AMD_SCORE_SCHEMA_VERSION",
    "AMD_SOL_SCHEMA_VERSION",
    "AMD_SOL_V2_SCHEMA_VERSION",
    ...
    "build_bound_graph",
    "build_amd_native_suite_report",
    "amd_hardware_model_from_dict",
    ...
    "amd_sol_bound_v2_from_dict",
    "estimate_work",
    "estimate_bound_work",
```

Only export Phase 48 types if tests or later internal callers need `from sol_execbench.core.scoring import ...`. If exported, add a grouped import block for `solar_derivation.py` and matching `__all__` entries. Do not expose any CLI option or public Pydantic field.

---

### `tests/sol_execbench/test_solar_derivation_evidence.py` (test, fixture-driven transform/parser)

**Analogs:** `tests/sol_execbench/test_amd_sol_v2.py`, `tests/sol_execbench/test_amd_bound_graph.py`, `tests/sol_execbench/test_solar_derivation_contract.py`

**Small Definition/Workload fixture pattern** from `test_amd_sol_v2.py` (lines 34-56):
```python
def _matmul_definition() -> Definition:
    return Definition(
        name="matmul_demo",
        axes={
            "M": {"type": "var"},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="def run(a, b):\n    return a @ b",
    )
```

Use local helper definitions for positive linear/GEMM evidence, degraded softmax/attention-like evidence, and unsupported dynamic/control-flow evidence. Keep tests CPU-only and deterministic.

**Serialization/immutability/round-trip pattern** from `test_amd_sol_v2.py` (lines 59-125):
```python
artifact = build_amd_sol_bound_v2_artifact(
    _matmul_definition(),
    _matmul_workload(),
    hardware,
    hardware_model_ref="default_amd_hardware_models.gfx1200",
)
payload = artifact.to_dict()

assert artifact.schema_version == AMD_SOL_V2_SCHEMA_VERSION
assert artifact.derived is True
assert payload["definition"] == "matmul_demo"
assert payload["workload_uuid"] == "matmul-workload"
...
loaded = amd_sol_bound_v2_from_dict(payload)

assert loaded.to_dict() == payload

bad_schema = dict(payload)
bad_schema["schema_version"] = "sol_execbench.amd_sol_bound.v3"
with pytest.raises(ValueError, match="invalid schema_version"):
    amd_sol_bound_v2_from_dict(bad_schema)
```

Copy this test structure for `SolarDerivationEvidence`: assert sidecar schema version, `derived is True`, group/tensor evidence fields, confidence/status, missing evidence, warning prefixes, frozen dataclass behavior, round-trip equality, invalid schema rejection, missing field rejection, wrong list/object type rejection, and invalid nested confidence/status rejection.

**Graph evidence behavior tests** from `test_amd_bound_graph.py` (lines 42-98 and 120-182):
```python
graph = build_bound_graph(_matmul_definition(), _matmul_workload())
payload = graph.to_dict()

assert isinstance(graph, BoundGraph)
assert isinstance(graph.nodes[0], BoundGraphNode)
assert isinstance(next(iter(graph.tensors.values())), BoundTensor)
assert payload["derived"] is True
assert payload["nodes"][0]["confidence"] == "supported"
assert payload["nodes"][0]["op_family"] == "gemm"
assert payload["tensors"]["input:a"]["shape"] == [2, 4]
```

```python
graph = build_bound_graph(definition, workload)

assert graph.nodes[0].op_family == OpFamily.UNSUPPORTED
assert graph.nodes[0].confidence == EstimateConfidence.UNSUPPORTED
assert graph.nodes[0].source_expression == "torch.linalg.inv(x)"
assert "unsupported_operator:torch.linalg.inv" in graph.warnings
```

Add tests that Phase 48 evidence preserves visible unsupported/degraded groups instead of dropping them. Assert stable IDs, source kinds, tensor shape/dtype, semantic axes when available, and explicit `missing_evidence`.

**Fixture loader integration pattern** from `test_solar_derivation_contract.py` (lines 203-220):
```python
def test_fixture_loader_does_not_execute_reference_text(tmp_path):
    fixture = _valid_fixture()
    marker = tmp_path / "executed"
    fixture["reference"] = f"raise SystemExit; open({str(marker)!r}, 'w').write('bad')"
    (tmp_path / "fixture.json").write_text(json.dumps(fixture))

    load_solar_derivation_fixtures(tmp_path)

    assert not marker.exists()


def test_fixture_matrix_covers_required_families_and_classes():
    fixtures = load_solar_derivation_fixtures()

    assert len(fixtures) >= 18
    for family in sorted(TARGET_FAMILIES):
```

Use `load_solar_derivation_fixtures()` as the fixture matrix source for expected family, subroles, confidence, status, required evidence, missing evidence, and warning prefixes. Do not execute fixture `reference` text unless it is converted into a `Definition.reference` in a controlled unit test.

---

### `tests/sol_execbench/test_public_contract_guardrails.py` (test, guardrail)

**Analog:** `tests/sol_execbench/test_public_contract_guardrails.py`

**Existing SOLAR noncanonical field guardrail** (lines 159-190):
```python
def test_v1_10_solar_derivation_fields_remain_noncanonical():
    definition = Definition(
        name="demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )
    workload = Workload(axes={"N": 16}, inputs={"x": {"type": "random"}}, uuid="w1")
    trace = Trace(
        definition="demo",
        workload=workload,
        solution="solution",
        evaluation=None,
    )
    forbidden = (
        "solar_derivation",
        "expected_subroles",
        "required_evidence",
        "missing_evidence",
        "warning_prefixes",
        "scope_boundary",
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        for field in forbidden:
            assert field not in payload
```

Extend the forbidden tuple with Phase 48 sidecar names such as `semantic_groups`, `semantic_axes`, `source_boundary`, `candidate_solution_execution`, `confidence_rationale`, `subroles`, and any selected schema field names from `solar_derivation.py`.

**CLI guardrail pattern** (lines 192-204):
```python
def test_primary_cli_does_not_expose_v1_10_solar_derivation_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for option in (
        "--solar-derivation",
        "--derive-solar",
        "--solar-fixtures",
        "--solar-contract",
        "--solar-sidecar",
    ):
        assert option not in help_text
```

Add any Phase 48 implementation-specific option names to this denial list if they exist in planning, but the preferred implementation adds no CLI options.

**Derived-artifact public schema guardrail** (lines 284-332):
```python
assert "bound_graph" not in definition.model_dump(mode="json")
assert "graph_nodes" not in definition.model_dump(mode="json")
assert "op_family" not in definition.model_dump(mode="json")
assert "op_bounds" not in definition.model_dump(mode="json")
assert "formula_kind" not in definition.model_dump(mode="json")
assert "read_bytes" not in definition.model_dump(mode="json")
assert "movement_bytes" not in definition.model_dump(mode="json")
assert "operator_work_estimates" not in definition.model_dump(mode="json")
assert "coverage_summary" not in definition.model_dump(mode="json")
assert "aggregate_bound" not in definition.model_dump(mode="json")
```

Follow this explicit assertion style rather than snapshotting public schemas. The guardrail should prove canonical `Definition`, `Workload`, and `Trace` dumps remain free of Phase 48 evidence fields.

## Shared Patterns

### Internal Sidecar Only
**Source:** `src/sol_execbench/core/scoring/amd_sol_v2.py` lines 107-140; `tests/sol_execbench/test_public_contract_guardrails.py` lines 159-204.
**Apply to:** `solar_derivation.py`, `test_solar_derivation_evidence.py`, `test_public_contract_guardrails.py`.

Sidecars use frozen dataclasses with `to_dict()` and parser helpers. Public Pydantic models, trace JSONL, and primary CLI help must not expose Phase 48 fields.

### Evidence Inputs
**Source:** `src/sol_execbench/core/scoring/amd_bound_graph.py` lines 253-287; `src/sol_execbench/core/scoring/amd_bound_estimates.py` lines 66-88.
**Apply to:** `build_solar_derivation_evidence()` and `derive_solar_derivation_evidence()`.

Evidence comes from `Definition`, `Workload`, `BoundGraph`, and `OperatorWorkEstimate`. Do not inspect or execute submitted solution code.

### Confidence And Status Vocabulary
**Source:** `src/sol_execbench/core/scoring/amd_sol_v2.py` lines 271-400; `tests/sol_execbench/solar_derivation_fixtures.py` lines 63-78.
**Apply to:** semantic group confidence, aggregate status, warning generation, parser validation, tests.

Use exactly `supported`, `inexact`, `unsupported` and `scored`, `degraded`, `unscored`. Stable warning prefixes should stay in the existing family: `graph_warning:`, `estimate_warning:`, `inexact_operator:`, `unsupported_operator:`, `aggregate_degraded:`, and `aggregate_unscored:`.

### Fixture-Driven Tests
**Source:** `tests/sol_execbench/solar_derivation_fixtures.py` lines 15-78 and 81-142; `tests/sol_execbench/test_solar_derivation_contract.py` lines 214-220.
**Apply to:** `test_solar_derivation_evidence.py`.

Use fixture expectations as the semantic contract: target family, expected subroles, confidence, status, required evidence, missing evidence, warning prefixes, and scope boundaries. Keep loader validation separate from evidence builder tests.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| None | n/a | n/a | Existing scoring sidecar, bound graph, estimate, fixture, and guardrail tests cover all Phase 48 file roles. |

## Metadata

**Analog search scope:** `src/sol_execbench/core/scoring/`, `tests/sol_execbench/`, phase artifacts under `.planning/phases/48-extraction-pipeline-and-semantic-provenance/`.
**Files scanned:** 12 required/context files plus scoring package export file and fixture loader.
**Pattern extraction date:** 2026-05-23
