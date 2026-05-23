# Phase 49: High-Confidence Family Modeling - Pattern Map

**Mapped:** 2026-05-23
**Files analyzed:** 7 required files plus 4 focused analog test/helper files
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/scoring/solar_derivation.py` | service / internal sidecar model | transform | `SolarEvidenceSource`, `SolarTensorEvidence`, `SolarSemanticGroupEvidence`, `SolarDerivationEvidence` in `solar_derivation.py` | exact |
| `src/sol_execbench/core/scoring/solar_derivation.py` | parse helper | request-response / transform | `solar_derivation_from_dict()` and nested `_group_from_dict()` parser helpers in `solar_derivation.py` | exact |
| `src/sol_execbench/core/scoring/solar_derivation.py` | family evidence integration | transform | `_semantic_group_evidence()`, `_required_evidence_for_group()`, `_subroles_for_group()` in `solar_derivation.py` | exact |
| `src/sol_execbench/core/scoring/amd_bound_estimates.py` | formula / byte estimate service | transform | `OperatorWorkEstimate`, `_gemm_estimate()`, `_data_movement_estimate()`, `_unsupported_estimate()` in `amd_bound_estimates.py` | exact |
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | graph extraction / classifier | transform | `OpFamily`, `BoundGraphNode`, `_CALL_CLASSIFIERS` in `amd_bound_graph.py` | role-match |
| `tests/sol_execbench/test_solar_derivation_evidence.py` | test | request-response / transform | contract fixture tests and handcrafted `BoundGraph`/`OperatorWorkEstimate` tests in `test_solar_derivation_evidence.py` | exact |
| `tests/sol_execbench/test_public_contract_guardrails.py` | contract guardrail test | request-response | v1.10 SOLAR noncanonical and score eligibility guardrails | exact |
| `tests/sol_execbench/test_amd_bound_estimates.py` | formula / byte estimate test | transform | formula input, byte, unsupported, dtype-width tests | exact |
| `tests/sol_execbench/fixtures/solar_derivation/*.json` | fixture data | file-I/O / transform | fixture loader schema in `solar_derivation_fixtures.py` | role-match |

## Pattern Assignments

### Family Formula / Byte Evidence Dataclasses

**Recommended file/function to extend:** `src/sol_execbench/core/scoring/solar_derivation.py`

**Closest analog:** frozen sidecar dataclasses in `solar_derivation.py`.

**Pattern to copy** (`src/sol_execbench/core/scoring/solar_derivation.py` lines 39-80):

```python
@dataclass(frozen=True)
class SolarEvidenceSource:
    """Provenance source for a SOLAR derivation evidence record."""

    kind: str
    detail: str
    node_id: str | None = None
    tensor_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "node_id": self.node_id,
            "tensor_id": self.tensor_id,
        }
```

```python
@dataclass(frozen=True)
class SolarTensorEvidence:
    """Tensor metadata and semantic-axis provenance for SOLAR derivation."""

    tensor_id: str
    name: str
    shape: tuple[int, ...] | None
    dtype: str
    semantic_axes: tuple[str, ...]
    source: SolarEvidenceSource
    producer_node_id: str | None
    missing_evidence: tuple[str, ...] = ()
```

**Apply to Phase 49:** add any family-specific formula/byte evidence as frozen dataclasses with explicit `to_dict()` conversion. Prefer names such as `SolarFamilyFormulaEvidence` and `SolarFamilyByteEvidence` if new models are needed. Keep values JSON-safe: tuples serialize as lists, enums serialize through `.value`, nested sidecar models serialize with `.to_dict()`.

**Do not copy into public schemas:** these are internal sidecar patterns only.

### Strict Parse Helpers

**Recommended file/function to extend:** `solar_derivation_from_dict()` and nested parser helpers in `src/sol_execbench/core/scoring/solar_derivation.py`.

**Closest analog:** top-level strict parser and nested exact-key parsers.

**Pattern to copy** (`src/sol_execbench/core/scoring/solar_derivation.py` lines 299-357):

```python
def solar_derivation_from_dict(payload: dict[str, Any]) -> SolarDerivationEvidence:
    """Parse an internal SOLAR derivation evidence sidecar payload."""
    if not isinstance(payload, dict):
        raise ValueError("SOLAR derivation evidence payload must be an object")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "derived",
            "definition",
            "workload_uuid",
            "groups",
            "tensors",
            "warnings",
            "source_boundary",
        },
        source="SOLAR derivation evidence",
    )
```

```python
schema_version = _parse_str(
    payload, "schema_version", source="SOLAR derivation evidence"
)
if schema_version != SOLAR_DERIVATION_SCHEMA_VERSION:
    raise ValueError(
        "SOLAR derivation evidence has invalid schema_version "
        f"'{schema_version}', expected '{SOLAR_DERIVATION_SCHEMA_VERSION}'"
    )
```

**Pattern to copy** (`src/sol_execbench/core/scoring/solar_derivation.py` lines 360-435):

```python
def _group_from_dict(payload: Any, index: int) -> SolarSemanticGroupEvidence:
    source = f"groups[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "family",
            "group_id",
            "node_ids",
            "subroles",
            "confidence",
            "status",
            "required_evidence",
            "missing_evidence",
            "warning_prefixes",
            "source",
            "rationale",
        },
        source=source,
    )
```

**Apply to Phase 49:** if formula and byte evidence become new nested fields, add exact-key parse helpers next to `_group_from_dict()` / `_tensor_from_dict()`. Reject unknown fields, invalid enum strings, empty strings, malformed shape/list values, and wrong booleans. Do not use permissive `dict(...)` passthrough for new evidence.

### Family Evidence Integration

**Recommended file/function to extend:** `_semantic_group_evidence()`, `_required_evidence_for_group()`, and `_subroles_for_group()` in `solar_derivation.py`.

**Closest analog:** semantic group derivation from `BoundGraph` plus `OperatorWorkEstimate`.

**Pattern to copy** (`src/sol_execbench/core/scoring/solar_derivation.py` lines 529-585):

```python
estimates_by_family: dict[str, list[OperatorWorkEstimate]] = {}
for estimate in estimates:
    estimates_by_family.setdefault(estimate.op_family.value, []).append(estimate)

groups: list[SolarSemanticGroupEvidence] = []
for group_index, (family, family_estimates) in enumerate(
    sorted(
        estimates_by_family.items(),
        key=lambda item: _first_estimate_node_id(item[1]),
    ),
    start=1,
):
    ordered_estimates = tuple(sorted(family_estimates, key=lambda item: item.node_id))
    nodes = tuple(
        nodes_by_id[estimate.node_id]
        for estimate in ordered_estimates
        if estimate.node_id in nodes_by_id
    )
```

**Pattern to copy** (`src/sol_execbench/core/scoring/solar_derivation.py` lines 588-611):

```python
for estimate in estimates:
    if estimate.formula_inputs:
        required.append(f"formula_inputs:{estimate.node_id}")
    if estimate.formula and estimate.formula != "0":
        required.append(f"formula:{estimate.node_id}")
    if estimate.total_bytes > 0.0:
        required.append(f"bytes:{estimate.node_id}")
    if estimate.axis_source is not None:
        required.append(f"axis:{estimate.node_id}")
return _unique_sorted(required)
```

**Apply to Phase 49:** family formula/byte evidence should be attached through the existing group and required/missing evidence machinery. Add family-specific subroles in `_subroles_for_group()` rather than creating a separate grouping pipeline. Preserve deterministic sorting by family first node, node ID, subrole name, and `_unique_sorted()` evidence lists.

### Confidence, Degradation, And Unsupported Logic

**Recommended file/function to extend:** `classify_solar_confidence()`.

**Closest analog:** current missing-evidence classification.

**Pattern to copy** (`src/sol_execbench/core/scoring/solar_derivation.py` lines 216-296):

```python
if not estimates:
    missing.append("estimate:operator_work")
for estimate in sorted(estimates, key=lambda item: item.node_id):
    if estimate.confidence == EstimateConfidence.UNSUPPORTED:
        missing.append(f"estimate:{estimate.node_id}")
    if not estimate.formula_inputs:
        missing.append(f"formula_inputs:{estimate.node_id}")
    if not estimate.formula or estimate.formula == "0":
        missing.append(f"formula:{estimate.node_id}")
    if estimate.total_bytes <= 0.0:
        missing.append(f"bytes:{estimate.node_id}")
    if estimate.axis_source is None:
        missing.append(f"axis:{estimate.node_id}")
    warning_prefixes.extend(estimate.warnings)
```

```python
elif missing:
    confidence = _worse_confidence(confidence, EstimateConfidence.INEXACT)
```

**Apply to Phase 49:** supported evidence requires visible family, dimensions, dtype, semantic axes, source provenance, formula inputs, nonzero bytes, and axis provenance. Missing mask semantics, grouping metadata, dynamic indices, dynamic kernel sizes, unknown dtype, or unknown memory behavior should degrade to `inexact` or `unsupported`; never mark supported by filling invented values.

### Formula And Byte Estimate Helpers

**Recommended file/function to extend:** `src/sol_execbench/core/scoring/amd_bound_estimates.py`.

**Closest analog:** `OperatorWorkEstimate` and per-family estimate helpers.

**Pattern to copy** (`src/sol_execbench/core/scoring/amd_bound_estimates.py` lines 21-63):

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

**Pattern to copy** (`src/sol_execbench/core/scoring/amd_bound_estimates.py` lines 91-158):

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
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.INEXACT,
```

**Pattern to copy** (`src/sol_execbench/core/scoring/amd_bound_estimates.py` lines 286-319):

```python
movement_kind = str(node.attributes.get("movement_kind") or _movement_kind_from_op_name(node))
if movement_kind == "materialized":
    movement_bytes = read_bytes + write_bytes
    rationale = "materialized data movement estimate for contiguous or copy-like operation"
elif movement_kind == "broadcast_view":
    movement_bytes = 0.0
    rationale = "broadcast view evidence with zero movement bytes"
else:
    movement_kind = "logical_view"
    movement_bytes = 0.0
    rationale = "logical view evidence with zero movement bytes"
```

**Apply to Phase 49:** add estimate helpers for promoted high-confidence families in `amd_bound_estimates.py`, then route from `_estimate_node()` after existing family branches. Use `formula_kind` values such as `attention_flops`, `convolution_flops`, and `embedding_positional_bytes` only when required inputs are visible. For linear projection, preserve first-class `op_family="linear_projection"` while reusing the GEMM-compatible `formula_kind="gemm_flops"` logic where dimensions are explicit. For memory-bound families, keep `flops=0.0` if appropriate but still emit dtype-aware `read_bytes`, `write_bytes`, `intermediate_bytes`, `movement_bytes`, and `total_bytes`.

### Graph Family Classification

**Recommended file/function to extend:** `OpFamily` and `_CALL_CLASSIFIERS` in `src/sol_execbench/core/scoring/amd_bound_graph.py`.

**Closest analog:** enum taxonomy and call classification.

**Pattern to copy** (`src/sol_execbench/core/scoring/amd_bound_graph.py` lines 27-44):

```python
class OpFamily(str, Enum):
    """Paper-aligned operation family for SOLAR graph extraction."""

    ATTENTION = "attention"
    MOE = "moe"
    NORMALIZATION = "normalization"
    EMBEDDING_POSITIONAL = "embedding_positional"
    LINEAR_PROJECTION = "linear_projection"
    GEMM = "gemm"
    MLP_ACTIVATION = "mlp_activation"
    CONVOLUTION = "convolution"
    SSM_MAMBA = "ssm_mamba"
    SOFTMAX = "softmax"
    REDUCTION = "reduction"
    ELEMENTWISE = "elementwise"
    DATA_MOVEMENT = "data_movement"
    DTYPE_CONVERSION = "dtype_conversion"
    UNSUPPORTED = "unsupported"
```

**Pattern to copy** (`src/sol_execbench/core/scoring/amd_bound_graph.py` lines 155-228):

```python
_CALL_CLASSIFIERS: tuple[tuple[set[str], _CallClassification], ...] = (
    (
        {"matmul", "mm", "bmm"},
        _CallClassification(OpFamily.GEMM, EstimateConfidence.SUPPORTED, "recognized matrix multiply"),
    ),
    (
        {"linear"},
        _CallClassification(
            OpFamily.LINEAR_PROJECTION,
            EstimateConfidence.SUPPORTED,
            "recognized linear projection",
        ),
    ),
```

**Apply to Phase 49:** reuse existing `OpFamily` values. Add classifiers for explicit `conv1d`, `conv2d`, `conv3d`, `embedding`, gather/indexing, rotary-like visible operations, and attention pieces only when the call name and metadata are structurally visible. Keep ambiguous compound patterns as `INEXACT` or `UNSUPPORTED` with warnings rather than over-classifying.

## Test Pattern Assignments

### Sidecar Round Trip And Strict Parser Tests

**Recommended file/function to extend:** `tests/sol_execbench/test_solar_derivation_evidence.py`.

**Closest analog:** `_contract_artifact()` and parser rejection tests.

**Pattern to copy** (`tests/sol_execbench/test_solar_derivation_evidence.py` lines 62-121):

```python
attention_group = SolarSemanticGroupEvidence(
    family="attention",
    group_id="attention_group_1",
    node_ids=("op_1", "op_2"),
    subroles=(qk_scores,),
    confidence="supported",
    status="scored",
    required_evidence=("shape:batch", "shape:sequence_q", "shape:sequence_k"),
    missing_evidence=(),
    warning_prefixes=(),
    source=_source(kind="definition", detail="reference structure"),
    rationale="Required attention subrole evidence is present.",
)
```

**Apply to Phase 49:** add round-trip coverage for new formula/byte fields. Mutate payloads to verify missing required field, unknown field, bad formula kind/status/confidence, invalid byte buckets, and malformed provenance all raise deterministic `ValueError`.

### Handcrafted BoundGraph / Estimate Tests

**Recommended file/function to extend:** `tests/sol_execbench/test_solar_derivation_evidence.py`.

**Closest analog:** projection graph and estimate helpers.

**Pattern to copy** (`tests/sol_execbench/test_solar_derivation_evidence.py` lines 606-719):

```python
def _projection_graph(
    *,
    node_id: str = "op_1",
    family: OpFamily = OpFamily.LINEAR_PROJECTION,
    missing_output_axis: bool = False,
    unsupported: bool = False,
) -> BoundGraph:
    input_shape = (2, 4)
    weight_shape = (4, 8)
    output_shape = None if unsupported else (2, 8)
    output_source = "tmp:linear" if missing_output_axis else "definition.outputs"
```

```python
def _projection_estimate(
    *,
    node_id: str = "op_1",
    family: OpFamily = OpFamily.LINEAR_PROJECTION,
    confidence: EstimateConfidence = EstimateConfidence.SUPPORTED,
    formula_inputs: dict[str, object] | None = None,
    axis_source: str | None = "workload.axes",
    total_bytes: float = 160.0,
    warnings: tuple[str, ...] = (),
) -> OperatorWorkEstimate:
```

**Apply to Phase 49:** create small graph/estimate helpers per promoted family rather than needing ROCm or solution execution. Cover positive, degraded, and unsupported cases for attention, convolution, embedding/positional/gather/rotary-like memory movement, and linear projection.

### Estimate Formula / Byte Tests

**Recommended file/function to extend:** `tests/sol_execbench/test_amd_bound_estimates.py` or a focused new family modeling test file.

**Closest analog:** formula and byte assertions.

**Pattern to copy** (`tests/sol_execbench/test_amd_bound_estimates.py` lines 143-157):

```python
estimate = estimate_bound_work(graph)[0]

assert estimate.formula_kind == "gemm_flops"
assert estimate.formula == "2*M*N*K"
assert estimate.formula_inputs == {"M": 2, "N": 8, "K": 4}
assert estimate.flops == 128.0
assert estimate.read_bytes == 160.0
assert estimate.write_bytes == 64.0
assert estimate.total_bytes == 224.0
assert estimate.confidence == EstimateConfidence.SUPPORTED
```

**Pattern to copy** (`tests/sol_execbench/test_amd_bound_estimates.py` lines 108-124):

```python
for dtype in (
    DType.FLOAT64,
    DType.FLOAT32,
    DType.FLOAT16,
    DType.BFLOAT16,
    DType.FLOAT8_E4M3FN,
    DType.FLOAT8_E5M2,
    DType.FLOAT4_E2M1,
    DType.FLOAT4_E2M1FN_X2,
    DType.INT64,
    DType.INT32,
    DType.INT16,
    DType.INT8,
    DType.BOOL,
):
    assert _dtype_bytes(dtype) is not None
```

**Apply to Phase 49:** assert exact formula text, formula inputs, dtype-aware bytes, byte buckets, confidence, and warnings for each family. Add missing-shape/dtype tests that verify no fabricated bytes and degraded/unsupported status.

### Fixture-Driven Family Coverage

**Recommended file/function to extend:** `tests/sol_execbench/fixtures/solar_derivation/*.json` and `tests/sol_execbench/solar_derivation_fixtures.py` only if the fixture schema must grow.

**Closest analog:** strict fixture loader.

**Pattern to copy** (`tests/sol_execbench/solar_derivation_fixtures.py` lines 15-78):

```python
TARGET_FAMILIES = frozenset(
    {
        "attention",
        "moe",
        "convolution",
        "ssm_mamba",
        "embedding_positional",
        "linear_projection",
    }
)
```

```python
VALID_CONFIDENCES = frozenset({"supported", "inexact", "unsupported"})
VALID_STATUSES = frozenset({"scored", "degraded", "unscored"})
EXPECTED_STATES_BY_FIXTURE_CLASS = {
    "positive": ("supported", "scored"),
    "degraded": ("inexact", "degraded"),
    "unsupported": ("unsupported", "unscored"),
    "negative": ("unsupported", "unscored"),
}
```

**Apply to Phase 49:** use existing positive/degraded/unsupported fixture classes. If adding formula/byte expectations to fixtures, update the loader with strict required keys and validator checks; avoid optional loose blobs.

## Shared Patterns

### Source Boundary

**Source:** `solar_derivation.py` and public guardrail tests.

**Apply to:** every Phase 49 sidecar path.

**Pattern to preserve** (`src/sol_execbench/core/scoring/solar_derivation.py` lines 28-36):

```python
SOLAR_DERIVATION_SCHEMA_VERSION = "sol_execbench.solar_derivation.v1"
SOLAR_DERIVATION_STATUSES = frozenset({"scored", "degraded", "unscored"})
SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS = frozenset(
    {
        "canonical_trace_jsonl",
        "public_schema",
        "candidate_solution_execution",
    }
)
```

**Pattern to preserve** (`tests/sol_execbench/test_solar_derivation_evidence.py` lines 589-603):

```python
builder_signature = inspect.signature(build_solar_derivation_evidence)
derive_signature = inspect.signature(derive_solar_derivation_evidence)
forbidden_terms = {"Solution", "solution_path", "candidate", "submitted_code"}

assert not forbidden_terms & set(builder_signature.parameters)
assert not forbidden_terms & set(derive_signature.parameters)
```

Phase 49 must continue deriving only from canonical `Definition`, `Workload`, reference-visible FX/AST structure, bound graph, and estimates. Do not accept solution code, paths, compiled artifacts, runtime traces, or candidate execution as evidence sources.

### Public Contract Guardrails

**Source:** `tests/sol_execbench/test_public_contract_guardrails.py`.

**Apply to:** any planner action that touches schema, CLI, scoring, docs, or exports.

**Pattern to preserve** (`tests/sol_execbench/test_public_contract_guardrails.py` lines 139-158):

```python
for expected in (
    "sidecar-only",
    "not paper-scale dataset extraction",
    "not hosted leaderboard readiness",
    "not NVIDIA Blackwell/B200 equivalence",
    "not new real-hardware validation",
):
    assert expected in contract
```

**Pattern to preserve** (`tests/sol_execbench/test_public_contract_guardrails.py` lines 161-200):

```python
for payload in (
    definition.model_dump(mode="json"),
    workload.model_dump(mode="json"),
    trace.model_dump(mode="json"),
):
    for field in forbidden:
        assert field not in payload
        assert field not in repr(payload)
```

**Pattern to preserve** (`tests/sol_execbench/test_public_contract_guardrails.py` lines 203-220):

```python
for option in (
    "--solar-derivation",
    "--derive-solar",
    "--solar-fixtures",
    "--solar-contract",
    "--solar-sidecar",
    "--semantic-provenance",
    "--solar-evidence",
    "--solar-confidence",
    "--solar-provenance",
    "--derive-solar-sidecar",
):
    assert option not in help_text
```

**Pattern to preserve** (`tests/sol_execbench/test_public_contract_guardrails.py` lines 223-277):

```python
assert v1_score.supported is True
assert v2_score.supported is True
assert v1_score.to_dict()["claim_level"] == "amd-native-derived"
assert v2_score.to_dict()["claim_level"] == "amd-native-derived"
assert "solar_derivation" not in v1_score.to_dict()["evidence_refs"]
assert "solar_derivation" not in v2_score.to_dict()["evidence_refs"]
assert "solar_derivation" not in v1_artifact.to_dict()
assert "solar_derivation" not in v2_artifact.to_dict()
```

### Naming And Style Conventions

- Dataclasses: `Solar...Evidence` for SOLAR sidecar models; `Operator...Estimate` for bound estimate models.
- Internal helpers: private snake_case helpers such as `_attention_estimate()`, `_convolution_estimate()`, `_family_formula_from_estimate()`, `_family_bytes_from_estimate()`.
- Formula kinds: lowercase snake_case strings ending in the thing measured, for example `gemm_flops`, `batched_gemm_flops`, `data_movement_bytes`.
- Warning prefixes: use existing prefixes from fixtures: `graph_warning:`, `estimate_warning:`, `inexact_operator:`, `unsupported_operator:`, `aggregate_degraded:`, `aggregate_unscored:`.
- Evidence keys: use deterministic `family:detail` strings in `required_evidence` and `missing_evidence`, matching existing keys such as `formula_inputs:op_1`, `bytes:op_1`, `axis:op_1`, `semantic_axes:input:x`.
- Ordering: sort nodes by `node_id`, sort groups by first estimate node ID, sort subroles by name, dedupe with `dict.fromkeys()` or `_unique_sorted()`.
- Serialization: convert tuple fields to lists in `to_dict()`, enum values to strings, and nested dataclasses through their own `to_dict()`.

## Anti-Patterns To Avoid

- Do not add new fields to `Definition`, `Workload`, `Trace`, canonical trace JSONL, or primary CLI output.
- Do not add public `sol-execbench` CLI options for SOLAR derivation in Phase 49.
- Do not change AMD-native score eligibility, `evidence_refs`, v1/v2 bound artifacts, or score claim levels.
- Do not infer from submitted candidate solution code, compile outputs, execution traces, or runtime behavior.
- Do not fabricate dimensions, dtypes, axes, mask semantics, grouping metadata, index semantics, or byte movement when absent.
- Do not replace `BoundGraph`, `BoundGraphNode`, `BoundTensor`, `OperatorWorkEstimate`, or `EstimateConfidence` with a parallel graph/estimate stack.
- Do not loosen strict parser contracts with permissive extra fields.
- Do not claim paper-scale dataset extraction, hosted leaderboard readiness, NVIDIA Blackwell/B200 equivalence, or new real-hardware validation.

## No Analog Found

| File / Concept | Role | Data Flow | Reason |
|----------------|------|-----------|--------|
| Dedicated compound attention formula model | service / model | transform | No existing complete compound attention estimator exists; build from `OperatorWorkEstimate`, `SolarSemanticGroupEvidence`, and `classify_solar_confidence()` patterns. |
| Dedicated convolution formula estimator | service | transform | `OpFamily.CONVOLUTION` exists, but there is no current `_convolution_estimate()` helper; follow `_gemm_estimate()` and `_pointwise_estimate()` structure. |
| Dedicated embedding/positional/gather/rotary byte estimator | service | transform | `OpFamily.EMBEDDING_POSITIONAL` exists, but no dedicated byte model exists; follow `_data_movement_estimate()` and dtype byte helpers. |

## Metadata

**Analog search scope:** `src/sol_execbench/core/scoring/`, `tests/sol_execbench/`, `tests/sol_execbench/fixtures/solar_derivation/`, `.planning/phases/49-high-confidence-family-modeling/`

**Files scanned:** 11

**Pattern extraction date:** 2026-05-23
