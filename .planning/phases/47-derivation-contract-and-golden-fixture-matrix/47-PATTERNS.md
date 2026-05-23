# Phase 47: Derivation Contract And Golden Fixture Matrix - Pattern Map

**Mapped:** 2026-05-23
**Files analyzed:** 22
**Analogs found:** 22 / 22

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `docs/internal/solar_derivation_contract.md` | docs/internal contract | transform | `docs/internal/rdna4_v1_9_validation_evidence.md` | exact |
| `tests/sol_execbench/solar_derivation_fixtures.py` | test utility | file-I/O | `src/sol_execbench/core/scoring/amd_sol_v2.py` | role-match |
| `tests/sol_execbench/test_solar_derivation_contract.py` | test | file-I/O, transform | `tests/sol_execbench/test_v1_9_validation_closure.py` | exact |
| `tests/sol_execbench/test_public_contract_guardrails.py` | test | request-response, transform | `tests/sol_execbench/test_public_contract_guardrails.py` | exact |
| `tests/sol_execbench/fixtures/solar_derivation/attention_positive.json` | fixture | file-I/O | `tests/sol_execbench/samples/*/definition.json` plus `tests/sol_execbench/test_amd_sol_v2.py` sidecar payload assertions | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/attention_degraded_partial_mask.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` degraded fixture semantics | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/attention_unsupported_dynamic_axes.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` unsupported semantics | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/moe_positive.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_bound_estimates.py` out-of-scope family vocabulary | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/moe_degraded_dynamic_routing.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` degraded warning semantics | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/moe_unsupported_taxonomy_only.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_bound_estimates.py` unsupported family tests | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/convolution_positive.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_bound_estimates.py` out-of-scope family vocabulary | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/convolution_degraded_missing_padding.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` degraded warning semantics | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/convolution_unsupported_dynamic_kernel.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` unsupported warning semantics | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_positive.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_bound_estimates.py` out-of-scope family vocabulary | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_degraded_missing_recurrence.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` degraded warning semantics | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_unsupported_custom_scan.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` unsupported warning semantics | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/embedding_positional_positive.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_bound_estimates.py` out-of-scope family vocabulary | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/embedding_positional_degraded_dynamic_indices.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` degraded warning semantics | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/embedding_positional_unsupported_missing_metadata.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` unsupported warning semantics | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/linear_projection_positive.json` | fixture | file-I/O | `src/sol_execbench/core/scoring/amd_bound_graph.py` family vocabulary and sidecar shape | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/linear_projection_degraded_missing_shape.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` degraded warning semantics | role-match |
| `tests/sol_execbench/fixtures/solar_derivation/linear_projection_unsupported_missing_metadata.json` | fixture | file-I/O | `tests/sol_execbench/test_amd_sol_v2.py` unsupported warning semantics | role-match |

## Pattern Assignments

### `docs/internal/solar_derivation_contract.md` (docs/internal contract, transform)

**Analog:** `docs/internal/rdna4_v1_9_validation_evidence.md`

**Internal evidence doc shape** (lines 1-13):
```markdown
# RDNA 4 v1.9 Validation Evidence

**Date:** 2026-05-23
**Scope:** v1.9 AMD SOL/SOLAR bound modeling completion
**Validation target:** RDNA 4 `gfx1200`

## Claim Boundary

This evidence record supports the v1.9 ROCm-port claim that AMD SOL/SOLAR bound
modeling artifacts are implemented and CPU-verified with RDNA 4-scoped hardware
model metadata. It does not claim NVIDIA B200 equivalence, upstream SOLAR
equivalence, leaderboard equivalence, CDNA 3 / MI300X real-hardware validation,
or CDNA 4 validation.
```

**Derived sidecar wording** (lines 32-57):
```markdown
## Derived Sample Run Shape

A small RDNA 4-scoped derived report run should emit:

- canonical trace JSON under the dataset output directory;
- AMD SOL bound artifact v2 sidecars under `out/amd-sol-bounds`;
- AMD-native score report JSON at `out/amd-score-report.json`.

Expected derived artifacts:

- `trace_jsonl`: remains the canonical benchmark output and is not modified by
  the v2 sidecars.
- `sol_execbench.amd_sol_bound.v2`: contains graph, rich estimate, per-op
  bound, aggregate, warning, coverage, and hardware model evidence.
```

**Inventory style** (lines 59-71):
```markdown
## Golden Coverage Inventory

The focused tests cover:

- matmul and batched matmul;
- elementwise chains and activations;
- reductions, normalization, and softmax;
- data movement, logical views, broadcast views, contiguous materialization,
  and dtype conversion;
- tuple outputs;
- unsupported operations;
```

**Apply:** Use the same short internal-doc sections: title, date/scope, claim boundary, fixture schema, family/state vocabulary, and golden matrix inventory. Do not put generated fixture expectations into canonical public schemas.

---

### `tests/sol_execbench/solar_derivation_fixtures.py` (test utility, file-I/O)

**Analog:** `src/sol_execbench/core/scoring/amd_sol_v2.py`

**Imports and contract constants** (lines 6-27):
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


AMD_SOL_V2_SCHEMA_VERSION = "sol_execbench.amd_sol_bound.v2"
AGGREGATE_STATUSES = frozenset({"scored", "degraded", "unscored"})
```

**Required-field parser pattern** (lines 171-207):
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
        source="AMD SOL v2 artifact",
    )
```

**Small validation helpers** (lines 502-523):
```python
def _require_keys(payload: dict[str, Any], required: set[str], *, source: str) -> None:
    for key in sorted(required):
        if key not in payload:
            raise ValueError(f"{source} missing required field: {key}")


def _parse_dict(payload: dict[str, Any], key: str, *, source: str) -> dict[str, Any]:
    value = payload[key]
    return _ensure_dict(value, source=f"{source}.{key}")


def _ensure_dict(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be an object")
    return value


def _parse_list(payload: dict[str, Any], key: str, *, source: str) -> list[Any]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{source}.{key} must be a list")
    return value
```

**String and enum validation style** (lines 526-602):
```python
def _parse_str_item(value: Any, *, source: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source} must be a string")
    if not value:
        raise ValueError(f"{source} must be non-empty")
    return value


def _parse_confidence(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> EstimateConfidence:
    raw = _parse_str(payload, key, source=source)
    try:
        return EstimateConfidence(raw)
    except ValueError as exc:
        valid_values = ", ".join(value.value for value in EstimateConfidence)
        raise ValueError(
            f"{source}.{key} has invalid confidence '{raw}', expected one of: {valid_values}"
        ) from exc
```

**Apply:** Keep this module under `tests/sol_execbench/`, import `json` and `Path`, set `FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "solar_derivation"`, load only sorted `*.json`, and validate required top-level and nested expectation fields. Avoid Pydantic and avoid executing fixture `reference` text.

---

### `tests/sol_execbench/test_solar_derivation_contract.py` (test, file-I/O/transform)

**Analog:** `tests/sol_execbench/test_v1_9_validation_closure.py`

**Repo-root plus text helper** (lines 1-10):
```python
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (REPO_ROOT / path).read_text()
```

**Doc content assertion pattern** (lines 13-28):
```python
def test_analysis_docs_explain_v2_sidecars_and_rdna4_scope():
    text = _text("docs/analysis.md")

    for expected in (
        "sol_execbench.amd_sol_bound.v2",
        "--amd-sol-bound-dir",
        "operator_work_estimates",
        "aggregate_bound",
        "coverage_summary",
        "hardware_validation_status",
        "model_validation_status",
        "RDNA 4 (`gfx1200`) is the only validation",
        "CDNA 3 / MI300X real-hardware validation and CDNA 4 validation",
    ):
        assert expected in text
```

**Forbidden claim assertion pattern** (lines 30-49):
```python
def test_v1_9_docs_do_not_make_forbidden_equivalence_or_validation_claims():
    combined = "\n".join(
        [
            _text("docs/analysis.md"),
            _text("docs/internal/rdna4_v1_9_validation_evidence.md"),
        ]
    )

    forbidden = (
        "NVIDIA B200 equivalence is validated",
        "upstream SOLAR equivalence is validated",
        "leaderboard equivalence is validated",
        "CDNA 3 / MI300X real-hardware validation is complete",
        "CDNA 4 validation is complete",
    )
    for phrase in forbidden:
        assert phrase not in combined
```

**Coverage inventory assertion pattern** (lines 52-72):
```python
def test_golden_bound_modeling_coverage_inventory_is_present():
    graph_tests = _text("tests/sol_execbench/test_amd_bound_graph.py")
    estimate_tests = _text("tests/sol_execbench/test_amd_bound_estimates.py")
    sol_v2_tests = _text("tests/sol_execbench/test_amd_sol_v2.py")
    evidence = _text("docs/internal/rdna4_v1_9_validation_evidence.md")
    combined = "\n".join([graph_tests, estimate_tests, sol_v2_tests, evidence])

    for expected in (
        "matmul",
        "batched matmul",
        "elementwise",
        "activation",
        "reduction",
        "normalization",
        "softmax",
        "data movement",
        "dtype conversion",
        "tuple outputs",
        "unsupported operations",
    ):
        assert expected in combined
```

**Apply:** Import `load_solar_derivation_fixtures` from the new test helper. Tests should cover fixture required fields, all six target families, positive/degraded/unsupported-or-negative classes per family, negative categories (`dynamic`, `partial`, `unsupported`, `taxonomy_only`, `missing_metadata`), stable warning prefixes, and doc claim boundaries.

---

### `tests/sol_execbench/test_public_contract_guardrails.py` (test, request-response/transform)

**Analog:** existing `tests/sol_execbench/test_public_contract_guardrails.py`

**Imports and repo path constants** (lines 1-35):
```python
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.data.definition import Definition

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPATIBILITY_INVENTORY = REPO_ROOT / "docs/internal/v1_4_compatibility_inventory.md"
```

**Primary CLI guardrail pattern** (lines 105-128):
```python
def test_primary_cli_does_not_expose_v1_6_derived_workflow_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for additive_non_primary_option in (
        "--amd-score-report",
        "--rocprofv3",
        "--timing-evidence",
        "--sol-bound",
        "--sol-bound-v2",
        "--bound-graph",
        "--extract-bound-graph",
        "--bound-estimates",
        "--formula-inputs",
        "--movement-bytes",
        "--operator-work-estimates",
        "--coverage-summary",
        "--aggregate-bound",
        "--hardware-model",
        "--amd-hardware-model",
        "--hardware-model-path",
    ):
        assert additive_non_primary_option not in help_text
```

**Canonical schema guardrail pattern** (lines 208-256):
```python
def test_definition_workload_trace_schemas_do_not_include_derived_artifact_fields():
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

    assert "hardware_model" not in definition.model_dump(mode="json")
    assert "bound_graph" not in definition.model_dump(mode="json")
    assert "operator_work_estimates" not in definition.model_dump(mode="json")
    assert "coverage_summary" not in definition.model_dump(mode="json")
    assert "aggregate_bound" not in definition.model_dump(mode="json")
    assert "bound_graph" not in workload.model_dump(mode="json")
    assert "operator_work_estimates" not in workload.model_dump(mode="json")
    assert "aggregate_bound" not in workload.model_dump(mode="json")
    assert "bound_graph" not in trace.model_dump(mode="json")
    assert "operator_work_estimates" not in trace.model_dump(mode="json")
    assert "aggregate_bound" not in trace.model_dump(mode="json")
```

**Inventory invariant style** (lines 282-292):
```python
def test_v1_4_compatibility_inventory_rejects_phase_19_public_drift():
    text = COMPATIBILITY_INVENTORY.read_text()
    for invariant in (
        "Do not add public `sol-execbench` CLI options or subcommands.",
        "Do not change Pydantic public field names",
        "Do not add fields to trace JSONL.",
        "Do not replace the eval driver",
        "Do not claim CDNA 3 hardware validation.",
        "Do not introduce the hip-execbench TypeScript/Zod runtime stack.",
    ):
        assert invariant in text
```

**Apply:** Add only targeted assertions for new SOLAR-derived contract terms staying out of canonical `Definition`, `Workload`, and `Trace` payloads and primary CLI options. Do not duplicate the full new fixture matrix in this file.

---

### `tests/sol_execbench/fixtures/solar_derivation/*.json` (fixture, file-I/O)

**Analog:** `tests/sol_execbench/test_amd_sol_v2.py` and `src/sol_execbench/core/scoring/amd_bound_graph.py`

**Existing family vocabulary** (`src/sol_execbench/core/scoring/amd_bound_graph.py` lines 27-44):
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

**Existing confidence vocabulary** (`src/sol_execbench/core/scoring/amd_hardware_models.py` lines 21-27):
```python
class EstimateConfidence(str, Enum):
    """Confidence level for hardware model estimates."""

    SUPPORTED = "supported"
    INEXACT = "inexact"
    UNSUPPORTED = "unsupported"
```

**Existing aggregate status vocabulary** (`src/sol_execbench/core/scoring/amd_sol_v2.py` lines 26-28):
```python
AMD_SOL_V2_SCHEMA_VERSION = "sol_execbench.amd_sol_bound.v2"
AGGREGATE_STATUSES = frozenset({"scored", "degraded", "unscored"})
```

**Sidecar payload fields to mirror conceptually** (`tests/sol_execbench/test_amd_sol_v2.py` lines 59-84):
```python
def test_v2_artifact_serializes_required_sidecar_fields():
    hardware = default_amd_hardware_models()["gfx1200"]

    artifact = build_amd_sol_bound_v2_artifact(
        _matmul_definition(),
        _matmul_workload(),
        hardware,
        hardware_model_ref="default_amd_hardware_models.gfx1200",
    )
    payload = artifact.to_dict()

    assert payload["schema_version"] == "sol_execbench.amd_sol_bound.v2"
    assert payload["derived"] is True
    assert payload["definition"] == "matmul_demo"
    assert payload["workload_uuid"] == "matmul-workload"
    assert payload["hardware_model_ref"] == "default_amd_hardware_models.gfx1200"
    assert payload["bound_graph"]["definition"] == "matmul_demo"
    assert payload["operator_work_estimates"][0]["formula_kind"] == "gemm_flops"
    assert payload["op_bounds"][0]["confidence"] == "supported"
    assert payload["aggregate_bound"]["status"] == "degraded"
    assert payload["coverage_summary"]["worst_confidence"] == "supported"
```

**Degraded warning semantics** (`tests/sol_execbench/test_amd_sol_v2.py` lines 179-202):
```python
assert artifact.aggregate_bound.status == "degraded"
assert artifact.aggregate_bound.scored is True
assert artifact.coverage_summary.total_ops == 3
assert artifact.coverage_summary.supported_ops == 0
assert artifact.coverage_summary.inexact_ops == 3
assert artifact.coverage_summary.unsupported_ops == 0
assert artifact.coverage_summary.worst_confidence.value == "inexact"
assert any(warning.startswith("inexact_operator:") for warning in artifact.warnings)
assert any(
    warning.startswith("aggregate_degraded:") for warning in artifact.warnings
)
```

**Unsupported warning semantics** (`tests/sol_execbench/test_amd_sol_v2.py` lines 225-236):
```python
assert artifact.coverage_summary.total_ops == 1
assert artifact.coverage_summary.unsupported_ops == 1
assert artifact.coverage_summary.worst_confidence.value == "unsupported"
assert artifact.op_bounds[0].confidence.value == "unsupported"
assert artifact.aggregate_bound.status == "unscored"
assert artifact.aggregate_bound.scored is False
assert any(
    warning.startswith("unsupported_operator:") for warning in artifact.warnings
)
assert any(
    warning.startswith("aggregate_unscored:") for warning in artifact.warnings
)
```

**Out-of-scope family baseline** (`tests/sol_execbench/test_amd_bound_estimates.py` lines 446-462):
```python
def test_out_of_scope_families_are_explicit_unsupported_estimates():
    for family in (
        OpFamily.ATTENTION,
        OpFamily.MOE,
        OpFamily.SSM_MAMBA,
        OpFamily.CONVOLUTION,
        OpFamily.EMBEDDING_POSITIONAL,
    ):
        graph = _single_node_graph(_unsupported_node(family))

        estimate = estimate_bound_work(graph)[0]

        assert estimate.confidence == EstimateConfidence.UNSUPPORTED
        assert estimate.flops == 0.0
        assert estimate.total_bytes == 0.0
        assert estimate.formula_kind == "unsupported"
        assert estimate.warnings == ("unsupported_family:torch.linalg.inv",)
```

**Apply:** Use one JSON file per fixture case. Each fixture should include stable fields like:

```json
{
  "case_id": "attention_positive_dense_qkv",
  "family": "attention",
  "fixture_class": "positive",
  "negative_category": null,
  "description": "Dense self-attention with explicit Q/K/V, softmax, PV, and output projection.",
  "source_kind": "reference_snippet",
  "reference": "def run(q, k, v, w_o): ...",
  "workload_axes": {"B": 2, "S": 16, "H": 4, "D": 32},
  "expectation": {
    "expected_family": "attention",
    "expected_subroles": ["q_projection", "k_projection", "v_projection"],
    "expected_confidence": "supported",
    "expected_status": "scored",
    "required_evidence": ["shape:batch", "shape:sequence", "shape:head_dim"],
    "missing_evidence": [],
    "warning_prefixes": [],
    "degradation_rationale": null
  },
  "scope_boundary": {
    "paper_scale_dataset": false,
    "hosted_leaderboard_ready": false,
    "nvidia_blackwell_b200_equivalence": false,
    "real_hardware_validation": false
  }
}
```

For degraded fixtures, use `expected_confidence: "inexact"`, `expected_status: "degraded"`, non-empty `missing_evidence` when applicable, non-empty `warning_prefixes` such as `inexact_operator:` or `aggregate_degraded:`, and a non-empty `degradation_rationale`.

For unsupported or negative fixtures, use `expected_confidence: "unsupported"`, `expected_status: "unscored"`, a non-empty `negative_category`, non-empty `missing_evidence`, warning prefixes such as `unsupported_operator:` and `aggregate_unscored:`, and no exception expectation.

## Shared Patterns

### Sidecar-Only Boundary

**Source:** `tests/sol_execbench/test_public_contract_guardrails.py` lines 208-256  
**Apply to:** `docs/internal/solar_derivation_contract.md`, `tests/sol_execbench/test_solar_derivation_contract.py`, `tests/sol_execbench/test_public_contract_guardrails.py`

```python
assert "bound_graph" not in definition.model_dump(mode="json")
assert "operator_work_estimates" not in definition.model_dump(mode="json")
assert "aggregate_bound" not in definition.model_dump(mode="json")
assert "bound_graph" not in workload.model_dump(mode="json")
assert "operator_work_estimates" not in workload.model_dump(mode="json")
assert "aggregate_bound" not in workload.model_dump(mode="json")
assert "bound_graph" not in trace.model_dump(mode="json")
assert "operator_work_estimates" not in trace.model_dump(mode="json")
assert "aggregate_bound" not in trace.model_dump(mode="json")
```

Phase 47 should add equivalent assertions for any new SOLAR derivation terms that must remain sidecar-only, such as derivation fixture expectations and SOLAR family evidence.

### Stable Warning Prefixes

**Source:** `src/sol_execbench/core/scoring/amd_sol_v2.py` lines 356-388  
**Apply to:** fixture JSON expectations and contract tests

```python
for warning in graph_warnings:
    warnings.append(f"graph_warning:{warning}")
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
```

Fixture tests should assert prefixes, not full warning bodies, so later implementation can keep useful detail after the prefix.

### Claim Boundary Docs

**Source:** `docs/internal/v1_4_compatibility_inventory.md` lines 122-130  
**Apply to:** internal contract doc and contract tests

```markdown
## Phase 19 Non-Goals

- Do not add public `sol-execbench` CLI options or subcommands.
- Do not change Pydantic public field names, required fields, or validation
  semantics.
- Do not add fields to trace JSONL.
- Do not replace the eval driver with a hip-execbench pipeline.
- Do not claim CDNA 3 hardware validation.
- Do not introduce the hip-execbench TypeScript/Zod runtime stack.
```

Use the same explicit non-goal style for: no paper-scale dataset extraction, no hosted leaderboard readiness, no NVIDIA Blackwell/B200 equivalence, and no new real-hardware validation.

### JSON File Reading

**Source:** `tests/sol_execbench/test_rocm_library_readiness_docs.py` lines 29-39 and 42-63  
**Apply to:** fixture loader and fixture-matrix tests

```python
compatibility_examples = [
    "examples/cutlass/gemm/solution_cutlass.json",
    "examples/cudnn/softmax/solution_cudnn.json",
    "examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json",
    "examples/cutile/jamba_attn_proj/solution_cutile.json",
]
for relative_path in compatibility_examples:
    data = json.loads((REPO_ROOT / relative_path).read_text())
    assert data["spec"]["languages"] == ["pytorch"], relative_path
    assert "compatibility example" in data["description"], relative_path
```

```python
for path in sorted((REPO_ROOT / "examples").glob("*/*/solution*.json")):
    data = json.loads(path.read_text())
    languages = set(data["spec"]["languages"])
    relative = str(path.relative_to(REPO_ROOT))
```

Use sorted glob iteration and include the path or case id in assertion messages where failures need review.

## No Analog Found

All Phase 47 files have usable analogs. There is no existing `tests/sol_execbench/fixtures/` directory, so the directory itself is new, but file loading, JSON parsing, docs assertions, sidecar vocabulary, and public guardrail patterns all have close codebase matches.

## Metadata

**Analog search scope:** `docs/internal/`, `tests/sol_execbench/`, `src/sol_execbench/core/scoring/`  
**Files scanned:** 44 listed files plus targeted `rg` matches for JSON loading, sidecar fields, aggregate statuses, family enums, and warning prefixes  
**Pattern extraction date:** 2026-05-23
