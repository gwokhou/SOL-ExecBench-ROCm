# Phase 78: Matrix Contract And Claim Guardrails - Pattern Map

**Mapped:** 2026-05-28
**Files analyzed:** 5 likely new/modified files
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/compatibility.py` | model, service, utility | transform, request-response preflight classification | `src/sol_execbench/core/bench/static_kernel_evidence.py` | exact |
| `tests/sol_execbench/test_rocm_compatibility_matrix.py` | test | transform, request-response contract checks | `tests/sol_execbench/test_static_kernel_evidence.py` | exact |
| `tests/sol_execbench/test_matrix_claim_guardrails.py` | test | transform, guardrail assertions | `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` | exact |
| `tests/sol_execbench/test_public_contract_guardrails.py` | test | public contract guardrail | `tests/sol_execbench/test_public_contract_guardrails.py` | exact |
| `docs/user/CLAIMS.md` | documentation | claim-boundary wording guardrail | `docs/user/CLAIMS.md` and `tests/sol_execbench/test_research_release_docs.py` | role-match |

## Pattern Assignments

### `src/sol_execbench/core/compatibility.py` (model/service/utility, transform)

**Analog:** `src/sol_execbench/core/bench/static_kernel_evidence.py`

**Use when:** Defining `sol_execbench.rocm_compatibility_matrix.v1`, `MatrixEntry`, `Target`, observed evidence models, bounded status/reason enums, claim flags, and pure classification helpers.

**Imports and strict model config pattern** (lines 5-20, 35-41):
```python
"""Strict diagnostic-only static kernel evidence sidecar contract."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings

STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION = "sol_execbench.static_kernel_evidence.v1"
_STATIC_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)
```

Copy this shape for a private compatibility model config. The matrix contract should likely use `extra="forbid"`, `frozen=True`, `strict=True`, and `use_attribute_docstrings=True` because Phase 78 explicitly wants bounded, strict public schema semantics.

**Bounded enum and string coercion pattern** (lines 44-79):
```python
class StaticKernelEvidenceStatus(str, Enum):
    """Aggregate and per-artifact status vocabulary."""

    COLLECTED = "collected"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    SKIPPED = "skipped"


class StaticKernelEvidenceReasonCode(str, Enum):
    """Stable reason-code vocabulary for static evidence outcomes."""

    STATIC_EVIDENCE_NOT_REQUESTED = "static_evidence_not_requested"
    STATIC_EVIDENCE_COLLECTED = "static_evidence_collected"
    TOOLCHAIN_UNAVAILABLE = "toolchain_unavailable"
    UNSUPPORTED_SOLUTION_TYPE = "unsupported_solution_type"
    EXTRACTOR_FAILED = "extractor_failed"


def _validate_status(value: object) -> object:
    if isinstance(value, str):
        return StaticKernelEvidenceStatus(value)
    return value
```

Use the same `str, Enum` style for `MatrixCompatibilityStatus` with exactly:
`host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, `not_tested`.
Use a separate reason-code enum so future phases can add precise classification without widening status.

**Diagnostic-only authority flags pattern** (lines 215-260):
```python
class StaticKernelEvidenceSidecar(BaseModelWithDocstrings):
    """Strict diagnostic-only static kernel evidence sidecar."""

    model_config = _STATIC_MODEL_CONFIG

    schema_version: Literal[STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION] = (
        STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION
    )
    status: StaticKernelEvidenceStatusField
    reason_code: StaticKernelEvidenceReasonCodeField
    diagnostic_only: Literal[True] = True
    correctness_authority: Literal[False] = False
    performance_authority: Literal[False] = False
    timing_authority: Literal[False] = False
    score_authority: Literal[False] = False
    paper_parity_authority: Literal[False] = False
    leaderboard_authority: Literal[False] = False

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""
        return self.model_dump(mode="json")
```

For Phase 78, preserve the diagnostic-only pattern but adjust flags to compatibility wording. At minimum include `diagnostic_only: Literal[True]`, `score_authority: Literal[False]`, `paper_parity_authority: Literal[False]`, and `leaderboard_authority: Literal[False]`. Add a native-host claim flag so Docker entries cannot imply native host validation.

**Builder pattern** (lines 264-280):
```python
def build_static_kernel_evidence_sidecar(
    *,
    status: StaticKernelEvidenceStatus | str,
    reason_code: StaticKernelEvidenceReasonCode | str,
    classification: StaticKernelEvidenceClassification | None = None,
    artifacts: Sequence[StaticKernelEvidenceArtifact] = (),
    tool_runs: Sequence[StaticKernelEvidenceToolRun] = (),
) -> StaticKernelEvidenceSidecar:
    """Build a strict static evidence sidecar without collecting artifacts."""

    return StaticKernelEvidenceSidecar(
        status=status,
        reason_code=reason_code,
        classification=classification or StaticKernelEvidenceClassification(),
        artifacts=list(artifacts),
        tool_runs=list(tool_runs),
    )
```

Use this for fixture-backed construction, for example `build_matrix_entry(...)` or `classify_matrix_entry(...)`, keeping it pure and CPU-safe.

**Optional observed evidence model pattern** from `src/sol_execbench/core/environment.py` (lines 28-160):
```python
ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION = "sol_execbench.environment_snapshot.v1"


class EnvironmentEvidenceStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class PytorchRocmSummary(BaseModelWithDocstrings):
    model_config = ConfigDict(use_attribute_docstrings=True)

    available: bool
    torch_version: str | None = None
    hip_version: str | None = None
    cuda_version: str | None = None
    device_count: int | None = None
    device_name: str | None = None
    gfx_target: str | None = None
    error: str | None = None


class EnvironmentSnapshot(BaseModelWithDocstrings):
    schema_version: str = ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION
    generated_at: str
    collection_status: EnvironmentEvidenceStatus
    tools: dict[str, ToolProbeResult] = Field(default_factory=dict)
    gpus: list[GpuEnvironmentSummary] = Field(default_factory=list)
    rocm: RocmEnvironmentSummary = Field(default_factory=RocmEnvironmentSummary)
    pytorch: PytorchRocmSummary | None = None
```

Copy the nested optional evidence style for observed host/container/Python/toolchain/GPU evidence. Do not merge observed fields into `Target`; keep requested and observed models separate.

**Routing/classification report pattern** from `src/sol_execbench/core/toolchain.py` (lines 61-183, 376):
```python
class ToolchainStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED_ARCH = "unsupported_arch"
    FAILED = "failed"


class ToolchainRoutingRequest(BaseModelWithDocstrings):
    evidence_level: ToolchainEvidenceLevel
    artifact_type: ToolchainArtifactType = ToolchainArtifactType.NONE
    gpu_architecture: str | None = None
    hardware_generation: str | None = None
    rocm_version: str | None = None


class ToolchainRoutingDecision(BaseModelWithDocstrings):
    tool_id: str
    status: ToolchainStatus
    reason_code: str
    reason: str
    selected: bool = False


class ToolchainRoutingReport(BaseModelWithDocstrings):
    schema_version: str = TOOLCHAIN_ROUTING_SCHEMA_VERSION
    generated_at: str
    diagnostic_only: bool = True
    correctness_authority: bool = False
    performance_authority: bool = False
    leaderboard_authority: bool = False
    request: ToolchainRoutingRequest
    selected_tool_id: str | None = None
    decisions: list[ToolchainRoutingDecision] = Field(default_factory=list)
```

Use this request/decision/report shape for matrix classification output if a helper returns status, reasons, and benchmark allowance. `benchmark_allowed` should be an explicit field derived from status and override settings, not inferred by callers from free text.

### `tests/sol_execbench/test_rocm_compatibility_matrix.py` (test, transform contract checks)

**Analog:** `tests/sol_execbench/test_static_kernel_evidence.py`

**Use when:** Testing schema serialization, strict unknown-field rejection, bounded matrix statuses, reason codes, claim flags, and representative nested evidence.

**Imports and expected vocabulary constants** (lines 1-53):
```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.static_kernel_evidence import (
    STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION,
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceStatus,
    build_static_kernel_evidence_sidecar,
)

EXPECTED_STATUSES = {
    "collected",
    "partial",
    "unavailable",
    "unsupported",
    "failed",
    "skipped",
}
```

Copy this test structure with `EXPECTED_MATRIX_STATUSES` locked to the six Phase 78 statuses.

**Representative fixture pattern** (lines 56-87):
```python
def _representative_sidecar() -> StaticKernelEvidenceSidecar:
    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        artifacts=[
            StaticKernelEvidenceArtifact(
                artifact_id="kernel-object",
                artifact_type="elf_object",
                status=StaticKernelEvidenceStatus.COLLECTED,
                reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
                source_path="build/kernel.o",
            )
        ],
    )
```

Use a `_representative_matrix_entry()` fixture with a `target` object, `observed` object, status, reason code, artifacts, and claim flags.

**Round-trip and strictness tests** (lines 89-110):
```python
def test_static_kernel_evidence_round_trips_strict_json_payload():
    sidecar = _representative_sidecar()
    payload = sidecar.model_dump(mode="json")

    assert payload["schema_version"] == STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION
    assert StaticKernelEvidenceSidecar.model_validate(payload) == sidecar


def test_static_kernel_evidence_rejects_unknown_top_level_and_nested_fields():
    payload = _representative_sidecar().model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        StaticKernelEvidenceSidecar.model_validate(payload)
```

Add the same top-level and nested unknown-field rejection for `MatrixEntry`, `Target`, and `observed`.

**Locked vocabulary and authority tests** (lines 112-129):
```python
def test_status_and_reason_code_vocabularies_are_locked():
    assert {status.value for status in StaticKernelEvidenceStatus} == EXPECTED_STATUSES
    assert {reason.value for reason in StaticKernelEvidenceReasonCode} == (
        EXPECTED_REASON_CODES
    )


def test_authority_fields_are_diagnostic_only_and_false_for_benchmark_truth():
    payload = _representative_sidecar().model_dump(mode="json")

    assert payload["diagnostic_only"] is True
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False
```

Use this exact assertion style for matrix claim flags. Include a negative Docker case asserting Docker evidence never yields `host_validated`.

### `tests/sol_execbench/test_matrix_claim_guardrails.py` (test, guardrail assertions)

**Analog:** `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py`

**Use when:** Proving compatibility evidence does not mutate canonical trace/scoring semantics and that mixed-version/debug override cannot produce clean claims.

**Canonical trace non-mutation pattern** (lines 63-80):
```python
def test_trace_summary_does_not_mutate_public_trace_schema():
    traces = [
        _trace(EvaluationStatus.PASSED, 1.5),
        _trace(EvaluationStatus.RUNTIME_ERROR),
    ]

    before = [trace.model_dump(mode="json") for trace in traces]
    summary = summarize_traces(traces)
    after = [trace.model_dump(mode="json") for trace in traces]

    assert before == after
    assert summary.total == 2
```

Adapt this to assert constructing/classifying a `MatrixEntry` leaves `Trace`, `EvaluationStatus`, correctness, timing, scoring, and public trace keys unchanged.

**Derived evidence labeling pattern** (lines 82-116):
```python
def test_derived_evidence_report_labels_noncanonical_output():
    before = [trace.model_dump(mode="json") for trace in traces]
    report = build_evidence_report(traces, diagnostics)
    after = [trace.model_dump(mode="json") for trace in traces]
    payload = report.to_dict()

    assert before == after
    assert report.derived is True
    assert payload["canonical_output"] == "trace_jsonl"
```

Matrix compatibility evidence should use the same sidecar/noncanonical assertion: it is diagnostic compatibility evidence, not a canonical trace JSONL field.

**Claim warning and sidecar boundary pattern** (lines 118-149):
```python
def test_amd_native_claims_receive_guardrail_warning():
    interpretation = interpret_sol_score(0.75, amd_native_claim=True)
    assert interpretation.claim_level == "amd-native-performance"
    assert interpretation.warning == AMD_PERFORMANCE_CLAIM_WARNING


def test_static_evidence_sidecar_construction_does_not_mutate_trace_or_scoring():
    before = trace.model_dump(mode="json")
    sidecar = build_static_kernel_evidence_sidecar(...)
    score = sol_score(t_k=1.0, t_b=2.0, t_sol=1.0)
    after = trace.model_dump(mode="json")

    assert sidecar.diagnostic_only is True
    assert sidecar.score_authority is False
    assert before == after
    assert "static_kernel_evidence" not in before
```

For Phase 78, add explicit cases:
- Docker entry with matching host/container versions may be `container_validated`, never `host_validated`.
- `mixed_version` defaults to `benchmark_allowed is False`.
- Mixed-version debug override may make probes allowed but still leaves `container_validated`, `host_validated`, score, paper-parity, and leaderboard authority false.

### `tests/sol_execbench/test_public_contract_guardrails.py` (test, public contract guardrail)

**Analog:** Existing same file.

**Use when:** Guarding that the new matrix contract does not add fields to canonical `Definition`, `Workload`, or `Trace` schemas and that public contract wording/capabilities remain explicit.

**Canonical key guardrail pattern** (lines 112-124, 216-221):
```python
CANONICAL_DEFINITION_KEYS = {
    "name",
    "op_type",
    "axes",
    "custom_inputs_entrypoint",
    "inputs",
    "outputs",
    "reference",
    "description",
    "hf_id",
}
CANONICAL_WORKLOAD_KEYS = {"axes", "inputs", "uuid", "tolerance"}
CANONICAL_TRACE_KEYS = {"definition", "workload", "solution", "evaluation"}


def test_canonical_definition_workload_trace_top_level_keys_are_exact():
    definition, workload, trace = _sample_definition_workload_trace()

    assert set(definition.model_dump(mode="json")) == CANONICAL_DEFINITION_KEYS
    assert set(workload.model_dump(mode="json")) == CANONICAL_WORKLOAD_KEYS
    assert set(trace.model_dump(mode="json")) == CANONICAL_TRACE_KEYS
```

Keep this as the public contract guard. If the matrix contract is exposed through evaluator metadata, add a capability token but do not add matrix fields to canonical trace/workload/definition payloads.

**Recursive key scanner pattern** (lines 140-151):
```python
def _json_object_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for nested in value.values():
            keys.update(_json_object_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_json_object_keys(nested))
        return keys
    return set()
```

Use this if tests need to assert matrix-only keys such as `target`, `observed`, `benchmark_allowed`, or compatibility claim flags do not leak into trace/scoring payloads.

**Evaluator contract capability pattern** from `src/sol_execbench/core/data/contract.py` (lines 28-83):
```python
SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION = "sol_execbench.evaluator_contract.v1"


class EvaluatorContract(BaseModelWithDocstrings):
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    schema_version: str
    contract_version: str
    capabilities: list[str] = Field(default_factory=list)
    compatibility_metadata_fields: list[str]
    source_boundary_claims: list[str]


def build_evaluator_contract() -> EvaluatorContract:
    return EvaluatorContract(
        schema_version=SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION,
        capabilities=[
            "trace.correctness.v1",
            "runtime.evidence.v1",
            "profiling.evidence.v1",
            "toolchain.routing.v1",
            "static_kernel_evidence.v1",
        ],
        ...
    )
```

If Phase 78 exposes a capability, follow this existing list pattern with a token like `rocm_compatibility_matrix.v1`; avoid changing canonical trace requirements.

### `docs/user/CLAIMS.md` (documentation, claim-boundary wording)

**Analog:** `docs/user/CLAIMS.md` and `tests/sol_execbench/test_research_release_docs.py`

**Use when:** Adding wording guardrails for container user-space validation versus native host validation.

**Claims boundary source pattern** (docs lines 7-18, 19-36, 49-69):
```markdown
## What Can Be Claimed Today

| Claim level | Allowed claim | Required evidence |
| --- | --- | --- |
| Static Kernel Evidence | A HIP/C++ run produced diagnostic static build artifacts and bounded routed extractor outputs. | ... diagnostic-only authority flags. |

## What Must Not Be Claimed Yet

- Static Kernel Evidence as correctness authority, performance authority,
  timing authority, score authority, paper-parity authority, or leaderboard
  authority.

## Reporting Language

- Say "Static Kernel Evidence" only for diagnostic static-analysis sidecars and
  persisted artifacts.
- Do not say "hardware validated" for schema/build support alone.
```

Add matrix-specific wording in the same sections:
- Allowed: "container ROCm user-space validated on recorded host driver/devices".
- Forbidden: Docker entries as native host ROCm validation.
- Forbidden: `host_validated` without direct native-host evidence.
- Forbidden: compatibility sidecars as score, paper-parity, or leaderboard authority.

**Docs guardrail test pattern** from `tests/sol_execbench/test_research_release_docs.py` (lines 9-39, 136-168):
```python
def _read_doc(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def test_claims_doc_defines_allowed_and_unsupported_claims():
    text = _read_doc("docs/user/CLAIMS.md")

    for allowed in (
        "ROCm-port evidence",
        "Runtime evidence",
        "Static Kernel Evidence",
    ):
        assert allowed in text

    for unsupported in (
        "NVIDIA B200",
        "official leaderboard parity",
        "full 235-problem paper validation",
    ):
        assert unsupported in text


def test_static_kernel_evidence_docs_define_usage_and_boundaries():
    text = _read_doc("docs/user/static_kernel_evidence.md")
    claims = _read_doc("docs/user/CLAIMS.md")

    for boundary in (
        "correctness authority",
        "performance authority",
        "score authority",
        "paper-parity authority",
        "leaderboard authority",
    ):
        assert boundary in text
        assert boundary in claims or boundary.replace("-", " ") in claims
```

Use this style for CPU-safe docs wording tests. The Phase 78 test should check exact presence of "container ROCm user-space" and "native host validation" boundary language, plus negative/forbidden wording if the docs add an explicit forbidden list.

## Shared Patterns

### Public Pydantic Schemas
**Source:** `src/sol_execbench/core/data/base_model.py` lines 22-30 and `src/sol_execbench/core/bench/static_kernel_evidence.py` lines 35-41  
**Apply to:** `src/sol_execbench/core/compatibility.py`

Use `BaseModelWithDocstrings`, `Field(default_factory=...)`, explicit schema version constants, and strict `ConfigDict` for the new matrix contract. Public model fields should have attribute docstrings so JSON schema remains documented.

### Bounded Status And Reason Codes
**Source:** `src/sol_execbench/core/bench/static_kernel_evidence.py` lines 44-79 and `src/sol_execbench/core/toolchain.py` lines 61-73  
**Apply to:** compatibility statuses, reason codes, preflight classification output

Use `str, Enum` with uppercase members and string values. Lock values in tests with `{status.value for status in EnumClass}`.

### Diagnostic-Only Authority Flags
**Source:** `src/sol_execbench/core/bench/static_kernel_evidence.py` lines 215-260; `src/sol_execbench/core/toolchain.py` lines 175-183  
**Apply to:** `MatrixEntry`, aggregate matrix report if introduced, and all claim guardrail tests

Compatibility sidecars should state diagnostic authority directly in the payload instead of relying on docs. `Literal[False]` is preferred for strict Phase 78 flags that must never be true.

### Requested vs Observed Evidence Separation
**Source:** `src/sol_execbench/core/toolchain.py` lines 133-173; `src/sol_execbench/core/environment.py` lines 42-160  
**Apply to:** `Target`, `ObservedCompatibilityEvidence`, host/container/Python/toolchain/GPU models

Follow request/decision and snapshot/summary separation. `Target` should represent requested validation configuration. `observed` should represent host/container/dependency/toolchain/GPU facts.

### CPU-Safe Contract Tests
**Source:** `tests/sol_execbench/test_static_kernel_evidence.py` lines 89-129, 153-181; `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` lines 63-149  
**Apply to:** all Phase 78 tests

Build representative objects in memory. Do not require Docker, ROCm devices, PyTorch ROCm imports, or subprocess probes for Phase 78.

### Public Contract Non-Mutation
**Source:** `tests/sol_execbench/test_public_contract_guardrails.py` lines 112-124, 140-151, 216-221  
**Apply to:** matrix contract tests and public contract guardrails

Assert the new compatibility model does not alter canonical trace/workload/definition top-level fields or scoring semantics.

### Documentation Claim Wording
**Source:** `docs/user/CLAIMS.md` lines 19-36 and 49-69; `tests/sol_execbench/test_research_release_docs.py` lines 9-39 and 136-168
**Apply to:** `docs/user/CLAIMS.md` and claim guardrail tests

Use exact wording assertions for allowed and forbidden claim language. Phrase positive claims narrowly and keep forbidden claims explicit.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| None | - | - | Existing sidecar schemas, status enums, guardrail tests, and docs claim-boundary tests provide sufficient analogs. |

## Metadata

**Analog search scope:** `src/sol_execbench/core/`, `src/sol_execbench/core/data/`, `src/sol_execbench/core/bench/`, `tests/sol_execbench/`, `docs/`
**Files scanned:** focused `rg` and `find` scans over source, tests, docs, and planning context
**Pattern extraction date:** 2026-05-28
