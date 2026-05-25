# Phase 73: Static Evidence Contract And Guardrails - Pattern Map

**Mapped:** 2026-05-25
**Files analyzed:** 6
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/bench/static_kernel_evidence.py` | model, utility | transform, request-response sidecar serialization | `src/sol_execbench/core/bench/rocm_profiler.py`; `src/sol_execbench/core/toolchain.py` | exact |
| `src/sol_execbench/core/data/contract.py` | model, config | request-response metadata serialization | `src/sol_execbench/core/data/contract.py` | exact |
| `tests/sol_execbench/test_static_kernel_evidence.py` | test | transform, request-response schema round trip | `tests/sol_execbench/test_rocm_profiler.py`; `tests/sol_execbench/test_toolchain_routing.py` | exact |
| `tests/sol_execbench/test_contract.py` | test | request-response metadata guardrail | `tests/sol_execbench/test_contract.py` | exact |
| `tests/sol_execbench/test_public_contract_guardrails.py` | test | request-response canonical-output guardrail | `tests/sol_execbench/test_public_contract_guardrails.py` | exact |
| `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` | test | transform no-mutation guardrail | `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` | exact |

## Pattern Assignments

### `src/sol_execbench/core/bench/static_kernel_evidence.py` (model, utility; transform/request-response)

**Analogs:** `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/data/base_model.py`

**Imports and schema constants pattern** (`src/sol_execbench/core/toolchain.py` lines 10-25):

```python
from __future__ import annotations

import fnmatch
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum

from pydantic import ConfigDict, Field

from .data.base_model import BaseModelWithDocstrings
from .environment import ProbeCompletedProcess


TOOLCHAIN_ROUTING_SCHEMA_VERSION = "sol_execbench.toolchain_routing.v1"
```

For the new bench module, copy the `Enum`, `ConfigDict`, `Field`, and `BaseModelWithDocstrings` style, but use absolute imports as in profiler tests when needed:

```python
from pydantic import ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings
```

**Base model pattern** (`src/sol_execbench/core/data/base_model.py` lines 29-32):

```python
class BaseModelWithDocstrings(BaseModel):
    """Base model with the attribute docstrings being extracted to the model JSON schema."""

    model_config = ConfigDict(use_attribute_docstrings=True)
```

Phase 73 should strengthen this locally for sidecar models with `ConfigDict(extra="forbid", frozen=True, strict=True, use_attribute_docstrings=True)` because the phase requires strict parsing and no silent field acceptance.

**Status enum pattern** (`src/sol_execbench/core/toolchain.py` lines 61-73):

```python
class ToolchainStatus(str, Enum):
    """Status vocabulary for routing decisions."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED_ARCH = "unsupported_arch"
    UNSUPPORTED_ARTIFACT = "unsupported_artifact"
    DEPRECATED = "deprecated"
    MIGRATED = "migrated"
    PLANNED = "planned"
    REJECTED = "rejected"
    FAILED = "failed"
```

Copy this `str, Enum` pattern for:

```python
class StaticKernelEvidenceStatus(str, Enum):
    COLLECTED = "collected"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    SKIPPED = "skipped"
```

Use another `str, Enum` for stable reason codes, grouped by prefix/category in names and values.

**Nested Pydantic model and list-shape pattern** (`src/sol_execbench/core/toolchain.py` lines 75-96):

```python
class ToolchainProbeResult(BaseModelWithDocstrings):
    """Bounded dynamic probe result for one tool."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    tool_id: str
    """Stable tool identifier."""
    command: list[str] = Field(default_factory=list)
    """Command attempted for the probe."""
    path: str | None = None
    """Resolved executable path when present."""
    status: ToolchainStatus
    """Probe status."""
    returncode: int | None = None
    """Process return code when a command was executed."""
    stdout_tail: str = ""
    """Bounded stdout tail."""
    stderr_tail: str = ""
    """Bounded stderr tail."""
    timeout_seconds: float | None = None
    """Probe timeout."""
```

Use this shape for `StaticKernelEvidenceArtifact`, `StaticKernelEvidenceToolRun`, `StaticKernelEvidenceKernel`, warning, source-reference, and classification models. Keep list fields present with `Field(default_factory=list)`.

**Authority boundary pattern** (`src/sol_execbench/core/toolchain.py` lines 175-197):

```python
class ToolchainRoutingReport(BaseModelWithDocstrings):
    """Toolchain routing report for one request."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    schema_version: str = TOOLCHAIN_ROUTING_SCHEMA_VERSION
    """Routing schema version."""
    generated_at: str
    """UTC timestamp when the report was generated."""
    diagnostic_only: bool = True
    """Routing is diagnostic metadata only."""
    correctness_authority: bool = False
    """Routing never proves correctness."""
    performance_authority: bool = False
    """Routing never proves performance."""
    leaderboard_authority: bool = False
    """Routing never proves leaderboard readiness."""
    request: ToolchainRoutingRequest
    """Requested routing target."""
    selected_tool_id: str | None = None
    """Selected tool when one is available."""
    decisions: list[ToolchainRoutingDecision] = Field(default_factory=list)
    """All considered routing decisions."""
```

Extend this exact pattern with the locked Phase 73 booleans: `diagnostic_only=True`, `correctness_authority=False`, `performance_authority=False`, `timing_authority=False`, `score_authority=False`, `paper_parity_authority=False`, and `leaderboard_authority=False`.

**Diagnostic sidecar payload pattern** (`src/sol_execbench/core/bench/rocm_profiler.py` lines 67-114):

```python
@dataclass(frozen=True)
class Rocprofv3ProfileResult:
    """Result metadata for optional `rocprofv3` artifact collection."""

    status: str
    command: tuple[str, ...]
    output_directory: Path
    output_file: str
    artifacts: tuple[Rocprofv3ProfileArtifact, ...] = ()
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    skipped_reason: str | None = None
    failed_reason: str | None = None
    working_directory: Path | None = None
    timeout_seconds: int | None = None
    profiler_available: bool | None = None
    schema_version: str = ROCPROFV3_PROFILE_SCHEMA_VERSION

    @property
    def succeeded(self) -> bool:
        """Whether profiler collection completed with registered artifacts."""
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostic sidecar payload."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "diagnostic_only": True,
            "score_authority": False,
            "command": list(self.command),
            "working_directory": (
                str(self.working_directory)
                if self.working_directory is not None
                else None
            ),
            "timeout_seconds": self.timeout_seconds,
            "output_directory": str(self.output_directory),
            "output_file": self.output_file,
            "profiler_available": self.profiler_available,
            "returncode": self.returncode,
            "stdout_tail": _tail(self.stdout),
            "stderr_tail": _tail(self.stderr),
            "skipped_reason": self.skipped_reason,
            "failed_reason": self.failed_reason,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }
```

Static evidence should be Pydantic rather than dataclass, but keep the same sidecar properties: schema version, status, authority booleans, command/tool provenance, bounded diagnostic strings, and artifacts as JSON-compatible data. Prefer `sidecar.model_dump(mode="json")` and a `to_dict()` wrapper if nearby contract consumers expect one.

**Nonfatal unavailable/failed helper pattern** (`src/sol_execbench/core/bench/rocm_profiler.py` lines 356-447):

```python
if not rocprofv3_available:
    return Rocprofv3ProfileResult(
        status="unavailable",
        command=tuple(command),
        output_directory=request.output_directory,
        output_file=request.output_file,
        skipped_reason=f"{request.executable} is not available on PATH",
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=False,
    )
...
except subprocess.TimeoutExpired as exc:
    return Rocprofv3ProfileResult(
        status="failed",
        command=tuple(command),
        output_directory=request.output_directory,
        output_file=request.output_file,
        stdout=_subprocess_text(exc.stdout),
        stderr=_subprocess_text(exc.stderr),
        failed_reason=(
            f"rocprofv3 command timed out after {request.timeout_seconds} seconds"
        ),
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=True,
    )
...
return Rocprofv3ProfileResult(
    status="success",
    command=tuple(command),
    output_directory=request.output_directory,
    output_file=request.output_file,
    artifacts=artifacts,
    returncode=completed.returncode,
    stdout=completed.stdout or "",
    stderr=completed.stderr or "",
    working_directory=request.working_directory,
    timeout_seconds=request.timeout_seconds,
    profiler_available=True,
)
```

Phase 73 should expose pure constructors for `collected`, `partial`, `unavailable`, `unsupported`, `failed`, and `skipped` states. They must return full valid sidecars with empty `artifacts`, `tool_runs`, `kernels`, `warnings`, and `source_refs` lists when no evidence is present.

---

### `src/sol_execbench/core/data/contract.py` (model/config; request-response metadata)

**Analog:** `src/sol_execbench/core/data/contract.py`

**Model and serialization pattern** (lines 32-64):

```python
class EvaluatorContract(BaseModelWithDocstrings):
    """SOL-owned evaluator and baseline export compatibility contract."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    schema_version: str
    """Schema identifier for the contract payload."""
    contract_version: str
    """Semantic contract version for HIP/SOL compatibility checks."""
    capabilities: list[str] = Field(default_factory=list)
    """Named capability tokens consumers must check explicitly."""
    trace_field_requirements: dict[str, list[str]]
    """Required trace field groups from the canonical trace JSONL contract."""
...
    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible contract payload."""
        return self.model_dump(mode="json")
```

**Capability list pattern** (lines 67-84):

```python
def build_evaluator_contract() -> EvaluatorContract:
    """Build the current SOL evaluator compatibility contract."""

    return EvaluatorContract(
        schema_version=SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION,
        contract_version=SOL_EXECBENCH_CONTRACT_VERSION,
        capabilities=[
            "trace.correctness.v1",
            "trace.timing.v1",
            "trace.scoring.v1",
            "baseline.measured_export.v1",
            "baseline.scoring_artifact.v1",
            "compatibility.metadata.v1",
            "failure_categories.v1",
            "runtime.evidence.v1",
            "profiling.evidence.v1",
            "toolchain.routing.v1",
        ],
```

Add only `"static_kernel_evidence.v1"` to `capabilities`. Preserve `SOL_EXECBENCH_CONTRACT_VERSION = "1.0"` (line 29) and do not add static evidence fields to canonical trace requirements.

**Boundary claims pattern** (lines 185-190):

```python
source_boundary_claims=[
    "SOL owns correctness, timing, scoring, trace semantics, and ROCm execution.",
    "HIP consumes this contract as external JSON and does not redefine benchmark truth.",
    "Contract metadata is emitted beside trace JSONL and is not part of Trace.",
    "Measured baseline registry evidence is distinct from SOL scoring baseline artifacts.",
    "Toolchain routing reports availability and provenance only; it is not correctness, performance, or score authority.",
],
```

Add a static-evidence boundary sentence only if needed by tests. Keep it clear that static evidence is optional diagnostic sidecar metadata and is not correctness, timing, scoring, paper-parity, or leaderboard authority.

---

### `tests/sol_execbench/test_static_kernel_evidence.py` (test; schema round trip and guardrails)

**Analogs:** `tests/sol_execbench/test_rocm_profiler.py`, `tests/sol_execbench/test_toolchain_routing.py`, `src/sol_execbench/core/data/solution.py`

**Import and focused unit-test style** (`tests/sol_execbench/test_rocm_profiler.py` lines 1-18):

```python
from __future__ import annotations

import subprocess
from collections.abc import Sequence

from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3CollectionRequest,
    Rocprofv3ProfileRequest,
    build_rocprofv3_profile_command,
    build_rocprofv3_command,
    build_timing_evidence,
    collect_rocprofv3_profile,
    collect_source_timing_evidence,
    discover_rocprofv3_artifacts,
    collect_rocprofv3_timing,
    parse_rocprofv3_csv,
    select_default_timing,
)
```

Use direct imports from `sol_execbench.core.bench.static_kernel_evidence` for all models/enums/helpers under test.

**Authority and JSON payload assertion pattern** (`tests/sol_execbench/test_rocm_profiler.py` lines 93-131):

```python
result = collect_rocprofv3_profile(request, runner=runner)
payload = result.to_dict()

assert result.succeeded is True
assert calls[0][-3:] == ["--", "python", "eval_driver.py"]
assert payload["schema_version"] == "sol_execbench.rocprofv3_profile.v1"
assert payload["diagnostic_only"] is True
assert payload["score_authority"] is False
assert payload["status"] == "success"
assert payload["working_directory"] == str(tmp_path)
assert payload["timeout_seconds"] == 30
assert payload["returncode"] == 0
assert payload["artifacts"][0]["kind"] == "rocpd"
assert payload["stderr_tail"] == "profiler note"
```

Adapt to assert `schema_version == "sol_execbench.static_kernel_evidence.v1"`, all Phase 73 authority booleans, status values, reason codes, stable empty list sections, and JSON round trip through `model_dump(mode="json")` plus `model_validate(...)`.

**Nonfatal unavailable/failed state pattern** (`tests/sol_execbench/test_rocm_profiler.py` lines 134-184):

```python
result = collect_rocprofv3_profile(
    request,
    rocprofv3_available=False,
    runner=runner,
)

assert result.succeeded is False
assert result.status == "unavailable"
assert result.returncode is None
assert result.skipped_reason == "rocprofv3 is not available on PATH"
assert result.to_dict()["artifacts"] == []
...
assert result.status == "failed"
assert payload["returncode"] == 22
assert payload["failed_reason"] == "rocprofv3 command failed with exit code 22"
assert payload["stderr_tail"] == "profiler failed"
assert payload["artifacts"][0]["kind"] == "trace_csv"
```

Write one test that parametrizes all top-level statuses: `collected`, `partial`, `unavailable`, `unsupported`, `failed`, and `skipped`. Each should build a valid sidecar and prove present empty lists.

**Routing report authority test pattern** (`tests/sol_execbench/test_toolchain_routing.py` lines 41-66):

```python
payload = report.model_dump(mode="json")

assert payload["schema_version"] == TOOLCHAIN_ROUTING_SCHEMA_VERSION
assert payload["selected_tool_id"] == "rocprofv3"
assert payload["diagnostic_only"] is True
assert payload["correctness_authority"] is False
assert payload["performance_authority"] is False
assert payload["leaderboard_authority"] is False
selected = [decision for decision in payload["decisions"] if decision["selected"]]
assert selected[0]["status"] == ToolchainStatus.AVAILABLE.value
```

Copy this assertion density for sidecar authority boundaries and selected status/reason assertions.

**Frozen Pydantic model pattern** (`src/sol_execbench/core/data/solution.py` lines 265-275):

```python
class Solution(BaseModelWithDocstrings):
    """A concrete implementation for a given Definition.
...
    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)
    """Treat Solution as immutable to safely memoize derived fields."""
```

Add tests proving static sidecars are frozen and strict: extra keys fail validation; mutating a sidecar field fails; incorrect scalar types do not coerce silently.

---

### `tests/sol_execbench/test_contract.py` (test; evaluator metadata guardrail)

**Analog:** `tests/sol_execbench/test_contract.py`

**Capability sets pattern** (lines 16-29):

```python
REQUIRED_CAPABILITIES = {
    "trace.correctness.v1",
    "trace.timing.v1",
    "trace.scoring.v1",
    "baseline.measured_export.v1",
    "baseline.scoring_artifact.v1",
    "compatibility.metadata.v1",
    "failure_categories.v1",
}
OPTIONAL_CAPABILITIES = {
    "runtime.evidence.v1",
    "profiling.evidence.v1",
    "toolchain.routing.v1",
}
```

Add `"static_kernel_evidence.v1"` to `OPTIONAL_CAPABILITIES`, not `REQUIRED_CAPABILITIES`.

**No contract bump pattern** (lines 32-51):

```python
def test_evaluator_contract_versions_are_stable():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert payload["schema_version"] == SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION
    assert payload["contract_version"] == SOL_EXECBENCH_CONTRACT_VERSION
    assert payload["schema_version"] == "sol_execbench.evaluator_contract.v1"
    assert payload["contract_version"] == "1.0"
...
def test_evaluator_contract_advertises_optional_evidence_without_bump():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert OPTIONAL_CAPABILITIES.issubset(set(payload["capabilities"]))
    assert payload["contract_version"] == "1.0"
```

Keep this exact structure. Add a specific assertion that static evidence is optional and the required contract version remains `"1.0"`.

**CLI contract payload parity pattern** (lines 103-111):

```python
def test_contract_cli_json_outputs_builder_payload_without_problem_directory():
    result = CliRunner().invoke(cli, ["contract", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    expected = build_evaluator_contract().model_dump(mode="json")
    assert payload == expected
    assert payload["schema_version"] == "sol_execbench.evaluator_contract.v1"
    assert REQUIRED_CAPABILITIES.issubset(set(payload["capabilities"]))
```

This already protects CLI contract output. Update expected capabilities only through the builder.

---

### `tests/sol_execbench/test_public_contract_guardrails.py` (test; canonical-output and primary CLI guardrails)

**Analog:** `tests/sol_execbench/test_public_contract_guardrails.py`

**Canonical keys and helper pattern** (lines 112-169):

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
...
def _sample_definition_workload_trace() -> tuple[Definition, Workload, Trace]:
    definition = make_definition(
        name="demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )
    workload = make_workload(axes={"N": 16}, inputs={"x": {"type": "random"}}, uuid="w1")
    trace = make_trace(
        definition="demo",
        workload=workload,
        solution="solution",
        evaluation=None,
    )
    return definition, workload, trace
```

Use the existing helper and constants for new static-evidence negative assertions.

**Exact canonical top-level key assertion** (lines 216-221):

```python
def test_canonical_definition_workload_trace_top_level_keys_are_exact():
    definition, workload, trace = _sample_definition_workload_trace()

    assert set(definition.model_dump(mode="json")) == CANONICAL_DEFINITION_KEYS
    assert set(workload.model_dump(mode="json")) == CANONICAL_WORKLOAD_KEYS
    assert set(trace.model_dump(mode="json")) == CANONICAL_TRACE_KEYS
```

Add static-evidence key names to the forbidden-key tests, not to canonical key sets.

**Forbidden key-space pattern** (lines 224-245):

```python
def test_canonical_trace_jsonl_excludes_derived_report_key_space():
    _, _, trace = _sample_definition_workload_trace()
    payload = trace.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True)
    forbidden_keys = {
        "baseline_export_fields",
        "capabilities",
        "compatibility_metadata_fields",
        "contract_version",
        "derived_evidence_refs",
        "failure_categories",
        "formula",
        "coverage",
        "runtime.evidence.v1",
        "environment_snapshot",
        "score_eligibility",
        *DERIVED_REPORT_EVIDENCE_REF_KEYS,
        *PHASE51_INTERNAL_PUBLIC_BOUNDARY_FIELDS,
    }

    assert _json_object_keys(payload).isdisjoint(forbidden_keys)
```

Add static-evidence forbidden names here, such as `"static_kernel_evidence"`, `"sol_execbench.static_kernel_evidence.v1"`, `"diagnostic_only"`, `"paper_parity_authority"`, `"leaderboard_authority"`, `"tool_runs"`, `"source_refs"`, and `"reason_code"`.

**Primary CLI exclusion pattern** (lines 274-297):

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

Add `--static-evidence` to this tuple or create a Phase 73-specific test with the same pattern. Do not add CLI flags in Phase 73.

**Sidecar-only schema exclusion pattern** (lines 961-1010):

```python
def test_definition_workload_trace_schemas_do_not_include_derived_artifact_fields():
...
    for field in PHASE51_INTERNAL_PUBLIC_BOUNDARY_FIELDS:
        assert field not in definition.model_dump(mode="json")
        assert field not in workload.model_dump(mode="json")
        assert field not in trace.model_dump(mode="json")
...
    assert "hardware_model_ref" not in trace.model_dump(mode="json")
```

Add a small static-evidence field tuple and assert it is absent from definition, workload, and trace payloads.

---

### `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` (test; no-mutation scoring/reporting guardrail)

**Analog:** `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py`

**Trace fixture pattern** (lines 27-55):

```python
def _workload(uuid: str) -> Workload:
    return Workload(axes={}, inputs={}, uuid=uuid)


def _trace(status: EvaluationStatus, latency_ms: float | None = None) -> Trace:
    performance = None
    correctness = None
    if status == EvaluationStatus.PASSED:
        performance = Performance(
            latency_ms=latency_ms or 1.0,
            reference_latency_ms=2.0,
            speedup_factor=2.0,
        )
        correctness = Correctness()
...
    return Trace(
        definition="demo",
        workload=_workload(status.value),
        solution="solution",
        evaluation=Evaluation(
            status=status,
            environment=Environment(hardware="AMD gfx1200", libs={}),
            timestamp="2026-05-22T00:00:00+08:00",
            correctness=correctness,
            performance=performance,
        ),
    )
```

Use this fixture style to compare trace payloads before and after any static-evidence model import/construction.

**No-mutation transform pattern** (lines 58-74):

```python
def test_trace_summary_does_not_mutate_public_trace_schema():
    traces = [
        _trace(EvaluationStatus.PASSED, 1.5),
        _trace(EvaluationStatus.PASSED, 2.5),
        _trace(EvaluationStatus.RUNTIME_ERROR),
    ]

    before = [trace.model_dump(mode="json") for trace in traces]
    summary = summarize_traces(traces)
    after = [trace.model_dump(mode="json") for trace in traces]

    assert before == after
    assert summary.total == 3
```

Add a static-evidence no-mutation test using the same `before`/`after` shape: construct or serialize a static sidecar while holding canonical traces and assert trace dumps remain identical.

**Derived evidence labeling pattern** (lines 77-105):

```python
before = [trace.model_dump(mode="json") for trace in traces]
report = build_evidence_report(traces, diagnostics)
after = [trace.model_dump(mode="json") for trace in traces]
payload = report.to_dict()

assert before == after
assert report.schema_version == DERIVED_EVIDENCE_SCHEMA_VERSION
assert report.derived is True
assert report.canonical_output == CANONICAL_BENCHMARK_OUTPUT
assert payload["canonical_output"] == "trace_jsonl"
```

Static evidence sidecar tests should make the same distinction: its payload is diagnostic sidecar data and must not replace or relabel canonical `trace_jsonl`.

**Score formula guardrail pattern** (lines 108-123):

```python
def test_sol_score_formula_stays_unchanged_for_existing_contract():
    assert sol_score(t_k=2.0, t_b=2.0, t_sol=1.0) == 0.5
    assert sol_score(t_k=1.0, t_b=2.0, t_sol=1.0) == 1.0
...
def test_benchmark_relative_scores_do_not_warn():
    interpretation = interpret_sol_score(0.75)
    assert interpretation.claim_level == "benchmark-relative"
    assert interpretation.warning is None
```

Add a guardrail that importing/building static evidence does not alter score formula output, score interpretation, or warning behavior.

## Shared Patterns

### Schema Versioning

**Source:** `src/sol_execbench/core/bench/rocm_profiler.py` lines 27-29 and `src/sol_execbench/core/data/contract.py` lines 28-29

```python
ROCPROFV3_EXECUTABLE = "rocprofv3"
ROCPROFV3_EVIDENCE_SCHEMA_VERSION = "sol_execbench.rocprofv3_timing.v1"
ROCPROFV3_PROFILE_SCHEMA_VERSION = "sol_execbench.rocprofv3_profile.v1"
...
SOL_EXECBENCH_CONTRACT_SCHEMA_VERSION = "sol_execbench.evaluator_contract.v1"
SOL_EXECBENCH_CONTRACT_VERSION = "1.0"
```

Apply as `STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION = "sol_execbench.static_kernel_evidence.v1"`. Do not change `SOL_EXECBENCH_CONTRACT_VERSION`.

### JSON Serialization

**Source:** `src/sol_execbench/core/data/contract.py` lines 62-64

```python
def to_dict(self) -> dict[str, Any]:
    """Return the JSON-compatible contract payload."""
    return self.model_dump(mode="json")
```

Use `model_dump(mode="json")` for Pydantic sidecars and test round trips with `model_validate(payload)`.

### Authority Boundaries

**Source:** `src/sol_execbench/core/toolchain.py` lines 184-191 and `tests/sol_execbench/test_toolchain_routing.py` lines 58-64

```python
diagnostic_only: bool = True
"""Routing is diagnostic metadata only."""
correctness_authority: bool = False
"""Routing never proves correctness."""
performance_authority: bool = False
"""Routing never proves performance."""
leaderboard_authority: bool = False
"""Routing never proves leaderboard readiness."""
```

```python
assert payload["diagnostic_only"] is True
assert payload["correctness_authority"] is False
assert payload["performance_authority"] is False
assert payload["leaderboard_authority"] is False
```

Add `timing_authority`, `score_authority`, and `paper_parity_authority` for static evidence.

### Stable Empty Sections

**Source:** `src/sol_execbench/core/toolchain.py` lines 111-127 and 196-197

```python
evidence_levels: list[ToolchainEvidenceLevel] = Field(default_factory=list)
artifact_types: list[ToolchainArtifactType] = Field(default_factory=list)
hardware_generations: list[str] = Field(default_factory=list)
gpu_arch_patterns: list[str] = Field(default_factory=list)
expected_binaries: list[str] = Field(default_factory=list)
probe_command: list[str] = Field(default_factory=list)
source_refs: list[str] = Field(default_factory=list)
...
decisions: list[ToolchainRoutingDecision] = Field(default_factory=list)
```

All static-evidence section fields should use `Field(default_factory=list)`, including `artifacts`, `tool_runs`, `kernels`, `warnings`, and `source_refs`.

### Canonical Trace Isolation

**Source:** `tests/sol_execbench/test_public_contract_guardrails.py` lines 216-245

```python
assert set(definition.model_dump(mode="json")) == CANONICAL_DEFINITION_KEYS
assert set(workload.model_dump(mode="json")) == CANONICAL_WORKLOAD_KEYS
assert set(trace.model_dump(mode="json")) == CANONICAL_TRACE_KEYS
...
assert _json_object_keys(payload).isdisjoint(forbidden_keys)
```

Add static-evidence names to the forbidden key space. Do not modify `Trace`, `Definition`, or `Workload` models.

### Primary CLI Isolation

**Source:** `tests/sol_execbench/test_public_contract_guardrails.py` lines 274-297

```python
result = CliRunner().invoke(cli, ["--help"])
assert result.exit_code == 0
help_text = result.output

for additive_non_primary_option in (
    "--amd-score-report",
    "--rocprofv3",
    "--timing-evidence",
    ...
):
    assert additive_non_primary_option not in help_text
```

Add `--static-evidence` to the exclusion list. Phase 73 must not add public CLI flags.

## No Analog Found

All requested files have close analogs in the codebase. The only pattern not already present exactly is strict Pydantic `extra="forbid", strict=True` on local models; this is a phase requirement from `73-RESEARCH.md`, and it should be layered onto the existing `BaseModelWithDocstrings` pattern.

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| None | n/a | n/a | All files have exact or near-exact local analogs. |

## Metadata

**Analog search scope:** `src/sol_execbench/core/bench`, `src/sol_execbench/core/data`, `src/sol_execbench/core/toolchain.py`, `tests/sol_execbench`
**Files scanned:** 8 primary analog files plus project instructions and phase context
**Pattern extraction date:** 2026-05-25
