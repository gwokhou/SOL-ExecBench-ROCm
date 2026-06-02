# Phase 41: Bound Model Contract And Hardware Artifacts - Pattern Map

**Mapped:** 2026-05-23
**Files analyzed:** 13
**Analogs found:** 13 / 13

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/scoring/amd_hardware_models.py` | service / utility | file-I/O + transform | `src/sol_execbench/core/scoring/baseline_artifact.py` | exact |
| `src/sol_execbench/core/scoring/amd_sol.py` | model / service | transform | `src/sol_execbench/core/scoring/amd_sol.py` | exact |
| `src/sol_execbench/core/scoring/amd_score.py` | service | transform | `src/sol_execbench/core/scoring/amd_score.py` | exact |
| `src/sol_execbench/core/scoring/__init__.py` | package facade | transform | `src/sol_execbench/core/scoring/__init__.py` | exact |
| `src/sol_execbench/data/__init__.py` | package marker | package resource | `src/sol_execbench/core/scoring/__init__.py` | role-match |
| `src/sol_execbench/data/amd_hardware_models/__init__.py` | package marker | package resource | `src/sol_execbench/core/scoring/__init__.py` | role-match |
| `src/sol_execbench/data/amd_hardware_models/gfx1200.json` | config / artifact | file-I/O | `src/sol_execbench/core/scoring/baseline_artifact.py` | role-match |
| `pyproject.toml` | config | package build | `pyproject.toml` | exact |
| `tests/sol_execbench/test_amd_hardware_models.py` | test | file-I/O + validation | `tests/sol_execbench/test_amd_sol_bounds.py` | role-match |
| `tests/sol_execbench/test_amd_sol_bounds.py` | test | transform + contract | `tests/sol_execbench/test_amd_sol_bounds.py` | exact |
| `tests/sol_execbench/test_amd_native_score.py` | test | transform + guardrail | `tests/sol_execbench/test_amd_native_score.py` | exact |
| `tests/sol_execbench/test_public_contract_guardrails.py` | test | contract guardrail | `tests/sol_execbench/test_public_contract_guardrails.py` | exact |
| `docs/analysis.md` | documentation | contract text | `docs/analysis.md` + guardrail tests | exact |

## Pattern Assignments

### `src/sol_execbench/core/scoring/amd_hardware_models.py` (service / utility, file-I/O + transform)

**Analog:** `src/sol_execbench/core/scoring/baseline_artifact.py`

**Imports pattern** (lines 6-11):
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
```

Use the same small-stdlib surface, plus `from importlib import resources` for packaged JSON resource reads.

**Dataclass artifact pattern** (lines 17-41):
```python
@dataclass(frozen=True)
class ScoringBaselineEntry:
    """Optimized baseline timing for one definition/workload pair."""

    definition: str
    workload_uuid: str
    latency_ms: float
    solution: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "latency_ms": self.latency_ms,
        }
        if self.solution:
            payload["solution"] = self.solution
        if self.source:
            payload["source"] = self.source
        return payload
```

Apply this style to an immutable hardware artifact/model class, but include the v2 fields `hardware_validation_status` and `model_validation_status`; do not emit `validation_status` in v2 JSON.

**External file loader pattern** (lines 77-87):
```python
def load_scoring_baseline_artifact(path: Path) -> ScoringBaselineArtifact:
    """Load a scoring baseline artifact from JSON."""
    payload = json.loads(path.read_text())
    return scoring_baseline_artifact_from_dict(payload, source=str(path))


def scoring_baseline_artifact_from_dict(
    payload: dict[str, Any],
    *,
    source: str | None = None,
) -> ScoringBaselineArtifact:
```

Copy this split for `load_amd_hardware_model(path: Path)` and `amd_hardware_model_from_dict(payload, source=...)`. Add a second thin reader using `importlib.resources.files("sol_execbench.data.amd_hardware_models").joinpath(f"{architecture}.json").read_text(encoding="utf-8")`.

**Validation/error pattern** (lines 101-116):
```python
entries_payload = payload.get("entries")
if not isinstance(entries_payload, list):
    raise ValueError("scoring baseline artifact requires an entries list")

for index, raw in enumerate(entries_payload):
    if not isinstance(raw, dict):
        raise ValueError(f"baseline entry {index} must be an object")
    try:
        definition = str(raw["definition"])
        workload_uuid = str(raw["workload_uuid"])
        latency_ms = float(raw["latency_ms"])
    except KeyError as exc:
        raise ValueError(f"baseline entry {index} missing {exc.args[0]}") from exc
    if latency_ms <= 0.0:
        raise ValueError(f"baseline entry {index} latency_ms must be positive")
```

Use explicit `ValueError` messages for missing provenance/source, non-positive `peak_tflops` / `memory_bandwidth_gbps`, unknown fields, unknown enum values, old `validation_status`, and filename/architecture mismatch.

---

### `src/sol_execbench/core/scoring/amd_sol.py` (model / service, transform)

**Analog:** `src/sol_execbench/core/scoring/amd_sol.py`

**Enum and dataclass style** (lines 20-33, 76-97):
```python
class EstimateConfidence(str, Enum):
    """Confidence level for graph and work estimates."""

    SUPPORTED = "supported"
    INEXACT = "inexact"
    UNSUPPORTED = "unsupported"


class HardwareValidationStatus(str, Enum):
    """Validation state for hardware model entries."""

    VALIDATED = "validated"
    PROVISIONAL = "provisional"
    UNVALIDATED = "unvalidated"
```

```python
@dataclass(frozen=True)
class AmdHardwareModel:
    """Architecture-specific AMD SOL model input."""

    architecture: str
    dtype_or_path: str
    peak_tflops: float
    memory_bandwidth_gbps: float
    source: str
    confidence: EstimateConfidence
    validation_status: HardwareValidationStatus
```

Keep `EstimateConfidence` and `HardwareValidationStatus` enum values stable for compatibility. Replace or extend `AmdHardwareModel` deliberately so downstream score warnings can distinguish hardware validation from model validation. If legacy `validation_status` stays as a property, make it compatibility-only and do not serialize it in v2 hardware JSON.

**Derived artifact serialization pattern** (lines 124-159):
```python
@dataclass(frozen=True)
class AmdSolBoundArtifact:
    """Auditable AMD SOL bound artifact for one workload."""

    definition: str
    workload_uuid: str
    hardware_model: AmdHardwareModel
    graph_nodes: tuple[GraphNode, ...]
    work_estimates: tuple[WorkEstimate, ...]
    op_bounds: tuple[OpSolBound, ...]
    schema_version: str = AMD_SOL_SCHEMA_VERSION
    derived: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "hardware_model": self.hardware_model.to_dict(),
            "graph_nodes": [node.to_dict() for node in self.graph_nodes],
            "work_estimates": [estimate.to_dict() for estimate in self.work_estimates],
            "op_bounds": [bound.to_dict() for bound in self.op_bounds],
            "aggregate_sol_bound_ms": self.aggregate_sol_bound_ms,
            "coverage_summary": self.coverage_summary.to_dict(),
        }
```

Use this exact derived-artifact style for bound artifact v2 serialization: explicit `schema_version`, `derived`, stable nested `to_dict()` calls, and no mutation of `Trace`.

**Compatibility facade target** (lines 184-190):
```python
def default_amd_hardware_models() -> dict[str, AmdHardwareModel]:
    """Return built-in AMD hardware model entries."""
    return {
        "gfx1200": AmdHardwareModel(
            architecture="gfx1200",
            dtype_or_path="bf16/fp32 mixed benchmark path",
```

Replace the hard-coded body with a facade over packaged JSON. Do not keep `gfx942` as a default v1.9 packaged model unless it is explicitly unvalidated and loaded through the same strict path; the phase context says `gfx1200` is the starting default artifact.

---

### `src/sol_execbench/core/scoring/amd_score.py` (service, transform)

**Analog:** `src/sol_execbench/core/scoring/amd_score.py`

**Warning constants pattern** (lines 23-43):
```python
AMD_SCORE_SCHEMA_VERSION = "sol_execbench.amd_native_score.v1"
AMD_SCORE_CLAIM_LEVEL = "amd-native-derived"
UNSUPPORTED_EVIDENCE_WARNING = (
    "AMD-native score evidence contains unsupported operations; do not present "
    "the score as complete hardware-performance validation."
)
UNVALIDATED_HARDWARE_WARNING = (
    "AMD hardware model is not validated; score is provisional derived evidence."
)
CDNA3_NO_VALIDATION_WARNING = (
    "CDNA3 full-suite validation has not been recorded for this ROCm port; do "
    "not present this report as a CDNA3 hardware-validation claim."
)
```

Update warning checks to use the v2 hardware/model validation fields. Keep warning strings focused on claim safety and avoid premature B200, SOLAR, leaderboard, MI300X-on-CDNA3, or CDNA 4 validation claims.

**Score construction pattern** (lines 125-175):
```python
def score_amd_native_workload(
    artifact: AmdSolBoundArtifact,
    *,
    measured_latency_ms: float | None,
    baseline_latency_ms: float | None,
    trace_ref: str | None = None,
    timing_evidence_ref: str | None = None,
    sol_bound_ref: str | None = None,
    baseline_ref: str | None = None,
    baseline_source: str = "scoring_baseline",
    hardware_model_ref: str | None = None,
) -> AmdNativeScore:
    """Build a guarded AMD-native score for one workload."""
    sol_bound_ms = artifact.aggregate_sol_bound_ms
    warnings = _warnings_for_artifact(artifact)
```

Preserve function signatures unless a compatibility shim is unavoidable. Full score-report warning integration is Phase 45 scope.

**Guardrail helper pattern** (lines 309-323):
```python
def _warnings_for_artifact(artifact: AmdSolBoundArtifact) -> list[str]:
    warnings: list[str] = []
    if any(
        estimate.confidence == EstimateConfidence.UNSUPPORTED
        for estimate in artifact.work_estimates
    ):
        warnings.append(UNSUPPORTED_EVIDENCE_WARNING)

    if artifact.hardware_model.validation_status != HardwareValidationStatus.VALIDATED:
        warnings.append(UNVALIDATED_HARDWARE_WARNING)

    if artifact.hardware_model.architecture.startswith("gfx94"):
        warnings.append(CDNA3_NO_VALIDATION_WARNING)

    return warnings
```

Change only the hardware validation predicate needed for the new fields. Keep CDNA guardrails conservative.

---

### `src/sol_execbench/core/scoring/__init__.py` (package facade, transform)

**Analog:** `src/sol_execbench/core/scoring/__init__.py`

**Facade import/export pattern** (lines 6-19, 41-69):
```python
from .amd_sol import (
    AMD_SOL_SCHEMA_VERSION,
    AmdHardwareModel,
    AmdSolBoundArtifact,
    EstimateConfidence,
    GraphNode,
    HardwareValidationStatus,
    OpSolBound,
    WorkEstimate,
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
    estimate_work,
    extract_graph,
)
```

```python
__all__ = [
    "AMD_SCORE_CLAIM_LEVEL",
    "AMD_SCORE_SCHEMA_VERSION",
    "AMD_SOL_SCHEMA_VERSION",
    "AmdHardwareModel",
    "default_amd_hardware_models",
]
```

If new loader functions are public, export them here deliberately. Keep existing exports for `AmdHardwareModel`, `HardwareValidationStatus`, and `default_amd_hardware_models`.

---

### `src/sol_execbench/data/__init__.py` and `src/sol_execbench/data/amd_hardware_models/__init__.py` (package marker, package resource)

**Analog:** `src/sol_execbench/core/scoring/__init__.py`

**Module header pattern** (lines 1-4):
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMD-native scoring support modules."""
```

Use the repository SPDX header and a short package docstring. These package markers exist so `importlib.resources` can address `sol_execbench.data.amd_hardware_models`.

---

### `src/sol_execbench/data/amd_hardware_models/gfx1200.json` (config / artifact, file-I/O)

**Analog:** `src/sol_execbench/core/scoring/baseline_artifact.py`

**Artifact shape pattern** (lines 66-74):
```python
return {
    "schema_version": self.schema_version,
    "derived": self.derived,
    "release": self.release,
    "source": self.source,
    "summary": self.summary,
    "entries": [entry.to_dict() for entry in self.entries],
}
```

The JSON artifact should be explicit and self-describing: `schema_version`, `architecture`, `dtype_or_path`, positive numeric peaks, bandwidth, source/provenance, confidence, `hardware_validation_status`, `model_validation_status`, and evidence references. Do not include `validation_status` in v2.

---

### `pyproject.toml` (config, package build)

**Analog:** `pyproject.toml`

**Current build backend pattern** (lines 1-3):
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Current package/project pattern** (lines 5-29):
```toml
[project]
name = "sol-execbench"
version = "1.0.2"
description = "ROCm-only port of SOL ExecBench for evaluating GPU kernel solutions on AMD hardware"
readme = "README.md"
requires-python = ">=3.12,<3.14"

[project.scripts]
sol-execbench = "sol_execbench.cli:cli"
sol-execbench-baseline = "sol_execbench.cli.baseline:cli"
```

Research notes no explicit Hatchling package-data stanza today. If packaged JSON is not included by default, add a minimal Hatchling include/files config and a packaging smoke test; otherwise leave `pyproject.toml` untouched.

---

### `tests/sol_execbench/test_amd_hardware_models.py` (test, file-I/O + validation)

**Analogs:** `tests/sol_execbench/test_amd_sol_bounds.py`, `src/sol_execbench/core/scoring/baseline_artifact.py`

**Imports/helper pattern** (lines 1-27):
```python
from __future__ import annotations

from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_sol import (
    AMD_SOL_SCHEMA_VERSION,
    EstimateConfidence,
    HardwareValidationStatus,
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
```

For loader tests, import the new loader functions from `amd_hardware_models` and use `tmp_path` for external JSON. Test packaged load, external-path load, unknown fields, missing source/provenance, non-positive values, invalid enum/status, old `validation_status`, and architecture/filename mismatch.

**Assertion style pattern** (lines 47-69):
```python
hardware = default_amd_hardware_models()["gfx1200"]

artifact = build_amd_sol_bound_artifact(definition, workload, hardware)
payload = artifact.to_dict()

assert artifact.schema_version == AMD_SOL_SCHEMA_VERSION
assert artifact.derived is True
assert payload["hardware_model"]["architecture"] == "gfx1200"
assert payload["hardware_model"]["validation_status"] == "provisional"
```

Keep tests direct and value-oriented. For v2, assert `hardware_validation_status` and `model_validation_status`, and assert `"validation_status" not in payload["hardware_model"]`.

---

### `tests/sol_execbench/test_amd_sol_bounds.py` (test, transform + contract)

**Analog:** `tests/sol_execbench/test_amd_sol_bounds.py`

**Bound artifact test pattern** (lines 47-69):
```python
def test_matmul_bound_artifact_records_graph_work_hardware_and_bounds():
    definition = _matmul_definition()
    workload = Workload(
        axes={"M": 2},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="matmul-workload",
    )
    hardware = default_amd_hardware_models()["gfx1200"]

    artifact = build_amd_sol_bound_artifact(definition, workload, hardware)
    payload = artifact.to_dict()

    assert artifact.schema_version == AMD_SOL_SCHEMA_VERSION
    assert artifact.derived is True
    assert payload["coverage_summary"]["supported_ops"] == 1
    assert payload["hardware_model"]["architecture"] == "gfx1200"
```

Update expected hardware payload fields to v2. Add regression coverage that `default_amd_hardware_models()` comes from packaged JSON and fails explicitly if the package artifact is invalid/missing where feasible without brittle monkeypatching.

**Trace immutability pattern** (lines 194-224):
```python
before = trace.model_dump(mode="json")

artifact = build_amd_sol_bound_artifact(
    _matmul_definition(),
    trace.workload,
    default_amd_hardware_models()["gfx1200"],
)
_ = artifact.to_dict()

assert trace.model_dump(mode="json") == before
```

Keep this invariant unchanged: derived AMD SOL artifacts must not mutate canonical `Trace` JSONL payloads.

---

### `tests/sol_execbench/test_amd_native_score.py` (test, transform + guardrail)

**Analog:** `tests/sol_execbench/test_amd_native_score.py`

**Evidence reference pattern** (lines 85-113):
```python
report = score_amd_native_workload(
    artifact,
    measured_latency_ms=1.5,
    baseline_latency_ms=2.0,
    trace_ref="traces/matmul.jsonl",
    timing_evidence_ref="timing/matmul.json",
    sol_bound_ref="sol/matmul.json",
    baseline_ref="baseline/reference.json",
    hardware_model_ref="hardware/gfx1200.json",
)

assert report.evidence_refs == {
    "timing": "timing/matmul.json",
    "sol_bound": "sol/matmul.json",
    "trace": "traces/matmul.jsonl",
    "baseline": "baseline/reference.json",
    "hardware_model": "hardware/gfx1200.json",
}
assert UNVALIDATED_HARDWARE_WARNING in report.warnings
```

Update warning assertions to reflect v2 hardware/model validation semantics. Keep evidence refs stable.

**Canonical trace no-mutation pattern** (lines 164-208):
```python
before = trace.model_dump(mode="json")

score = score_amd_native_trace_workload(
    trace,
    artifact,
    trace_ref="traces.json",
    timing_evidence_ref="timing.json",
    sol_bound_ref="sol.json",
    baseline_ref="trace.reference_latency_ms",
    hardware_model_ref="artifact.hardware_model",
)

assert score.score is not None
assert trace.model_dump(mode="json") == before
```

Preserve this for any compatibility shim.

---

### `tests/sol_execbench/test_public_contract_guardrails.py` (test, contract guardrail)

**Analog:** `tests/sol_execbench/test_public_contract_guardrails.py`

**Public schema guard pattern** (lines 38-79):
```python
solution = Solution(
    name="demo",
    definition="demo_problem",
    author="tester",
    spec={
        "languages": ["hip_cpp"],
        "target_hardware": ["gfx1200", "gfx942"],
        "entry_point": "kernel.hip::run",
        "compile_options": {"hip_cflags": ["-O3"]},
    },
    sources=[
        {"path": "kernel.hip", "content": 'extern "C" __global__ void k() {}'}
    ],
)

dumped = solution.model_dump(mode="json")
assert dumped["spec"]["target_hardware"] == ["gfx1200", "gfx942"]
```

Use this style to guard Definition/Workload/Solution/Trace public contract drift. Keep assertions on concrete serialized keys.

**CLI help guard pattern** (lines 82-116):
```python
result = CliRunner().invoke(cli, ["--help"])
assert result.exit_code == 0
help_text = result.output
for expected_option in (
    "Usage:",
    "--definition",
    "--workload",
    "--solution",
    "--config",
    "--compile-timeout",
    "--timeout",
    "--output",
    "--json",
    "--lock-clocks",
    "--keep-staging",
    "--verbose",
):
    assert expected_option in help_text
for unexpected_option in ("diagnose", "profile", "hip-bench"):
    assert unexpected_option not in help_text
```

Add v1.9 checks that no hardware-model path CLI/dataset option is exposed in Phase 41.

**Claim guard pattern** (lines 144-151, 236-242):
```python
project = Path(".planning/PROJECT.md").read_text()
requirements = Path(".planning/REQUIREMENTS.md").read_text()
analysis = Path("docs/analysis.md").read_text()

assert "not NVIDIA B200, SOLAR, or leaderboard equivalence claims" in analysis
```

```python
handoff = Path(".planning/CDNA3-VALIDATION-HANDOFF.md").read_text()
project = Path(".planning/PROJECT.md").read_text()
assert "hardware validation remains deferred" in project
assert "CDNA3 full-suite validation has not been recorded" in CDNA3_NO_VALIDATION_WARNING
```

Extend these grep-style tests to block premature B200, upstream SOLAR, leaderboard-equivalence, MI300X-on-CDNA3 validation, and CDNA 4 validation claims in v1.9 docs or outputs.

---

### `docs/analysis.md` (documentation, contract text)

**Analog:** existing guardrail tests over `docs/analysis.md`

**Derived-score doc guard pattern** (`tests/sol_execbench/test_amd_native_score.py` lines 320-324):
```python
text = (REPO_ROOT / "docs" / "analysis.md").read_text()

assert "AMD-native score reports are derived artifacts" in text
assert "not NVIDIA B200, SOLAR, or leaderboard equivalence claims" in text
```

If docs need updates, write claim boundaries in the same concrete language the tests assert. Keep Phase 41 docs limited to artifact contract, validation-status semantics, packaged defaults, and deferred validation claims.

## Shared Patterns

### Strict Artifact Loading
**Source:** `src/sol_execbench/core/scoring/baseline_artifact.py` lines 77-116  
**Apply to:** `amd_hardware_models.py`, `gfx1200.json`, `test_amd_hardware_models.py`

Use separate reader and parser functions. Reader loads JSON; parser validates shape and produces immutable dataclasses. Raise `ValueError` with clear field-specific messages.

### Derived Artifact Serialization
**Source:** `src/sol_execbench/core/scoring/amd_sol.py` lines 124-159 and `src/sol_execbench/core/scoring/amd_score.py` lines 46-79  
**Apply to:** `AmdHardwareModel`, `AmdSolBoundArtifact`, score compatibility checks

Derived artifacts use frozen dataclasses, `to_dict()` methods, explicit schema/version fields, and nested conversion. They are not canonical trace JSONL.

### Public Compatibility Facades
**Source:** `src/sol_execbench/core/scoring/__init__.py` lines 6-70 and `src/sol_execbench/core/scoring/amd_sol.py` lines 184-190  
**Apply to:** `amd_sol.py`, `amd_hardware_models.py`, `core/scoring/__init__.py`

Keep public imports stable. Move implementation details into the new module while preserving compatibility entry points such as `default_amd_hardware_models()`.

### Claim And Contract Guardrails
**Source:** `tests/sol_execbench/test_public_contract_guardrails.py` lines 38-151 and 236-242  
**Apply to:** public schema/CLI/Trace tests, docs guardrails, score warning tests

Guardrails should assert concrete strings and serialized keys. They should fail on accidental public CLI options, schema field drift, trace mutation, or premature validation/equivalence claims.

## No Analog Found

No files lacked a usable analog. The weakest match is the new JSON resource path because no `src/sol_execbench/data/` package exists yet; use `baseline_artifact.py` for artifact shape and `core/scoring/__init__.py` for package marker style.

## Metadata

**Analog search scope:** `src/sol_execbench/`, `tests/sol_execbench/`, `docs/`, `pyproject.toml`  
**Files scanned:** 40+ source/test/docs/config candidates via `find` and `rg`  
**Pattern extraction date:** 2026-05-23
