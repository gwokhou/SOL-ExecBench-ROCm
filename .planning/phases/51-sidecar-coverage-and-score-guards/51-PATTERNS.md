# Phase 51: Sidecar Coverage And Score Guards - Pattern Map

**Mapped:** 2026-05-23
**Files analyzed:** 5 implementation targets, 3 test targets
**Analogs found:** 8 / 8

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/scoring/solar_derivation.py` | model, parser, utility | transform, file-I/O sidecar | `SolarDerivationEvidence`, `_group_from_dict`, parser helpers in same file | exact |
| `src/sol_execbench/core/scoring/solar_derivation.py` coverage dataclasses | model | transform | `AmdSolV2CoverageSummary` in `amd_sol_v2.py` | exact |
| `src/sol_execbench/core/scoring/solar_derivation.py` aggregate status helper | utility | transform | `_aggregate_for_bounds` in `amd_sol_v2.py` | exact |
| `src/sol_execbench/core/scoring/amd_score.py` | service | request-response style scoring transform | `score_amd_native_workload`, `_warnings_for_artifact` | exact |
| `tests/sol_execbench/test_solar_derivation_evidence.py` | test | parser round-trip, transform | existing SOLAR parser/round-trip tests | exact |
| `tests/sol_execbench/test_amd_sol_v2.py` | test | aggregate status, sidecar round-trip | v2 coverage and unscored tests | role-match |
| `tests/sol_execbench/test_public_contract_guardrails.py` | test | public contract guardrail | SOLAR noncanonical and score eligibility tests | exact |

## Recommended Files And Functions To Extend

| Target | Recommended Change Boundary |
|--------|-----------------------------|
| `src/sol_execbench/core/scoring/solar_derivation.py` | Add coverage dataclasses near `SolarConfidenceClassification` / `SolarDerivationEvidence`; add `coverage` field to `SolarDerivationEvidence`; compute coverage in `derive_solar_derivation_evidence()` from `groups`; parse with strict exact-key helpers. |
| `src/sol_execbench/core/scoring/amd_score.py` | Add optional internal SOLAR sidecar/coverage input to score builders only if needed; keep default behavior and existing evidence ref keys unchanged; return `score=None` only for explicit unscored SOLAR evidence. |
| `tests/sol_execbench/test_solar_derivation_evidence.py` | Extend `_contract_artifact()` / `_contract_payload()` and parser rejection parametrization for every new field. |
| `tests/sol_execbench/test_amd_sol_v2.py` | Copy aggregate degraded/unscored semantics for score guard expectations, but keep AMD SOL v2 artifact schema separate. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Extend forbidden canonical/public fields and score eligibility assertions. Do not add primary CLI expectations for new options. |

## Pattern Assignments

### `src/sol_execbench/core/scoring/solar_derivation.py` coverage dataclasses (model, transform)

**Analog:** `src/sol_execbench/core/scoring/solar_derivation.py`

**Imports and constants pattern** (lines 8-35):
```python
from dataclasses import dataclass
from math import isfinite
from typing import Any

SOLAR_DERIVATION_SCHEMA_VERSION = "sol_execbench.solar_derivation.v1"
SOLAR_DERIVATION_STATUSES = frozenset({"scored", "degraded", "unscored"})
```

**Frozen dataclass + explicit JSON-safe `to_dict()` pattern** (lines 201-238):
```python
@dataclass(frozen=True)
class SolarSemanticGroupEvidence:
    status: str
    missing_evidence: tuple[str, ...]
    warning_prefixes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "missing_evidence": list(self.missing_evidence),
            "warning_prefixes": list(self.warning_prefixes),
            "formula_evidence": [
                evidence.to_dict() for evidence in self.formula_evidence
            ],
            "byte_evidence": [evidence.to_dict() for evidence in self.byte_evidence],
            "bound_evidence": [evidence.to_dict() for evidence in self.bound_evidence],
        }
```

**Coverage summary analog** (from `src/sol_execbench/core/scoring/amd_sol_v2.py` lines 80-104):
```python
@dataclass(frozen=True)
class AmdSolV2CoverageSummary:
    total_ops: int
    supported_ops: int
    inexact_ops: int
    unsupported_ops: int
    op_family_counts: dict[str, int]
    confidence_counts_by_family: dict[str, dict[str, int]]
    worst_confidence: EstimateConfidence

    def to_dict(self) -> dict[str, object]:
        return {
            "op_family_counts": dict(sorted(self.op_family_counts.items())),
            "confidence_counts_by_family": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(self.confidence_counts_by_family.items())
            },
            "worst_confidence": self.worst_confidence.value,
        }
```

**Apply to Phase 51:** Create compact SOLAR coverage dataclasses that aggregate over `SolarDerivationEvidence.groups`, not graph nodes. Include family counts, status counts, missing evidence, warnings, estimated/degraded/unsupported node IDs, and provenance records. Sort dict keys and tuple/list values in `to_dict()`.

### `src/sol_execbench/core/scoring/solar_derivation.py` parser helpers (parser, file-I/O sidecar)

**Analog:** `solar_derivation_from_dict()` and `_group_from_dict()`

**Top-level exact-key parser pattern** (lines 454-512):
```python
def solar_derivation_from_dict(payload: dict[str, Any]) -> SolarDerivationEvidence:
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

**Nested status validation pattern** (lines 515-541):
```python
raw = _ensure_dict(payload, source=source)
_require_exact_keys(raw, {..., "status", "missing_evidence", "warning_prefixes", ...}, source=source)
status = _parse_str(raw, "status", source=source)
if status not in SOLAR_DERIVATION_STATUSES:
    valid = ", ".join(sorted(SOLAR_DERIVATION_STATUSES))
    raise ValueError(f"{source}.status has invalid status '{status}', expected one of: {valid}")
```

**Shared parser primitives** (lines 1881-1938, 1976-1983):
```python
def _require_exact_keys(payload: dict[str, Any], allowed: frozenset[str] | set[str], *, source: str) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ValueError(f"{source} contains unknown field(s): {', '.join(unknown)}")
    _require_keys(payload, allowed, source=source)

def _parse_str_tuple(payload: dict[str, Any], key: str, *, source: str) -> tuple[str, ...]:
    return tuple(
        _parse_str_item(item, source=f"{source}.{key}[{index}]")
        for index, item in enumerate(_parse_list(payload, key, source=source))
    )

def _ensure_json_scalar(value: object, *, source: str) -> None:
    if value is None or isinstance(value, str):
        return
```

**Apply to Phase 51:** Use `_require_exact_keys`, not `_require_keys`, for new SOLAR coverage payloads because TEST-03 requires strict exact-key behavior. Add `_coverage_from_dict()` and nested helpers in this file; avoid importing AMD SOL v2 parser helpers because their `_require_keys()` currently allows unknown fields.

### `src/sol_execbench/core/scoring/solar_derivation.py` coverage aggregation (utility, transform)

**Analog:** `src/sol_execbench/core/scoring/amd_sol_v2.py`

**Aggregate status pattern** (lines 271-312):
```python
if not op_bounds:
    return AmdSolV2AggregateBound(status="unscored", scored=False, ...)
if any(bound.confidence == EstimateConfidence.UNSUPPORTED for bound in op_bounds):
    return AmdSolV2AggregateBound(status="unscored", scored=False, ...)
if any(bound.confidence == EstimateConfidence.INEXACT for bound in op_bounds):
    return AmdSolV2AggregateBound(status="degraded", scored=True, ...)
return AmdSolV2AggregateBound(status="scored", scored=True, ...)
```

**Coverage count pattern** (lines 315-353):
```python
op_family_counts: dict[str, int] = {}
confidence_counts: dict[str, dict[str, int]] = {}
for estimate in estimates:
    family = estimate.op_family.value
    confidence = estimate.confidence.value
    op_family_counts[family] = op_family_counts.get(family, 0) + 1
    counts = confidence_counts.setdefault(family, {"supported": 0, "inexact": 0, "unsupported": 0})
    counts[confidence] = counts.get(confidence, 0) + 1
```

**Apply to Phase 51:** Aggregate from `SolarSemanticGroupEvidence.status`, `missing_evidence`, `warning_prefixes`, `formula_evidence`, `byte_evidence`, and `bound_evidence`. Status precedence should be `unscored` over `degraded` over `scored`; empty groups should be explicit `unscored`, not missing sidecar data.

### `src/sol_execbench/core/scoring/amd_score.py` score guards (service, request-response transform)

**Analog:** `score_amd_native_workload()` and `_warnings_for_artifact()`

**Score None guard pattern** (lines 151-204):
```python
score_value = None
if isinstance(artifact, AmdSolBoundV2Artifact) and not artifact.aggregate_bound.scored:
    if UNSCORED_SOL_BOUND_WARNING not in warnings:
        warnings.append(UNSCORED_SOL_BOUND_WARNING)
elif _has_complete_numeric_inputs(...):
    score_value = sol_score(t_k=measured_latency_ms, t_b=baseline_latency_ms, t_sol=sol_bound_ms)
else:
    warnings.append(INCOMPLETE_EVIDENCE_WARNING)
```

**Warning preservation pattern** (lines 346-357):
```python
if isinstance(artifact, AmdSolBoundV2Artifact):
    warnings = list(artifact.warnings)
    if artifact.aggregate_bound.status == "degraded":
        warnings.append(DEGRADED_SOL_BOUND_WARNING)
    elif artifact.aggregate_bound.status == "unscored":
        warnings.append(UNSCORED_SOL_BOUND_WARNING)
    return _unique(warnings)
```

**Boundary:** If Phase 51 wires SOLAR coverage into score guards, add it as internal optional evidence and preserve current behavior when no SOLAR sidecar is supplied. `unscored` SOLAR coverage may force `score=None`; `degraded` SOLAR coverage must keep the numeric score if AMD-native numeric inputs are complete and append warnings. Do not add `solar_derivation`, `formula_evidence`, `byte_evidence`, or `bound_evidence` to public `evidence_refs`.

## TEST-03 Parser / Round-Trip Coverage Map

| New Field Kind | Test Location | Required Assertions |
|----------------|---------------|---------------------|
| top-level coverage object on SOLAR sidecar | `test_solar_derivation_round_trip_preserves_provenance` | `solar_derivation_from_dict(payload).to_dict() == payload`; schema remains v1 unless intentionally bumped by plan. |
| coverage family counts | new/extended SOLAR coverage test | Dict ordering deterministic; counts reflect `groups[*].family`. |
| coverage status counts | new/extended SOLAR coverage test | `scored`, `degraded`, `unscored` keys present with integer JSON values. |
| missing pattern records | parser rejection + round-trip tests | Lists serialize from tuples; missing malformed strings/non-lists rejected. |
| unsupported pattern records | parser rejection + unscored test | Unsupported records preserve group ID/node IDs and force aggregate `unscored`. |
| degraded node records | degraded test | Degraded records preserve warnings/missing evidence and do not imply `score=None`. |
| estimated node records | scored/positive test | Estimated records are derived from formula/byte/bound evidence and sorted by node ID. |
| source/provenance records | round-trip test | Use exact-key nested parser; reject unknown provenance fields. |
| aggregate status | invalid status parametrization | Reject values outside `SOLAR_DERIVATION_STATUSES`. |
| deterministic serialization | existing deterministic test lines 1149-1176 | Reversed input order yields identical coverage payload; nested lists sorted. |

**Existing test patterns to copy:**

Round-trip and required-field rejection from `tests/sol_execbench/test_solar_derivation_evidence.py` lines 173-230.

Unknown field rejection from lines 233-276.

Malformed sidecar evidence rejection from lines 413-462.

Status semantics from lines 937-1100.

Deterministic ordering from lines 1149-1176.

AMD SOL v2 sidecar round-trip and aggregate rejection from `tests/sol_execbench/test_amd_sol_v2.py` lines 89-125.

## Shared Patterns

### Deterministic Ordering
**Source:** `solar_derivation.py` `_unique_sorted()` lines 1863-1864 and deterministic tests lines 1149-1176.
**Apply to:** coverage warnings, missing evidence, unsupported/degraded/estimated record lists, family/status count maps.

### Status Vocabulary
**Source:** `SOLAR_DERIVATION_STATUSES` in `solar_derivation.py` line 34 and `_status_for_confidence()` lines 1832-1837.
**Apply to:** every new aggregate coverage status and parser validation.

### Sidecar-Only Boundary
**Source:** `source_boundary` defaults in `solar_derivation.py` lines 1867-1872 and public guardrails lines 183-291.
**Apply to:** any new internal coverage field. Keep canonical `Definition`, `Workload`, `Trace`, trace JSONL, and primary CLI help unchanged.

### Score Claim Boundary
**Source:** `amd_score.py` lines 176-191 and public guardrails lines 297-469.
**Apply to:** score guard integration. Preserve `claim_level == "amd-native-derived"` and existing evidence refs.

## Public Contract Anti-Patterns To Avoid

- Do not add SOLAR coverage fields to Pydantic public models (`Definition`, `Workload`, `Trace`) or canonical trace JSONL.
- Do not expose new primary CLI options such as `--solar-coverage`, `--solar-sidecar`, or `--derive-solar-sidecar` in this phase.
- Do not store `formula_kind` or family-specific evidence names as schema keys; keep them as sidecar values, following `test_phase50_formula_kinds_remain_sidecar_values_not_schema_keys`.
- Do not treat missing SOLAR sidecar data as equivalent to explicit `unscored` coverage. `unscored` must be machine-verifiable evidence.
- Do not silently score explicit `unscored` SOLAR evidence.
- Do not suppress degraded warnings while returning a numeric score.
- Do not add NVIDIA equivalence, hosted leaderboard, paper-scale dataset, or hardware-validation claims.
- Do not add dependencies for this phase.

## No Analog Found

None. All Phase 51 work has close analogs in existing SOLAR sidecar evidence, AMD SOL v2 coverage aggregation, AMD-native score guards, and public contract tests.

## Metadata

**Analog search scope:** `src/sol_execbench/core/scoring/`, `tests/sol_execbench/`
**Files scanned:** 8 direct files plus targeted symbol search
**Pattern extraction date:** 2026-05-23
