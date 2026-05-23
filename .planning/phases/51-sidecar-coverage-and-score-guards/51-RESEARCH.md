# Phase 51: Sidecar Coverage And Score Guards - Research

**Researched:** 2026-05-23
**Domain:** Internal SOLAR derivation sidecar coverage, aggregate score state, and AMD-native score guards
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Implementation Decisions

### Sidecar Coverage Scope
- Aggregate existing SOLAR derivation semantic groups into family-aware coverage
  evidence.
- Coverage must expose extraction provenance, missing patterns, unsupported
  patterns, degraded nodes, estimated nodes, family counts, and status counts.
- Coverage should reuse Phase 48-50 group-local formula, byte, bound,
  confidence, warning, and missing-evidence records.
- Do not re-run extraction from candidate solution code and do not create a
  second derivation graph.

### Machine-Verifiable Status Semantics
- Aggregate SOLAR evidence must expose explicit `scored`, `degraded`, and
  `unscored` states.
- `scored` requires supported groups with complete score-eligible evidence.
- `degraded` preserves warning and missing-evidence detail when evidence is
  useful but incomplete.
- `unscored` carries explicit unsupported/unscored evidence and must be
  distinguishable from missing sidecar data.

### AMD-Native Score Guard Semantics
- AMD-native scoring should return `None` for unscored SOLAR evidence.
- AMD-native score reports should preserve warnings for degraded SOLAR evidence.
- Degraded evidence must not be silently treated as fully scored.
- This phase may wire internal sidecar coverage into score guard behavior, but
  must preserve AMD-native-derived claim boundaries and avoid public schema
  drift.

### Parse And Serialize Coverage
- TEST-03 requires sidecar parse/serialize round-trip tests for every new
  machine-verifiable derivation evidence field.
- New coverage/aggregate/score-guard sidecar fields must use strict exact-key
  parser behavior, deterministic ordering, and JSON-safe values.
- Existing Phase 48-50 evidence parser coverage must remain green.

### Public Boundary
- Keep canonical `Definition`, `Workload`, `Trace`, primary CLI help, and
  canonical trace JSONL unchanged.
- Do not add public primary CLI options for SOLAR derivation coverage in this
  phase.
- Do not add new dependencies, paper-scale dataset claims, hardware-validation
  claims, hosted leaderboard claims, or NVIDIA equivalence claims.

### the agent's Discretion
- Exact internal dataclass names, helper names, and sidecar field names are at
  the agent's discretion if they remain deterministic, parseable, and consistent
  with existing scoring conventions.
- The planner may split work by coverage model, score guard integration, and
  parser/guardrail closure.

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

- Dataset-runner reporting closure and user-facing documentation remain Phase
  52.
- Public claim guardrails for paper parity, hardware validation, hosted
  leaderboard readiness, and NVIDIA equivalence are finalized in Phase 52.
- Real hardware validation and paper-scale extraction remain out of scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REPORT-01 | SOLAR sidecars report family-aware coverage, extraction provenance, missing patterns, unsupported patterns, degraded nodes, and estimated nodes. | Add internal coverage dataclasses under `solar_derivation.py` that aggregate existing `SolarDerivationEvidence.groups`, `tensors`, and `warnings`; do not create a new extractor. [VERIFIED: .planning/REQUIREMENTS.md][VERIFIED: codebase grep] |
| REPORT-02 | Aggregate SOLAR evidence remains machine-verifiable through parseable `scored`, `degraded`, and `unscored` states. | Reuse the existing `SOLAR_DERIVATION_STATUSES` values and make a top-level aggregate status object with explicit `score_eligible` boolean plus node/group traceability. [VERIFIED: .planning/REQUIREMENTS.md][VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py] |
| REPORT-03 | AMD-native scoring returns `None` for unscored SOLAR evidence and preserves warnings for degraded SOLAR evidence. | Extend AMD-native score guards to accept optional internal SOLAR coverage/status input; preserve current v2 behavior where unscored aggregate bounds produce `score=None` and degraded bounds keep warnings. [VERIFIED: .planning/REQUIREMENTS.md][VERIFIED: src/sol_execbench/core/scoring/amd_score.py] |
| TEST-03 | Sidecar parse and serialize round-trip tests cover every new machine-verifiable derivation evidence field. | Add round-trip, unknown-field rejection, invalid-status rejection, deterministic ordering, degraded, and unscored coverage tests to `test_solar_derivation_evidence.py`. [VERIFIED: .planning/REQUIREMENTS.md][VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py] |
</phase_requirements>

## Summary

Implement Phase 51 as an internal extension of `SolarDerivationEvidence`, not as a public AMD SOL v2 replacement and not as a parallel derivation pipeline. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md] The existing sidecar already has group-level `status`, `confidence`, `required_evidence`, `missing_evidence`, `warning_prefixes`, formula evidence, byte evidence, bound evidence, tensors, warnings, and source-boundary fields; Phase 51 should add a compact coverage/aggregate model computed from those records. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py]

Primary recommendation: add `SolarCoverageSummary`, `SolarFamilyCoverage`, and `SolarAggregateStatus` frozen dataclasses in `solar_derivation.py`, attach them to `SolarDerivationEvidence.to_dict()`, parse them with exact-key helpers, and add a narrow optional AMD score guard adapter that maps aggregate `unscored` to `score=None` while preserving degraded warnings. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py][VERIFIED: src/sol_execbench/core/scoring/amd_score.py]

## Project Constraints (from AGENTS.md)

- Python package code lives under `src/sol_execbench/`; scoring implementation belongs under `src/sol_execbench/core/scoring/`. [VERIFIED: AGENTS.md]
- Tests belong under `tests/sol_execbench/` for package behavior. [VERIFIED: AGENTS.md]
- Use Python 3.12+, Ruff style, `snake_case` for functions and modules, and `PascalCase` for classes and dataclasses. [VERIFIED: AGENTS.md]
- Pytest is the test framework; use focused unit tests for schema and scoring guard logic. [VERIFIED: AGENTS.md]
- Preserve SOL ExecBench benchmark semantics and public schemas unless a ROCm-specific change is unavoidable. [VERIFIED: AGENTS.md]
- Do not commit proprietary kernels, credentials, Hugging Face tokens, downloaded datasets, local cache, build output, or benchmark output. [VERIFIED: AGENTS.md]
- Phase work should stay inside GSD workflow artifacts; this research produces only `.planning/phases/51-sidecar-coverage-and-score-guards/51-RESEARCH.md`. [VERIFIED: AGENTS.md][VERIFIED: user request]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| SOLAR sidecar coverage aggregation | Python package / scoring internals | Tests | Coverage is derived from existing in-process derivation groups and is serialized as an internal sidecar. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py] |
| Aggregate `scored`/`degraded`/`unscored` semantics | Python package / scoring internals | AMD score guard | The same status vocabulary already exists in derivation and AMD SOL v2 aggregate bound code. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py][VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] |
| AMD-native score guard behavior | Python package / score reporting | Tests | `score_amd_native_workload()` already owns score suppression and warning preservation for unscored/degraded AMD SOL v2 artifacts. [VERIFIED: src/sol_execbench/core/scoring/amd_score.py] |
| Public contract guardrails | Tests | Python data models / CLI | Existing guardrails assert sidecar fields stay absent from `Definition`, `Workload`, `Trace`, primary CLI help, and canonical trace JSONL. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python standard library `dataclasses` | Python 3.12.13 available locally | Frozen internal sidecar records | Existing scoring sidecars use frozen dataclasses with explicit `to_dict()` methods. [VERIFIED: environment probe][VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py] |
| Pytest | Existing project test framework | Parser, score guard, and public boundary tests | AGENTS.md and existing tests use pytest for scoring and schema coverage. [VERIFIED: AGENTS.md][VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py] |
| Existing `sol_execbench.core.scoring` modules | In-repo | SOLAR derivation, AMD SOL v2 artifacts, AMD score reports | Phase constraints prohibit new dependencies and require preserving existing public behavior. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md][VERIFIED: codebase grep] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| No new external package | n/a | Keep implementation dependency-free | Required by Phase 51 context and v1.10 out-of-scope list. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md][VERIFIED: .planning/REQUIREMENTS.md] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Internal sidecar coverage dataclasses | Extend public `Definition`, `Workload`, `Trace`, or primary CLI output | Rejected because public schemas, CLI help, and canonical trace JSONL must stay unchanged. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md] |
| Aggregating existing groups | Re-run graph extraction or candidate solution analysis | Rejected because coverage must derive from `SolarDerivationEvidence.groups` and candidate execution is forbidden. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md] |
| Warning-text-only status inference | Explicit aggregate status fields | Rejected because REPORT-02 requires parseable `scored`, `degraded`, and `unscored` states. [VERIFIED: .planning/REQUIREMENTS.md] |

**Installation:** No install command; Phase 51 adds no dependencies. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md]

**Version verification:** Python 3.12.13 and `uv 0.11.15` are available; no package registry verification is required because no external package is recommended. [VERIFIED: environment probe]

## Package Legitimacy Audit

Not applicable. Phase 51 installs no external packages and recommends no package additions. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md]

## Architecture Patterns

### System Architecture Diagram

```text
Definition + Workload
        |
        v
build_bound_graph() + estimate_bound_work()
        |
        v
derive_solar_derivation_evidence()
        |
        v
SolarDerivationEvidence.groups / tensors / warnings
        |
        v
SolarCoverageSummary + SolarAggregateStatus
        |
        +--> strict sidecar to_dict()/from_dict() round trip
        |
        +--> optional AMD score guard input
                  |
                  +--> aggregate status == unscored -> score None + unscored warning
                  +--> aggregate status == degraded -> numeric score allowed + warnings preserved
                  +--> aggregate status == scored -> normal AMD-native score path
```

### Recommended Project Structure

```text
src/sol_execbench/core/scoring/
├── solar_derivation.py     # Add internal coverage and aggregate status dataclasses/parser helpers
├── amd_score.py            # Add narrow optional score guard adapter for SOLAR aggregate status
└── amd_sol_v2.py           # Keep existing AMD SOL v2 aggregate semantics unchanged unless reused as reference

tests/sol_execbench/
├── test_solar_derivation_evidence.py      # TEST-03 coverage parse/serialize/strict parser tests
├── test_amd_native_score.py               # unscored -> None and degraded warning preservation
├── test_amd_sol_v2.py                     # existing aggregate status regression coverage
└── test_public_contract_guardrails.py     # public schema/CLI/trace guardrails
```

### Pattern 1: Internal Coverage Model Over Existing Groups

**What:** Add coverage as deterministic aggregate sidecar data computed from `SolarSemanticGroupEvidence` records. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py]

**When to use:** Always during `derive_solar_derivation_evidence()` after groups and warnings are known. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py]

**Recommended fields:** `total_groups`, `family_counts`, `status_counts`, `estimated_node_ids`, `degraded_node_ids`, `unsupported_node_ids`, `missing_patterns`, `unsupported_patterns`, `warning_prefixes`, and per-family records with `family`, `group_ids`, `node_ids`, `statuses`, `missing_evidence`, `warnings`, and `source_refs`. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md][VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py]

**Example:**

```python
# Source: src/sol_execbench/core/scoring/solar_derivation.py existing pattern
@dataclass(frozen=True)
class SolarCoverageSummary:
    total_groups: int
    family_counts: dict[str, int]
    status_counts: dict[str, int]
    estimated_node_ids: tuple[str, ...]
    degraded_node_ids: tuple[str, ...]
    unsupported_node_ids: tuple[str, ...]
```

### Pattern 2: Parseable Aggregate Status

**What:** Add `SolarAggregateStatus(status, score_eligible, reason, group_ids, node_ids, warnings)` with `status in {"scored", "degraded", "unscored"}`. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py][VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py]

**Rules:** If no groups or any group is `unscored`, aggregate is `unscored` and `score_eligible=False`; else if any group is `degraded`, aggregate is `degraded` and `score_eligible=True`; else aggregate is `scored` and `score_eligible=True`. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md][VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py]

**Reason strings:** Use stable parseable values such as `missing semantic group evidence`, `unsupported semantic evidence present`, `incomplete semantic evidence present`, and `all semantic evidence is score eligible`. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py]

### Pattern 3: Score Guard Adapter

**What:** Keep `score_amd_native_workload()` as owner of AMD-native score suppression, but allow internal SOLAR aggregate status to guard score computation. [VERIFIED: src/sol_execbench/core/scoring/amd_score.py]

**Recommended signature shape:** Add an optional keyword-only argument such as `solar_aggregate_status: SolarAggregateStatus | None = None` or a small protocol-free dict extracted by the caller; do not add a public CLI option. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md][ASSUMED]

**Behavior:** If SOLAR aggregate status is `unscored`, return `score=None` even when numeric timing and AMD SOL bounds exist; if status is `degraded`, keep numeric score eligibility but append/preserve deterministic degraded warnings. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md][VERIFIED: src/sol_execbench/core/scoring/amd_score.py]

### Anti-Patterns to Avoid

- **Adding coverage to canonical models:** `Definition`, `Workload`, and `Trace` are public schemas and already have guardrails proving sidecar fields stay absent. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
- **Inferring aggregate state from warning text only:** Warning strings exist, but REPORT-02 requires parseable aggregate states. [VERIFIED: .planning/REQUIREMENTS.md]
- **Letting degraded evidence silently look fully scored:** Degraded evidence must preserve warnings and remain distinct from scored evidence. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md]
- **Treating missing sidecar as unscored evidence:** `unscored` must be explicit unsupported/unscored evidence, while absent sidecar data is a different integration condition. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Coverage source data | A second graph extractor | Aggregate `SolarDerivationEvidence.groups` | Existing groups already carry family, status, missing evidence, warnings, node IDs, and source provenance. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py] |
| Parser framework | New schema dependency | Existing `_require_exact_keys`, `_parse_*`, and dataclass `to_dict()` pattern | Phase constraints prohibit new dependencies and parser style already exists. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py][VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md] |
| Score math | New SOL score implementation | Existing `sol_score()` call inside `score_amd_native_workload()` | AMD score reports already use central SOL score math and guard incomplete/unscored inputs. [VERIFIED: src/sol_execbench/core/scoring/amd_score.py] |
| Public reporting closure | New user-facing CLI/docs | Phase 52 | Dataset-runner reporting, public docs, and final claim guardrails are explicitly deferred. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md] |

**Key insight:** Phase 51 is a sidecar integrity and score-eligibility phase, not a modeling phase; using existing groups avoids creating mismatched coverage and scoring interpretations. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md][VERIFIED: Phase 49/50 verification reports]

## Common Pitfalls

### Pitfall 1: Status Drift Between Group And Aggregate Semantics
**What goes wrong:** A group can be `unscored` while aggregate state is still treated as score-eligible. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md]
**Why it happens:** Current group status is parseable, but there is no top-level derivation aggregate status yet. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py]
**How to avoid:** Derive aggregate status directly from sorted groups with `unscored > degraded > scored` precedence. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py]
**Warning signs:** Score report has numeric score while coverage has unsupported groups. [VERIFIED: .planning/REQUIREMENTS.md]

### Pitfall 2: Strict Parser Regression
**What goes wrong:** New coverage fields serialize but are not parsed, or unknown nested keys are accepted. [VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py]
**Why it happens:** Every new dataclass needs matching exact-key parser helpers and invalid-payload tests. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py]
**How to avoid:** Add parser tests for top-level `coverage_summary`, `aggregate_status`, per-family coverage, missing required fields, unknown fields, invalid statuses, invalid counts, and non-list node IDs. [VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py]
**Warning signs:** `to_dict()` includes a field that `solar_derivation_from_dict()` ignores or cannot reconstruct. [VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py]

### Pitfall 3: Public Surface Leakage
**What goes wrong:** Internal sidecar fields appear in canonical trace JSONL, primary CLI help, public data models, or AMD-native score evidence refs. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
**Why it happens:** Internal sidecar names can be accidentally added to public artifact refs or CLI flags. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
**How to avoid:** Extend the forbidden-field list with `coverage_summary`, `aggregate_status`, `family_counts`, `status_counts`, `degraded_node_ids`, `unsupported_node_ids`, `estimated_node_ids`, and score-guard field names. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
**Warning signs:** Primary CLI help contains `--solar-*` or canonical trace JSON contains derivation coverage names. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

### Pitfall 4: Duplicated Degraded Warnings
**What goes wrong:** Score reports contain repeated degraded or unscored warnings after both AMD SOL v2 and SOLAR sidecar guards run. [VERIFIED: src/sol_execbench/core/scoring/amd_score.py]
**Why it happens:** `_warnings_for_artifact()` already deduplicates AMD SOL v2 warnings via `_unique()`. [VERIFIED: src/sol_execbench/core/scoring/amd_score.py]
**How to avoid:** Route SOLAR score-guard warnings through the same unique-warning pattern. [VERIFIED: src/sol_execbench/core/scoring/amd_score.py]
**Warning signs:** `DEGRADED_SOL_BOUND_WARNING` or `UNSCORED_SOL_BOUND_WARNING` appears twice in `AmdNativeScore.warnings`. [VERIFIED: tests/sol_execbench/test_amd_native_score.py]

## Code Examples

### Aggregate Status Helper

```python
# Source: existing precedence mirrors src/sol_execbench/core/scoring/amd_sol_v2.py
def _aggregate_status_for_groups(groups: tuple[SolarSemanticGroupEvidence, ...]) -> SolarAggregateStatus:
    if not groups:
        return SolarAggregateStatus(
            status="unscored",
            score_eligible=False,
            reason="missing semantic group evidence",
            group_ids=(),
            node_ids=(),
            warnings=("aggregate_unscored:missing semantic group evidence",),
        )
    if any(group.status == "unscored" for group in groups):
        return SolarAggregateStatus(
            status="unscored",
            score_eligible=False,
            reason="unsupported semantic evidence present",
            group_ids=tuple(group.group_id for group in groups if group.status == "unscored"),
            node_ids=_coverage_node_ids(groups, status="unscored"),
            warnings=_coverage_warnings(groups, status="unscored"),
        )
```

### Strict Parser Extension

```python
# Source: existing exact-key parser pattern in src/sol_execbench/core/scoring/solar_derivation.py
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
        "coverage_summary",
        "aggregate_status",
    },
    source="SOLAR derivation evidence",
)
```

### Score Guard Integration

```python
# Source: existing score suppression path in src/sol_execbench/core/scoring/amd_score.py
if solar_aggregate_status is not None and solar_aggregate_status.status == "unscored":
    score_value = None
    warnings.extend(solar_aggregate_status.warnings)
elif solar_aggregate_status is not None and solar_aggregate_status.status == "degraded":
    warnings.extend(solar_aggregate_status.warnings)
    # Existing numeric-input guard still controls whether a provisional score can be computed.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Group-local status only in `SolarSemanticGroupEvidence` | Add top-level `SolarAggregateStatus` over all groups | Phase 51 | Consumers can distinguish score eligibility without scanning group details manually. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py][VERIFIED: .planning/REQUIREMENTS.md] |
| AMD SOL v2 aggregate status guards only AMD SOL v2 artifacts | Optional SOLAR aggregate status can guard AMD-native scoring too | Phase 51 | Unscored SOLAR sidecars can force `score=None`; degraded sidecars preserve warnings. [VERIFIED: src/sol_execbench/core/scoring/amd_score.py][VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md] |
| Coverage summary in AMD SOL v1/v2 counts operation confidence | SOLAR derivation sidecar coverage counts semantic families, statuses, missing patterns, unsupported patterns, degraded nodes, and estimated nodes | Phase 51 | Coverage becomes family-aware and provenance-aware for derivation sidecars. [VERIFIED: src/sol_execbench/core/scoring/amd_sol.py][VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py][VERIFIED: .planning/REQUIREMENTS.md] |

**Deprecated/outdated:** No deprecated APIs were identified for this phase; do not introduce new public CLI options or canonical schema fields. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The implementation may use an optional keyword-only score guard argument rather than a new public artifact type. | Architecture Patterns | Planner may need to choose a different internal call shape if maintainers prefer a separate adapter helper. |

## Open Questions

1. **Should `solar_derivation_from_dict()` remain backward-compatible with Phase 48-50 sidecars lacking new Phase 51 fields?**
   - What we know: Existing tests currently build payloads without coverage fields, and strict exact-key parsing rejects unknown fields. [VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py]
   - What's unclear: Phase 51 can either bump/require the new internal schema shape or provide a compatibility normalization path for older in-repo sidecars. [ASSUMED]
   - Recommendation: Prefer strict new required fields because the sidecar schema version stays internal and tests can update fixtures; add explicit migration only if existing persisted sidecars must be read. [ASSUMED]

2. **Should score guard evidence refs include a SOLAR sidecar reference?**
   - What we know: Phase 51 says avoid public schema drift, while Phase 52 owns final evidence-reference and public reporting closure. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md]
   - What's unclear: Whether an internal score report may include a `solar_derivation` evidence ref before Phase 52. [ASSUMED]
   - Recommendation: Do not add `solar_derivation` to AMD-native score `evidence_refs` in Phase 51; keep the guard internal and let Phase 52 decide public report evidence references. [ASSUMED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | Dataclasses and tests | yes | 3.12.13 | none |
| uv | Test command execution | yes | 0.11.15 | use direct `pytest` only if uv is unavailable |
| pytest | Automated verification | yes | installed on PATH | `uv run pytest` remains preferred |

**Missing dependencies with no fallback:** None. [VERIFIED: environment probe]

**Missing dependencies with fallback:** None. [VERIFIED: environment probe]

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 per Phase 51 validation contract [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-VALIDATION.md] |
| Config file | `pyproject.toml` [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-VALIDATION.md] |
| Quick run command | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-VALIDATION.md] |
| Full suite command | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-VALIDATION.md] |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| REPORT-01 | Family-aware coverage exposes provenance, missing patterns, unsupported patterns, degraded nodes, estimated nodes, family counts, and status counts. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "coverage or aggregate or round_trip" -n 0 -x` | yes [VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py] |
| REPORT-02 | Aggregate SOLAR status is parseable as `scored`, `degraded`, or `unscored`. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "aggregate or status" -n 0 -x` | yes [VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py] |
| REPORT-03 | Unscored SOLAR evidence returns `None`; degraded SOLAR evidence preserves warnings. | unit | `uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_v2.py -k "solar or degraded or unscored" -n 0 -x` | yes [VERIFIED: tests/sol_execbench/test_amd_native_score.py][VERIFIED: tests/sol_execbench/test_amd_sol_v2.py] |
| TEST-03 | New sidecar fields round-trip and reject malformed payloads. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` | yes [VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py] |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-VALIDATION.md]
- **Per wave merge:** Full suite command from Phase 51 validation contract. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-VALIDATION.md]
- **Phase gate:** Full suite green plus Ruff over touched scoring/test files before `$gsd-verify-work`. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-VALIDATION.md]

### Wave 0 Gaps
- [ ] Add tests in `tests/sol_execbench/test_solar_derivation_evidence.py` for every new coverage and aggregate field. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-VALIDATION.md]
- [ ] Add AMD-native score guard tests in `tests/sol_execbench/test_amd_native_score.py` or `tests/sol_execbench/test_public_contract_guardrails.py` for SOLAR aggregate degraded/unscored inputs. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-VALIDATION.md]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No auth surface in this phase. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md] |
| V3 Session Management | no | No sessions or network service in this phase. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md] |
| V4 Access Control | no | Internal local scoring artifacts only. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py] |
| V5 Input Validation | yes | Strict exact-key parsers and type/value validation for sidecar JSON. [VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py] |
| V6 Cryptography | no | No cryptography introduced. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md] |

### Known Threat Patterns for Sidecar Parsing

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Unknown fields smuggle public or unsupported claims | Tampering | Exact-key parser rejection at every nested level. [VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py] |
| Candidate execution affects derivation evidence | Elevation of privilege / Tampering | Keep builder signatures limited to `Definition` and `Workload`; existing test checks no candidate parameters. [VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py] |
| Unsupported evidence receives numeric score | Tampering | Aggregate `unscored` must force AMD-native score `None`. [VERIFIED: .planning/REQUIREMENTS.md] |
| Public claim boundary drift | Spoofing | Public guardrails reject SOLAR sidecar field leakage into canonical schemas, CLI, and trace JSONL. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |

## Sources

### Primary (HIGH confidence)
- `.planning/ROADMAP.md` - Phase 51 scope and v1.10 milestone boundary. [VERIFIED: codebase grep]
- `.planning/REQUIREMENTS.md` - REPORT-01, REPORT-02, REPORT-03, TEST-03, and out-of-scope constraints. [VERIFIED: codebase grep]
- `.planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md` - locked decisions, public boundary, and implementation discretion. [VERIFIED: codebase grep]
- `.planning/phases/51-sidecar-coverage-and-score-guards/51-VALIDATION.md` - validation commands and Wave 0 gaps. [VERIFIED: codebase grep]
- `.planning/phases/49-high-confidence-family-modeling/49-VERIFICATION.md` - verified Phase 49 group-local formula/byte/bound evidence. [VERIFIED: codebase grep]
- `.planning/phases/50-degraded-complex-family-modeling/50-VERIFICATION.md` - verified Phase 50 degraded/unsupported MoE and SSM behavior. [VERIFIED: codebase grep]
- `src/sol_execbench/core/scoring/solar_derivation.py` - current sidecar dataclasses, strict parser, statuses, confidence mapping, and warnings. [VERIFIED: codebase grep]
- `src/sol_execbench/core/scoring/amd_score.py` - current AMD-native score suppression and warning preservation. [VERIFIED: codebase grep]
- `src/sol_execbench/core/scoring/amd_sol_v2.py` - current aggregate `scored`/`degraded`/`unscored` status model and coverage summary. [VERIFIED: codebase grep]
- `tests/sol_execbench/test_solar_derivation_evidence.py` - current parser, round-trip, sidecar boundary, and candidate-boundary tests. [VERIFIED: codebase grep]
- `tests/sol_execbench/test_public_contract_guardrails.py` - public schema/CLI/trace guardrails. [VERIFIED: codebase grep]
- `tests/sol_execbench/test_amd_native_score.py` and `tests/sol_execbench/test_amd_sol_v2.py` - degraded/unscored AMD score behavior. [VERIFIED: codebase grep]

### Secondary (MEDIUM confidence)
- None. No web or third-party docs were needed because Phase 51 is dependency-free internal code research. [VERIFIED: .planning/phases/51-sidecar-coverage-and-score-guards/51-CONTEXT.md]

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new packages; implementation uses existing Python dataclasses and pytest patterns. [VERIFIED: AGENTS.md][VERIFIED: src/sol_execbench/core/scoring/solar_derivation.py]
- Architecture: HIGH - extension points are explicit in `solar_derivation.py`, `amd_score.py`, and AMD SOL v2 aggregate behavior. [VERIFIED: codebase grep]
- Pitfalls: HIGH - parser, score guard, and public boundary risks are covered by existing tests and phase constraints. [VERIFIED: tests/sol_execbench/test_solar_derivation_evidence.py][VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

**Research date:** 2026-05-23
**Valid until:** 2026-06-22
