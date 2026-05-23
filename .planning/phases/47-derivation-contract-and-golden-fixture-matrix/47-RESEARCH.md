# Phase 47: Derivation Contract And Golden Fixture Matrix - Research

**Researched:** 2026-05-23
**Domain:** SOLAR derivation contract, golden fixtures, pytest guardrails
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Implementation Decisions

### Fixture Matrix Scope
- Store fixtures under `tests/sol_execbench/fixtures/solar_derivation/` as JSON fixtures with a Python helper loader.
- Cover each target family with at least three fixture classes: positive, degraded, and unsupported or negative. Families are attention, MoE, convolution, SSM/Mamba, embedding or positional, and linear projection.
- Fixture expectations should record expected family, subroles, SOLAR state, required evidence, missing evidence, and stable warning prefixes. Phase 47 should not require complete FLOP or byte golden numbers.
- The fixture matrix must explicitly verify paper-aligned derivation behavior without claiming paper-scale dataset extraction, hosted leaderboard readiness, NVIDIA Blackwell/B200 equivalence, or new real-hardware validation.

### Contract Artifact Shape
- Add `docs/internal/solar_derivation_contract.md` covering the sidecar-only contract, family states, fixture schema, and claim boundaries.
- Add a test-side helper that loads fixture JSON and validates required base fields so the contract is executable.
- Reuse existing confidence and aggregate vocabulary: `SUPPORTED`, `INEXACT`, `UNSUPPORTED`, and `scored`, `degraded`, `unscored`.
- Express negative fixtures with `expected_status`, `missing_evidence`, and stable warning prefixes rather than exception expectations.

### Phase 47 Execution Boundary
- Do not implement real extractor or modeling changes in Phase 47. Leave extraction and modeling implementation for Phases 48-50.
- Tests should validate fixture schema completeness, coverage across all required families and states, claim-boundary documentation, and fixture expectations that later phases can consume.
- Avoid production scoring-code changes unless a pure constant or type export is needed. Prefer docs/internal, fixtures, and tests.
- Phase 47 is complete when phase artifacts, the internal contract doc, fixtures, loader tests, and focused pytest checks are committed.

### the agent's Discretion
- The exact JSON field names and loader helper names are at the agent's discretion as long as they are stable, clear, and aligned with existing pytest conventions.
- The exact split between one fixture file per case and grouped fixture files is at the agent's discretion, provided tests can identify family/state coverage precisely.

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

- Real extraction infrastructure is deferred to Phase 48.
- High-confidence family formula implementation is deferred to Phase 49.
- MoE and SSM/Mamba modeling implementation is deferred to Phase 50.
- Sidecar coverage and score guard integration is deferred to Phase 51.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TEST-01 | Golden derivation fixtures cover attention, MoE, convolution, SSM/Mamba, embedding or positional patterns, and linear projection. | Fixture matrix must include all six target families and at least one positive fixture per family. [CITED: .planning/REQUIREMENTS.md] |
| TEST-02 | Negative and degradation fixtures cover dynamic, partial, unsupported, taxonomy-only, and missing-metadata cases. | Fixture matrix must include degradation/negative cases for those five categories and use expected status plus missing evidence rather than exception expectations. [CITED: .planning/REQUIREMENTS.md] |
</phase_requirements>

## Summary

Phase 47 should create an executable test contract, not extraction logic: add an internal SOLAR derivation contract doc, JSON fixtures under `tests/sol_execbench/fixtures/solar_derivation/`, and a focused pytest loader/validator that proves fixture coverage and claim boundaries before Phases 48-50 implement recognizers or formulas. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

The correct vocabulary already exists in the codebase: operation families include `attention`, `moe`, `convolution`, `ssm_mamba`, `embedding_positional`, and `linear_projection`; confidence values are `supported`, `inexact`, and `unsupported`; aggregate sidecar statuses are `scored`, `degraded`, and `unscored`; warnings use stable prefixes such as `graph_warning:`, `estimate_warning:`, `inexact_operator:`, `unsupported_operator:`, `aggregate_degraded:`, and `aggregate_unscored:`. [VERIFIED: codebase grep]

**Primary recommendation:** Use one JSON file per fixture case plus a test-side loader module, and validate the fixture matrix with CPU-only pytest checks over schema, family/state coverage, warning-prefix stability, and no-claim documentation. [VERIFIED: codebase grep]

## Project Constraints (from AGENTS.md)

- Source code lives under `src/sol_execbench/`; tests live under `tests/sol_execbench/`; docs live under `docs/`; downloaded benchmark assets belong in `data/`. [CITED: AGENTS.md]
- Use Python 3.12+ and Ruff style; keep changes consistent with nearby modules and avoid broad refactors. [CITED: AGENTS.md]
- Use `snake_case` for functions, variables, and modules; `PascalCase` for classes and Pydantic models; descriptive test names such as `test_rejects_invalid_solution_schema`. [CITED: AGENTS.md]
- Pytest is the test framework; place related tests under `tests/sol_execbench/`; prefer small unit tests for schema and driver logic. [CITED: AGENTS.md]
- Environment-sensitive markers exist for `cpp`, `requires_rocm`, `requires_rdna4`, `requires_cdna3`, and legacy `requires_cutile`; Phase 47 should not need them because fixture validation is CPU-only. [CITED: AGENTS.md]
- Do not commit proprietary kernels, credentials, Hugging Face tokens, downloaded datasets, local cache, build output, or benchmark output. [CITED: AGENTS.md]
- Before file-changing work, use a GSD workflow entry point; this research was spawned by the GSD phase planning workflow. [CITED: AGENTS.md]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Fixture schema and JSON cases | Tests | Docs | Fixtures are contract inputs for later tests and should live under `tests/sol_execbench/fixtures/solar_derivation/`. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |
| Loader and schema completeness validation | Tests | — | Phase 47 needs executable validation without production extraction changes. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |
| SOLAR derivation contract prose | Docs | Tests | `docs/internal/solar_derivation_contract.md` should define sidecar-only behavior, family states, fixture schema, and claim boundaries. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |
| Existing scoring vocabulary alignment | Core scoring | Tests | `OpFamily`, `EstimateConfidence`, and AMD SOL v2 aggregate statuses already define the vocabulary fixtures must reuse. [VERIFIED: codebase grep] |
| Public/canonical schema protection | Tests | Docs | Existing guardrail tests prove derived artifacts stay separate from `Definition`, `Workload`, `Trace`, `Solution`, and primary CLI contracts. [VERIFIED: codebase grep] |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.12.13 local; project requires `>=3.12,<3.14` | Fixture loader and tests | Repository baseline is Python 3.12+. [VERIFIED: local command; CITED: pyproject.toml] |
| pytest | 9.0.2 local; project requires `>=9.0.2` in dev group | Contract and fixture matrix tests | Existing test suite uses pytest and repo config defines pytest markers/addopts. [VERIFIED: local command; CITED: pyproject.toml] |
| stdlib `json` / `pathlib` | Python stdlib | Load fixture JSON files deterministically | Existing tests use stdlib file and JSON helpers for contract checks. [VERIFIED: codebase grep] |
| Existing scoring enums/constants | Current repo | Align family, confidence, status, and warning vocabulary | `OpFamily`, `EstimateConfidence`, `AGGREGATE_STATUSES`, and warning prefixes already exist. [VERIFIED: codebase grep] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Ruff | On demand via `uv run --with ruff ruff check .` | Lint/format new Python helper/tests | AGENTS.md lists Ruff commands for style checks. [CITED: AGENTS.md] |
| Pydantic | `>=2.12.5` project dependency | Existing public schemas only | Phase 47 should not add Pydantic models unless production type export becomes unavoidable. [CITED: pyproject.toml; CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JSON fixture files | Python dict literals in tests | JSON makes the fixture matrix inspectable and machine-readable for later phases; context explicitly requires JSON fixtures. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |
| Test-side lightweight validator | Production Pydantic model | Production model would expand scoring surface during a contract-only phase; Phase 47 should prefer docs/internal, fixtures, and tests. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |
| One grouped mega-file | One JSON file per fixture case | Per-case files make coverage and failures easier to identify; grouped files remain allowed if tests can identify family/state precisely. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |

**Installation:** No new package installation is recommended for Phase 47. [VERIFIED: codebase grep]

## Package Legitimacy Audit

No external packages should be installed in Phase 47. Existing project dependencies already provide pytest and stdlib JSON/path handling is sufficient. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md; CITED: pyproject.toml]

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| None | — | — | — | — | Not run | No install required |

**Packages removed due to slopcheck [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```text
Phase 47 inputs
  -> CONTEXT.md locked decisions
  -> REQUIREMENTS TEST-01/TEST-02
  -> existing scoring vocabulary
        |
        v
docs/internal/solar_derivation_contract.md
  -> sidecar-only contract
  -> family/state vocabulary
  -> claim boundaries
        |
        v
tests/sol_execbench/fixtures/solar_derivation/*.json
  -> positive fixtures by family
  -> degraded fixtures
  -> unsupported/negative fixtures
        |
        v
tests/sol_execbench/solar_derivation_fixtures.py
  -> load JSON
  -> validate required base fields
  -> expose fixture list to tests
        |
        v
tests/sol_execbench/test_solar_derivation_contract.py
  -> schema completeness
  -> family/state coverage
  -> warning-prefix stability
  -> docs claim-boundary assertions
```

### Recommended Project Structure

```text
docs/internal/
└── solar_derivation_contract.md        # v1.10 sidecar-only derivation contract

tests/sol_execbench/
├── solar_derivation_fixtures.py        # test-side JSON loader and validator
├── test_solar_derivation_contract.py   # executable contract tests
└── fixtures/
    └── solar_derivation/
        ├── attention_positive.json
        ├── attention_degraded_partial_mask.json
        ├── attention_unsupported_dynamic_axes.json
        ├── moe_positive.json
        ├── moe_degraded_dynamic_routing.json
        ├── moe_unsupported_taxonomy_only.json
        ├── convolution_positive.json
        ├── convolution_degraded_missing_padding.json
        ├── convolution_unsupported_dynamic_kernel.json
        ├── ssm_mamba_positive.json
        ├── ssm_mamba_degraded_missing_recurrence.json
        ├── ssm_mamba_unsupported_custom_scan.json
        ├── embedding_positional_positive.json
        ├── embedding_positional_degraded_dynamic_indices.json
        ├── embedding_positional_unsupported_missing_metadata.json
        ├── linear_projection_positive.json
        ├── linear_projection_degraded_missing_shape.json
        └── linear_projection_unsupported_missing_metadata.json
```

The exact filenames are discretionary, but the matrix must cover all required families and positive/degraded/unsupported-or-negative classes. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

### Pattern 1: Sidecar-Only Contract

**What:** The contract says SOLAR derivation evidence belongs in derived sidecars or opt-in reports, not canonical `definition.json`, `workload.jsonl`, trace JSONL, solution schemas, or primary CLI behavior. [VERIFIED: codebase grep]

**When to use:** Every Phase 47 fixture and doc statement should describe expected sidecar evidence, never public schema mutation. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

**Example:**

```python
# Source: tests/sol_execbench/test_public_contract_guardrails.py
assert "bound_graph" not in definition.model_dump(mode="json")
assert "operator_work_estimates" not in workload.model_dump(mode="json")
assert "aggregate_bound" not in trace.model_dump(mode="json")
```

### Pattern 2: Fixture Expectations Over Real Extraction

**What:** Fixtures should encode expected evidence and state, not execute actual recognizers. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

**When to use:** For each family, store a minimal reference/workload sketch and expected outcome fields for later implementation phases. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

**Recommended fixture shape:**

```json
{
  "case_id": "attention_positive_dense_qkv",
  "family": "attention",
  "fixture_class": "positive",
  "description": "Dense self-attention with explicit Q/K/V, softmax, PV, and output projection.",
  "source_kind": "reference_snippet",
  "reference": "def run(q, k, v, w_o): ...",
  "workload_axes": {"B": 2, "S": 16, "H": 4, "D": 32},
  "expectation": {
    "expected_family": "attention",
    "expected_subroles": ["q_projection", "k_projection", "v_projection", "qk_scores", "softmax", "pv_aggregation", "output_projection"],
    "expected_confidence": "supported",
    "expected_status": "scored",
    "required_evidence": ["shape:batch", "shape:sequence_q", "shape:sequence_k", "shape:head_dim"],
    "missing_evidence": [],
    "warning_prefixes": [],
    "degradation_rationale": null
  },
  "scope_boundary": {
    "paper_scale_dataset": false,
    "real_hardware_validation": false,
    "leaderboard_ready": false,
    "nvidia_b200_equivalence": false
  }
}
```

### Pattern 3: Stable Warning Prefixes

**What:** Fixtures should assert stable prefixes, not full free-form warning bodies. Existing artifacts already use prefix categories. [VERIFIED: codebase grep]

**When to use:** Negative/degraded fixtures should require prefixes such as `graph_warning:`, `estimate_warning:`, `inexact_operator:`, `unsupported_operator:`, `aggregate_degraded:`, or `aggregate_unscored:`. [VERIFIED: codebase grep]

**Example:**

```python
# Source: src/sol_execbench/core/scoring/amd_sol_v2.py
assert any(warning.startswith("aggregate_degraded:") for warning in warnings)
assert any(warning.startswith("aggregate_unscored:") for warning in warnings)
```

### Anti-Patterns to Avoid

- **Implementing real recognizers in Phase 47:** This phase defines the contract and fixtures; extraction and modeling are explicitly deferred. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]
- **Golden FLOP/byte numbers in Phase 47:** The context says fixtures should not require complete FLOP or byte golden numbers yet. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]
- **Exception-based negative fixtures:** Negative cases should use `expected_status`, `missing_evidence`, and warning prefixes rather than exception expectations. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]
- **Claim overreach:** Docs/fixtures must not imply paper-scale extraction, hosted leaderboard readiness, NVIDIA Blackwell/B200 equivalence, or new real-hardware validation. [CITED: .planning/REQUIREMENTS.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON parsing | Custom parser | Python stdlib `json` | Fixture files are plain JSON and stdlib parsing is enough. [VERIFIED: codebase grep] |
| Public schema validation | New production Pydantic models | Test-side required-field validator | Phase 47 should avoid production scoring-code changes unless a pure constant/type export is unavoidable. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |
| Family vocabulary | New strings independent of code | `OpFamily` values: `attention`, `moe`, `convolution`, `ssm_mamba`, `embedding_positional`, `linear_projection` | Existing code already defines paper-aligned family names. [VERIFIED: codebase grep] |
| Confidence/status vocabulary | New status names | `supported`, `inexact`, `unsupported`, `scored`, `degraded`, `unscored` | Existing artifacts and tests already enforce this vocabulary. [VERIFIED: codebase grep] |
| Claim guardrails | Manual review only | pytest checks that read docs/fixtures | Existing project uses explicit guardrail tests for public contracts and claims. [VERIFIED: codebase grep] |

**Key insight:** The hard part in Phase 47 is not modeling math; it is freezing a precise, executable contract so later extractor/modeling work cannot silently redefine what "supported", "degraded", "unsupported", and family coverage mean. [CITED: .planning/research/SUMMARY.md]

## Common Pitfalls

### Pitfall 1: Fixture Matrix Becomes Taxonomy-Only

**What goes wrong:** Fixtures assert only `family: attention` or `family: moe` and do not list subroles, required evidence, missing evidence, status, or rationale. [CITED: .planning/research/PITFALLS.md]

**Why it happens:** `OpFamily` already contains target family names, so it is easy to confuse labels with derivation support. [VERIFIED: codebase grep]

**How to avoid:** Require every fixture expectation to include `expected_family`, `expected_subroles`, `expected_confidence`, `expected_status`, `required_evidence`, `missing_evidence`, `warning_prefixes`, and `degradation_rationale` when degraded/unsupported. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

**Warning signs:** A negative fixture has an empty `missing_evidence` list or a degraded fixture has no stable warning prefix. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

### Pitfall 2: Overclaiming Paper Equivalence

**What goes wrong:** The contract doc says or implies full SOLAR parity, paper benchmark parity, leaderboard readiness, NVIDIA B200/Blackwell equivalence, or new hardware validation. [CITED: .planning/REQUIREMENTS.md]

**Why it happens:** The milestone uses "paper-aligned SOLAR", while project scope explicitly excludes the paper-scale dataset, hosted leaderboard, and hardware-equivalence claims. [CITED: .planning/STATE.md]

**How to avoid:** Put scope-boundary booleans in fixture expectations and add doc tests for forbidden claims. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

**Warning signs:** Docs use "paper-equivalent", "leaderboard-ready", "B200 equivalent", or "hardware validated" without a deferral/negative qualifier. [CITED: .planning/research/PITFALLS.md]

### Pitfall 3: Negative Fixtures Expect Exceptions

**What goes wrong:** Tests define negative behavior as load/parse exceptions instead of as machine-verifiable unscored/degraded evidence. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

**Why it happens:** Schema tests often use exceptions, but SOLAR degradation semantics require visible evidence, not failure. [VERIFIED: codebase grep]

**How to avoid:** Invalid fixture schema can raise in the loader, but valid negative cases should parse and assert `expected_status`, `missing_evidence`, and warning prefixes. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

**Warning signs:** A dynamic/partial/unsupported/taxonomy-only/missing-metadata case is represented only by `pytest.raises`. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

### Pitfall 4: Tests Depend On Hardware Or Dataset Extraction

**What goes wrong:** Phase 47 validation requires ROCm hardware, paper-scale extraction, or dataset assets. [CITED: .planning/REQUIREMENTS.md]

**Why it happens:** SOLAR is tied to performance analysis, but this phase is contract-only. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

**How to avoid:** Keep fixtures small, static, JSON-backed, and CPU-only; do not mark tests with ROCm/hardware markers. [CITED: AGENTS.md]

**Warning signs:** New Phase 47 tests use `requires_rocm`, `requires_rdna4`, `requires_cdna3`, Docker, or `data/SOL-ExecBench`. [CITED: AGENTS.md]

## Code Examples

### Loader Skeleton

```python
# Source: recommended pattern from existing pathlib/json pytest style in tests/sol_execbench.
from __future__ import annotations

import json
from pathlib import Path

FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "solar_derivation"


REQUIRED_TOP_LEVEL = {
    "case_id",
    "family",
    "fixture_class",
    "description",
    "source_kind",
    "expectation",
    "scope_boundary",
}


def load_solar_derivation_fixtures() -> tuple[dict[str, object], ...]:
    fixtures = []
    for path in sorted(FIXTURE_ROOT.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = REQUIRED_TOP_LEVEL - payload.keys()
        if missing:
            raise ValueError(f"{path.name} missing required fields: {sorted(missing)}")
        fixtures.append(payload)
    return tuple(fixtures)
```

### Coverage Assertions

```python
# Source: Phase 47 contract and TEST-01/TEST-02 requirements.
TARGET_FAMILIES = {
    "attention",
    "moe",
    "convolution",
    "ssm_mamba",
    "embedding_positional",
    "linear_projection",
}

NEGATIVE_CATEGORIES = {
    "dynamic",
    "partial",
    "unsupported",
    "taxonomy_only",
    "missing_metadata",
}


def test_fixture_matrix_covers_required_families_and_classes():
    fixtures = load_solar_derivation_fixtures()
    families = {fixture["family"] for fixture in fixtures}
    assert TARGET_FAMILIES <= families

    for family in TARGET_FAMILIES:
        classes = {
            fixture["fixture_class"]
            for fixture in fixtures
            if fixture["family"] == family
        }
        assert {"positive", "degraded"} <= classes
        assert "unsupported" in classes or "negative" in classes
```

### Docs Claim Boundary Assertions

```python
# Source: existing public contract guardrail test style.
from pathlib import Path


def test_solar_contract_preserves_v1_10_scope_boundaries():
    text = Path("docs/internal/solar_derivation_contract.md").read_text()
    assert "sidecar" in text
    assert "not paper-scale dataset extraction" in text
    assert "not hosted leaderboard readiness" in text
    assert "not NVIDIA Blackwell/B200 equivalence" in text
    assert "not new real-hardware validation" in text
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v1.9 sidecars model known primitive families and already carry coverage/status/warnings. | v1.10 Phase 47 should freeze the future SOLAR family contract before adding recognizers. | v1.10 roadmap, 2026-05-23 | Planner should sequence docs/fixtures/tests before production extraction. [CITED: .planning/ROADMAP.md] |
| Current target family names exist as taxonomy in `OpFamily`. | Phase 47 fixtures must distinguish taxonomy-only from supported/degraded derivation. | v1.10 requirements, 2026-05-23 | Taxonomy-only cases belong in negative/degradation matrix. [VERIFIED: codebase grep; CITED: .planning/REQUIREMENTS.md] |
| Existing aggregate statuses are `scored`, `degraded`, `unscored`. | Phase 47 should reuse these exactly and avoid inventing new state names. | v1.9 code already present | Later phases can consume fixture expectations without status migration. [VERIFIED: codebase grep] |

**Deprecated/outdated:**
- Treating SOLAR fixture coverage as docs-only is insufficient; Phase 47 needs executable pytest validation. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]
- Treating negative behavior as exceptions is out of scope for valid negative/degradation fixtures; expected evidence should remain parseable. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | One JSON file per fixture case is the recommended split. [ASSUMED] | Architecture Patterns | If maintainers prefer grouped files, planner can use grouped JSON while preserving coverage-test precision. |
| A2 | Recommended fixture field names such as `fixture_class`, `expected_subroles`, and `scope_boundary` are acceptable. [ASSUMED] | Architecture Patterns / Code Examples | If naming preferences differ, rename consistently before implementation; locked semantic requirements stay the same. |
| A3 | No production constant/type export is needed for Phase 47. [ASSUMED] | Standard Stack | If tests need canonical shared family/status constants, planner may add a pure export with no scoring behavior change. |

## Open Questions

1. **Fixture granularity**
   - What we know: One file per case and grouped files are both allowed. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]
   - What's unclear: Maintainer preference for review ergonomics.
   - Recommendation: Use one file per case for clearer coverage failures unless implementation finds excessive duplication.

2. **Exact positive expectation for MoE and SSM/Mamba**
   - What we know: Positive fixtures are required per target family class, but real modeling is deferred. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md]
   - What's unclear: Whether "positive" for inherently complex families should mean eventual `inexact/degraded` or a narrowly supported static case.
   - Recommendation: Define one narrowly static positive fixture and separate dynamic degraded fixtures for MoE and SSM/Mamba.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | Test helper and pytest | yes | 3.12.13 | — |
| uv | Running pytest/Ruff commands | yes | 0.11.15 | Direct `python -m pytest` only if env already synced |
| pytest | Fixture/contract tests | yes | 9.0.2 | — |
| ROCm GPU | Phase 47 validation | no requirement | — | Keep tests CPU-only |
| Docker | Phase 47 validation | no requirement | — | Do not use |

**Missing dependencies with no fallback:** none.

**Missing dependencies with fallback:** none.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 [VERIFIED: local command] |
| Config file | `pyproject.toml` [CITED: pyproject.toml] |
| Quick run command | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py -x` |
| Full suite command | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TEST-01 | Fixture matrix covers attention, MoE, convolution, SSM/Mamba, embedding/positional, and linear projection positives. | unit/schema | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py::test_fixture_matrix_covers_required_families_and_classes -x` | no, Wave 0 |
| TEST-02 | Negative/degradation fixtures cover dynamic, partial, unsupported, taxonomy-only, and missing-metadata cases. | unit/schema | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py::test_fixture_matrix_covers_negative_and_degraded_cases -x` | no, Wave 0 |
| TEST-02 | Each degraded/negative fixture records expected status, missing evidence, degradation rationale, and warning prefixes. | unit/schema | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py::test_negative_and_degraded_fixtures_have_executable_expectations -x` | no, Wave 0 |
| TEST-01/02 | Contract doc states sidecar-only scope and no paper-scale/hardware/leaderboard/B200 claims. | docs guardrail | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py::test_contract_doc_preserves_claim_boundaries -x` | no, Wave 0 |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py -x`
- **Per wave merge:** `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py`
- **Phase gate:** Focused Phase 47 tests plus existing public contract guardrails green before `$gsd-verify-work`.

### Wave 0 Gaps

- [ ] `tests/sol_execbench/solar_derivation_fixtures.py` - shared fixture loader/validator.
- [ ] `tests/sol_execbench/test_solar_derivation_contract.py` - executable contract checks for TEST-01 and TEST-02.
- [ ] `tests/sol_execbench/fixtures/solar_derivation/*.json` - required matrix fixtures.
- [ ] `docs/internal/solar_derivation_contract.md` - internal sidecar-only contract.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Phase 47 does not add auth surfaces. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |
| V3 Session Management | no | Phase 47 does not add sessions. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |
| V4 Access Control | no | Phase 47 does not add service/API authorization. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |
| V5 Input Validation | yes | Validate fixture JSON required fields and enum-like values in test-side loader. [VERIFIED: codebase grep] |
| V6 Cryptography | no | Phase 47 does not add crypto. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |

### Known Threat Patterns for Fixture/Contract Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Fixture path traversal or loading unintended files | Tampering | Load only `*.json` from fixed `tests/sol_execbench/fixtures/solar_derivation/` using `Path` and no user-controlled path input. [VERIFIED: codebase grep] |
| Executing reference snippets from fixture JSON | Elevation of privilege | Phase 47 loader must parse JSON only and never `exec` fixture `reference` text. [CITED: .planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md] |
| Overclaiming derived evidence as validation | Spoofing/Repudiation | Add doc tests for forbidden claims and explicit scope-boundary fields. [CITED: .planning/REQUIREMENTS.md] |

## Sources

### Primary (HIGH confidence)

- `.planning/phases/47-derivation-contract-and-golden-fixture-matrix/47-CONTEXT.md` - locked user decisions, phase boundary, fixture/documentation shape.
- `.planning/REQUIREMENTS.md` - TEST-01, TEST-02, v1.10 exclusions, traceability.
- `.planning/ROADMAP.md` - Phase 47 goal, dependencies, success criteria.
- `.planning/STATE.md` - current milestone state and deferred scope.
- `.planning/research/SUMMARY.md` - milestone research summary and recommended phase shape.
- `AGENTS.md` - repository layout, commands, testing conventions, security/configuration instructions.
- `pyproject.toml` - Python, dependency, pytest, and Ruff configuration.
- `src/sol_execbench/core/scoring/amd_bound_graph.py` - `OpFamily`, `BoundGraph`, warning behavior.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - `OperatorWorkEstimate`, formula fields, confidence/warnings.
- `src/sol_execbench/core/scoring/amd_sol_v2.py` - schema version, aggregate statuses, coverage summary, warning prefixes.
- `tests/sol_execbench/test_amd_bound_graph.py` - taxonomy and extraction tests.
- `tests/sol_execbench/test_amd_sol_v2.py` - aggregate state and warning tests.
- `tests/sol_execbench/test_public_contract_guardrails.py` - canonical schema and CLI guardrails.

### Secondary (MEDIUM confidence)

- `.planning/research/ARCHITECTURE.md` - milestone architecture recommendations.
- `.planning/research/FEATURES.md` - target family and fixture landscape.
- `.planning/research/PITFALLS.md` - claim, coverage, and degradation pitfalls.
- `docs/analysis.md` - existing AMD SOL v2 sidecar and coverage semantics.

### Tertiary (LOW confidence)

- None used for Phase 47 recommendations.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies; versions verified locally and in `pyproject.toml`.
- Architecture: HIGH - phase is constrained to docs, fixtures, and tests; existing sidecar vocabulary is verified in code.
- Pitfalls: HIGH - local research and guardrail tests already identify overclaiming, taxonomy-only support, and public contract drift risks.

**Research date:** 2026-05-23
**Valid until:** 2026-06-22 for Phase 47 planning, or sooner if Phase 48 changes the derivation vocabulary before Phase 47 executes.
