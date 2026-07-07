# SOLAR Derivation Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Split the oversized SOLAR derivation evidence module into focused scoring modules while preserving the existing public API.

**Architecture:** Keep `src/sol_execbench/core/scoring/solar_derivation.py` as a facade that re-exports the same public names. Move dataclasses/constants into models, payload parsing into parsing, evidence construction into builders, and group coverage/status helpers into coverage so existing callers keep importing from `solar_derivation`.

**Tech Stack:** Python 3.12, dataclasses, pytest, Ruff.

---

### Task 1: Add Split-Module Compatibility Test

**Files:**
- Modify: `tests/sol_execbench/test_solar_derivation_evidence.py`

- [x] **Step 1: Write the failing test**

Add a test near the existing parser round-trip tests:

```python
def test_solar_derivation_split_modules_export_facade_symbols():
    from sol_execbench.core.scoring import solar_derivation
    from sol_execbench.core.scoring.solar_derivation_builders import (
        build_solar_derivation_evidence as split_build,
    )
    from sol_execbench.core.scoring.solar_derivation_models import (
        SolarDerivationEvidence as split_evidence,
    )
    from sol_execbench.core.scoring.solar_derivation_parsing import (
        solar_derivation_from_dict as split_from_dict,
    )

    assert split_evidence is solar_derivation.SolarDerivationEvidence
    assert split_build is solar_derivation.build_solar_derivation_evidence
    assert split_from_dict is solar_derivation.solar_derivation_from_dict
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py::test_solar_derivation_split_modules_export_facade_symbols -q`

Expected: FAIL with `ModuleNotFoundError` for the new split modules.

### Task 2: Move Models

**Files:**
- Create: `src/sol_execbench/core/scoring/solar_derivation_models.py`
- Modify: `src/sol_execbench/core/scoring/solar_derivation.py`

- [x] **Step 1: Move constants and dataclasses**

Move these names into `solar_derivation_models.py`:

```python
SOLAR_DERIVATION_SCHEMA_VERSION
SOLAR_DEFAULT_AMD_HARDWARE_MODEL_REF
SOLAR_BOUND_LIMITING_RESOURCES
SolarEvidenceSource
SolarTensorEvidence
SolarSubroleEvidence
SolarFormulaEvidence
SolarByteEvidence
SolarBoundEvidence
SolarSemanticGroupEvidence
SolarConfidenceClassification
SolarCoverageSourceRef
SolarFamilyCoverage
SolarCoveragePattern
SolarCoverageSummary
SolarAggregateStatus
SolarDerivationEvidence
```

Import helper functions from `solar_derivation_status` directly where model `to_dict()` methods need them. Import coverage/status builders lazily inside `SolarDerivationEvidence.to_dict()` to avoid import cycles.

- [x] **Step 2: Re-export moved names from facade**

Update `solar_derivation.py` to import the moved names from `solar_derivation_models.py`.

### Task 3: Move Parser

**Files:**
- Create: `src/sol_execbench/core/scoring/solar_derivation_parsing.py`
- Modify: `src/sol_execbench/core/scoring/solar_derivation.py`

- [x] **Step 1: Move payload parser functions**

Move `solar_derivation_from_dict()` and all `_parse_*`, `_ensure_*`, `_require_*`, and `*_from_dict()` helpers into `solar_derivation_parsing.py`. Import model classes and constants from `solar_derivation_models.py`.

- [x] **Step 2: Re-export parser from facade**

Update `solar_derivation.py` to import `solar_derivation_from_dict` from `solar_derivation_parsing.py`.

### Task 4: Move Builders and Coverage Helpers

**Files:**
- Create: `src/sol_execbench/core/scoring/solar_derivation_builders.py`
- Create: `src/sol_execbench/core/scoring/solar_derivation_coverage.py`
- Modify: `src/sol_execbench/core/scoring/solar_derivation.py`

- [x] **Step 1: Move public builders and evidence construction**

Move `build_solar_derivation_evidence()`, `derive_solar_derivation_evidence()`, `classify_solar_confidence()`, and evidence/subrole/source helper functions into `solar_derivation_builders.py`.

- [x] **Step 2: Move coverage/status helpers**

Move `_coverage_for_groups()`, coverage pattern/source-ref helpers, `_aggregate_status_for_groups()`, and warning/status utility wrappers into `solar_derivation_coverage.py`.

- [x] **Step 3: Re-export builder names from facade**

Update `solar_derivation.py` to import the public builder functions from `solar_derivation_builders.py`.

### Task 5: Verify Behavior and Style

**Files:**
- Test: `tests/sol_execbench/test_solar_derivation_evidence.py`
- Test: `tests/sol_execbench/test_amd_native_score.py`

- [x] **Step 1: Run focused scoring tests**

Run: `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_native_score.py -q`

Expected: all selected tests pass.

- [x] **Step 2: Run Ruff on touched files**

Run: `uv run --with ruff ruff check src/sol_execbench/core/scoring/solar_derivation.py src/sol_execbench/core/scoring/solar_derivation_models.py src/sol_execbench/core/scoring/solar_derivation_parsing.py src/sol_execbench/core/scoring/solar_derivation_builders.py src/sol_execbench/core/scoring/solar_derivation_coverage.py tests/sol_execbench/test_solar_derivation_evidence.py`

Expected: `All checks passed!`

- [x] **Step 3: Review diff**

Run: `git diff --stat` and `git diff --check`

Expected: facade is smaller, new modules are focused, and there are no whitespace errors.
