---
phase: 47-derivation-contract-and-golden-fixture-matrix
reviewed: 2026-05-23T04:36:01Z
depth: standard
files_reviewed: 22
files_reviewed_list:
  - docs/internal/solar_derivation_contract.md
  - tests/sol_execbench/solar_derivation_fixtures.py
  - tests/sol_execbench/test_solar_derivation_contract.py
  - tests/sol_execbench/test_public_contract_guardrails.py
  - tests/sol_execbench/fixtures/solar_derivation/attention_degraded_partial_mask.json
  - tests/sol_execbench/fixtures/solar_derivation/attention_positive.json
  - tests/sol_execbench/fixtures/solar_derivation/attention_unsupported_dynamic_axes.json
  - tests/sol_execbench/fixtures/solar_derivation/convolution_degraded_missing_padding.json
  - tests/sol_execbench/fixtures/solar_derivation/convolution_positive.json
  - tests/sol_execbench/fixtures/solar_derivation/convolution_unsupported_dynamic_kernel.json
  - tests/sol_execbench/fixtures/solar_derivation/embedding_positional_degraded_dynamic_indices.json
  - tests/sol_execbench/fixtures/solar_derivation/embedding_positional_positive.json
  - tests/sol_execbench/fixtures/solar_derivation/embedding_positional_unsupported_missing_metadata.json
  - tests/sol_execbench/fixtures/solar_derivation/linear_projection_degraded_missing_shape.json
  - tests/sol_execbench/fixtures/solar_derivation/linear_projection_positive.json
  - tests/sol_execbench/fixtures/solar_derivation/linear_projection_unsupported_missing_metadata.json
  - tests/sol_execbench/fixtures/solar_derivation/moe_degraded_dynamic_routing.json
  - tests/sol_execbench/fixtures/solar_derivation/moe_positive.json
  - tests/sol_execbench/fixtures/solar_derivation/moe_unsupported_taxonomy_only.json
  - tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_degraded_missing_recurrence.json
  - tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_positive.json
  - tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_unsupported_custom_scan.json
findings:
  critical: 3
  warning: 1
  info: 0
  total: 4
status: findings
---

# Phase 47: Code Review Report

**Reviewed:** 2026-05-23T04:36:01Z
**Depth:** standard
**Files Reviewed:** 22
**Status:** findings

## Summary

Reviewed the SOLAR derivation contract document, fixture loader/validator, guardrail tests, and the full golden fixture JSON matrix. The checked-in fixtures currently validate and the targeted test suite passes, but the contract boundary is not stable enough for downstream phases: the Markdown schema contradicts the executable fixture schema, and the validator accepts fixture states that the written contract explicitly forbids.

Verification run:

```bash
uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py
```

Result: 27 passed.

## Critical Issues

### CR-01: Markdown fixture schema contradicts executable fixture schema

**File:** `docs/internal/solar_derivation_contract.md:103`

**Issue:** The documented schema requires `category` and scope fields named `leaderboard_ready` / `nvidia_b200_equivalence`, but the executable loader requires `negative_category`, `hosted_leaderboard_ready`, and `nvidia_blackwell_b200_equivalence` (`tests/sol_execbench/solar_derivation_fixtures.py:25`, `tests/sol_execbench/solar_derivation_fixtures.py:51`). The actual JSON fixtures also use the loader names, for example `tests/sol_execbench/fixtures/solar_derivation/attention_positive.json:5` and `tests/sol_execbench/fixtures/solar_derivation/attention_positive.json:20`. This makes the written contract an unsafe source of truth for later extractor/modeling work: an implementation following the document will emit fixture sidecars that fail the validator and guardrail tests.

**Fix:**

```json
{
  "case_id": "attention_positive_dense_qkv",
  "family": "attention",
  "fixture_class": "positive",
  "negative_category": null,
  "scope_boundary": {
    "paper_scale_dataset": false,
    "hosted_leaderboard_ready": false,
    "nvidia_blackwell_b200_equivalence": false,
    "real_hardware_validation": false
  }
}
```

Update the example and required-field lists in `docs/internal/solar_derivation_contract.md` to match the loader and fixtures, or rename the loader/fixtures consistently if the document is intended to be canonical.

### CR-02: Scope-boundary validator allows forbidden positive claims

**File:** `tests/sol_execbench/solar_derivation_fixtures.py:132`

**Issue:** The contract says all scope-boundary values must be `false` for Phase 47 fixtures (`docs/internal/solar_derivation_contract.md:177`), but `validate_solar_derivation_fixture()` only verifies that each value is a boolean. A fixture with `"real_hardware_validation": true` or `"hosted_leaderboard_ready": true` passes the validator, so the reusable contract loader does not enforce the public claim guardrails it is supposed to protect.

**Fix:**

```python
for key in sorted(REQUIRED_SCOPE_BOUNDARY):
    if not isinstance(scope_boundary[key], bool):
        raise ValueError(f"{source}.scope_boundary.{key} must be a boolean")
    if scope_boundary[key] is not False:
        raise ValueError(f"{source}.scope_boundary.{key} must be false")
```

Add a regression test that mutates one `_valid_fixture()["scope_boundary"]` value to `True` and asserts `validate_solar_derivation_fixture()` rejects it.

### CR-03: Fixture class is not tied to confidence/status semantics

**File:** `tests/sol_execbench/solar_derivation_fixtures.py:149`

**Issue:** `_validate_expectation()` validates that `expected_confidence` and `expected_status` are individually valid vocabulary values, but it never enforces the class-to-state contract. A `positive` fixture can pass with `expected_confidence: "unsupported"` and `expected_status: "unscored"` as long as its rationale is null and evidence lists are shaped correctly; a `degraded` fixture can also pass with `supported` / `scored`. That breaks the golden matrix contract where positive means supported/scored, degraded means inexact/degraded, and unsupported/negative means unsupported/unscored.

**Fix:**

```python
expected_states = {
    "positive": ("supported", "scored"),
    "degraded": ("inexact", "degraded"),
    "unsupported": ("unsupported", "unscored"),
    "negative": ("unsupported", "unscored"),
}
expected_confidence, expected_status = expected_states[fixture_class]
if confidence != expected_confidence:
    raise ValueError(
        f"{source}.expected_confidence must be {expected_confidence} for {fixture_class}"
    )
if status != expected_status:
    raise ValueError(
        f"{source}.expected_status must be {expected_status} for {fixture_class}"
    )
```

Add parameterized validator tests for the invalid cross-products, especially positive/unscored and degraded/scored.

## Warnings

### WR-01: Test helper import can resolve to the wrong module in polluted environments

**File:** `tests/sol_execbench/test_solar_derivation_contract.py:14`

**Issue:** Both Phase 47 test files append the test directory to the end of `sys.path` and then import `solar_derivation_fixtures` as a top-level module (`tests/sol_execbench/test_public_contract_guardrails.py:37`). Because `append()` gives existing path entries priority, a same-named installed or repository-root module can shadow the intended test helper. That makes these guardrail tests brittle in developer environments and CI jobs with extra path entries.

**Fix:** Prefer a package-safe import path, or place the helper in a normal test package/conftest module. If path mutation is kept, insert the test directory at the front and guard against duplicates:

```python
TEST_DIR = str(Path(__file__).resolve().parent)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)
```

---

_Reviewed: 2026-05-23T04:36:01Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
