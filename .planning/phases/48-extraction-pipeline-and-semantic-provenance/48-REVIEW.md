---
phase: 48-extraction-pipeline-and-semantic-provenance
reviewed: 2026-05-23T05:38:32Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - src/sol_execbench/core/scoring/solar_derivation.py
  - tests/sol_execbench/test_solar_derivation_evidence.py
  - tests/sol_execbench/test_public_contract_guardrails.py
findings:
  critical: 0
  warning: 2
  info: 0
  total: 2
status: issues_found
---

# Phase 48: Code Review Report

**Reviewed:** 2026-05-23T05:38:32Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Narrative Findings (AI reviewer)

## Summary

Reviewed the SOLAR derivation sidecar implementation and the targeted evidence/public-contract guardrail tests. The scoped tests pass (`34 passed in 4.37s`), but the parser still accepts malformed or drifting payloads that the current tests do not cover.

## Warnings

### WR-01: Sidecar Parser Silently Accepts Unknown Schema Fields

**File:** `src/sol_execbench/core/scoring/solar_derivation.py:303`

**Issue:** `solar_derivation_from_dict()` and the nested parsers only check that required keys exist. They never reject unknown top-level, group, subrole, tensor, source, or `source_boundary` fields. That makes schema drift easy to miss: a payload can carry new claim or provenance fields, parse successfully, and then have those fields silently dropped by `to_dict()`. This is risky for the Phase 48 contract because derived SOLAR evidence is explicitly noncanonical and claim-boundary sensitive.

**Fix:** Add allowed-key validation alongside required-key validation, matching the stricter pattern already used by `amd_hardware_models._require_keys()`, and apply it recursively to all nested sidecar objects.

```python
def _require_exact_keys(
    payload: dict[str, Any],
    allowed: frozenset[str] | set[str],
    *,
    source: str,
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ValueError(f"{source} contains unknown field(s): {', '.join(unknown)}")
    _require_keys(payload, allowed, source=source)
```

### WR-02: Shape Parser Accepts Boolean Dimensions

**File:** `src/sol_execbench/core/scoring/solar_derivation.py:998`

**Issue:** `_parse_shape()` uses `isinstance(item, int)`, which accepts `True` and `False` because `bool` is a subclass of `int` in Python. A malformed sidecar can therefore parse `shape: [true, 4]` as valid evidence and retain a boolean dimension, weakening schema validation around tensor metadata and downstream confidence decisions.

**Fix:** Require the concrete type to be `int` and reject invalid dimensions explicitly.

```python
for index, item in enumerate(value):
    if type(item) is not int:
        raise ValueError(f"{source}.{key}[{index}] must be an integer")
    if item < 0:
        raise ValueError(f"{source}.{key}[{index}] must be non-negative")
    shape.append(item)
```

---

_Reviewed: 2026-05-23T05:38:32Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
