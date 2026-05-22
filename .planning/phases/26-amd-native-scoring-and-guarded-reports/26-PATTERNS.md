# Phase 26 Pattern Map

**Status:** Complete  
**Date:** 2026-05-22

## Code Patterns

| New work | Closest existing pattern | Notes |
|----------|--------------------------|-------|
| `core/scoring/amd_score.py` | `core/baseline.py`, `core/reporting.py` | Pure dataclasses with `to_dict()` and no trace mutation. |
| Score guardrails | `core/scoring_guardrails.py` | Preserve concise warning strings and explicit claim levels. |
| Suite report aggregation | `core/reporting.py` | Derived report with schema version, `derived=True`, `canonical_output=trace_jsonl`. |
| Tests | `test_trace_reporting_and_score_guardrails.py`, `test_baseline_comparison.py` | Assert report shape and that public contracts remain unchanged. |

## Constraints

- Keep implementation independent of GPU availability.
- Keep all values JSON-serializable through `to_dict()`.
- Do not introduce new CLI surface unless later explicitly requested.
