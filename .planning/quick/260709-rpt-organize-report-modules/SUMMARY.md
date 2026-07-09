---
status: complete
completed: 2026-07-09
---

# Organize Report Modules

## Summary

Grouped clear prefix-based `core/reports` feature clusters into subpackages:

- `consistency/`
- `matrix_diff/`
- `trust_summary/`
- `claim_upgrade/`
- `evaluation_stability/`

Updated source, tests, scripts, docs, and boundary allowlists to use the new
package paths. Kept shared report infrastructure (`reporting.py`,
`report_payloads.py`) at the reports package top level.

## Verification

- `uv run pytest tests/sol_execbench/core/reports tests/sol_execbench/core/evidence/test_v1_19_evidence_examples.py tests/sol_execbench/core/evidence/test_v1_20_evidence_quality_docs.py tests/sol_execbench/core/bench/test_agent_feedback.py tests/sol_execbench/core/bench/test_profile_summary.py tests/sol_execbench/core/reports/test_payload_schema_boundaries.py`
  - 126 passed
- `uv run --with ruff ruff check .`
  - passed
- `uv run ty check`
  - passed
- `uv run pytest tests/`
  - 1885 passed, 41 skipped
