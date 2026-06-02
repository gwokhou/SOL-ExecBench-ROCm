---
quick_id: 260601-q02
slug: strengthen-static-evidence-profile-claim-guardrails
status: complete
completed: 2026-06-01
---

# Summary

Completed static evidence and profiler claim-boundary guardrail work.

- Normalized diagnostic-only wording for static evidence and `rocprofv3`
  profile sidecars.
- Added public contract guardrails for forbidden authority wording and required
  diagnostic-only phrasing.
- Updated the concern map to reflect the remaining phrase-test residual risk.

Verification recorded in the plan:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/sol_execbench/test_public_contract_guardrails.py`
