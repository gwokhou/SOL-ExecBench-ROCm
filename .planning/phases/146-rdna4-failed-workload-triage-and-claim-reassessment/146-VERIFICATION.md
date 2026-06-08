---
phase: 146
title: RDNA4 failed workload triage and claim reassessment
status: verified
verified: 2026-06-08
---

# Phase 146 Verification

## Commands

```bash
uv run python - <<'PY'
import json
from pathlib import Path

summary = json.loads(
    Path("out/rdna4-failure-triage-v131/phase146-failure-triage.json").read_text()
)
assert summary["failed_workload_records"] == 146
assert sum(summary["failure_class_counts"].values()) == 146
assert summary["unique_failed_problems"] == 35
assert summary["claim_reassessment"]["stronger_rdna4_claim_allowed"] is False
PY

UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py::test_phase_141_rdna4_public_claims_stay_bounded -q
```

## Result

- Failure-triage summary checks passed.
- Public guardrail test passed.
- Public docs keep denominator and failure counts visible while adding v1.31
  classification context.

## Verification Conclusion

Phase 146 satisfies `RDNA4-FU-FAIL-01` and `RDNA4-FU-CLAIM-01`. The failed
workloads are classified, and public claims remain bounded.
