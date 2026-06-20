---
status: complete
completed: 2026-06-20
---

# Summary

Fixed the full pytest failures from the requested test run.

## Changes

- Re-exported `_write_blocked_sidecar` from
  `scripts/run_rdna4_profiler_timing_batch.py` so compatibility imports expose
  the internal helper expected by timing isolation tests.
- Restored the `bounded`/deferred evidence wording expected by v1.21 release
  documentation guardrails.
- Made the low-precision dtype assertions compare stable dtype names, avoiding
  brittle torch dtype identity behavior observed only during the full parallel
  suite.

## Verification

- `uv run pytest tests/sol_execbench/core/bench/test_timing_isolation.py::TestPidLockContentionInSidecar tests/sol_execbench/test_research_release_docs.py::test_v1_21_docs_keep_debt_reduction_separate_from_external_claims`
  - `3 passed`
- `uv run pytest tests/sol_execbench/test_low_precision_compatibility.py tests/sol_execbench/core/bench/test_timing_isolation.py::TestPidLockContentionInSidecar tests/sol_execbench/test_research_release_docs.py::test_v1_21_docs_keep_debt_reduction_separate_from_external_claims`
  - `13 passed`
- `uv run pytest tests/`
  - `1710 passed, 63 skipped in 179.62s`
