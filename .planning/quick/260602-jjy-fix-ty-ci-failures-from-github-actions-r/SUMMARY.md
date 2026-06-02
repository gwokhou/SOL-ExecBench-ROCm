---
quick_id: 260602-jjy
slug: fix-ty-ci-failures-from-github-actions-r
status: complete
completed_at: "2026-06-02T06:30:00Z"
---

# Quick Task 260602-jjy: Fix Ty CI failures

## Result

Fixed the GitHub Actions failure from run 26798761166, job 79000755601. The
observed failure was `uv run ty check`; after fixing Ty diagnostics, the
workflow's subsequent CPU-safe pytest commands also exposed stale contract
drift, which was fixed in the same CI repair pass.

## Changes

- Added precise typing for static evidence aggregation, runtime evidence,
  matrix/docker schema literals, dynamic monkeypatching, dict narrowing, and
  nullable model accesses.
- Updated tests to make intentional dynamic or invalid payload cases explicit
  to Ty without changing assertions.
- Allowed exact ROCm system compile flags for native examples:
  `-I/opt/rocm/include` and `-L/opt/rocm/lib`.
- Restored public guardrail wording and planning compatibility text required by
  CPU-safe audit tests.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check` - passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .` - passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench --ignore=tests/sol_execbench/driver/test_eval_driver.py --ignore=tests/sol_execbench/test_e2e.py` - 1286 passed, 57 skipped.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/examples/test_examples.py -k consistency` - 14 passed.
