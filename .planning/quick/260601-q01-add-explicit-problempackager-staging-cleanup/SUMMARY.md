---
quick_id: 260601-q01
slug: add-explicit-problempackager-staging-cleanup
status: complete
completed: 2026-06-01
---

# Summary

Completed explicit `ProblemPackager` staging cleanup work.

- Added explicit `ProblemPackager.close()` and context-manager support.
- Preserved `__del__` as a best-effort fallback for backwards compatibility.
- Added lifecycle tests for explicit cleanup, repeated close, context-manager
  cleanup, and `keep_output_dir` behavior.
- Updated the concern map to reflect the narrowed cleanup risk.

Verification recorded in the plan:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/driver/test_problem_packager.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/driver/problem_packager.py tests/sol_execbench/driver/test_problem_packager.py`
