---
quick_id: 260601-q04
slug: extract-eval-driver-shape-dtype-helper
status: complete
completed: 2026-06-01
---

# Summary

Completed eval-driver shape/dtype helper extraction.

- Extracted `check_output_shape_dtype()` into importable correctness helpers.
- Replaced the generated eval driver inline shape/dtype loop while preserving
  status priority and trace behavior.
- Added helper and template integration coverage.

Verification recorded in the plan:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/core/bench/test_correctness.py tests/sol_execbench/driver/test_eval_driver.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/bench/correctness.py src/sol_execbench/driver/templates/eval_driver.py tests/sol_execbench/core/bench/test_correctness.py tests/sol_execbench/driver/test_eval_driver.py`
