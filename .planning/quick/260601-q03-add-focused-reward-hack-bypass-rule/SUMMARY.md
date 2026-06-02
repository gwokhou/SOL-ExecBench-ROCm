---
quick_id: 260601-q03
slug: add-focused-reward-hack-bypass-rule
status: complete
completed: 2026-06-01
---

# Summary

Completed focused reward-hack static-review expansion.

- Added blocking for `os.system`, `os.popen`, `os.spawn*`, `os.exec*`, and
  `pty.spawn` process-execution patterns.
- Added malicious and allowed-case reward-hack unit coverage.
- Preserved existing allowed cases and did not change trace schema or runtime
  evaluation semantics.

Verification recorded in the plan:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/driver/test_eval_driver.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/bench/reward_hack.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/driver/test_eval_driver.py`
