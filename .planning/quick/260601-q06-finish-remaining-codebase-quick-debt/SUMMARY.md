---
quick_id: 260601-q06
slug: finish-remaining-codebase-quick-debt
status: complete
completed: 2026-06-01
---

# Summary

Completed the remaining quick-scope codebase debt pass:

- Main CLI now calls `ProblemPackager.close()` on normal completion and known
  compile/evaluation/no-trace exits.
- Eval driver output invocation is delegated to importable
  `call_and_collect_outputs()` with focused unit coverage.
- Reward-hack static review now blocks common `__import__`, `importlib`, and
  `getattr(os, ...)` process-execution spellings.
- `.planning/codebase/CONCERNS.md` now reflects the narrowed but still nonzero
  residual risks.

Verification:

- `test_utils.py`: 2 passed.
- `test_reward_hack.py`: 30 passed.
- `test_eval_driver.py`: 18 passed when run alone with the GPU driver visible.
- `test_utils.py` + `test_reward_hack.py`: 32 passed.
- Ruff passed on all touched Python files.

Note: running `test_utils.py`, `test_reward_hack.py`, and `test_eval_driver.py`
together produced environment-sensitive eval-driver failures where subprocesses
reported CPU/no active Triton driver. `test_eval_driver.py` passed when rerun
alone immediately afterward, so this was treated as a GPU runtime isolation issue,
not a code regression in this quick task.
