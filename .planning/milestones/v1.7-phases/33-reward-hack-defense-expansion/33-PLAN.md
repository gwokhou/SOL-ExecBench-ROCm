# Phase 33 Plan: Reward-Hack Defense Expansion

**Status:** Complete

## Tasks

- [x] Add structured static source review issues and review payloads.
- [x] Detect hidden async/non-default stream patterns before user-code import.
- [x] Detect semantic output cache patterns using data pointers, content keys,
  cache dictionaries, or hashing.
- [x] Detect unauthorized file I/O, embedded payload decoding, dynamic native
  loading, subprocess, and network access patterns.
- [x] Detect precision downgrade abuse for float32 output contracts.
- [x] Integrate blocking static review into the evaluation driver before
  importing submitted Python source.
- [x] Preserve existing runtime guards for monkey-patching, lazy outputs,
  thread injection, and eval-driver integrity.
- [x] Add focused pure and subprocess tests for malicious and legitimate
  fixtures.
- [x] Document reward-hack review behavior.

## Verification

- `uv run pytest tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/driver/test_eval_driver.py`
- `uv run ruff check src/sol_execbench/core/bench/reward_hack.py src/sol_execbench/driver/templates/eval_driver.py tests/sol_execbench/core/bench/test_reward_hack.py tests/sol_execbench/driver/test_eval_driver.py`
