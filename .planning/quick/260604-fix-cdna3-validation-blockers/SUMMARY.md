---
status: complete
completed_at: "2026-06-04"
slug: fix-cdna3-validation-blockers
---

# Summary

Fixed the CPU-device synchronization bug, narrowed the static source review
exception to only resolved `triton.language.load`, and replaced the RMSNorm
HIP example's warp-shuffle reduction with a conservative shared-memory
reduction for `gfx942` portability. Synced the embedded HIP sources in
`examples/hip_cpp/rmsnorm/solution_hip.json`.

Verification:

- `uv run pytest tests/sol_execbench/core/bench/test_utils.py tests/sol_execbench/core/bench/test_reward_hack.py -q`
- `uv run pytest tests/sol_execbench/driver/test_eval_driver.py::test_torch_compile_no_reward_hack -q`
- `uv run --with ruff ruff check src/sol_execbench/core/bench/utils.py src/sol_execbench/core/bench/reward_hack.py tests/sol_execbench/core/bench/test_reward_hack.py`

