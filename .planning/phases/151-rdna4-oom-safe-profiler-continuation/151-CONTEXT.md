---
status: complete
---

# Phase 151 Context

RDNA4 profiler-backed timing coverage reached `60/235` after a real
`rocprofv3` continuation run, but two host OOM events killed large
`eval_driver.py` Python processes at roughly 22-23 GiB RSS. The batch runner
must avoid repeating known OOM targets and must make blocked profiler attempts
visible to the same 235-problem coverage ledger.

This phase does not relax the profiler-backed claim boundary. Only full
workload coverage with `rocprofv3` kernel activity rows can count as
`profiler_backed`.
