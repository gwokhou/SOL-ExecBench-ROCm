# Quick Task: Fix rocprofv3 TMPDIR segfault

## Goal

Close the root cause for RDNA4 profiler failures reported as:

- `rocprofv3 command failed with exit code -11`
- `rocprofv3 did not produce a CSV timing output`

## Plan

- Reproduce the failing profiler path with one known target.
- Isolate environment differences between batch and direct `rocprofv3`.
- Fix the batch runner environment so staged subprocesses receive an absolute temp directory.
- Use the active Python interpreter for staged eval driver execution.
- Verify with unit tests and real ROCm profiler repros.
