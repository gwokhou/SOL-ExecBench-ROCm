# Quick Task: Fix profiler resource guards

## Goal

Fix the RDNA4 profiler batch resource issues found after `/tmp` filled:

- Count ROCm GPUs correctly from `rocm-smi --showid`.
- Avoid HIP API trace CSV growth by default.
- Clean per-target staging directories by default.
- Compact profiler timing sidecars by default so parsed rows do not duplicate large CSVs.

## Verification

- Run focused timing isolation and profiler batch tests.
- Confirm defaults preserve explicit opt-in paths for debugging.
