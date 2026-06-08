# Phase 138 Blocked

## Status

Blocked on 2026-06-07 before full dataset execution.

## Blocking Conditions

1. Missing dataset assets.
   - Expected default path: `data/SOL-ExecBench/benchmark`
   - Observed: only `data/` exists.
   - Rechecked 2026-06-07T15:41:14Z: `find data -maxdepth 4 -type d`
     still reports only `data`.

2. ROCm device nodes unavailable inside `uv run`.
   - Outer shell host tools see RDNA4 `gfx1200`.
   - `UV_CACHE_DIR=/tmp/uv-cache uv run python ...` reports:
     - `/dev/kfd False`
     - `/dev/dri False`
     - HIP `7.1.25424`
     - `torch.cuda.is_available() False`
     - device count `0`
   - Rechecked 2026-06-07T15:41:14Z with the same result.

## Required User/Environment Action

- Provide or create the migrated local SOL ExecBench benchmark dataset under
  `data/SOL-ExecBench/benchmark`, or give the exact alternate dataset path.
- Run Codex/`uv run` in an environment with ROCm device passthrough so the same
  Python process that launches `scripts/run_dataset.py` can access `/dev/kfd`,
  `/dev/dri`, and `gfx1200`.

## Non-Claim

This blocker is not RDNA4 full dataset validation evidence and must not be used
to mark RDNA4 benchmark-grade validation complete.
