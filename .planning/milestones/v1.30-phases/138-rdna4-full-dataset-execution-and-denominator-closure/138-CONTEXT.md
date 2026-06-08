# Phase 138: RDNA4 Full Dataset Execution and Denominator Closure - Context

## Scope

Phase 138 owns RDNA4 full dataset execution and complete denominator closure for
requirement RDNA4-VAL-04. It must produce real run artifacts, traces, summaries,
and execution-closure records for every selected problem/workload outcome.

## Preconditions From Earlier Phases

- Phase 136 defined the RDNA4 denominator and added explicit long-tail
  exclusion controls.
- Phase 137 created the long-running runbook and confirmed the host can see
  RDNA4 `gfx1200` through ROCm host tools.
- Phase 137 also confirmed the current `uv`/pytest execution environment
  cannot see `/dev/kfd` or `/dev/dri`; PyTorch ROCm reports HIP `7.1.25424`
  but zero visible devices in that environment.

## Current Local State

Preflight on 2026-06-07 found:

- `data/` exists.
- `data/SOL-ExecBench/benchmark` does not exist.
- No migrated SOL ExecBench benchmark dataset is available under the default
  local path.
- A full RDNA4 dataset run would be invalid without local benchmark assets and
  GPU device visibility in the exact command environment.

## Blocking Conditions

Phase 138 cannot complete in the current environment until both are true:

1. The user provides a local migrated dataset path, expected default:
   `data/SOL-ExecBench/benchmark`.
2. The command environment used by `uv run scripts/run_dataset.py` can access
   ROCm device nodes and PyTorch ROCm sees `gfx1200`.

## Claim Boundary

This context is not full dataset validation evidence. It is a blocker record
and execution plan seed. Do not mark RDNA4 dataset validation complete from
this phase until real traces and execution-closure artifacts exist.
