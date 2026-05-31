---
quick_id: 260531-uki
slug: add-remaining-requires-rocm-e2e-coverage
status: completed
created: 2026-05-31T14:00:36Z
---

# Quick Task 260531-uki: Add remaining requires_rocm E2E coverage

## Goal

Add focused `requires_rocm` regressions for the highest-value remaining E2E
gaps after `260531-u2s`:

1. Public `sol-execbench` HIP/C++ compile/evaluate path.
2. Public `sol-execbench --static-evidence auto` sidecar path.
3. `scripts/run_dataset.py` existing-pass reuse and `--rerun` closure behavior.

## Tasks

| Task | Files | Action | Verify | Done |
| --- | --- | --- | --- | --- |
| 1 | `tests/examples/test_rocm_cli_paths.py` | Add a one-workload HIP/C++ CLI test over `examples/hip_cpp/rmsnorm`. | Targeted ROCm test passes. | done |
| 2 | `tests/examples/test_rocm_cli_paths.py` | Add static evidence assertions for the HIP/C++ CLI run. | Static evidence sidecar exists and is diagnostic-only. | done |
| 3 | `tests/examples/test_rocm_cli_paths.py` | Add dataset runner first-run, existing-pass skip, and `--rerun` closure assertions. | Full `requires_rocm` set passes. | done |

## Scope Boundaries

- No Docker build/run.
- No dependency relock.
- No new hardware target claims.
- ROCm/GPU access is limited to `requires_rocm` pytest execution.
