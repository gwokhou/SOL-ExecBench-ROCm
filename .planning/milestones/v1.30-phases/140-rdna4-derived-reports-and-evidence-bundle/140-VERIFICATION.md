---
phase: 140
title: RDNA4 derived reports and evidence bundle
status: verified
verified_at: 2026-06-08
---

# Phase 140 Verification

## Commands

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_derived_isolated.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_run_dataset_execution_closure.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/run_derived_isolated.py tests/sol_execbench/test_run_derived_isolated.py`
- `systemd-run --user --wait --collect --unit sol-derived-amd-score-final --property MemoryMax=20G --property MemorySwapMax=0 --same-dir -- env UV_CACHE_DIR=/home/guohao/.cache/uv uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --phase derived -o out/rdna4-full-dataset/run --long-tail-exclusions out/rdna4-derived-reports/rdna4-derived-long-tail-exclusions.json --amd-score-report out/rdna4-derived-reports/amd-score.json --amd-sol-bound-dir out/rdna4-derived-reports/amd-sol-bound --solar-derivation out/rdna4-derived-reports/solar-derivation`
- `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/report_paper_denominator.py ...`
- `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/report_parity_gaps.py ...`
- `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/report_amd_bound_sanity.py ...`
- `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/report_consistency.py ...`
- `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/report_claim_upgrade.py ...`
- `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/report_trust_summary.py ...`

## Results

- 56 targeted tests passed.
- Ruff passed for the isolated derived runner and its tests.
- Final AMD score generation completed in a systemd user unit with result
  `success`; memory cap was 20G and swap cap was 0.
- Sidecar coverage audit found 1895 traces, 1839 traces with both sidecars, 56
  temporarily excluded traces, and 0 unexcluded missing sidecars.
- Bundle manifest records checksums for score, reports, exclusion config,
  isolated-run logs/status, Phase 138 closure/summary, and Phase 139 stability.

## Residual Risk

- `out/` artifacts are local generated evidence and are not committed.
- Several derived sidecar builders remain memory-heavy and are temporarily
  excluded for Phase 140: see
  `out/rdna4-derived-reports/rdna4-derived-long-tail-exclusions.json`.
- Consistency/trust reports intentionally preserve blocker status rather than
  upgrading claims.
