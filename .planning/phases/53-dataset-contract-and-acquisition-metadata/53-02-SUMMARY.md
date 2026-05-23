# Phase 53-02 Summary: Downloader CLI And Idempotency

**Status:** complete
**Completed:** 2026-05-23

## Delivered

- Updated `scripts/download_solexecbench.py` with repeatable `--category`,
  `--output-root`, `--manifest`, `--revision`, `--force`, and `--verify-only`.
- Changed the downloader default root to `data/SOL-ExecBench/benchmark`.
- Added compare-before-write behavior that reuses identical files, rejects
  divergent local files unless `--force` is explicit, and never deletes unknown
  files.
- Added manifest writing after download or verify-only layout inspection.
- Aligned `scripts/download_data.sh` with the canonical SOL-ExecBench output
  root while preserving the separate FlashInfer trace download.

## Verification

```bash
uv run pytest tests/sol_execbench/test_download_solexecbench.py tests/sol_execbench/test_dataset_contract.py -x
bash -n scripts/download_data.sh
```

Result: `10 passed`; shell syntax valid.
Follow-up code review fixes added unsafe remote problem-name rejection and
fail-fast shell wrapper coverage; the focused downloader suite now has 10 tests.
