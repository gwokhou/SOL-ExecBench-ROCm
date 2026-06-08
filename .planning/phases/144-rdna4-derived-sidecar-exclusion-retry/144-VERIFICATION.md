---
phase: 144
title: RDNA4 derived sidecar exclusion retry
status: verified
verified: 2026-06-08
---

# Phase 144 Verification

## Commands

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_run_derived_isolated.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/run_derived_isolated.py tests/sol_execbench/test_run_derived_isolated.py
systemd-run --user --collect --unit sol-phase144-derived-retry --same-dir --setenv=UV_CACHE_DIR=/home/guohao/.cache/uv -- /home/guohao/.cargo/bin/uv run scripts/run_derived_isolated.py data/SOL-ExecBench/benchmark -o out/rdna4-full-dataset/run --problem-id-file out/rdna4-derived-retry-v131/logs/retry-problem-ids.txt --amd-score-report out/rdna4-derived-retry-v131/amd-score.json --amd-sol-bound-dir out/rdna4-derived-retry-v131/amd-sol-bound --solar-derivation out/rdna4-derived-retry-v131/solar-derivation --gpu-architecture gfx1200 --uv-bin /home/guohao/.cargo/bin/uv --uv-cache-dir /home/guohao/.cache/uv --launch-mode systemd --memory-max 24G --memory-swap-max 0 --status-jsonl out/rdna4-derived-retry-v131/logs/isolated-derived-status.jsonl --log-file out/rdna4-derived-retry-v131/logs/isolated-derived.log --continue-on-failure
```

## Results

- Focused pytest: 5 passed.
- Focused Ruff check: passed.
- Isolated retry status: 8 problem attempts recorded.
- Full problem retries recovered: 1.
- Memory-blocked problem retries: 7, all reported `Finished with result:
  oom-kill`, `status=137`, `Memory peak: 24.0G`, and `Memory swap peak: 0B`
  in the retry log.
- Retry sidecars generated: 76 AMD SOL v2 files and 76 SOLAR derivation files.
- Codex/calling shell stability: preserved; OOM stayed inside transient
  systemd child units.

## Verification Conclusion

The Phase 144 retry path is verified for targeted problem filtering,
isolated/memory-capped execution, and auditable recovered-vs-blocked
classification. Remaining OOM rows should stay as explicit memory blockers
unless a larger-memory host or a more memory-efficient derived generator rerun
changes the evidence.

