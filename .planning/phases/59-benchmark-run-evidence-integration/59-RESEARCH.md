# Phase 59 Research: Benchmark Run Evidence Integration

**Researched:** 2026-05-25
**Status:** Complete

## Current Output Path

`src/sol_execbench/cli/main.py` parses eval-driver stdout into canonical
`Trace` objects and writes exactly those traces to stdout and/or `--output`.
This is the compatibility-sensitive path used by downstream consumers.

## Recommended Integration

Use an opt-in sidecar:

- `SOLEXECBENCH_ENV_SNAPSHOT_PATH=/path/to/file.json` writes the snapshot there.
- `SOLEXECBENCH_ENV_SNAPSHOT=1` writes next to `--output` as
  `<output>.environment.json`.
- If enabled without a path and without `--output`, skip with a stderr warning
  rather than polluting stdout.

This gives tests and future consumers a stable integration point while keeping
default CLI behavior identical.

## Verification Focus

- Default `--json` / `--output` emits unchanged trace JSONL.
- Enabled sidecar writes snapshot JSON and does not alter benchmark exit status.
- Collection failure or unavailable tools is represented in sidecar metadata,
  not benchmark status.

