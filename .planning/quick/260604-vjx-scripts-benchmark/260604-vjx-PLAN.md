# Quick Task 260604-vjx: 修复 scripts 重型脚本性能问题，保持 benchmark 语义正确

## Scope

Fix performance issues found in heavyweight scripts without changing benchmark semantics.

## Plan

1. Replace full in-memory subprocess output capture for long script commands with bounded tail capture and optional stdout artifact streaming.
2. Remove redundant I/O in derived AMD/SOLAR report generation while preserving serialized sidecars and score inputs.
3. Cache bundle checksum reads and stream/truncate workload reads where only a bounded prefix is needed.

## Verification

- Run focused tests for dataset runner, release/prerelease scripts, and execution closure.
- Run static import/compile checks for modified scripts if focused tests are unavailable.
