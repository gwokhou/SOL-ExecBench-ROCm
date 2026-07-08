---
status: in-progress
date: 2026-07-08
---

# Quick Task 260708-dti: Reduce Source-File Coupling

## Scope

Enforce existing import-boundary tests and remove the AMD SOL v1 model/coverage
cycle without changing public artifact payloads.

## Tasks

1. Run the focused module-boundary test and confirm it fails on the current
   two-node cycle.
2. Refactor AMD SOL v1 coverage summary computation so models do not import
   coverage logic.
3. Run focused boundary and AMD SOL tests.
