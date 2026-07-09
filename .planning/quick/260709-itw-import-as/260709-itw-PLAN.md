---
quick_id: 260709-itw
slug: import-as
status: planned
---

# Quick Task 260709-itw: 优化不必要的 import as

## Goal

Remove import aliases where the alias is identical to the imported name.

## Tasks

1. Run Ruff `PLC0414` to identify redundant import aliases.
2. Remove only `name as name` aliases without changing actual renamed imports.
3. Re-run Ruff `PLC0414` to verify no redundant aliases remain.
