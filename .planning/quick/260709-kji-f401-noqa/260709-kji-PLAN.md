---
quick_id: 260709-kji
slug: f401-noqa
status: planned
---

# Quick Task 260709-kji: 清理不必要的 F401 noqa 重导出

## Goal

Replace unnecessary `F401` suppression comments with explicit export intent where
the import exists for facade compatibility.

## Tasks

1. Inspect remaining `F401` suppressions in source, scripts, and tests.
2. Convert facade compatibility imports to explicit `__all__` exports where useful.
3. Keep only necessary `F401` suppressions that preserve runtime behavior.
4. Run focused lint and related tests.
