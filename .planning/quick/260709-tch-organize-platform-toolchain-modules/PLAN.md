---
status: complete
created: 2026-07-09
---

# Organize Platform Toolchain Modules

## Goal

Reduce flat `src/sol_execbench/core/platform/` layout by moving the clear
`toolchain*` feature cluster into a subpackage.

## Scope

- Move `toolchain.py` to `toolchain/__init__.py`.
- Move `toolchain_models.py` to `toolchain/models.py`.
- Move `toolchain_probes.py` to `toolchain/probes.py`.
- Move `toolchain_registry.py` to `toolchain/registry.py`.
- Move `toolchain_routing.py` to `toolchain/routing.py`.
- Update imports, docs, and tests.

## Out of Scope

- Other platform clusters (`environment`, `diagnostics`, `compatibility`).
- Behavioral changes.
- Backward compatibility facade modules for old flat submodule paths.

## Verification

- Focused toolchain/platform tests.
- Ruff check.
- Ty check.
- Full pytest if focused validation passes.
