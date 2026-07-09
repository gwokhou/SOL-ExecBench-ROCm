---
status: complete
created: 2026-07-09
---

# Organize Platform Matrix Modules

## Goal

Reduce flat `src/sol_execbench/core/platform/` layout by moving the clearest
matrix-related feature clusters into subpackages.

## Scope

- Move `dependency_matrix*` modules into `platform/dependency_matrix/`.
- Move `docker_matrix*` modules into `platform/docker_matrix/`.
- Update imports, tests, scripts, docs, and command module references.

## Out of Scope

- Other platform clusters (`compatibility`, `environment`, `toolchain`,
  `diagnostics`).
- Behavioral changes.
- Backward compatibility facade modules for old flat submodule paths.

## Verification

- Focused platform matrix tests.
- Ruff check.
- Ty check.
- Full pytest if focused validation passes.
