# Quick Task Context

Task: Safely clean unnecessary `import ... as ...` alias noise introduced by
module-layout compatibility facades.

Scope:

- Keep legacy import paths and `sys.modules[__name__]` facade behavior intact.
- Replace facade helper aliases such as `_import_module` and `_sys` with direct
  `importlib` and `sys` imports.
- Remove clearly redundant same-name aliases such as `load_json as load_json`.
- Do not remove aliases that carry local meaning, avoid name collisions, or make
  tests more explicit.

Verification:

- Run Ruff.
- Run focused import smoke for canonical and legacy module paths.
- Run focused tests that cover CLI facades, report modules, and provenance.
