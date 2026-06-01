# Phase 107: Staged User Import Isolation - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase changes Python solution loading so staged user files are imported by
unique file-based module identities instead of ordinary dotted names that can
collide with preexisting `sys.modules` entries.

</domain>

<decisions>
## Implementation Decisions

### Import Isolation
- Use `importlib.util.spec_from_file_location()` for Python solution entry
  files.
- Generate deterministic unique module names from solution content and entry
  path.
- Preserve the existing native ROCm shared-object loading path.
- Keep the staging directory on `sys.path` for existing absolute sibling import
  compatibility, but do not import the entry module through `import_module()`.

### Package Compatibility
- Support simple files such as `kernel.py::run`.
- Support package-like paths such as `pkg/kernel.py::run` by registering a
  unique synthetic package chain for relative imports.
- Do not change solution schema or public entry-point format.

### Testing Boundary
- Add focused `eval_runtime` tests for collisions with `sys.modules`.
- Cover both simple file and package-like entry paths.
- Assert that the loaded function comes from the staged file, not from a
  preexisting module with the same ordinary name.

### the agent's Discretion
The exact helper names and unique module-name format are implementation
details, as long as names are stable for one staged solution and cannot collide
with ordinary user module names.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `parse_entry_point()` already separates entry file and function symbol.
- `solution.hash()` returns a deterministic content hash suitable for
  generating unique module names.
- `tests/sol_execbench/core/bench/test_eval_runtime.py` already covers
  `load_user_function()`.

### Established Patterns
- Native ROCm solution loading uses `spec_from_file_location()` for the compiled
  module.
- Python solution schemas require entry points to reference a source file with
  a `.py` suffix.

### Integration Points
- `load_user_function()` currently inserts staging into `sys.path`, converts the
  entry path to a dotted module name, and calls `importlib.import_module()`.
- The generated eval driver calls `load_user_function()` during staged
  evaluation.

</code_context>

<specifics>
## Specific Ideas

Keep the change scoped to runtime import mechanics and tests. Do not alter
solution JSON schema, trace contracts, or eval driver template behavior.

</specifics>

<deferred>
## Deferred Ideas

No broader sandboxing or import allowlist policy in this phase.

</deferred>
