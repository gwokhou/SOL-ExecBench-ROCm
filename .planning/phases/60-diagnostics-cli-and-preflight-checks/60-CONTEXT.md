# Phase 60: Diagnostics CLI and Preflight Checks - Context

**Gathered:** 2026-05-25
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Phase 60 adds a standalone diagnostics command and lightweight preflight checks
on top of the Phase 58/59 environment evidence layer. It must run without a
problem directory or solution.
</domain>

<decisions>
## Implementation Decisions

- Add `sol-execbench doctor --json` as the public diagnostic command.
- Keep benchmark invocation defaults unchanged.
- Emit machine-readable JSON only for v1.13; human formatting can be added
  later if needed.
- GPU smoke checks are explicit and best-effort: unavailable hardware/tools
  return structured statuses instead of failing the command.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/milestones/v1.13-REQUIREMENTS.md`
- `src/sol_execbench/core/environment.py`
- `src/sol_execbench/cli/main.py`
- `tests/conftest.py`
- `docs/rocm.md`
</canonical_refs>

