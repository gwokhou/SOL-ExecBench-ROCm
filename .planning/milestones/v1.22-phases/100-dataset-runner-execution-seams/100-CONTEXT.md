# Phase 100: Dataset Runner Execution Seams - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

<domain>
## Phase Boundary
Maintainers can evolve dataset-scale execution through importable runner helpers while `scripts/run_dataset.py` preserves existing user-facing behavior. Scope is limited to DATASET-01..04: runner abstraction, token/AST-aware source wrapping without global `stream` mutation, report/closure helper seams, and CLI-compatible scheduling/report seams.
</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
All implementation choices are at the agent's discretion - pure infrastructure phase. Preserve public CLI behavior, output filenames, sidecar contracts, and existing benchmark semantics. Prefer focused helper extraction under `src/sol_execbench/core/dataset/` with tests near existing dataset runner coverage.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/dataset/run_state.py`, `run_closure.py`, `evidence_refs.py`, `execution_closure.py`, etc.
- `scripts/run_dataset.py` is current orchestration script.
- Tests: `tests/sol_execbench/test_run_dataset_execution_closure.py`, `test_run_dataset_amd_score.py`, `test_dataset_run_state.py`, `test_dataset_run_closure.py`.

### Established Patterns
- Use snake_case, typed helpers, Pydantic/dataclasses as appropriate.
- Keep CLI as thin orchestration and push reusable logic into package helpers.
- Preserve compatibility and sidecar schemas unless explicitly required.

### Integration Points
- `scripts/run_dataset.py`
- `src/sol_execbench/core/dataset/`
- trace/closure/scoring sidecars under existing tests and docs.
</code_context>

<specifics>
## Specific Ideas
No specific requirements - infrastructure phase. Follow ROADMAP success criteria and keep changes narrow and test-backed.
</specifics>

<deferred>
## Deferred Ideas
- Hard sandbox/multi-tenant execution.
- CDNA3/MI300X/CDNA4 or native-host validation.
- Paper-scale parity and leaderboard readiness.
</deferred>
