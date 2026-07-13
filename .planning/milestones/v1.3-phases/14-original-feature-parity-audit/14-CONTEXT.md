# Phase 14: Original Feature Parity Audit - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Compare NVIDIA SOL ExecBench's public functionality against this ROCm port and
turn the comparison into a maintained project artifact.
</domain>

<decisions>
## Implementation Decisions

- Treat upstream NVIDIA support as the reference for public surfaces, not a
  runtime compatibility target.
- Classify each original solution category instead of flattening all NVIDIA
  paths into a generic unsupported bucket.
- Protect the comparison with lightweight tests so future documentation changes
  do not erase known gaps or scope boundaries.

### the agent's Discretion
The exact document location and test shape are at the agent's discretion as long
as they preserve public contract clarity.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `README.md`, `docs/user/solution.md`, `docs/user/trace.md`, and `docs/user/compliance.md`
  already document ROCm public surfaces.
- `tests/sol_execbench/test_public_contract_guardrails.py` establishes the
  local pattern for documentation and public-contract checks.

### Established Patterns
- Documentation-only guardrails are tested with simple file-content assertions.
- NVIDIA runtime paths are described as unsupported or compatibility examples
  rather than silently removed.

### Integration Points
- New parity documentation belongs under `docs/`.
- Tests belong under `tests/sol_execbench/`.
</code_context>

<specifics>
## Specific Ideas

- Include CLI, dataset runner, data download, schemas, traces, examples,
  SOL-Score, and solution categories.
- Explicitly exclude CDNA 3 hardware validation.
</specifics>

<deferred>
## Deferred Ideas

- Implementing AMD scoring and baseline workflow belongs to Phase 15.
- ROCm library category readiness belongs to Phase 16.
</deferred>
