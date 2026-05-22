# Phase 17: hip-execbench Practice Adaptation - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Update the `hip-execbench` practice map to reflect v1.3 adaptations and keep
accepted/rejected engineering experience explicit.
</domain>

<decisions>
## Implementation Decisions

- Accept trace-file baseline comparison as an additive SOL ExecBench ROCm
  workflow.
- Keep statistical significance and HTML reporting deferred until their data and
  dependency contracts are explicit.
- Continue rejecting schema, trace, and CLI replacement patterns from
  `hip-execbench`.

### the agent's Discretion
Practice-map wording and tests may be concise as long as accepted and rejected
decisions remain concrete.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/internal/hip_execbench_practice_map.md` already records v1.2 accepted and
  rejected practices.
- Phase 15 added the public baseline CLI that this phase should classify.

### Established Patterns
- Internal docs can be protected with focused file-content tests.

### Integration Points
- Update the practice map and add regression tests under `tests/sol_execbench/`.
</code_context>

<specifics>
## Specific Ideas

- Promote baseline comparison from deferred to accepted, but keep the narrower
  SOL ExecBench implementation boundaries.
</specifics>

<deferred>
## Deferred Ideas

- HTML/Plotly reports.
- Direct Mann-Whitney U integration without repeated-sample trace contracts.
</deferred>
