# Phase 11: ROCm Diagnostics and Failure Reporting - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Improve ROCm operator clarity by adapting suitable diagnostic routing,
structured error, and reporting ideas while preserving current command and file
contracts.
</domain>

<decisions>
## Implementation Decisions

- Add internal helper modules rather than new CLI flags or trace fields.
- Keep diagnostics pure and testable through dependency injection.
- Use stage-aware failure messages for future call sites without changing
  current subprocess behavior.
</decisions>

<code_context>
## Existing Code Insights

`ProblemPackager` already detects local gfx targets for offload flags.
`cli/main.py` emits user-facing compile/eval messages. Trace output is a stable
Pydantic JSONL contract. Internal helpers can live under `sol_execbench.core`.
</code_context>

<specifics>
## Specific Ideas

- Add ROCm tool readiness helpers.
- Add profiler backend selection as descriptive readiness metadata.
- Add pure trace summary helpers for local reporting.
</specifics>

<deferred>
## Deferred Ideas

- Wiring diagnostics into a new public command.
- Changing normal trace JSONL output.
</deferred>
