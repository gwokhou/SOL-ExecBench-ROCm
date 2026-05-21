# Phase 10: Practice Harvest and Adaptation Map - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Convert the `hip-execbench` inspection into a concrete adaptation map that
identifies which practices are safe, useful, and compatible with SOL ExecBench
ROCm's existing benchmark contracts.
</domain>

<decisions>
## Implementation Decisions

- Treat `hip-execbench` as an engineering-practice reference, not a replacement
  architecture.
- Preserve public CLI behavior, Pydantic schemas, trace JSONL output, examples,
  and benchmark semantics.
- Accept internal-only practices around diagnostics, reporting transforms,
  structured failure surfaces, and comparison discipline.
- Reject or defer practices that would introduce new public commands, schemas,
  or agent JSON contracts.
</decisions>

<code_context>
## Existing Code Insights

SOL ExecBench ROCm is a Python package with Click/Rich CLI entry points,
Pydantic v2 data contracts, a staging `ProblemPackager`, and subprocess eval
drivers. The sibling `hip-execbench` project is TypeScript and has useful
internal modules for profiler routing, structured errors, agent/report builders,
and baseline comparisons.
</code_context>

<specifics>
## Specific Ideas

- Write an adaptation map under `docs/internal/`.
- Use the map to guide later phases toward pure helpers, tests, and docs.
- Keep CDNA 3 real hardware validation deferred.
</specifics>

<deferred>
## Deferred Ideas

- New public benchmark subcommands.
- Wholesale schema or CLI replacement.
- Agent-specific JSON output as a replacement for trace JSONL.
- Real CDNA 3 `gfx94*` validation.
</deferred>
