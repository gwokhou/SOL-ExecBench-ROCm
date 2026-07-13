# Phase 15: AMD Scoring and Baseline Workflow - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Make baseline comparison usable from public trace JSONL files and clarify how
SOL-Score-style values may be interpreted on AMD hardware.
</domain>

<decisions>
## Implementation Decisions

- Preserve existing `sol-execbench` evaluation CLI behavior.
- Add baseline comparison as an adjacent public command instead of changing
  trace JSONL or evaluation output.
- Match candidate and baseline traces by `definition` plus `workload.uuid`.
- Use explicit WIN/PARITY/LOSS thresholds inspired by `hip-execbench`, without
  importing its TypeScript pipeline or report schema.
- Emit warnings for attempted AMD-native performance claims unless a dedicated
  AMD interpretation model is documented.

### the agent's Discretion
The exact module names and output formatting are flexible as long as the trace
schema remains unchanged and tests cover contract stability.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Trace` Pydantic models can parse existing JSONL lines.
- `core/scoring_guardrails.py` already centralizes AMD claim warnings.
- `core/reporting.py` uses pure helpers that do not mutate trace objects.

### Established Patterns
- Public CLI entry points are Click commands exposed through `pyproject.toml`.
- Tests use `CliRunner` and temporary JSON fixtures.

### Integration Points
- New baseline helpers belong in `src/sol_execbench/core/`.
- New public command belongs under `src/sol_execbench/cli/`.
- User-facing workflow belongs in `docs/internal/analysis.md`.
</code_context>

<specifics>
## Specific Ideas

- Provide text and JSON output.
- Preserve trace JSONL as the source artifact.
- Guard against unsupported AMD hardware-performance claims.
</specifics>

<deferred>
## Deferred Ideas

- Statistical significance testing can be added later if repeated trace samples
  become a public requirement.
- CDNA 3 hardware-performance claims remain out of scope.
</deferred>
