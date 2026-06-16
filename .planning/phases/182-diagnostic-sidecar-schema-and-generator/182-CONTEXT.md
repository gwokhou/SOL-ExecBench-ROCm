# Phase 182: Diagnostic Sidecar Schema and Generator - Context

**Gathered:** 2026-06-16
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase defines and writes the minimal strict
`sol_execbench.agent_feedback.v1` diagnostic sidecar beside trace outputs. It
does not implement freshness identity checks, artifact checksums, HIP adapter
logic, or governance fixture suites beyond local schema/authority validation.

</domain>

<decisions>
## Implementation Decisions

### Schema Scope
- Use strict frozen Pydantic models with `extra="forbid"`.
- Model authority fields as literal diagnostic-only/false flags.
- Summarize canonical trace statuses and optional profile/static evidence
  status without embedding raw trace rows, profiler dumps, source, or prompt
  text.

### CLI Integration
- Write `trace.jsonl.agent-feedback.json` only when an explicit trace output
  path exists.
- Keep sidecar write failures nonfatal, matching profile/static evidence
  sidecar behavior.
- Avoid new CLI flags; the sidecar is optional trace-adjacent metadata.

### the agent's Discretion
Recommendation wording and SOL-side bottleneck labels may be conservative and
minimal in this phase. HIP owns downstream taxonomy mapping.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_write_profile_sidecar` and `_write_static_evidence_sidecar` in
  `src/sol_execbench/cli/main.py` provide nonfatal writer patterns.
- `StaticKernelEvidenceSidecar` shows strict diagnostic-only authority fields.
- Existing CLI helper tests in `test_cli_environment_snapshot.py` cover sidecar
  path and payload helpers without GPU execution.

### Established Patterns
- Optional sidecars are persisted beside output traces using
  `output_file.with_name(f"{output_file.name}.<suffix>.json")`.
- Diagnostic helper failures print warnings and never change evaluation status.

### Integration Points
- `build_agent_feedback_sidecar()` receives already-parsed traces and optional
  profile/static evidence metadata.
- `_evaluate_cli()` writes the sidecar after canonical trace/profile/static
  sidecars are handled.

</code_context>

<specifics>
## Specific Ideas

Keep the first schema small: status, reason code, authority, summary, items,
limitations, and compact source refs. Phase 183 will extend identity/citation
richness.

</specifics>

<deferred>
## Deferred Ideas

- Freshness identity and checksum references are Phase 183.
- Contradictory-authority fixture matrix is Phase 184.
- HIP-facing fixture package is Phase 185.

</deferred>
