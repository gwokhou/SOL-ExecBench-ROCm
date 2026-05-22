# Phase 13: Public Contract Compatibility Guardrails - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Prove that v1.2 internal improvements did not break supported public schemas,
CLI usage, examples, trace JSONL contracts, or CDNA 3 validation deferral
language.
</domain>

<decisions>
## Implementation Decisions

- Add tests around existing public contracts rather than changing contracts.
- Preserve documented HIP-facing examples.
- Keep CDNA 3 validation deferral visible in project docs and handoff material.
</decisions>

<code_context>
## Existing Code Insights

The public contract is defined by Pydantic data models, Click help/options,
example paths, and trace JSONL serialization. Existing docs include
`.planning/CDNA3-VALIDATION-HANDOFF.md`.
</code_context>

<specifics>
## Specific Ideas

- Test representative solution, workload, and trace model round-trips.
- Test current CLI help includes existing options and no new `diagnose`
  subcommand.
- Test public HIP example paths remain HIP-facing.
- Test CDNA 3 deferred language remains present.
</specifics>

<deferred>
## Deferred Ideas

- Hardware validation.
- Public contract expansion.
</deferred>
