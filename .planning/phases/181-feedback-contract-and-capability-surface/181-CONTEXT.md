# Phase 181: Feedback Contract and Capability Surface - Context

**Gathered:** 2026-06-16
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase adds the contract-level discovery surface for optional SOL
agent-feedback/profile-summary sidecars. It does not implement sidecar
generation, freshness validation, HIP adapter behavior, or any canonical Trace
JSONL schema change.

</domain>

<decisions>
## Implementation Decisions

### Contract Boundary
- Add optional capability tokens only; keep `contract_version` at `1.0`.
- Preserve existing `trace_field_requirements`, correctness fields, timing
  fields, scoring fields, and evaluation status vocabulary byte-for-byte.
- Add source-boundary statements that define feedback/profile-summary sidecars
  as diagnostic metadata only.
- Do not introduce new runtime dependencies.

### Documentation
- Add a dedicated evaluator-contract document describing capability discovery,
  canonical-trace non-changes, and authority boundaries.
- Cross-reference the new document from existing developer documentation.
- Document that HIP owns adapter normalization, `ProfileDigest`, strategy
  hints, and runtime prompt assembly.

### the agent's Discretion
All naming details are at the agent's discretion as long as capability tokens
are explicit, versioned, and stable for HIP v1.26 planning.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/data/contract.py` already builds the GPU-free
  evaluator contract.
- `tests/sol_execbench/test_contract.py` already protects version stability,
  optional capabilities, and canonical trace field groups.
- `docs/trace.md` and `docs/DEVELOPMENT.md` already explain canonical trace and
  diagnostic sidecar boundaries.

### Established Patterns
- Optional diagnostic evidence capabilities are advertised without bumping the
  semantic contract version.
- Existing tests assert optional sidecars do not enter canonical correctness,
  timing, or scoring fields.

### Integration Points
- `sol-execbench contract --json` is the public discovery command.
- HIP should consume the contract as external JSON and should not redefine SOL
  benchmark truth.

</code_context>

<specifics>
## Specific Ideas

Use capability tokens for `agent_feedback.sidecar.v1` and
`profile_summary.sidecar.v1`. Treat these as optional support signals only.

</specifics>

<deferred>
## Deferred Ideas

- Sidecar schema and generator implementation is Phase 182.
- Freshness identity and stale-state validation is Phase 183.
- Governance fixtures and HIP-facing examples are Phases 184-185.

</deferred>
