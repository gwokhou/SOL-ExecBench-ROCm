# Phase 178: Derived Script Parallelism - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

The derived isolation script runs multiple problem subprocesses concurrently via ThreadPoolExecutor, improving throughput for CPU-bound derived sidecar generation without affecting GPU correctness. Thread-safe JSONL writes, preserved --resume/--continue-on-failure semantics, and configurable --jobs flag.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase.

</decisions>

<code_context>
## Existing Code Insights

Codebase context will be gathered during plan-phase research.

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase.

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase.

</deferred>
