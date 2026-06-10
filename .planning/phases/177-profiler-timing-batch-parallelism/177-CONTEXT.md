# Phase 177: Profiler Timing Batch Parallelism - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

The profiler timing batch script stages problems in parallel CPU threads while keeping GPU profiling strictly serial, eliminating the manual multi-instance workflow and its timing bias. CPU-side staging (JSON parsing, ProblemPackager construction, temp directory setup) uses ThreadPoolExecutor while GPU subprocess calls remain strictly serial with architectural enforcement.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions.

</decisions>

<code_context>
## Existing Code Insights

Codebase context will be gathered during plan-phase research.

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. Refer to ROADMAP phase description and success criteria.

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase.

</deferred>
