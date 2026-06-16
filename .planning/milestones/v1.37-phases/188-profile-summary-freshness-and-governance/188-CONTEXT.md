# Phase 188: Profile Summary Freshness and Governance - Context

**Gathered:** 2026-06-16
**Status:** Ready for planning

<domain>
HIP consumers need enough identity and citations to reject stale or contradictory profile summaries while preserving canonical trace authority.
</domain>

<decisions>
- Include trace path, generated timestamp, contract version, optional run id, and citation checksums.
- Use compact artifact paths so prompt-facing summaries do not leak raw temp directories.
- Classify stale, malformed, missing, unavailable, partial, and contradictory-authority sidecars as diagnostic states.
- Authority flags default to diagnostic-only and cannot be set to promote benchmark claims.
</decisions>

<code_context>
- Profile-summary model and validator helpers live in `src/sol_execbench/core/bench/profile_summary.py`.
- CLI sidecar tests can construct local trace/profile/artifact files without requiring ROCm.
</code_context>

<specifics>
- Need freshness validator coverage for matching and stale identities.
- Need governance guardrail coverage for contradictory authority flags.
- Need artifact citations for trace, profile metadata, and profiler artifacts.
</specifics>

<deferred>
Profile-summary freshness does not validate source-level identity or replace canonical trace validation.
</deferred>
