# Phase 187: Profile Summary Producer and CLI Persistence - Context

**Gathered:** 2026-06-16
**Status:** Ready for planning

<domain>
The sidecar must be emitted trace-adjacent during CLI evaluation without changing benchmark execution, trace output, score semantics, raw profiler metadata, static evidence, or agent-feedback sidecars.
</domain>

<decisions>
- Persist profile summaries as `<trace>.profile-summary.json`.
- Keep `<trace>.profile.json` as the existing raw rocprofv3 metadata sidecar.
- Treat summary write failures as nonfatal warnings.
- Build normalized metadata from existing `Rocprofv3ProfileResult` objects only.
</decisions>

<code_context>
- CLI orchestration is in `src/sol_execbench/cli/main.py`.
- Existing `_write_profile_sidecar` persists raw profiler metadata.
- Existing `_evaluate_cli` already sequences profile sidecar, static evidence, environment, and agent feedback.
</code_context>

<specifics>
- Add path helper for `<trace>.profile-summary.json`.
- Add writer that computes run identity from the canonical trace when available.
- Ensure skipped, failed, unavailable, and artifact-empty profiles still serialize explicit diagnostic state.
</specifics>

<deferred>
No changes to GPU execution, scorer behavior, or profiler invocation are included in this phase.
</deferred>
