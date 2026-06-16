# Phase 189: HIP Profile Summary Fixtures and Docs - Context

**Gathered:** 2026-06-16
**Status:** Ready for planning

<domain>
HIP Playground needs stable SOL-side fixtures and ingestion guidance to implement adapter tests for profile-summary sidecars without depending on ROCm hardware.
</domain>

<decisions>
- Provide fixtures for valid, partial, unavailable, stale, malformed, missing, and contradictory-authority cases.
- Keep fixtures deterministic and free of raw profiler dumps, raw trace rows, source code, and absolute temporary paths.
- Document HIP mapping as diagnostic adapter input, not benchmark authority.
</decisions>

<code_context>
- Fixture package belongs under `tests/sol_execbench/fixtures/profile_summary/`.
- HIP-facing docs belong in `docs/profile_summary_sidecar.md`.
- CPU-safe fixture validation belongs under `tests/sol_execbench/`.
</code_context>

<specifics>
- Include explicit safe unknown handling language.
- Explain mapping into consumer-side `ProfileDigest`-style inputs.
- Document that v1 does not include occupancy/register/LDS/bandwidth/cache/utilization bottleneck diagnosis.
</specifics>

<deferred>
Hardware-counter-derived bottleneck docs and fixtures remain deferred until profiler taxonomy and thresholds are defined.
</deferred>
