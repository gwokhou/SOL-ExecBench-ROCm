# Phase 186: Profile Summary Contract and Schema - Context

**Gathered:** 2026-06-16
**Status:** Ready for planning

<domain>
v1.37 turns `profile_summary.sidecar.v1` from a reserved future capability into a concrete optional diagnostic artifact. The sidecar must be safe for HIP Playground consumers while preserving canonical Trace JSONL and the existing rocprofv3 metadata sidecar as separate authorities.
</domain>

<decisions>
- The schema version is `sol_execbench.profile_summary.v1`.
- The evaluator capability token is `profile_summary.sidecar.v1`.
- Profile summaries are diagnostic-only and cannot act as correctness, timing, performance, score, evidence-tier, release-gate, cutover, paper-parity, leaderboard, or claim-upgrade authority.
- The raw `<trace>.profile.json` rocprofv3 metadata remains the lower-level profiler metadata artifact; the new sidecar is normalized and bounded.
</decisions>

<code_context>
- CLI capability documentation lives in `docs/EVALUATOR-CONTRACT.md`.
- Existing profile metadata is represented by `Rocprofv3ProfileResult`.
- New schema and builder code belongs near benchmark/profile helpers under `src/sol_execbench/core/bench/`.
- Existing sidecar tests are in `tests/sol_execbench/test_cli_environment_snapshot.py` and contract tests are in `tests/sol_execbench/test_contract.py`.
</code_context>

<specifics>
- Need strict Pydantic models with bounded statuses, reason codes, identity, authority, metrics, limitations, and artifact citations.
- Need tests proving trace rows and existing profile metadata semantics do not change.
- Need documentation to stop describing `profile_summary.sidecar.v1` as only reserved future capability.
</specifics>

<deferred>
Profiler-counter-derived bottleneck taxonomy remains out of scope for v1.37 and is tracked as future requirements PDIAG-F01/PDIAG-F02.
</deferred>
