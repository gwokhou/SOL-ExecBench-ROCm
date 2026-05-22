# Phase 12: Scoring and Baseline Comparison Review - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Review `hip-execbench` scoring and baseline-comparison practices and adapt only
guardrails that improve SOL ExecBench ROCm interpretation without creating
unsupported AMD performance claims.
</domain>

<decisions>
## Implementation Decisions

- Keep the existing `sol_score` formula unchanged.
- Add interpretation metadata and warnings around AMD-native performance claims.
- Treat richer baseline comparison as future scope unless explicitly approved.
</decisions>

<code_context>
## Existing Code Insights

`src/sol_execbench/sol_score.py` contains the existing score formula. The
project docs already warn that AMD-native scoring or roofline interpretation is
future work.
</code_context>

<specifics>
## Specific Ideas

- Add a small scoring guardrail module.
- Add tests proving the existing score formula remains unchanged.
- Add tests for AMD performance claim warning behavior.
</specifics>

<deferred>
## Deferred Ideas

- New baseline-comparison CLI.
- Statistical significance reports as public output.
- AMD-native roofline model.
</deferred>
