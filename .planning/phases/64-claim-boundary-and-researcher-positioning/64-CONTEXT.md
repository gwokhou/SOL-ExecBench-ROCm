# Phase 64: Claim Boundary and Researcher Positioning - Context

**Gathered:** 2026-05-25
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Make the project's current claims, non-claims, evidence requirements, and
research positioning explicit and test-protected.
</domain>

<decisions>
## Implementation Decisions

- Treat claim wording as part of the public benchmark contract.
- Prefer one central `docs/CLAIMS.md` page over scattering claim rules across
  release notes.
- Add tests that check the most important allowed and forbidden claim language.
</decisions>

<code_context>
## Existing Code Insights

- Existing claim boundaries already appear in `docs/analysis.md`,
  `docs/original_parity.md`, and planning artifacts.
- Public contract guardrail tests already validate claim-boundary phrases in
  docs and reports.
</code_context>

<specifics>
## Specific Ideas

- Define allowed claim levels: ROCm-port evidence, runtime evidence, profiling
  evidence, AMD-native-derived evidence, and research-preview evidence.
- Define unsupported claims: B200 parity, upstream SOLAR equivalence, full
  paper validation, unvalidated hardware claims, and profiling-as-score.
</specifics>

<deferred>
## Deferred Ideas

- Do not build static ISA/RGA claim categories in this phase; keep them as
  future upgrade rules.
</deferred>

