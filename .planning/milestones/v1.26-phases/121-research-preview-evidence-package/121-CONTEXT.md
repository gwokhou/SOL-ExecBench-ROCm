# Phase 121: Research Preview Evidence Package - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning
**Mode:** Autonomous smart discuss, documentation/evidence phase

<domain>
## Phase Boundary

This phase delivers a researcher-facing evidence package that explains how to interpret the v1.26 prerelease as a bounded research preview. It documents methodology, benchmark scope, hardware scope, evidence surfaces, representative commands, produced artifacts, and explicit non-claims.

</domain>

<decisions>
## Implementation Decisions

### Research Interpretation
- Present v1.26 as engineering prerelease plus research preview, not a stable benchmark authority release.
- Keep Trace JSONL as the canonical run artifact.
- Distinguish AMD-derived SOL/score evidence from upstream SOLAR parity and paper-scale validation.
- Treat release validation, readiness reports, profiles, static evidence, Matrix reports, closure, consistency, claim-upgrade, and trust summaries as bounded sidecar evidence.

### Hardware Scope
- State that RDNA4 has recorded prerelease evidence where archived artifacts exist.
- State that MI300X is the concrete CDNA3 `gfx942` target, not a separate architecture target.
- State that full MI300X validation remains deferred and CDNA4 validation remains unavailable.

### Traceability
- Link representative commands for first-run trace generation, release validation, artifact bundle generation, readiness gates, and bounded dataset slices.
- Map commands to expected artifacts and their authority class.

### the agent's Discretion
- The report layout and exact section names are at the agent's discretion as long as RESEARCH-01 through RESEARCH-03 are covered.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/RESEARCHER-GUIDE.md` already explains researcher workflows and artifact interpretation.
- `docs/CLAIMS.md`, `docs/rocm.md`, `docs/prerelease_artifact_bundle.md`, and `docs/prerelease_readiness.md` now define the core public boundaries.
- `scripts/build_prerelease_artifact_bundle.py` and `scripts/check_prerelease_readiness.py` provide representative release evidence commands.

### Established Patterns
- Research-facing docs use tables that separate what artifacts prove from what they do not prove.
- Guardrail tests read docs directly and assert required wording.

### Integration Points
- Add `docs/research_preview.md`.
- Link it from `docs/RESEARCHER-GUIDE.md` and prerelease docs.
- Add a focused docs test for research preview coverage.

</code_context>

<specifics>
## Specific Ideas

- Include a command-to-artifact traceability table.
- Include a scope table separating canonical, diagnostic-only, provisional, deferred, and unavailable evidence.
- Include clear non-claims for full 235-problem paper validation, upstream SOLAR parity, leaderboard readiness, hard sandboxing, full MI300X validation, and CDNA4 validation.

</specifics>

<deferred>
## Deferred Ideas

- Public prerelease page assembly belongs to Phase 122.
- Running new dataset or hardware validation is out of scope.

</deferred>
