# Phase 125: Compliance And Attribution Documentation - Context

**Gathered:** 2026-06-02
**Status:** Ready for planning
**Mode:** Autonomous from roadmap, Phase 123 policy, and Phase 124 cleanup

<domain>

## Phase Boundary

Align public compliance, attribution, release, and research-preview
documentation with the provenance policy.

</domain>

<decisions>

## Implementation Decisions

- `docs/user/provenance.md` is the canonical provenance policy.
- `docs/user/compliance.md` should summarize the license and attribution boundary.
- README and release/research materials should link the provenance policy.
- Public wording must avoid implying NVIDIA or AMD endorsement.

</decisions>

<code_context>

## Existing Code Insights

The relevant public docs are `README.md`, `docs/user/compliance.md`,
`docs/user/research_preview.md`, `docs/user/public_prerelease.md`, and
`docs/releases/v1_26_prerelease_draft.md`.

</code_context>

<deferred>

## Deferred Ideas

Automated gate enforcement remains Phase 126.

</deferred>
