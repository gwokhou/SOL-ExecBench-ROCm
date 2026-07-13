# Phase 122: Public Publishing Materials - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning
**Mode:** Autonomous smart discuss, documentation/release-materials phase

<domain>
## Phase Boundary

This phase delivers public publishing materials for the v1.26 engineering prerelease and research preview. It prepares a GitHub prerelease draft or equivalent release-page source, links the artifact bundle/readiness/research-preview/support docs, and keeps wording bounded to the project's actual evidence.

</domain>

<decisions>
## Implementation Decisions

### Public Release Material
- Provide a Markdown release draft that maintainers can paste into GitHub Releases or adapt for another public release page.
- Include explicit placeholders for artifact bundle and readiness report links.
- Link the support matrix, claim boundaries, first-run guide, timing semantics, researcher guide, research preview, known limitations, artifact bundle, and readiness gate docs.

### Wording Boundary
- Use "engineering prerelease and research preview".
- Avoid stable benchmark authority, paper parity, upstream SOLAR parity, hosted leaderboard readiness, hard sandbox authority, completed MI300X validation, or CDNA4 validation claims.
- Keep MI300X as the concrete CDNA3 `gfx942` hardware target, not a separate architecture peer.

### Integration
- Add a public publishing guide and release draft under `docs/`.
- Update README navigation.
- Add focused docs tests.

### the agent's Discretion
- Exact file names and release draft structure are at the agent's discretion as long as PUBLISH-01 through PUBLISH-03 are covered.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/internal/v1_25_release_notes.md` and `docs/internal/v1_25_prerelease_checklist.md` provide earlier prerelease shape.
- `docs/internal/prerelease_artifact_bundle.md`, `docs/internal/prerelease_readiness.md`, and `docs/user/research_preview.md` provide the new v1.26 publication evidence.
- README already has a documentation index.

### Established Patterns
- Public docs use concise Markdown with explicit claim boundaries.
- Tests directly inspect docs for required links and forbidden/allowed wording.

### Integration Points
- Add `docs/user/public_prerelease.md`.
- Add `docs/releases/v1_26_prerelease_draft.md`.
- Update README docs list.
- Add `tests/sol_execbench/test_public_prerelease_docs.py`.

</code_context>

<specifics>
## Specific Ideas

- Include a pre-publication checklist: generate bundle, run readiness, attach/link artifacts, verify claims.
- Include a link checklist for required public docs.
- Include a "Not claimed" section in the release draft.

</specifics>

<deferred>
## Deferred Ideas

- Actually creating a GitHub Release remains a maintainer action outside this repo edit.
- Uploading artifacts to remote storage is not performed by this phase.

</deferred>
