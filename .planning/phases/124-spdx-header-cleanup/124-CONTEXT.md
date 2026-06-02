# Phase 124: SPDX Header Cleanup - Context

**Gathered:** 2026-06-02
**Status:** Ready for planning
**Mode:** Autonomous from Phase 123 policy and manifest

<domain>

## Phase Boundary

Apply provenance-based SPDX/copyright cleanup across active files without
removing required upstream notices or misattributing independent ROCm work.

</domain>

<decisions>

## Implementation Decisions

- Use `provenance.toml` as the source of truth for this phase.
- Files in `nvidia_notice.allowed` keep NVIDIA attribution and receive project
  attribution as derivative-modified candidates.
- Files in `nvidia_notice.cleanup_candidates` replace NVIDIA-only attribution
  with project attribution.
- Preserve `SPDX-License-Identifier: Apache-2.0`.
- Preserve shebang lines in scripts.

</decisions>

<code_context>

## Existing Code Insights

Phase 123 tests currently verify all NVIDIA header files are classified. Phase
124 must update those tests so allowed files require NVIDIA plus project
attribution, while cleanup candidates require project attribution and no NVIDIA
file copyright line.

</code_context>

<deferred>

## Deferred Ideas

Release gate integration remains Phase 126.

</deferred>
